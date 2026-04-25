"""
dong_stage2.py — Generic Dong-style kinematics (Stage 2) applied to FK link positions.

Implements Dong Block 1 + Block 3 in batched PyTorch, using a YAML hand config
that maps link names to anatomical roles. Works for any hand with a palm frame
and finger chains of length >= 2.

Block 1 (Dong Eq. 5-7):  FK positions -> wrist-local frame
Block 3 MCP (Eq. 19-22): finger root orientation in wrist frame (absolute)
Block 3 PIP (Eq. 29-33): PIP flexion angle Ry(beta) relative to MCP frame
Block 3 DIP (Eq. 31,34): DIP flexion angle Ry(beta) relative to PIP frame
TIP: identity (no DOF)

Input:
    fk_out : {link_name: Tensor[B, 4, 4]}  from robot_randomizer.py Stage 1
    config  : dict loaded from hand_configs/*.yaml

Output:
    quats       : Tensor[B, N, 4]  flat quaternion array (wxyz, w>=0)
    joint_labels: list[str]        name of each slot in the N dimension
    meta        : dict with R_wrist and per-finger data
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_kinematics as pk
import yaml


EPS = 1e-8


# ============================================================
# Geometry helpers (all batched: first dim = B)
# ============================================================

def _normalize(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, dim=-1, eps=EPS)


def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """[B, 3, 3] -> [B, 4] wxyz with w >= 0."""
    q = pk.matrix_to_quaternion(R)  # wxyz
    q = torch.where(q[:, :1] < 0, -q, q)
    return q


def _rotation_y(beta: torch.Tensor) -> torch.Tensor:
    """Ry(beta) for each sample. beta: [B] -> [B, 3, 3]."""
    c = beta.cos()
    s = beta.sin()
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    # column-major stack matching Dong Eq. 33
    row0 = torch.stack([ c, z, s], dim=-1)
    row1 = torch.stack([ z, o, z], dim=-1)
    row2 = torch.stack([-s, z, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # [B, 3, 3]


# ============================================================
# Block 1: wrist frame
# ============================================================

def block1_wrist_frame(
    wrist_pos: torch.Tensor,
    index_mcp: torch.Tensor,
    middle_mcp: torch.Tensor,
    ring_mcp: torch.Tensor,
) -> torch.Tensor:
    """
    Dong Eq. 5-7 (batched).

    Y0 = normalize(ring_mcp - index_mcp)
    Z0 = normalize(normalize(middle_mcp - wrist) x Y0)
    X0 = Y0 x Z0
    R_wrist = [X0 | Y0 | Z0]   columns, [B, 3, 3]
    """
    Y0 = _normalize(ring_mcp - index_mcp)
    v  = _normalize(middle_mcp - wrist_pos)
    Z0 = _normalize(torch.linalg.cross(v, Y0))
    X0 = _normalize(torch.linalg.cross(Y0, Z0))
    return torch.stack([X0, Y0, Z0], dim=-1)  # [B, 3, 3]


def world_to_local(p: torch.Tensor, wrist_pos: torch.Tensor, R_wrist: torch.Tensor) -> torch.Tensor:
    """Transform world positions to wrist-local frame. p: [B, 3] -> [B, 3]."""
    diff = p - wrist_pos
    return torch.bmm(R_wrist.transpose(-1, -2), diff.unsqueeze(-1)).squeeze(-1)


# ============================================================
# Block 3: finger kinematics
# ============================================================

def block3_mcp(mcp_local: torch.Tensor, pip_local: torch.Tensor) -> torch.Tensor:
    """
    Dong Eq. 20-22: MCP frame in wrist-local coords.

    Xi = normalize(pip - mcp)
    Yi = normalize(Z0_local x Xi)   where Z0_local = [0,0,1]
    Zi = normalize(Xi x Yi)
    R_mcp = [Xi | Yi | Zi]          [B, 3, 3]
    """
    Xi = _normalize(pip_local - mcp_local)
    Z0 = torch.zeros_like(Xi)
    Z0[..., 2] = 1.0
    Yi = _normalize(torch.linalg.cross(Z0, Xi))
    Zi = _normalize(torch.linalg.cross(Xi, Yi))
    return torch.stack([Xi, Yi, Zi], dim=-1)  # [B, 3, 3]


def block3_pip_dip(
    mcp_local: torch.Tensor,
    pip_local: torch.Tensor,
    dip_local: torch.Tensor,
    tip_local: torch.Tensor,
    R_mcp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dong Eq. 29-34: PIP and DIP flexion angles (batched).

    Returns R_pip [B,3,3], R_dip [B,3,3] — both Ry(beta).
    """
    R_mcp_T = R_mcp.transpose(-1, -2)  # [B, 3, 3]

    def to_mcp(p: torch.Tensor) -> torch.Tensor:
        diff = p - mcp_local
        return torch.bmm(R_mcp_T, diff.unsqueeze(-1)).squeeze(-1)

    dj  = to_mcp(pip_local)   # PIP in MCP frame [B,3]
    dj1 = to_mcp(dip_local)   # DIP in MCP frame
    dj2 = to_mcp(tip_local)   # TIP in MCP frame

    # PIP angle: between (DIP-PIP) and (PIP from MCP origin)
    xj   = dj1 - dj
    xjm1 = dj
    cos_pip = (xj * xjm1).sum(-1) / (xj.norm(dim=-1).clamp(min=EPS) * xjm1.norm(dim=-1).clamp(min=EPS))
    beta_pip = torch.arccos(cos_pip.clamp(-1 + EPS, 1 - EPS))

    # DIP angle: between (TIP-DIP) and (DIP-PIP)
    xj1 = dj2 - dj1
    cos_dip = (xj1 * xj).sum(-1) / (xj1.norm(dim=-1).clamp(min=EPS) * xj.norm(dim=-1).clamp(min=EPS))
    beta_dip = torch.arccos(cos_dip.clamp(-1 + EPS, 1 - EPS))

    return _rotation_y(beta_pip), _rotation_y(beta_dip)


def block3_pip_only(
    mcp_local: torch.Tensor,
    pip_local: torch.Tensor,
    tip_local: torch.Tensor,
    R_mcp: torch.Tensor,
) -> torch.Tensor:
    """PIP angle when DIP doesn't exist (2-link fingers). Returns R_pip [B,3,3]."""
    R_mcp_T = R_mcp.transpose(-1, -2)

    def to_mcp(p: torch.Tensor) -> torch.Tensor:
        return torch.bmm(R_mcp_T, (p - mcp_local).unsqueeze(-1)).squeeze(-1)

    dj  = to_mcp(pip_local)
    dj1 = to_mcp(tip_local)
    xj   = dj1 - dj
    xjm1 = dj
    cos_pip = (xj * xjm1).sum(-1) / (xj.norm(dim=-1).clamp(min=EPS) * xjm1.norm(dim=-1).clamp(min=EPS))
    beta_pip = torch.arccos(cos_pip.clamp(-1 + EPS, 1 - EPS))
    return _rotation_y(beta_pip)


# ============================================================
# Main Stage 2 entry point
# ============================================================

def run_stage2(
    fk_out: dict[str, torch.Tensor],
    config: dict,
) -> tuple[torch.Tensor, list[str], dict]:
    """
    Apply Dong Block 1 + Block 3 to FK link positions.

    Args:
        fk_out  : {link_name: Tensor[B, 4, 4]} from Stage 1
        config  : dict loaded from hand_configs/*.yaml

    Returns:
        quats        : Tensor[B, N, 4]  wxyz quaternions (w >= 0)
        joint_labels : list[str]        label for each of the N slots
        meta         : dict             R_wrist, per-finger rotations
    """
    B = next(iter(fk_out.values())).shape[0]
    device = next(iter(fk_out.values())).device

    def pos(link: str) -> torch.Tensor:
        return fk_out[link][:, :3, 3]  # [B, 3]

    # ---- Block 1 ----
    wrist_pos   = pos(config["wrist_link"])
    index_mcp   = pos(config["frame_index_mcp"])
    middle_mcp  = pos(config["frame_middle_mcp"])
    ring_mcp    = pos(config["frame_ring_mcp"])

    R_wrist = block1_wrist_frame(wrist_pos, index_mcp, middle_mcp, ring_mcp)  # [B,3,3]

    def local(link: str) -> torch.Tensor:
        return world_to_local(pos(link), wrist_pos, R_wrist)

    # ---- Block 3 per finger ----
    quats_list: list[torch.Tensor] = []
    labels: list[str] = []
    finger_meta: dict = {}

    for finger_name, finger_cfg in config["fingers"].items():
        chain = finger_cfg["chain"]
        n = len(chain)

        if n < 2:
            continue  # need at least MCP + PIP

        mcp_l = local(chain[0])
        pip_l = local(chain[1])

        # MCP orientation in wrist frame
        R_mcp = block3_mcp(mcp_l, pip_l)
        q_mcp = _mat_to_quat(R_mcp)
        quats_list.append(q_mcp)
        labels.append(f"{finger_name}_mcp")

        if n == 2:
            finger_meta[finger_name] = {"R_mcp": R_mcp}
            continue

        if n == 3:
            tip_l = local(chain[2])
            R_pip = block3_pip_only(mcp_l, pip_l, tip_l, R_mcp)
            q_pip = _mat_to_quat(R_pip)
            quats_list.append(q_pip)
            labels.append(f"{finger_name}_pip")
            finger_meta[finger_name] = {"R_mcp": R_mcp, "R_pip": R_pip}
            continue

        # n >= 4: MCP + PIP + DIP (+ optional TIP)
        dip_l = local(chain[2])
        tip_l = local(chain[3])
        R_pip, R_dip = block3_pip_dip(mcp_l, pip_l, dip_l, tip_l, R_mcp)
        q_pip = _mat_to_quat(R_pip)
        q_dip = _mat_to_quat(R_dip)
        quats_list.append(q_pip)
        labels.append(f"{finger_name}_pip")
        quats_list.append(q_dip)
        labels.append(f"{finger_name}_dip")
        finger_meta[finger_name] = {"R_mcp": R_mcp, "R_pip": R_pip, "R_dip": R_dip}

    quats = torch.stack(quats_list, dim=1)  # [B, N, 4]

    meta = {
        "R_wrist": R_wrist,
        "fingers": finger_meta,
        "wrist_pos": wrist_pos,
    }
    return quats, labels, meta


# ============================================================
# Config loader
# ============================================================

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
