"""
Dong-style kinematics for robot FK output (PyTorch, batched).

Implements Dong & Payandeh (2025) Blocks 1 + 3 applied to URDF FK link positions.
Input: world-frame joint positions from pytorch_kinematics FK.
Output: wxyz quaternions (w >= 0) per joint, in wrist-local frame.

This is the robot-side counterpart of human/kinematics/dong_kinematics.py.
Different framework (torch vs numpy), different input (FK vs MediaPipe),
no calibration (robots are deterministic from URDF).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_kinematics as pk

from cross_emb.rotations import matrix_to_rot6d

EPS = 1e-8


def _dong_normalize(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, dim=-1, eps=EPS)


def _dong_mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """[B, 3, 3] -> [B, 4] wxyz with w >= 0."""
    q = pk.matrix_to_quaternion(R)
    q = torch.where(q[:, :1] < 0, -q, q)
    return q


def _dong_rotation_y(beta: torch.Tensor) -> torch.Tensor:
    """Ry(beta) for each sample. beta: [B] -> [B, 3, 3]."""
    c = beta.cos()
    s = beta.sin()
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    row0 = torch.stack([ c, z, s], dim=-1)
    row1 = torch.stack([ z, o, z], dim=-1)
    row2 = torch.stack([-s, z, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _dong_block1_wrist_frame(
    wrist_pos: torch.Tensor,
    index_mcp: torch.Tensor,
    middle_mcp: torch.Tensor,
    ring_mcp: torch.Tensor,
) -> torch.Tensor:
    """Dong Eq. 5-7: FK positions -> wrist frame R_wrist [B, 3, 3]."""
    Y0 = _dong_normalize(ring_mcp - index_mcp)
    v  = _dong_normalize(middle_mcp - wrist_pos)
    Z0 = _dong_normalize(torch.linalg.cross(v, Y0))
    X0 = _dong_normalize(torch.linalg.cross(Y0, Z0))
    return torch.stack([X0, Y0, Z0], dim=-1)


def _dong_world_to_local(p: torch.Tensor, wrist_pos: torch.Tensor, R_wrist: torch.Tensor) -> torch.Tensor:
    diff = p - wrist_pos
    return torch.bmm(R_wrist.transpose(-1, -2), diff.unsqueeze(-1)).squeeze(-1)


def _dong_block3_mcp(mcp_local: torch.Tensor, pip_local: torch.Tensor) -> torch.Tensor:
    """Dong Eq. 20-22: MCP frame [B, 3, 3]."""
    Xi = _dong_normalize(pip_local - mcp_local)
    Z0 = torch.zeros_like(Xi); Z0[..., 2] = 1.0
    Yi = _dong_normalize(torch.linalg.cross(Z0, Xi))
    Zi = _dong_normalize(torch.linalg.cross(Xi, Yi))
    return torch.stack([Xi, Yi, Zi], dim=-1)


def _dong_block3_pip_dip(
    mcp_local: torch.Tensor,
    pip_local: torch.Tensor,
    dip_local: torch.Tensor,
    tip_local: torch.Tensor,
    R_mcp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dong Eq. 29-34: PIP and DIP flexion angles. Returns R_pip, R_dip [B,3,3]."""
    R_mcp_T = R_mcp.transpose(-1, -2)

    def to_mcp(p: torch.Tensor) -> torch.Tensor:
        return torch.bmm(R_mcp_T, (p - mcp_local).unsqueeze(-1)).squeeze(-1)

    dj  = to_mcp(pip_local)
    dj1 = to_mcp(dip_local)
    dj2 = to_mcp(tip_local)

    xj   = dj1 - dj
    xjm1 = dj
    cos_pip = (xj * xjm1).sum(-1) / (xj.norm(dim=-1).clamp(min=EPS) * xjm1.norm(dim=-1).clamp(min=EPS))
    beta_pip = torch.arccos(cos_pip.clamp(-1 + EPS, 1 - EPS))

    xj1 = dj2 - dj1
    cos_dip = (xj1 * xj).sum(-1) / (xj1.norm(dim=-1).clamp(min=EPS) * xj.norm(dim=-1).clamp(min=EPS))
    beta_dip = torch.arccos(cos_dip.clamp(-1 + EPS, 1 - EPS))

    return _dong_rotation_y(beta_pip), _dong_rotation_y(beta_dip)


def _dong_block3_pip_only(
    mcp_local: torch.Tensor,
    pip_local: torch.Tensor,
    tip_local: torch.Tensor,
    R_mcp: torch.Tensor,
) -> torch.Tensor:
    """PIP angle for 2-link fingers. Returns R_pip [B,3,3]."""
    R_mcp_T = R_mcp.transpose(-1, -2)

    def to_mcp(p: torch.Tensor) -> torch.Tensor:
        return torch.bmm(R_mcp_T, (p - mcp_local).unsqueeze(-1)).squeeze(-1)

    dj  = to_mcp(pip_local)
    dj1 = to_mcp(tip_local)
    xj   = dj1 - dj
    xjm1 = dj
    cos_pip = (xj * xjm1).sum(-1) / (xj.norm(dim=-1).clamp(min=EPS) * xjm1.norm(dim=-1).clamp(min=EPS))
    beta_pip = torch.arccos(cos_pip.clamp(-1 + EPS, 1 - EPS))
    return _dong_rotation_y(beta_pip)


def dong_run_stage2(
    fk_out: dict[str, torch.Tensor],
    config: dict,
) -> tuple[torch.Tensor, list[str], dict]:
    """
    Dong Block 1 + Block 3 applied to FK link positions.

    Args:
        fk_out : {link_name: Tensor[B, 4, 4]} from RobotLoader.run_fk()
        config : hand config dict (from YAML via _load_hand_config)

    Returns:
        quats        : Tensor[B, N, 4]  wxyz quaternions (w >= 0)
        joint_labels : list[str]        label for each slot
        meta         : dict             R_wrist, per-finger rotations,
                                        tips [B,F,3] (wrist-local, unnormalized),
                                        tip_labels list[str],
                                        chain_positions {finger: [B, L, 3]}
    """
    def pos(link: str) -> torch.Tensor:
        return fk_out[link][:, :3, 3]

    wrist_pos  = pos(config["wrist_link"])
    index_mcp  = pos(config["frame_index_mcp"])
    middle_mcp = pos(config["frame_middle_mcp"])
    ring_mcp   = pos(config["frame_ring_mcp"])

    R_wrist = _dong_block1_wrist_frame(wrist_pos, index_mcp, middle_mcp, ring_mcp)

    def local(link: str) -> torch.Tensor:
        return _dong_world_to_local(pos(link), wrist_pos, R_wrist)

    quats_list: list[torch.Tensor] = []
    rot6_list: list[torch.Tensor] = []
    labels: list[str] = []
    finger_meta: dict = {}
    tips_list: list[torch.Tensor] = []
    tip_labels: list[str] = []
    chain_positions: dict[str, torch.Tensor] = {}

    for finger_name, finger_cfg in config["fingers"].items():
        chain = finger_cfg["chain"]
        n = len(chain)
        if n < 2:
            continue

        chain_local = [local(link) for link in chain]
        chain_positions[finger_name] = torch.stack(chain_local, dim=1)  # [B, L, 3]

        mcp_l = chain_local[0]
        pip_l = chain_local[1]
        tip_l = chain_local[-1]
        tips_list.append(tip_l)
        tip_labels.append(finger_name)

        R_mcp = _dong_block3_mcp(mcp_l, pip_l)
        q_mcp = _dong_mat_to_quat(R_mcp)
        quats_list.append(q_mcp)
        rot6_list.append(matrix_to_rot6d(R_mcp))
        labels.append(f"{finger_name}_mcp")

        if n == 2:
            finger_meta[finger_name] = {"R_mcp": R_mcp}
            continue

        if n == 3:
            R_pip = _dong_block3_pip_only(mcp_l, pip_l, tip_l, R_mcp)
            q_pip = _dong_mat_to_quat(R_pip)
            quats_list.append(q_pip)
            rot6_list.append(matrix_to_rot6d(R_pip))
            labels.append(f"{finger_name}_pip")
            finger_meta[finger_name] = {"R_mcp": R_mcp, "R_pip": R_pip}
            continue

        dip_l = local(chain[2])
        R_pip, R_dip = _dong_block3_pip_dip(mcp_l, pip_l, dip_l, tip_l, R_mcp)
        q_pip = _dong_mat_to_quat(R_pip)
        q_dip = _dong_mat_to_quat(R_dip)
        quats_list.append(q_pip)
        rot6_list.append(matrix_to_rot6d(R_pip))
        labels.append(f"{finger_name}_pip")
        quats_list.append(q_dip)
        rot6_list.append(matrix_to_rot6d(R_dip))
        labels.append(f"{finger_name}_dip")
        finger_meta[finger_name] = {"R_mcp": R_mcp, "R_pip": R_pip, "R_dip": R_dip}

    quats = torch.stack(quats_list, dim=1)
    rot6  = torch.stack(rot6_list, dim=1)
    tips  = torch.stack(tips_list, dim=1)

    meta = {
        "R_wrist": R_wrist,
        "fingers": finger_meta,
        "wrist_pos": wrist_pos,
        "tips": tips,
        "tip_labels": tip_labels,
        "chain_positions": chain_positions,
        "rot6": rot6,
    }
    return quats, labels, meta


# Alias kept for backward compat with scripts that import _dong_run_stage2 directly.
_dong_run_stage2 = dong_run_stage2
