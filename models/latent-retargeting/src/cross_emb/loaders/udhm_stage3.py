"""UDHM stage-3: decompose stage-2 per-joint rotations into the 22-slot
named per-axis angle vector (UDHM, Fang et al. 2026).

Peer of ``dong_run_stage2`` (dong_math.py): a PURE, embodiment-agnostic function.
Input  = stage-2 per-joint rotations (quat wxyz or rot6) + their labels.
Output = [B, 22] named angles, radians, normalized by pi -> ~[-1, 1].

The 22-slot contract (slot order + conventions) lives in
``robot/hand-configs/udhm_canonical_22dof.yaml``. ``UDHM22_SLOTS`` below mirrors
that order; ``test_udhm_stage3.py`` guards that they do not drift.

Angle math (matches Dong & Payandeh Eq. 24, read from the joint rotation):
  - MCP frame R = [Xi, Yi, Zi] columns, wrist-local (Z = palm normal n_p):
      flex (beta)  = arccos(Zi_z) = arccos(R[2, 2])   # rotation toward palm, e_lat
      abd  (gamma) = arccos(Yi_y) = arccos(R[1, 1])   # rotation around n_p
  - PIP/DIP are pure Ry(beta) hinges: flex = atan2(R[0, 2], R[0, 0]) (signed).
  - TIP / unmapped labels are ignored.

NOT yet consumed by S_k -- this only PRODUCES the vector. Wiring into the metric
is a separate step.
"""
from __future__ import annotations

import math

import torch

from cross_emb.rotations import quat_wxyz_to_matrix, rot6d_to_matrix

# 22 canonical slot names, in order. MIRRORS udhm_canonical_22dof.yaml (guarded by test).
UDHM22_SLOTS: list[str] = [
    "thumb_cmc_flex", "thumb_cmc_spread", "thumb_mcp_flex", "thumb_mcp_abd", "thumb_ip_flex",
    "index_mcp_abd", "index_mcp_flex", "index_pip_flex", "index_dip_flex",
    "middle_mcp_abd", "middle_mcp_flex", "middle_pip_flex", "middle_dip_flex",
    "ring_mcp_abd", "ring_mcp_flex", "ring_pip_flex", "ring_dip_flex",
    "pinky_mcp_abd", "pinky_mcp_flex", "pinky_pip_flex", "pinky_dip_flex", "pinky_twist",
]
_SLOT_IDX: dict[str, int] = {name: i for i, name in enumerate(UDHM22_SLOTS)}

# Fill map: stage-2 label "{finger}_{joint}" -> {component: slot_name}.
# "flex" is present for every joint; "abd" only for 2-DoF MCP-type roots.
# THUMB nuance: Dong's "thumb_mcp" is the CMC (landmark 1) -> fills thumb_cmc_*;
# "thumb_pip" is the anatomical MCP; "thumb_dip" is the IP.
# thumb_mcp_abd (slot 3) and pinky_twist (slot 21) stay zero -- Dong does not model them.
_FILL: dict[str, dict[str, str]] = {
    "thumb_mcp":  {"flex": "thumb_cmc_flex",  "abd": "thumb_cmc_spread"},
    "thumb_pip":  {"flex": "thumb_mcp_flex"},
    "thumb_dip":  {"flex": "thumb_ip_flex"},
    "index_mcp":  {"flex": "index_mcp_flex",  "abd": "index_mcp_abd"},
    "index_pip":  {"flex": "index_pip_flex"},
    "index_dip":  {"flex": "index_dip_flex"},
    "middle_mcp": {"flex": "middle_mcp_flex", "abd": "middle_mcp_abd"},
    "middle_pip": {"flex": "middle_pip_flex"},
    "middle_dip": {"flex": "middle_dip_flex"},
    "ring_mcp":   {"flex": "ring_mcp_flex",   "abd": "ring_mcp_abd"},
    "ring_pip":   {"flex": "ring_pip_flex"},
    "ring_dip":   {"flex": "ring_dip_flex"},
    "pinky_mcp":  {"flex": "pinky_mcp_flex",  "abd": "pinky_mcp_abd"},
    "pinky_pip":  {"flex": "pinky_pip_flex"},
    "pinky_dip":  {"flex": "pinky_dip_flex"},
}

_ACOS_EPS = 1e-7  # keep arccos input strictly inside (-1, 1) for finite gradients


def _pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """[B, N, F] (F=4 quat wxyz, or F=6 rot6) -> [B, N, 3, 3] rotation matrices."""
    f = pose.shape[-1]
    if f == 4:
        return quat_wxyz_to_matrix(pose)
    if f == 6:
        return rot6d_to_matrix(pose)
    raise ValueError(f"pose last dim must be 4 (quat) or 6 (rot6), got {f}")


def udhm_run_stage3(pose: torch.Tensor, joint_labels: list[str]) -> torch.Tensor:
    """Decompose stage-2 per-joint rotations into the 22-slot named-angle vector.

    Args:
        pose:         [B, N, F] per-joint rotations (F=4 quat wxyz or F=6 rot6),
                      wrist-local, as produced by stage-2 (dong_run_stage2 / Dong CSV).
        joint_labels: length-N list of "{finger}_{joint}" labels for each slot.

    Returns:
        [B, 22] named angles in radians, normalized by pi. Slot order = UDHM22_SLOTS.
        Unmapped joints (e.g. "*_tip") and the always-zero slots (thumb_mcp_abd,
        pinky_twist) are left at 0.
    """
    if pose.dim() != 3:
        raise ValueError(f"pose must be [B, N, F], got shape {tuple(pose.shape)}")
    if pose.shape[1] != len(joint_labels):
        raise ValueError(
            f"joint_labels ({len(joint_labels)}) must match pose N ({pose.shape[1]})"
        )

    B = pose.shape[0]
    R = _pose_to_matrix(pose)  # [B, N, 3, 3]
    out = pose.new_zeros(B, len(UDHM22_SLOTS))

    for n, label in enumerate(joint_labels):
        fill = _FILL.get(label)
        if fill is None:  # e.g. "*_tip" -- no DoF
            continue
        Rn = R[:, n]  # [B, 3, 3]
        if "abd" in fill:  # 2-DoF MCP-type root: Dong Eq. 24
            flex = torch.arccos(Rn[:, 2, 2].clamp(-1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS))
            abd = torch.arccos(Rn[:, 1, 1].clamp(-1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS))
            out[:, _SLOT_IDX[fill["flex"]]] = flex
            out[:, _SLOT_IDX[fill["abd"]]] = abd
        else:  # 1-DoF hinge (PIP/DIP): pure Ry(beta), signed angle
            out[:, _SLOT_IDX[fill["flex"]]] = torch.atan2(Rn[:, 0, 2], Rn[:, 0, 0])

    return out / math.pi
