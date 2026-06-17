"""human_to_udhm -- the human path to the UDHM 22-slot vector (corrected stage-3).

Peer of robot_primitives.robot_to_udhm. The human has only landmarks, so joint
angles are INFERRED from geometry: landmarks -> Dong stage-2 (positions ->
per-joint rotations) -> here (rotations -> named angles). This function is the
"rotations -> angles" step.

It mirrors udhm_run_stage3's contract (same labels, same UDHM22 slots, same /pi
normalization) but fixes the angle math to be SIGNED, matching the UDHM paper
(Fang et al.), where every coordinate is a signed DoF (abduction/adduction,
flexion/extension):
  - MCP (2-DoF): the old stage-3 used ``arccos(R[i,i])`` -> range [0, pi], which
    DROPS the sign (abduction could only ever be positive). Here we use the signed
    ``atan2`` of the off-diagonal term instead -> the abduction/adduction direction
    is preserved. Magnitude matches the old arccos; the sign is recovered.
  - PIP/DIP (1-DoF hinge): already signed in stage-3 (atan2); kept identical.

Convention (same as udhm_run_stage3 / Dong & Payandeh Eq. 24, frame Ry(flex)Rz(abd)):
    R[2,2] = cos(flex), R[0,2] = sin(flex)   -> flex = atan2(R[0,2], R[2,2])
    R[1,1] = cos(abd),  R[1,0] = sin(abd)    -> abd  = atan2(R[1,0], R[1,1])

The existing udhm_run_stage3 is left untouched; this is a separate, clean path so
human and robot converge on the SAME signed UDHM convention.
"""
from __future__ import annotations

import math

import torch

from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS, _FILL, _SLOT_IDX, _pose_to_matrix

# Radial-side fingers (at -Y of the palm lateral axis): their away-from-middle
# abduction azimuth is negative, so flip it to "away = +". Fixed by anatomy.
_RADIAL_FINGERS = {"index", "thumb"}


def human_to_udhm(pose: torch.Tensor, joint_labels: list[str]) -> torch.Tensor:
    """Dong stage-2 per-joint rotations -> signed UDHM 22-slot vector.

    Args:
        pose:         [B, N, F] per-joint rotations (F=4 quat wxyz or F=6 rot6),
                      wrist-local, as produced by dong_run_stage2.
        joint_labels: length-N "{finger}_{joint}" labels (e.g. "index_mcp").

    Returns:
        [B, 22] signed named angles, radians normalized by pi. Slot order =
        UDHM22_SLOTS. Unmapped joints and always-zero slots stay 0.
    """
    if pose.dim() != 3:
        raise ValueError(f"pose must be [B, N, F], got {tuple(pose.shape)}")
    if pose.shape[1] != len(joint_labels):
        raise ValueError(f"joint_labels ({len(joint_labels)}) must match pose N ({pose.shape[1]})")

    R = _pose_to_matrix(pose)  # [B, N, 3, 3]
    out = pose.new_zeros(pose.shape[0], len(UDHM22_SLOTS))

    for n, label in enumerate(joint_labels):
        fill = _FILL.get(label)
        if fill is None:  # e.g. "*_tip" -- no DoF
            continue
        Rn = R[:, n]  # [B, 3, 3]
        if "abd" in fill:  # 2-DoF MCP-type root: SIGNED (was arccos, unsigned)
            # Dong's MCP frame R = [Xi | Yi | Zi], Xi = bone (column 0). The old
            # arccos read |abd| = bone azimuth in the palm plane and |flex| = bone
            # elevation out of it. Signed versions of the SAME quantities:
            xi_x, xi_y, xi_z = Rn[:, 0, 0], Rn[:, 1, 0], Rn[:, 2, 0]
            abd = torch.atan2(xi_y, xi_x)                         # azimuth, signed
            flex = torch.atan2(xi_z, torch.hypot(xi_x, xi_y))     # elevation, signed
            # Abduction convention = "away from middle finger = +" (DexGrasp). The
            # palm frame's lateral axis Y = (ring_mcp - index_mcp), so radial-side
            # fingers (index, thumb) sit at -Y: their away-from-middle azimuth is
            # negative -> flip them so spreading away is positive, like the ulnar
            # fingers (ring, pinky, +Y side). Anatomy is fixed, so this is a constant.
            if label.split("_")[0] in _RADIAL_FINGERS:
                abd = -abd
            out[:, _SLOT_IDX[fill["flex"]]] = flex
            out[:, _SLOT_IDX[fill["abd"]]] = abd
        else:  # 1-DoF hinge (PIP/DIP): signed, identical to stage-3
            out[:, _SLOT_IDX[fill["flex"]]] = torch.atan2(Rn[:, 0, 2], Rn[:, 0, 0])

    return out / math.pi
