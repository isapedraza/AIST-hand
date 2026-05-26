"""Rotation representation helpers for cross-embodiment training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_wxyz_to_matrix(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert wxyz unit quaternions to rotation matrices.

    Args:
        q: Tensor with final dimension 4, ordered as ``[w, x, y, z]``.

    Returns:
        Tensor with shape ``q.shape[:-1] + (3, 3)``.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected final quaternion dim 4, got {q.shape[-1]}")
    q = F.normalize(q, dim=-1, eps=eps)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Represent a rotation matrix by its first two columns."""
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [...,3,3], got {tuple(R.shape)}")
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def quat_wxyz_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to the R6 representation."""
    return matrix_to_rot6d(quat_wxyz_to_matrix(q))


def rot6d_to_matrix(r6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map R6 vectors to SO(3) with Gram-Schmidt orthonormalization."""
    if r6.shape[-1] != 6:
        raise ValueError(f"Expected final R6 dim 6, got {r6.shape[-1]}")
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=eps)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def d_r_yan_quat(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """Yan-style quaternion D_R: sum_j 1 - <q_a, q_b>^2."""
    if q_a.shape != q_b.shape:
        raise ValueError(f"Shape mismatch: {tuple(q_a.shape)} vs {tuple(q_b.shape)}")
    q_a = F.normalize(q_a, dim=-1)
    q_b = F.normalize(q_b, dim=-1)
    dot = (q_a * q_b).sum(dim=-1)
    return (1.0 - dot.square()).sum(dim=-1)


def d_r_rot6d(r6_a: torch.Tensor, r6_b: torch.Tensor) -> torch.Tensor:
    """R6 version of Yan D_R, computed on SO(3) without quaternions.

    For relative angle theta, quaternion ``1 - dot^2`` equals
    ``sin^2(theta/2) = (1 - cos(theta)) / 2``. This function reconstructs
    rotation matrices from R6 and applies that equivalent expression.
    """
    if r6_a.shape != r6_b.shape:
        raise ValueError(f"Shape mismatch: {tuple(r6_a.shape)} vs {tuple(r6_b.shape)}")
    R_a = rot6d_to_matrix(r6_a)
    R_b = rot6d_to_matrix(r6_b)
    R_rel = torch.matmul(R_a.transpose(-1, -2), R_b)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return ((1.0 - cos_theta) * 0.5).sum(dim=-1)


def d_r_pose(pose_a: torch.Tensor, pose_b: torch.Tensor, rot_repr: str) -> torch.Tensor:
    """Dispatch Yan-style rotation distance for the active pose representation."""
    if rot_repr == "quat":
        return d_r_yan_quat(pose_a, pose_b)
    if rot_repr == "r6":
        return d_r_rot6d(pose_a, pose_b)
    raise ValueError(f"Unsupported rotation representation: {rot_repr}")
