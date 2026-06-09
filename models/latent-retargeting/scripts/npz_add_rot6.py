#!/usr/bin/env python3
"""
Add rot6 field to an existing DONG_CACHE npz.

Reads existing quats [N, K, 4] (wxyz, w>=0), converts via
quat_wxyz_to_matrix -> matrix_to_rot6d, writes rot6 [N, K, 6].
No URDF or FK required -- pure numpy/torch conversion.

Usage:
    python models/latent-retargeting/scripts/npz_add_rot6.py \
        --input  robot/shadow-hand/datasets/processed/valid_robot_poses_eigengrasp_dong.npz \
        --output robot/shadow-hand/datasets/processed/valid_robot_poses_eigengrasp_dong_r6.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """[N, K, 4] wxyz -> [N, K, 3, 3]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 6] first two columns."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None, help="Default: <input stem>_r6.npz")
    p.add_argument("--batch", type=int, default=500_000, help="Rows per batch for memory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path  = Path(args.input).expanduser().resolve()
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else in_path.with_name(in_path.stem + "_r6.npz")
    )
    print(f"[npz_add_rot6] input  = {in_path}")
    print(f"[npz_add_rot6] output = {out_path}")

    data = np.load(in_path)
    print(f"[npz_add_rot6] fields: {list(data.files)}")
    quats = data["quats"]                         # [N, K, 4]
    N, K, _ = quats.shape
    print(f"[npz_add_rot6] N={N:,}  K={K}")

    rot6_out = np.empty((N, K, 6), dtype=np.float32)
    n_batches = (N + args.batch - 1) // args.batch
    for bi in range(n_batches):
        i = bi * args.batch
        j = min(i + args.batch, N)
        R = quat_wxyz_to_matrix(quats[i:j])       # [B, K, 3, 3]
        rot6_out[i:j] = matrix_to_rot6d(R)        # [B, K, 6]
        if bi % max(1, n_batches // 20) == 0 or bi == n_batches - 1:
            print(f"[npz_add_rot6] {j:>10,}/{N:,}  ({100*j/N:.1f}%)")

    print(f"[npz_add_rot6] writing {out_path} ...")
    save_kwargs = {k: data[k] for k in data.files}
    save_kwargs["rot6"] = rot6_out
    np.savez(out_path, **save_kwargs)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[npz_add_rot6] done. size={size_mb:,.1f} MB")
    print(f"[npz_add_rot6] fields written: {list(save_kwargs.keys())}")


if __name__ == "__main__":
    main()
