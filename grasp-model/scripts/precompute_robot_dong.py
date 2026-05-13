#!/usr/bin/env python3
"""
Precompute robot Dong features (quats, chain positions, fingertip positions)
for an existing valid_robot_poses NPZ. Output is a new NPZ with the same `q`
plus extra arrays so RobotLoader skips runtime FK + stage2 in `sample_dong`.

Input  NPZ schema:
    q              [N, J]      float32   (untouched, byte-identical in output)

Output NPZ schema:
    q              [N, J]      float32
    quats          [N, K, 4]   float32   Dong quaternions per joint (wxyz, w>=0)
    chain          [N, F, 4, 3] float32  chain link positions, wrist-local, /hand_length
    tips           [N, F, 3]   float32   fingertip positions, wrist-local, /hand_length
    mcp_angles     [N, F, 2]   float32   Dong Eq.24: [beta, gamma] per finger, radians
    pip_angles     [N, F]      float32   Dong PIP flexion beta per finger, radians
    dip_angles     [N, F]      float32   Dong DIP flexion beta per finger, radians (0 if no DIP)
    joint_labels   [K]         str       label per quaternion slot
    tip_labels     [F]         str       finger names (order matches mcp/pip/dip_angles)
    hand_config    str                   resolved path used at generation time
    hand_length    float32                hand_length used to normalize

Usage:
    python grasp-model/scripts/precompute_robot_dong.py \
        --input  grasp-model/data/processed/valid_robot_poses_eigengrasp.npz \
        --urdf   /home/yareeez/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf \
        --hand_config grasp-model/data/hand_configs/shadow_hand_right.yaml \
        --batch_size 20000

The original NPZ is left untouched. Output is written next to the input with
suffix `_dong.npz` unless `--output` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to existing q-only NPZ.")
    p.add_argument("--urdf", required=True, help="Path to robot URDF.")
    p.add_argument("--hand_config", required=True, help="Path to hand YAML config.")
    p.add_argument("--output", default=None, help="Output NPZ. Default: <input>_dong.npz.")
    p.add_argument("--batch_size", type=int, default=20000, help="Per-batch FK size.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit", type=int, default=None, help="Optional cap on poses.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root / "grasp-model/scripts"))

    from robot_loader import RobotLoader, _dong_run_stage2, _load_hand_config

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else in_path.with_name(in_path.stem + "_dong_euler.npz")
    )
    if out_path.exists():
        print(f"[precompute] WARNING: overwriting {out_path}")

    print(f"[precompute] input  = {in_path}")
    print(f"[precompute] output = {out_path}")
    print(f"[precompute] device = {device}")

    data = np.load(in_path)
    q_all = data["q"]
    if args.limit is not None:
        q_all = q_all[: args.limit]
    N, J = q_all.shape
    print(f"[precompute] poses={N:,}  J={J}  batch_size={args.batch_size}")

    loader = RobotLoader(args.urdf, device=device)
    config = _load_hand_config(args.hand_config)

    # Probe to discover label sets and Dong tensor shapes for this robot/config.
    with torch.no_grad():
        q_probe = torch.from_numpy(q_all[:1]).to(device)
        fk_probe = loader.run_fk(q_probe)
        quats_probe, joint_labels, meta_probe = _dong_run_stage2(fk_probe, config)
        hand_length = loader._get_hand_length(config)
        tip_labels = list(meta_probe["tip_labels"])
        K = quats_probe.shape[1]
        F = len(tip_labels)
    print(f"[precompute] joint_labels({K})={joint_labels}")
    print(f"[precompute] tip_labels({F})={tip_labels}")
    print(f"[precompute] hand_length={hand_length:.6f}")

    quats_out      = np.zeros((N, K, 4),    dtype=np.float32)
    chain_out      = np.zeros((N, F, 4, 3), dtype=np.float32)
    tips_out       = np.zeros((N, F, 3),    dtype=np.float32)
    mcp_angles_out = np.zeros((N, F, 2),    dtype=np.float32)  # [beta, gamma] per finger
    pip_angles_out = np.zeros((N, F),       dtype=np.float32)
    dip_angles_out = np.zeros((N, F),       dtype=np.float32)

    n_batches = (N + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        i = bi * args.batch_size
        j = min(i + args.batch_size, N)
        q_batch = torch.from_numpy(q_all[i:j]).to(device)
        with torch.no_grad():
            fk    = loader.run_fk(q_batch)
            quats, _, meta = _dong_run_stage2(fk, config)
            tips  = meta["tips"] / hand_length
            chain_per_finger = {
                f: meta["chain_positions"][f] / hand_length for f in tip_labels
            }
        quats_out[i:j] = quats.cpu().numpy()
        tips_out[i:j]  = tips.cpu().numpy()
        for fi, f in enumerate(tip_labels):
            chain_out[i:j, fi] = chain_per_finger[f].cpu().numpy()
            beta_mcp, gamma_mcp = meta["mcp_angles"][f]
            mcp_angles_out[i:j, fi, 0] = beta_mcp.cpu().numpy()
            mcp_angles_out[i:j, fi, 1] = gamma_mcp.cpu().numpy()
            pip_angles_out[i:j, fi] = meta["pip_angles"][f].cpu().numpy()
            dip_angles_out[i:j, fi] = meta["dip_angles"][f].cpu().numpy()
        if bi % max(1, n_batches // 50) == 0 or bi == n_batches - 1:
            pct = 100.0 * (j) / N
            print(f"[precompute] batch {bi+1:4d}/{n_batches}  poses {j:>10,}/{N:,}  ({pct:5.1f}%)")

    print(f"[precompute] writing {out_path} ...")
    np.savez(
        out_path,
        q             = q_all.astype(np.float32, copy=False),
        quats         = quats_out,
        chain         = chain_out,
        tips          = tips_out,
        mcp_angles    = mcp_angles_out,
        pip_angles    = pip_angles_out,
        dip_angles    = dip_angles_out,
        joint_labels  = np.array(joint_labels),
        tip_labels    = np.array(tip_labels),
        hand_config   = np.array(str(Path(args.hand_config).expanduser().resolve())),
        hand_length   = np.array(hand_length, dtype=np.float32),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[precompute] done. size={size_mb:,.1f} MB")


if __name__ == "__main__":
    main()
