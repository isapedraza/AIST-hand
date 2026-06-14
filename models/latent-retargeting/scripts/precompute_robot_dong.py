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
    rot6           [N, K, 6]   float32   6D rotation repr (Zhou 2019) per joint
    chain          [N, F, 4, 3] float32  chain link positions, wrist-local, /hand_length
    tips           [N, F, 3]   float32   fingertip positions, wrist-local, /hand_length
    joint_labels   [K]         str       label per quaternion slot
    tip_labels     [F]         str       finger names
    hand_config    str                   resolved path used at generation time
    hand_length    float32                hand_length used to normalize

Usage:
    python models/latent-retargeting/scripts/precompute_robot_dong.py \
        --input  robot/hands/shadow_hand/datasets/processed/valid_robot_poses_eigengrasp.npz \
        --urdf   /home/yareeez/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf \
        --hand_config robot/hand-configs/shadow.yaml \
        --batch_size 20000

The original NPZ is left untouched. Output is written next to the input with
suffix `_dong.npz` unless `--output` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cross_emb.loaders.robot_loader import RobotLoader, _dong_run_stage2, _load_hand_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to existing q-only NPZ.")
    p.add_argument("--urdf", required=True, help="Path to robot URDF.")
    p.add_argument("--hand_config", required=True, help="Path to hand YAML config.")
    p.add_argument("--output", default=None, help="Output NPZ. Default: <input>_dong.npz.")
    p.add_argument("--batch_size", type=int, default=20000, help="Per-batch FK size.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit", type=int, default=None, help="Optional cap on poses.")
    p.add_argument("--save-rot6", action="store_true",
                   help="Also store rot6 [N,K,6]. Default OFF: the loader recomputes rot6 "
                        "on-the-fly from quats (cache_keys has no rot6), so storing it is "
                        "dead weight (~528MB for shadow@1M).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else in_path.with_name(in_path.stem + "_dong.npz")
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
        # Chain link count varies per hand (leap/shadow=4 [MCP,PIP,DIP,TIP];
        # barrett=2 [med,dist]). Probe the max so chain_out fits any morphology.
        L = max(meta_probe["chain_positions"][f].shape[1] for f in tip_labels)
    print(f"[precompute] joint_labels({K})={joint_labels}")
    print(f"[precompute] tip_labels({F})={tip_labels}  chain_len(L)={L}")
    print(f"[precompute] hand_length={hand_length:.6f}")

    quats_out = np.zeros((N, K, 4),     dtype=np.float32)
    rot6_out  = np.zeros((N, K, 6),     dtype=np.float32) if args.save_rot6 else None
    chain_out = np.zeros((N, F, L, 3),  dtype=np.float32)
    tips_out  = np.zeros((N, F, 3),     dtype=np.float32)

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
        if rot6_out is not None:
            rot6_out[i:j] = meta["rot6"].cpu().numpy()
        tips_out[i:j]  = tips.cpu().numpy()
        for fi, f in enumerate(tip_labels):
            cpf = chain_per_finger[f].cpu().numpy()
            chain_out[i:j, fi, :cpf.shape[1]] = cpf   # pad shorter chains with zeros
        if bi % max(1, n_batches // 50) == 0 or bi == n_batches - 1:
            pct = 100.0 * (j) / N
            print(f"[precompute] batch {bi+1:4d}/{n_batches}  poses {j:>10,}/{N:,}  ({pct:5.1f}%)")

    print(f"[precompute] writing {out_path} ...")
    save_kw = dict(
        q             = q_all.astype(np.float32, copy=False),
        quats         = quats_out,
        chain         = chain_out,
        tips          = tips_out,
        joint_labels  = np.array(joint_labels),
        tip_labels    = np.array(tip_labels),
        hand_config   = np.array(str(Path(args.hand_config).expanduser().resolve())),
        hand_length   = np.array(hand_length, dtype=np.float32),
    )
    if rot6_out is not None:
        save_kw["rot6"] = rot6_out
    np.savez(out_path, **save_kw)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[precompute] done. size={size_mb:,.1f} MB")


if __name__ == "__main__":
    main()
