#!/usr/bin/env python3
"""
Offline generator: collision-free Shadow Hand poses.

Samples random joint configurations via RobotLoader, filters out poses
with self-collisions (MuJoCo ncon > 0), and saves valid poses to NPZ.

Usage:
    python generate_valid_robot_poses.py
    python generate_valid_robot_poses.py --n_target 1000000 --batch 2000

Output:
    grasp-model/data/processed/valid_robot_poses.npz
        q  : float32 [N, 24]  joint angles in RobotLoader chain order
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "grasp-model/scripts"))

from robot_loader import RobotLoader

URDF_PATH = Path("/home/yareeez/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf")
MJCF_PATH = REPO_ROOT / "third_party/mujoco_menagerie/shadow_hand/right_hand.xml"
OUT_PATH  = REPO_ROOT / "grasp-model/data/processed/valid_robot_poses.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_target", type=int, default=100_000,
                   help="Number of valid poses to collect (default: 100k for quick test)")
    p.add_argument("--batch",    type=int, default=2_000,
                   help="Poses sampled per iteration")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--out",      type=str, default=None,
                   help="Override output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader = RobotLoader(str(URDF_PATH))
    model  = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data   = mujoco.MjData(model)

    valid:         list[np.ndarray] = []
    total_sampled: int = 0
    t0 = time.time()
    last_print = t0
    LOG_INTERVAL = 30.0  # seconds

    print(f"Target: {args.n_target:,} valid poses")
    print(f"Batch:  {args.batch:,} per iteration")
    print(f"Output: {out_path}")
    print()

    while len(valid) < args.n_target:
        q, _ = loader.sample_q(args.batch, seed=None)
        q_np = q.cpu().numpy().astype(np.float32)
        total_sampled += args.batch

        for i in range(args.batch):
            data.qpos[:] = q_np[i]
            mujoco.mj_forward(model, data)
            if data.ncon == 0:
                valid.append(q_np[i])
            if len(valid) >= args.n_target:
                break

        now = time.time()
        n_valid = len(valid)
        if now - last_print >= LOG_INTERVAL or n_valid >= args.n_target:
            elapsed   = now - t0
            rate      = 100.0 * n_valid / max(1, total_sampled)
            eta_poses = args.n_target - n_valid
            eta_sec   = elapsed / max(1, n_valid) * eta_poses
            print(
                f"  valid={n_valid:>8,} / {args.n_target:,}  "
                f"sampled={total_sampled:>8,}  "
                f"acceptance={rate:.1f}%  "
                f"elapsed={elapsed:.0f}s  "
                f"eta={eta_sec:.0f}s"
            )
            last_print = now

    q_valid = np.stack(valid[: args.n_target], axis=0)  # [N, 24]
    np.savez_compressed(out_path, q=q_valid)

    elapsed = time.time() - t0
    acceptance = 100.0 * args.n_target / total_sampled
    print()
    print(f"Done.")
    print(f"  saved:       {out_path}")
    print(f"  shape:       {q_valid.shape}")
    print(f"  total sampled: {total_sampled:,}")
    print(f"  acceptance:  {acceptance:.1f}%")
    print(f"  time:        {elapsed:.0f}s")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  file size:   {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
