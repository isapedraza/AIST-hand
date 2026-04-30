from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_EIGENGRASPS = Path(
    "grasp-model/data/processed/"
    "dexonomy_shadow_eigengrasps_balanced_sample.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path, help="Dexonomy .npy file containing qpos arrays [N,29].")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--qpos-key", default="grasp_qpos")
    parser.add_argument("--eigengrasps", type=Path, default=DEFAULT_EIGENGRASPS)
    parser.add_argument("--k", type=int, nargs="*", default=[6, 9, 12, 17, 22])
    args = parser.parse_args()

    target = np.load(args.target, allow_pickle=True).item()
    eigen = np.load(args.eigengrasps, allow_pickle=False)

    if args.qpos_key not in target:
        available = ", ".join(sorted(str(key) for key in target.keys()))
        raise KeyError(f"Missing {args.qpos_key}. Available keys: {available}")
    target_qpos = target[args.qpos_key]
    if len(target_qpos.shape) != 2 or target_qpos.shape[1] != 29:
        raise ValueError(f"Expected {args.qpos_key} [N,29], got {target_qpos.shape}")
    if args.row < 0 or args.row >= target_qpos.shape[0]:
        raise IndexError(f"row={args.row} out of bounds for {args.qpos_key} with {target_qpos.shape[0]} rows")

    q22 = target_qpos[args.row, 7:].astype(np.float64)
    low = eigen["joint_low"].astype(np.float64)
    high = eigen["joint_high"].astype(np.float64)
    mean = eigen["mean_norm"].astype(np.float64)
    components = eigen["components_norm"].astype(np.float64)
    names = [str(name) for name in eigen["joint_names"]]

    qnorm = (q22 - low) / (high - low)
    centered = qnorm - mean

    print(f"target={args.target}")
    print(f"qpos_key={args.qpos_key}")
    print(f"row={args.row} rows={target_qpos.shape[0]}")
    print("q22=" + " ".join(f"{value:+.4f}" for value in q22))

    for k in args.k:
        coeffs = centered @ components[:k].T
        recon_norm = mean + coeffs @ components[:k]
        recon = np.clip(recon_norm, 0.0, 1.0) * (high - low) + low
        err = recon - q22

        print()
        print(f"k={k}")
        print("coeffs=" + " ".join(f"{value:+.4f}" for value in coeffs))
        print(
            f"mae={np.mean(np.abs(err)):.5f} "
            f"rmse={np.sqrt(np.mean(err * err)):.5f} "
            f"max_abs={np.max(np.abs(err)):.5f}"
        )
        order = np.argsort(np.abs(err))[::-1][:8]
        print("worst=" + ", ".join(f"{names[i]}:{err[i]:+.4f}" for i in order))
        print("q22_recon=" + " ".join(f"{value:+.4f}" for value in recon))


if __name__ == "__main__":
    main()
