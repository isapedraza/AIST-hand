#!/usr/bin/env python3
"""
Offline generator: collision-free Shadow Hand poses via eigengrasp sampling.

Samples joint configurations by drawing uniformly in eigengrasp coefficient
space [coeff_p01, coeff_p99], decoding to q24, and filtering self-collisions
(MuJoCo ncon > 0).

Usage:
    python generate_valid_robot_poses.py
    python generate_valid_robot_poses.py --n_target 1000000 --batch 2000

Output:
    robot/hands/shadow_hand/datasets/processed/valid_robot_poses.npz
        q  : float32 [N, 24]  joint angles in RobotLoader chain order
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

MJCF_PATH    = REPO_ROOT / "third_party/mujoco_menagerie/shadow_hand/right_hand.xml"
OUT_PATH     = REPO_ROOT / "robot/hands/shadow_hand/datasets/processed/valid_robot_poses.npz"
DEFAULT_EIGEN = (
    REPO_ROOT
    / "robot/hands/shadow_hand/datasets/processed"
    / "dexonomy_shadow_eigengrasps_balanced_phase_open_close_coeffstats_sample.npz"
)


def load_eigengrasps(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {"mean_norm", "components_norm", "joint_low", "joint_high", "coeff_p01", "coeff_p99"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    out = {k: data[k].astype(np.float64) for k in required}
    # coeff_mean/coeff_std needed for Gaussian sampling (probabilistic-PCA prior).
    for opt in ("coeff_mean", "coeff_std"):
        if opt in data.files:
            out[opt] = data[opt].astype(np.float64)
    return out


def sample_eigengrasp_batch(
    rng: np.random.Generator,
    eigen: dict[str, np.ndarray],
    n_knobs: int,
    batch_size: int,
    n_pad: int,
    sampling: str = "gaussian",
    trunc: float = 3.0,
) -> np.ndarray:
    """Sample in eigengrasp coefficient space -> decode to joint angles.

    sampling="gaussian": draw coeffs ~ N(coeff_mean, coeff_std), truncated at +-trunc sigma.
        This is the generative model implied by PCA (probabilistic PCA, Tipping & Bishop
        1999) and matches the real grasp density (clustered, not flat). Standard prior in
        synergy-based grasp samplers (Dimou et al. 2022, Mousavian et al. 2019).
    sampling="uniform": legacy U(coeff_p01, coeff_p99); over-represents extremes/corners.

    n_pad = MJCF.nq - n_eigengrasp_joints, prepended as zeros (robot-agnostic).
    Shadow: eigengrasp has 22 finger joints, MJCF nq=24 -> n_pad=2 (WRJ2/WRJ1=0).
    Allegro: eigengrasp has 16 finger joints, MJCF nq=16 -> n_pad=0 (no wrist).
    """
    if sampling == "gaussian":
        if "coeff_mean" not in eigen or "coeff_std" not in eigen:
            raise KeyError("Gaussian sampling needs coeff_mean/coeff_std in the NPZ.")
        mu  = eigen["coeff_mean"][:n_knobs]
        std = eigen["coeff_std"][:n_knobs]
        z = rng.standard_normal(size=(batch_size, n_knobs))
        z = np.clip(z, -trunc, trunc)                                # truncate tails
        coeffs = mu + z * std                                         # [B, n_knobs]
    elif sampling == "uniform":
        p01 = eigen["coeff_p01"][:n_knobs]
        p99 = eigen["coeff_p99"][:n_knobs]
        coeffs = rng.uniform(p01, p99, size=(batch_size, n_knobs))   # [B, n_knobs]
    else:
        raise ValueError(f"unknown sampling '{sampling}' (use gaussian|uniform)")
    mean   = eigen["mean_norm"]                                       # [J]
    comps  = eigen["components_norm"][:n_knobs]                       # [n_knobs, J]
    q_norm = mean + coeffs @ comps                                    # [B, J]
    q_norm = np.clip(q_norm, 0.0, 1.0)
    low    = eigen["joint_low"]
    high   = eigen["joint_high"]
    q_joints = (q_norm * (high - low) + low).astype(np.float32)       # [B, J]
    if n_pad > 0:
        pad = np.zeros((batch_size, n_pad), dtype=np.float32)
        return np.concatenate([pad, q_joints], axis=1)               # [B, J+n_pad]
    return q_joints                                                   # [B, J]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_target",    type=int,  default=100_000,
                   help="Number of valid poses to collect (default: 100k for quick test)")
    p.add_argument("--batch",       type=int,  default=2_000,
                   help="Poses sampled per iteration")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--n-knobs",     type=int,  default=9,
                   help="Number of eigengrasp components to sample (default: 9 = 91%% variance)")
    p.add_argument("--sampling",    choices=("gaussian", "uniform"), default="uniform",
                   help="Coeff sampling: uniform (even coverage of the reachable range, "
                        "for retargeter triplets) or gaussian (prob-PCA prior, matches grasp density)")
    p.add_argument("--trunc",       type=float, default=3.0,
                   help="Truncate Gaussian at +-trunc sigma (gaussian only)")
    p.add_argument("--contact-tol", type=float, default=0.0,
                   help="Self-penetration tolerance in m. 0 = accept only ncon==0 (clean "
                        "menagerie MJCF: leap/shadow). >0 = accept depth <= baseline+tol "
                        "(needed for auto-converted URDF with structural overlap: barrett).")
    p.add_argument("--eigengrasps", type=Path, default=DEFAULT_EIGEN,
                   help="Path to eigengrasps NPZ with coeff_p01/coeff_p99")
    p.add_argument("--mjcf",        type=Path, default=MJCF_PATH,
                   help="Path to robot MJCF for collision filtering (default: Shadow)")
    p.add_argument("--out",         type=str,  default=None,
                   help="Override output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eigen = load_eigengrasps(args.eigengrasps)
    rng   = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data  = mujoco.MjData(model)

    # Robot-agnostic wrist padding: MJCF qpos may have extra leading DOFs (Shadow
    # WRJ2/WRJ1) not present in the finger-only eigengrasp basis.
    n_eigen_joints = int(eigen["mean_norm"].shape[0])
    n_pad = model.nq - n_eigen_joints
    if n_pad < 0:
        raise ValueError(f"MJCF nq ({model.nq}) < eigengrasp joints ({n_eigen_joints})")
    print(f"MJCF: {args.mjcf}  nq={model.nq}  eigengrasp_joints={n_eigen_joints}  n_pad={n_pad}")

    # Collision acceptance. Clean menagerie MJCFs reach ncon==0 (use that). Auto-converted
    # URDFs (barrett) always report shallow structural contacts at adjacent-link joints, so
    # we accept poses whose deepest penetration stays within baseline+contact_tol.
    def max_penetration() -> float:
        if data.ncon == 0:
            return 0.0
        return -min(float(data.contact[i].dist) for i in range(data.ncon))

    if args.contact_tol > 0.0:
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        baseline = max_penetration()
        allow = baseline + args.contact_tol
        print(f"Collision: depth<= {allow*1000:.1f}mm (baseline {baseline*1000:.1f}mm + tol {args.contact_tol*1000:.1f}mm)")
    else:
        allow = None
        print("Collision: ncon==0 (clean MJCF)")

    def accept() -> bool:
        if allow is None:
            return data.ncon == 0
        return max_penetration() <= allow

    valid:         list[np.ndarray] = []
    total_sampled: int = 0
    t0 = time.time()
    last_print = t0
    LOG_INTERVAL = 30.0  # seconds

    print(f"Target:      {args.n_target:,} valid poses")
    print(f"Batch:       {args.batch:,} per iteration")
    print(f"Eigengrasps: {args.eigengrasps}")
    print(f"Knobs:       {args.n_knobs}")
    print(f"Sampling:    {args.sampling}" + (f" (trunc +-{args.trunc}sigma)" if args.sampling == "gaussian" else ""))
    print(f"Output:      {out_path}")
    print()

    while len(valid) < args.n_target:
        q_np = sample_eigengrasp_batch(rng, eigen, args.n_knobs, args.batch, n_pad,
                                       sampling=args.sampling, trunc=args.trunc)
        total_sampled += args.batch

        for i in range(args.batch):
            data.qpos[:] = q_np[i]
            mujoco.mj_forward(model, data)
            if accept():
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
