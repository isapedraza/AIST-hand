#!/usr/bin/env python3
"""
Build Barrett Hand eigengrasps (PCA) from the MultiDex (GenDexGrasp) dataset.

MultiDex stores each grasp as (qpos[17], object_name, robot_name) where
qpos = [trans(3), rot6d(6), joints(8)]; joints at [9:17].

The MultiDex barrett joint order/signs do NOT match the dex-urdf bhand_model.urdf.
We remap data -> urdf (verified visually + by joint-range structure, 2026-06-13):

    data per-finger order = (med, dist, spread); urdf = (spread, med, dist).
    urdf_qpos = [ -d2, -d0, -d1, +d5, -d3, -d4, -d6, -d7 ]

(data joints are all >=0; urdf med/dist are [-,0] so negate; spread f1 negate, f2 keep.)
GATE A (range check) is an objective test of the remap signs: wrong signs -> values
fall outside urdf limits.

MultiDex has only final grasps (no phases), so open/close come 100% from synthetic
anchors (--synthetic-qpos-npz, key qpos8), unlike Shadow/Leap 3-phase pipelines.

Usage:
    python build_barrett_eigengrasps.py \
        --multidex robot/hands/barrett_hand/datasets/raw/MultiDex_barrett.pt \
        --out robot/hands/barrett_hand/datasets/processed/eigengrasp_barrett.npz \
        [--synthetic-qpos-npz open.npz close.npz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Canonical joint order = dex-urdf bhand_model.urdf qpos order.
JOINTS8 = [
    "finger_1_prox_joint", "finger_1_med_joint", "finger_1_dist_joint",
    "finger_2_prox_joint", "finger_2_med_joint", "finger_2_dist_joint",
    "finger_3_med_joint", "finger_3_dist_joint",
]
# Limits from dex-urdf bhand_model.urdf (jnt_range).
LOW8 = np.array([-3.140, -2.440, -0.785, 0.000, -2.440, -0.785, -2.440, -0.785], dtype=np.float64)
HIGH8 = np.array([0.000, 0.000, 0.000, 3.140, 0.000, 0.000, 0.000, 0.000], dtype=np.float64)

MULTIDEX_JOINTS_SLICE = slice(9, 17)  # [trans(3), rot6d(6), joints(8)]
# Remap: urdf_qpos[k] = sign * data[src]. (src into data joints d0..d7)
REMAP = [(2, -1.0), (0, -1.0), (1, -1.0), (5, +1.0), (3, -1.0), (4, -1.0), (6, -1.0), (7, -1.0)]


def remap_data_to_urdf(joints_data: np.ndarray) -> np.ndarray:
    """[N,8] MultiDex joints -> [N,8] in dex-urdf order/signs."""
    out = np.empty_like(joints_data)
    for k, (src, sign) in enumerate(REMAP):
        out[:, k] = sign * joints_data[:, src]
    return out


def load_multidex_joints(path: Path) -> np.ndarray:
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d["metadata"]
    Q = torch.stack([m[0] for m in meta]).numpy()  # [N, 17]
    if Q.shape[1] != 17:
        raise ValueError(f"Expected MultiDex barrett qpos dim 17, got {Q.shape[1]}")
    data_joints = Q[:, MULTIDEX_JOINTS_SLICE].astype(np.float64)  # [N, 8]
    return remap_data_to_urdf(data_joints)


def gate_a_order_check(joints: np.ndarray, tol: float = 0.1) -> None:
    """GATE A: remapped columns must fit urdf limits (objective remap-sign check)."""
    jmin, jmax = joints.min(axis=0), joints.max(axis=0)
    bad = []
    for i in range(8):
        lo_ok = jmin[i] >= LOW8[i] - tol
        hi_ok = jmax[i] <= HIGH8[i] + tol
        span_ok = (jmax[i] - jmin[i]) > 0.1 * (HIGH8[i] - LOW8[i])
        if not (lo_ok and hi_ok and span_ok):
            bad.append(f"  [{i}] {JOINTS8[i]}: obs=[{jmin[i]:+.3f},{jmax[i]:+.3f}] "
                       f"limit=[{LOW8[i]:+.3f},{HIGH8[i]:+.3f}] lo={lo_ok} hi={hi_ok} span={span_ok}")
    if bad:
        raise SystemExit("GATE A FAILED: remapped joints do not fit urdf limits "
                         "(remap signs/order wrong):\n" + "\n".join(bad))
    print("GATE A PASSED: all 8 remapped columns fit urdf limits (remap verified).")


def clip_to_limits(joints: np.ndarray) -> np.ndarray:
    clipped = np.clip(joints, LOW8, HIGH8)
    n = int((clipped != joints).any(axis=1).sum())
    print(f"clipped {n} rows ({100.*n/joints.shape[0]:.2f}%); max overshoot={np.abs(clipped-joints).max():.4f}")
    return clipped


def compute_pca(qpos: np.ndarray):
    scale = HIGH8 - LOW8
    qnorm = (qpos - LOW8) / scale
    mean = qnorm.mean(axis=0)
    centered = qnorm - mean
    _, sv, comps = np.linalg.svd(centered, full_matrices=False)
    expl = sv * sv / max(1, centered.shape[0] - 1)
    ratio = expl / expl.sum()
    return mean, comps, ratio, np.cumsum(ratio)


def compute_coeff_stats(qpos, mean, comps):
    scale = HIGH8 - LOW8
    c = ((qpos - LOW8) / scale - mean) @ comps.T
    return {
        "coeff_min": c.min(0).astype(np.float32), "coeff_p01": np.percentile(c, 1, 0).astype(np.float32),
        "coeff_p05": np.percentile(c, 5, 0).astype(np.float32), "coeff_mean": c.mean(0).astype(np.float32),
        "coeff_std": c.std(0).astype(np.float32), "coeff_p95": np.percentile(c, 95, 0).astype(np.float32),
        "coeff_p99": np.percentile(c, 99, 0).astype(np.float32), "coeff_max": c.max(0).astype(np.float32),
    }


def gate_b_roundtrip(qpos, mean, comps, n_check=200, thresh=1e-3):
    scale = HIGH8 - LOW8
    s = qpos[:n_check]
    c = ((s - LOW8) / scale - mean) @ comps.T
    recon = (mean + c @ comps) * scale + LOW8
    err = np.abs(recon - s).max()
    if err > thresh:
        raise SystemExit(f"GATE B FAILED: round-trip error {err:.6f} > {thresh}.")
    print(f"GATE B PASSED: round-trip max error {err:.2e}.")


def load_synthetic_npz(paths: list[Path]) -> np.ndarray:
    rows = []
    for p in paths:
        data = np.load(p, allow_pickle=False)
        if "qpos8" not in data.files:
            raise KeyError(f"{p}: missing 'qpos8' (has {list(data.files)})")
        q = data["qpos8"].astype(np.float64)
        print(f"  synthetic {p.name}: {q.shape[0]} rows")
        rows.append(q)
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 8), dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multidex", type=Path,
                    default=Path("robot/hands/barrett_hand/datasets/raw/MultiDex_barrett.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path("robot/hands/barrett_hand/datasets/processed/eigengrasp_barrett.npz"))
    ap.add_argument("--synthetic-qpos-npz", type=Path, nargs="*", default=[],
                    help="Synthetic open/close NPZ (key qpos8) appended before PCA.")
    args = ap.parse_args()

    print(f"multidex = {args.multidex}")
    joints = load_multidex_joints(args.multidex)
    print(f"loaded + remapped MultiDex barrett joints: {joints.shape}")
    # tol=0.45: GenDexGrasp's barrett URDF allows distal flexion ~1.19 while dex-urdf
    # caps it at 0.785 (version discrepancy, like allegro's thumb-CMC). The remap SIGNS
    # are still verified (dist negative, spread-f2 positive); the span check still
    # catches gross scrambles. clip_to_limits then clamps the overshoot to dex-urdf.
    gate_a_order_check(joints, tol=0.45)
    joints = clip_to_limits(joints)

    synthetic = load_synthetic_npz(args.synthetic_qpos_npz)
    if synthetic.shape[0] > 0:
        synthetic = np.clip(synthetic, LOW8, HIGH8)
        qpos = np.concatenate([joints, synthetic], axis=0)
        print(f"appended {synthetic.shape[0]} synthetic rows -> total {qpos.shape[0]}")
    else:
        qpos = joints
        print("no synthetic rows (bare build for remap/order verification)")

    mean, comps, ratio, cum = compute_pca(qpos)
    stats = compute_coeff_stats(qpos, mean, comps)
    gate_b_roundtrip(joints, mean, comps)

    print("\nPCA explained variance:")
    for i in range(8):
        print(f"  PC{i+1:02d}: ratio={ratio[i]:.5f}  cum={cum[i]:.5f}")
    for thr in (0.80, 0.90, 0.95, 0.99):
        print(f"  k_for_{int(thr*100)}={int(np.searchsorted(cum, thr) + 1)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        source="multidex_barrett+synthetic" if synthetic.shape[0] else "multidex_barrett",
        multidex_path=str(args.multidex),
        n_multidex=joints.shape[0],
        n_synthetic=int(synthetic.shape[0]),
        remap=np.array(REMAP, dtype=np.float32),
        joint_names=np.array(JOINTS8),
        joint_low=LOW8.astype(np.float32),
        joint_high=HIGH8.astype(np.float32),
        mean_norm=mean.astype(np.float32),
        components_norm=comps.astype(np.float32),
        explained_ratio=ratio.astype(np.float32),
        explained_cumulative=cum.astype(np.float32),
        q_min=qpos.min(0).astype(np.float32),
        q_max=qpos.max(0).astype(np.float32),
        q_mean=qpos.mean(0).astype(np.float32),
        **stats,
    )
    print(f"\nsaved = {args.out}")


if __name__ == "__main__":
    main()
