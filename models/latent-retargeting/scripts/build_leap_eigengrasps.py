#!/usr/bin/env python3
"""
Build Leap Hand eigengrasps (PCA) from BODex grasp dataset.

BODex stores each grasp as a dict with keys:
    grasp_qpos   [N, 23]  float32  (7 wrist: 3 pos + 4 quat) + (16 finger joints)
    pregrasp_qpos / squeeze_qpos — same layout, not used here.

We take finger joints = grasp_qpos[:, 7:23] in MJCF/pk order:
    [0] if_mcp  [1] if_rot  [2] if_pip  [3] if_dip
    [4] mf_mcp  [5] mf_rot  [6] mf_pip  [7] mf_dip
    [8] rf_mcp  [9] rf_rot  [10] rf_pip  [11] rf_dip
    [12] th_cmc [13] th_axl [14] th_mcp [15] th_ipl

This order matches the menagerie MJCF qpos order AND pytorch_kinematics
chain order for leap_hand_right.urdf (verified: pk traverses "1","0","2","3"
per finger which maps to mcp,rot,pip,dip by limit comparison).

Limits are from the menagerie MJCF (what BODex simulation used).

Safety gates (same pattern as build_allegro_eigengrasps.py):
  GATE A: per-column min/max of BODex joints must lie within MJCF limits.
  GATE B: round-trip PCA reconstruction of real grasps must be near-exact.

Usage:
    python models/latent-retargeting/scripts/build_leap_eigengrasps.py \
        --bodex  robot/hands/leap_hand/datasets/raw/BODex_leap.tar.gz \
        --out    robot/hands/leap_hand/datasets/processed/eigengrasp_leap.npz
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np

JOINTS16 = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]

# Limits from mujoco_menagerie/leap_hand/right_hand.xml (ground truth for BODex).
_FINGER_LIMITS = [(-0.314, 2.230), (-1.047, 1.047), (-0.506, 1.885), (-0.366, 2.042)]
_THUMB_LIMITS  = [(-0.349, 2.094), (-0.349, 2.094), (-0.470, 2.443), (-1.340, 1.880)]
_ALL_LIMITS    = _FINGER_LIMITS * 3 + _THUMB_LIMITS
LOW16  = np.array([lo for lo, _ in _ALL_LIMITS], dtype=np.float64)
HIGH16 = np.array([hi for _, hi in _ALL_LIMITS], dtype=np.float64)

JOINTS_SLICE = slice(7, 23)  # grasp_qpos layout: [wrist_pos(3), wrist_quat(4), joints(16)]


def load_bodex_joints(tar_path: Path) -> np.ndarray:
    """Stream BODex tar.gz, extract all grasp_qpos[:, 7:23] across all objects."""
    rows: list[np.ndarray] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".npy")]
        print(f"[load] {len(members)} npy files in archive")
        for i, member in enumerate(members):
            f = tf.extractfile(member)
            if f is None:
                continue
            buf = io.BytesIO(f.read())
            d = np.load(buf, allow_pickle=True).item()
            q = d["grasp_qpos"]  # [N, 23]
            rows.append(q[:, JOINTS_SLICE].astype(np.float64))
            if (i + 1) % 200 == 0 or i == len(members) - 1:
                total = sum(r.shape[0] for r in rows)
                print(f"  [{i+1}/{len(members)}] total poses so far: {total:,}")
    joints = np.concatenate(rows, axis=0)
    print(f"[load] total: {joints.shape}")
    return joints


def gate_a(joints: np.ndarray) -> None:
    """Check span only — catches order scrambles without assuming exact limits.

    BODex may use a slightly different Leap URDF than the menagerie one, so
    absolute limit comparison is fragile for thumb joints. Span mismatch
    (a flexion range ~2.5 rad in an abduction slot ~2.1 rad) still catches
    a real scramble since swapped joints have very different spans.
    """
    jmin, jmax = joints.min(axis=0), joints.max(axis=0)
    ref_spans  = HIGH16 - LOW16
    bad = []
    for i in range(16):
        obs_span = jmax[i] - jmin[i]
        span_ok  = obs_span > 0.15 * ref_spans[i]  # must span >=15% of expected range
        if not span_ok:
            bad.append(
                f"  [{i:2}] {JOINTS16[i]}: obs_span={obs_span:.3f}  "
                f"ref_span={ref_spans[i]:.3f}  ratio={obs_span/ref_spans[i]:.2f}"
            )
    if bad:
        raise SystemExit("GATE A FAILED — joint order likely wrong (span too small):\n" + "\n".join(bad))
    print("GATE A PASSED: all 16 columns have reasonable spans (order verified).")


def compute_data_limits(joints: np.ndarray, buf: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Data-derived limits with a small buffer. Robust to URDF version differences."""
    span = joints.max(axis=0) - joints.min(axis=0)
    low  = joints.min(axis=0) - buf * span
    high = joints.max(axis=0) + buf * span
    # Clip to menagerie MJCF limits so we never exceed what MuJoCo allows.
    low  = np.maximum(low,  LOW16)
    high = np.minimum(high, HIGH16)
    return low.astype(np.float64), high.astype(np.float64)


def clip_to_limits(joints: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    clipped = np.clip(joints, low, high)
    n = int((clipped != joints).any(axis=1).sum())
    print(f"clipped {n} rows ({100.*n/joints.shape[0]:.2f}%); "
          f"max overshoot={np.abs(clipped - joints).max():.4f}")
    return clipped


def compute_pca(qpos: np.ndarray, low: np.ndarray, high: np.ndarray):
    scale = high - low
    qnorm = (qpos - low) / scale
    mean  = qnorm.mean(axis=0)
    centered = qnorm - mean
    _, sv, comps = np.linalg.svd(centered, full_matrices=False)
    expl  = sv * sv / max(1, centered.shape[0] - 1)
    ratio = expl / expl.sum()
    cum   = np.cumsum(ratio)
    return mean, comps, ratio, cum


def compute_coeff_stats(qpos, mean, comps, low, high):
    scale = high - low
    qnorm = (qpos - low) / scale
    c = (qnorm - mean) @ comps.T
    return {
        "coeff_min":  c.min(axis=0).astype(np.float32),
        "coeff_p01":  np.percentile(c,  1, axis=0).astype(np.float32),
        "coeff_p05":  np.percentile(c,  5, axis=0).astype(np.float32),
        "coeff_mean": c.mean(axis=0).astype(np.float32),
        "coeff_std":  c.std(axis=0).astype(np.float32),
        "coeff_p95":  np.percentile(c, 95, axis=0).astype(np.float32),
        "coeff_p99":  np.percentile(c, 99, axis=0).astype(np.float32),
        "coeff_max":  c.max(axis=0).astype(np.float32),
    }


def gate_b(qpos, mean, comps, low, high, n_check=200, thresh=1e-3):
    scale  = high - low
    sample = qpos[:n_check]
    qnorm  = (sample - low) / scale
    coeffs = (qnorm - mean) @ comps.T
    recon  = mean + coeffs @ comps
    recon  = recon * scale + low
    err    = np.abs(recon - sample).max()
    if err > thresh:
        raise SystemExit(f"GATE B FAILED: round-trip error {err:.6f} > {thresh}.")
    print(f"GATE B PASSED: round-trip max error {err:.2e}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodex", type=Path,
                    default=Path("robot/hands/leap_hand/datasets/raw/BODex_leap.tar.gz"))
    ap.add_argument("--out", type=Path,
                    default=Path("robot/hands/leap_hand/datasets/processed/eigengrasp_leap.npz"))
    args = ap.parse_args()

    print(f"bodex = {args.bodex}")
    joints = load_bodex_joints(args.bodex)
    gate_a(joints)
    low, high = compute_data_limits(joints)
    print(f"data limits (with buf): low={low.round(3)}  high={high.round(3)}")
    joints = clip_to_limits(joints, low, high)
    mean, comps, ratio, cum = compute_pca(joints, low, high)
    stats = compute_coeff_stats(joints, mean, comps, low, high)
    gate_b(joints, mean, comps, low, high)

    print("\nPCA explained variance:")
    for i in range(16):
        print(f"  PC{i+1:02d}: ratio={ratio[i]:.5f}  cum={cum[i]:.5f}")
    for thr in (0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, thr) + 1)
        print(f"  k_for_{int(thr*100)}={k}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        source         = "bodex_leap",
        bodex_path     = str(args.bodex),
        n_poses        = joints.shape[0],
        joint_names    = np.array(JOINTS16),
        joint_low      = low.astype(np.float32),
        joint_high     = high.astype(np.float32),
        mean_norm      = mean.astype(np.float32),
        components_norm= comps.astype(np.float32),
        explained_ratio= ratio.astype(np.float32),
        explained_cumulative = cum.astype(np.float32),
        q_min          = joints.min(axis=0).astype(np.float32),
        q_max          = joints.max(axis=0).astype(np.float32),
        q_mean         = joints.mean(axis=0).astype(np.float32),
        **stats,
    )
    print(f"\nsaved = {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
