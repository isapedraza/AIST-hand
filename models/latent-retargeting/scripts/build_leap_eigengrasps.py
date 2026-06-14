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

JOINTS_SLICE = slice(7, 23)  # qpos layout: [wrist_pos(3), wrist_quat(4), joints(16)]

# Like Shadow (analyze_dexonomy_eigengrasps.py): use all 3 grasp phases, not just
# the final grasp. pregrasp = hand approaching (more OPEN), grasp = closed on the
# object, squeeze = tighter. Each BODex npy has equal rows per phase, so loading
# all three is automatically phase-balanced 1:1:1 -> the PCA basis spans open->close.
DEFAULT_PHASES = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")


def load_bodex_phases(tar_path: Path, phases: tuple[str, ...] = DEFAULT_PHASES) -> dict[str, np.ndarray]:
    """Stream BODex tar.gz, return {phase: [N,16]} qpos[:, 7:23] across all objects."""
    rows: dict[str, list[np.ndarray]] = {p: [] for p in phases}
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".npy")]
        print(f"[load] {len(members)} npy files in archive  phases={phases}")
        for i, member in enumerate(members):
            f = tf.extractfile(member)
            if f is None:
                continue
            buf = io.BytesIO(f.read())
            d = np.load(buf, allow_pickle=True).item()
            for p in phases:
                if p in d:
                    rows[p].append(d[p][:, JOINTS_SLICE].astype(np.float64))
            if (i + 1) % 200 == 0 or i == len(members) - 1:
                total = sum(r.shape[0] for chunks in rows.values() for r in chunks)
                print(f"  [{i+1}/{len(members)}] total poses so far: {total:,}")
    out = {p: np.concatenate(chunks, axis=0) for p, chunks in rows.items() if chunks}
    print(f"[load] per-phase rows: { {p: a.shape[0] for p, a in out.items()} }")
    return out


def subsample_balance(phases_data: dict[str, np.ndarray], cap: int | None,
                      seed: int) -> np.ndarray:
    """Phase-balance: cap each phase to `cap` rows (random, no replacement), concat.

    Mirrors Shadow's rows_per_weight_unit: keeping real data modest so the appended
    synthetic open/close are a meaningful fraction (and thus land inside p01/p99).
    """
    rng = np.random.default_rng(seed)
    parts = []
    for p, arr in phases_data.items():
        if cap is not None and arr.shape[0] > cap:
            arr = arr[rng.choice(arr.shape[0], cap, replace=False)]
        parts.append(arr)
        print(f"  phase {p}: kept {arr.shape[0]} rows")
    return np.concatenate(parts, axis=0)


def load_synthetic_npz(paths: list[Path]) -> np.ndarray:
    """Load synthetic open/close NPZ files (key 'qpos16') appended before PCA."""
    rows: list[np.ndarray] = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        if "qpos16" not in d.files:
            raise KeyError(f"{p} missing 'qpos16' (has {list(d.files)})")
        q = d["qpos16"].astype(np.float64)
        if q.ndim != 2 or q.shape[1] != 16:
            raise ValueError(f"Expected qpos16 [N,16] in {p}, got {q.shape}")
        print(f"  synthetic {p.name}: {q.shape[0]} rows")
        rows.append(q)
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 16), dtype=np.float64)


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
    _proc = Path("robot/hands/leap_hand/datasets/processed")
    ap.add_argument("--synthetic-qpos-npz", type=Path, nargs="*",
                    default=[_proc / "synthetic_open_leap.npz", _proc / "synthetic_close_leap.npz"],
                    help="Synthetic open/close NPZ (key qpos16) appended before PCA. "
                         "BODex never reaches a flat open hand, so these anchor the extremes.")
    ap.add_argument("--max-rows-per-phase", type=int, default=100000,
                    help="Downsample each BODex phase to this many rows (Shadow-style). "
                         "Keeps real data modest so synthetic extremes land inside p01/p99. "
                         "0 = no cap (use all rows).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"bodex = {args.bodex}")
    phases_data = load_bodex_phases(args.bodex)
    cap = args.max_rows_per_phase if args.max_rows_per_phase > 0 else None
    joints = subsample_balance(phases_data, cap, args.seed)
    print(f"[balance] real total after phase cap: {joints.shape}")
    gate_a(joints)  # order check on BODex only (before appending synthetic)

    synthetic = load_synthetic_npz(args.synthetic_qpos_npz)
    if synthetic.shape[0] > 0:
        synthetic = np.clip(synthetic, LOW16, HIGH16)
        joints = np.concatenate([joints, synthetic], axis=0)
        print(f"appended {synthetic.shape[0]} synthetic rows -> total {joints.shape[0]}")
    else:
        print("no synthetic rows appended")

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
        source         = "bodex_leap+synthetic",
        bodex_path     = str(args.bodex),
        n_poses        = joints.shape[0],
        n_synthetic    = int(synthetic.shape[0]),
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
