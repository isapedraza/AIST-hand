#!/usr/bin/env python3
"""
Build Allegro eigengrasps (PCA) from the MultiDex grasp dataset.

Analog of analyze_dexonomy_eigengrasps.py, but for the Allegro hand sourced from
MultiDex (GenDexGrasp). MultiDex stores each grasp as a tuple
(qpos[25], object_name, robot_name) where qpos = [trans(3), rot6d(6), joints(16)].
We take joints = qpos[9:25] in URDF declaration order (joint_0.0 .. joint_15.0),
which is verified positionally identical to the dex-urdf URDF, the pytorch_kinematics
chain order, and the wonik_allegro MJCF qpos order. NO remapping anywhere.

Allegro has NO wrist joints (unlike Shadow's WRJ1/WRJ2): q16 directly, no padding.

Pipeline: load MultiDex joints -> (optional) append synthetic open/close ->
normalize to [0,1] via URDF limits -> PCA -> coeff percentiles -> save NPZ with the
same schema generate_valid_robot_poses.py expects.

Safety gates (the joint-order failure mode is silent garbage):
  GATE A (pre-PCA): per-column min/max of MultiDex joints must match URDF limits.
  GATE B (post-PCA): round-trip reconstruction of real grasps must be near-exact.

Usage:
    python build_allegro_eigengrasps.py \
        --multidex /tmp/MultiDex/allegro/allegro.pt \
        --out robot/hands/allegro_hand/datasets/processed/eigengrasp_allegro.npz \
        [--synthetic-qpos-npz open.npz close.npz]
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import torch


# Canonical joint order = joint_0.0 .. joint_15.0 (URDF declaration = pk chain = MJCF).
# Per-finger: [j0=abduction(axis Z), j1=MCP flex, j2=PIP flex, j3=DIP flex].
# Fingers in order: index(0-3), middle(4-7), ring(8-11), thumb(12-15).
JOINTS16 = [f"joint_{i}.0" for i in range(16)]

# Limits straight from dex-urdf/robots/hands/allegro_hand/allegro_hand_right.urdf
# (verified positionally identical to the wonik_allegro MJCF jnt_range).
_FINGER_LIMITS = [(-0.47, 0.47), (-0.196, 1.61), (-0.174, 1.709), (-0.227, 1.618)]
_THUMB_LIMITS = [(0.263, 1.396), (-0.105, 1.163), (-0.189, 1.644), (-0.162, 1.719)]
_ALL_LIMITS = _FINGER_LIMITS * 3 + _THUMB_LIMITS  # index, middle, ring, thumb
LOW16 = np.array([lo for lo, _ in _ALL_LIMITS], dtype=np.float64)
HIGH16 = np.array([hi for _, hi in _ALL_LIMITS], dtype=np.float64)

# MultiDex qpos layout: [trans(3), rot6d(6), joints(16)] -> joints at [9:25].
JOINTS_SLICE = slice(9, 25)


def load_multidex_joints(path: Path) -> np.ndarray:
    """Load MultiDex allegro.pt -> [N, 16] joint angles in canonical order."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d["metadata"]
    Q = torch.stack([m[0] for m in meta]).numpy()  # [N, 25]
    if Q.shape[1] != 25:
        raise ValueError(f"Expected MultiDex qpos dim 25, got {Q.shape[1]}")
    joints = Q[:, JOINTS_SLICE].astype(np.float64)  # [N, 16]
    if joints.shape[1] != 16:
        raise ValueError(f"Expected 16 joints, got {joints.shape[1]}")
    return joints


# --- BODex 3-phase loader (alternative source to MultiDex) ---------------------
# BODex stores grasp_qpos [N,23] = wrist(7) + joints(16); slice [7:23]. Like Shadow,
# use all 3 phases (pregrasp=more OPEN, grasp, squeeze) for natural open->close span.
DEFAULT_PHASES = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
BODEX_JOINTS_SLICE = slice(7, 23)


def load_bodex_phases(tar_path: Path, phases: tuple[str, ...] = DEFAULT_PHASES) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {p: [] for p in phases}
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".npy")]
        print(f"[bodex] {len(members)} npy files  phases={phases}")
        for i, member in enumerate(members):
            f = tf.extractfile(member)
            if f is None:
                continue
            d = np.load(io.BytesIO(f.read()), allow_pickle=True).item()
            for p in phases:
                if p in d:
                    rows[p].append(d[p][:, BODEX_JOINTS_SLICE].astype(np.float64))
            if (i + 1) % 200 == 0 or i == len(members) - 1:
                total = sum(r.shape[0] for chunks in rows.values() for r in chunks)
                print(f"  [{i+1}/{len(members)}] total poses so far: {total:,}")
    out = {p: np.concatenate(chunks, axis=0) for p, chunks in rows.items() if chunks}
    print(f"[bodex] per-phase rows: { {p: a.shape[0] for p, a in out.items()} }")
    return out


def subsample_balance(phases_data: dict[str, np.ndarray], cap: int | None, seed: int) -> np.ndarray:
    """Phase-balance: cap each phase to `cap` rows (random), concat. Mirrors Shadow."""
    rng = np.random.default_rng(seed)
    parts = []
    for p, arr in phases_data.items():
        if cap is not None and arr.shape[0] > cap:
            arr = arr[rng.choice(arr.shape[0], cap, replace=False)]
        parts.append(arr)
        print(f"  phase {p}: kept {arr.shape[0]} rows")
    return np.concatenate(parts, axis=0)


def gate_a_order_check(joints: np.ndarray, tol: float = 0.1) -> None:
    """GATE A: per-column range must match URDF limits, else abort (order is wrong).

    tol=0.1 absorbs small limit discrepancies between the GenDexGrasp Allegro URDF
    and dex-urdf (e.g. thumb joint_12 differs by ~0.06) while still catching a real
    order scramble: a misplaced joint mismatches its slot by >1.0 (a flexion range
    [-0.2,1.7] in an abduction slot [+-0.47], or vice versa), far beyond tol.
    """
    jmin = joints.min(axis=0)
    jmax = joints.max(axis=0)
    bad = []
    for i in range(16):
        lo_ok = jmin[i] >= LOW16[i] - tol
        hi_ok = jmax[i] <= HIGH16[i] + tol
        # Span: observed range must be a sizeable fraction of the joint's limit range.
        # A swap changes span dramatically (abd span 0.94 vs flex span ~1.8).
        span_ok = (jmax[i] - jmin[i]) > 0.2 * (HIGH16[i] - LOW16[i])
        if not (lo_ok and hi_ok and span_ok):
            bad.append(
                f"  [{i:2}] {JOINTS16[i]}: observed=[{jmin[i]:+.3f},{jmax[i]:+.3f}] "
                f"limit=[{LOW16[i]:+.3f},{HIGH16[i]:+.3f}] "
                f"lo_ok={lo_ok} hi_ok={hi_ok} span_ok={span_ok}"
            )
    if bad:
        raise SystemExit(
            "GATE A FAILED: MultiDex joint order does not match URDF limits.\n"
            "Joint order is WRONG -> samples would be silent garbage. Aborting.\n"
            + "\n".join(bad)
        )
    print("GATE A PASSED: all 16 columns match URDF limits (order verified, tol=0.1).")


def clip_to_limits(joints: np.ndarray) -> np.ndarray:
    """Clip MultiDex joints to dex-urdf limits (our MJCF/FK feasible space).

    GenDexGrasp grasps are valid in their URDF but a few thumb-CMC values slightly
    exceed dex-urdf's tighter limit. Clip so the PCA basis stays in feasible space.
    """
    clipped = np.clip(joints, LOW16, HIGH16)
    n_clipped = int((clipped != joints).any(axis=1).sum())
    frac = 100.0 * n_clipped / joints.shape[0]
    max_overshoot = float(np.abs(clipped - joints).max())
    print(f"clipped {n_clipped} rows ({frac:.2f}%) to URDF limits; "
          f"max overshoot={max_overshoot:.4f}")
    return clipped


def compute_pca(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize to [0,1] via URDF limits, then SVD. Mirrors analyze_dexonomy."""
    scale = HIGH16 - LOW16
    qnorm = (qpos - LOW16) / scale
    mean = qnorm.mean(axis=0)
    centered = qnorm - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    explained = (singular_values * singular_values) / max(1, centered.shape[0] - 1)
    ratio = explained / explained.sum()
    cumulative = np.cumsum(ratio)
    return mean, components, ratio, cumulative


def compute_coeff_stats(qpos: np.ndarray, mean: np.ndarray, components: np.ndarray) -> dict:
    scale = HIGH16 - LOW16
    qnorm = (qpos - LOW16) / scale
    coeffs = (qnorm - mean) @ components.T
    return {
        "coeff_min": coeffs.min(axis=0).astype(np.float32),
        "coeff_p01": np.percentile(coeffs, 1, axis=0).astype(np.float32),
        "coeff_p05": np.percentile(coeffs, 5, axis=0).astype(np.float32),
        "coeff_mean": coeffs.mean(axis=0).astype(np.float32),
        "coeff_std": coeffs.std(axis=0).astype(np.float32),
        "coeff_p95": np.percentile(coeffs, 95, axis=0).astype(np.float32),
        "coeff_p99": np.percentile(coeffs, 99, axis=0).astype(np.float32),
        "coeff_max": coeffs.max(axis=0).astype(np.float32),
    }


def gate_b_roundtrip(qpos: np.ndarray, mean: np.ndarray, components: np.ndarray,
                     n_check: int = 200, thresh: float = 1e-3) -> None:
    """GATE B: project real grasps to coeffs, reconstruct, compare. Catches order bugs."""
    scale = HIGH16 - LOW16
    sample = qpos[:n_check]
    qnorm = (sample - LOW16) / scale
    coeffs = (qnorm - mean) @ components.T          # full basis -> exact reconstruction
    recon_norm = mean + coeffs @ components
    recon = recon_norm * scale + LOW16
    err = np.abs(recon - sample).max()
    if err > thresh:
        raise SystemExit(
            f"GATE B FAILED: round-trip reconstruction error {err:.6f} > {thresh}. "
            "Normalization/PCA inconsistent -> aborting."
        )
    print(f"GATE B PASSED: round-trip max error {err:.2e} (< {thresh}).")


def load_synthetic_npz(paths: list[Path]) -> np.ndarray:
    """Load synthetic open/close NPZ files with key 'qpos16' (joint order check)."""
    rows = []
    for p in paths:
        data = np.load(p, allow_pickle=False)
        if "qpos16" not in data.files:
            raise KeyError(f"{p}: missing 'qpos16'")
        names = [str(s) for s in data["joint_names"]] if "joint_names" in data.files else JOINTS16
        if list(names) != JOINTS16:
            raise ValueError(f"{p}: joint_names do not match canonical JOINTS16 order")
        q = data["qpos16"].astype(np.float64)
        print(f"  synthetic {p.name}: {q.shape[0]} rows")
        rows.append(q)
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 16), dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multidex", type=Path, default=Path("/tmp/MultiDex/allegro/allegro.pt"))
    ap.add_argument("--bodex", type=Path, default=None,
                    help="BODex_allegro.tar.gz. If set, use BODex 3-phase data instead of MultiDex.")
    ap.add_argument("--max-rows-per-phase", type=int, default=100000,
                    help="Downsample each BODex phase to this many rows (0 = no cap). BODex only.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("robot/hands/allegro_hand/datasets/processed/eigengrasp_allegro.npz"))
    ap.add_argument("--synthetic-qpos-npz", type=Path, nargs="*", default=[],
                    help="Synthetic open/close NPZ files (key qpos16) appended before PCA.")
    args = ap.parse_args()

    if args.bodex is not None:
        print(f"bodex = {args.bodex}")
        phases_data = load_bodex_phases(args.bodex)
        cap = args.max_rows_per_phase if args.max_rows_per_phase > 0 else None
        joints = subsample_balance(phases_data, cap, args.seed)
        print(f"loaded BODex joints (phase-balanced): {joints.shape}")
    else:
        print(f"multidex = {args.multidex}")
        joints = load_multidex_joints(args.multidex)
        print(f"loaded MultiDex joints: {joints.shape}")

    # GATE A: order check on raw joints (before mixing anything in).
    gate_a_order_check(joints)

    # Clip to dex-urdf feasible space (a few thumb-CMC values overshoot GenDexGrasp's URDF).
    joints = clip_to_limits(joints)

    synthetic = load_synthetic_npz(args.synthetic_qpos_npz)
    if synthetic.shape[0] > 0:
        # clip synthetic to limits (jitter may exceed) before appending
        synthetic = np.clip(synthetic, LOW16, HIGH16)
        qpos = np.concatenate([joints, synthetic], axis=0)
        print(f"appended {synthetic.shape[0]} synthetic rows -> total {qpos.shape[0]}")
    else:
        qpos = joints
        print("no synthetic rows (run bare for order verification)")

    mean, components, ratio, cumulative = compute_pca(qpos)
    coeff_stats = compute_coeff_stats(qpos, mean, components)

    # GATE B on the MultiDex rows only (real grasps).
    gate_b_roundtrip(joints, mean, components)

    print("\npca explained variance:")
    for i in range(16):
        print(f"  PC{i+1:02d}: ratio={ratio[i]:.5f} cum={cumulative[i]:.5f}")
    for thr in (0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cumulative, thr) + 1)
        print(f"  k_for_{int(thr*100)}={k}")

    print("\ntop PC loadings:")
    for pc in range(min(6, components.shape[0])):
        order = np.argsort(np.abs(components[pc]))[::-1][:6]
        terms = ", ".join(f"{JOINTS16[j]}:{components[pc][j]:+.3f}" for j in order)
        print(f"  PC{pc+1:02d}: {terms}")

    # Tag provenance by the source actually used, so a BODex-built npz is not
    # mislabeled as MultiDex (the two are compared side by side, never overwritten).
    use_bodex = args.bodex is not None
    src_tag = "bodex_allegro" if use_bodex else "multidex_allegro"
    src_path = str(args.bodex) if use_bodex else str(args.multidex)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        source=src_tag,
        multidex_path=src_path,
        n_multidex=joints.shape[0],
        n_synthetic=synthetic.shape[0],
        synthetic_qpos_npz=np.array([str(p) for p in args.synthetic_qpos_npz]),
        joint_names=np.array(JOINTS16),
        joint_low=LOW16.astype(np.float32),
        joint_high=HIGH16.astype(np.float32),
        mean_norm=mean.astype(np.float32),
        components_norm=components.astype(np.float32),
        explained_ratio=ratio.astype(np.float32),
        explained_cumulative=cumulative.astype(np.float32),
        q_min=qpos.min(axis=0).astype(np.float32),
        q_max=qpos.max(axis=0).astype(np.float32),
        q_mean=qpos.mean(axis=0).astype(np.float32),
        **coeff_stats,
    )
    print(f"\nsaved = {args.out}")


if __name__ == "__main__":
    main()
