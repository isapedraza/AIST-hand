#!/usr/bin/env python3
"""
Generate synthetic open/close Allegro hand poses to anchor the eigengrasp PCA basis.

MultiDex only contains final grasps (no pregrasp/squeeze phases), so the PCA basis
would not span the fully-open hand. We synthesize open + close poses parametrically
and collision-filter them against the wonik_allegro MJCF (ncon==0), then feed them
to build_allegro_eigengrasps.py via --synthetic-qpos-npz.

No menagerie keyframe is needed: for Allegro the near-zero pose is an open flat hand
(flexion joints at 0 = extended fingers, abduction at 0 = neutral). Thumb CMC
(joint_12) has a positive lower limit (0.263), so "open" uses its low value there.

Joint order = canonical JOINTS16 (joint_0.0 .. joint_15.0), imported from
build_allegro_eigengrasps to guarantee a single source of truth.

Usage:
    python generate_synthetic_open_close_allegro.py \
        --out-open  .../synthetic_open_allegro.npz \
        --out-close .../synthetic_close_allegro.npz \
        --rows 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

# Single source of truth for joint order + limits (verified against URDF/MJCF).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_allegro_eigengrasps import JOINTS16, LOW16, HIGH16  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MJCF_PATH = REPO_ROOT / "third_party/mujoco_menagerie/wonik_allegro/right_hand.xml"

# Joint role indices in canonical order [index(0-3), middle(4-7), ring(8-11), thumb(12-15)].
ABD_IDX = [0, 4, 8]                       # finger abduction (spread), neutral=0
FLEX_IDX = [1, 2, 3, 5, 6, 7, 9, 10, 11]  # finger MCP/PIP/DIP flexion, open=0
THUMB_CMC = 12                            # thumb opposition/rotation (low=0.263)
THUMB_FLEX = [13, 14, 15]                 # thumb chain flexion


def open_target() -> np.ndarray:
    t = np.zeros(16, dtype=np.float64)
    t[THUMB_CMC] = LOW16[THUMB_CMC]      # thumb out (its lower limit ~0.263)
    return t  # flexion/abduction at 0 = extended flat open hand


def close_target() -> np.ndarray:
    t = np.zeros(16, dtype=np.float64)
    for i in FLEX_IDX:
        t[i] = 0.85 * HIGH16[i]          # fingers curled
    t[THUMB_CMC] = 0.85 * HIGH16[THUMB_CMC]  # thumb opposed
    for i in THUMB_FLEX:
        t[i] = 0.7 * HIGH16[i]           # thumb chain curled
    # abduction stays ~0 (fingers together)
    return t


def sample_around(target: np.ndarray, n: int, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n poses around target with uniform jitter, clipped to limits."""
    span = (HIGH16 - LOW16)[None, :]
    noise = rng.uniform(-jitter, jitter, size=(n, 16)) * span
    q = target[None, :] + noise
    return np.clip(q, LOW16, HIGH16)


def collision_ok(model, data, q16: np.ndarray, tol: float) -> bool:
    """Accept if no deep self-penetration. Light contact (dist >= -tol) is allowed,
    since a closed hand naturally touches palm/fingers. tol in meters."""
    data.qpos[:] = q16
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        return True
    min_dist = min(float(data.contact[i].dist) for i in range(data.ncon))
    return min_dist >= -tol


def generate(name: str, target: np.ndarray, rows: int, jitter: float,
             seed: int, out: Path, contact_tol: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    assert model.nq == 16, f"Allegro MJCF nq expected 16, got {model.nq}"
    rng = np.random.default_rng(seed)

    accepted: list[np.ndarray] = []
    generated = 0
    max_candidates = rows * 200
    while len(accepted) < rows and generated < max_candidates:
        batch = sample_around(target, 4096, jitter, rng)
        generated += batch.shape[0]
        for q in batch:
            if collision_ok(model, data, q.astype(np.float64), contact_tol):
                accepted.append(q.astype(np.float32))
                if len(accepted) >= rows:
                    break
    if len(accepted) < rows:
        print(f"WARNING: only {len(accepted)}/{rows} {name} poses (acceptance low). "
              "Consider lowering jitter.")
    qpos16 = np.stack(accepted[:rows], axis=0)
    acc = 100.0 * len(accepted) / max(1, generated)
    print(f"{name}: accepted {qpos16.shape[0]} / generated {generated} "
          f"(acceptance {acc:.1f}%)")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        synthetic_pose_name=f"synthetic_{name}_allegro",
        source="parametric_target_plus_jitter_mjcf_collision_filtered",
        mjcf=str(MJCF_PATH),
        seed=seed,
        jitter=jitter,
        target=target.astype(np.float32),
        joint_names=np.array(JOINTS16),
        joint_low=LOW16.astype(np.float32),
        joint_high=HIGH16.astype(np.float32),
        qpos16=qpos16,
    )
    print(f"saved = {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    proc = REPO_ROOT / "robot/hands/allegro_hand/datasets/processed"
    ap.add_argument("--out-open", type=Path, default=proc / "synthetic_open_allegro.npz")
    ap.add_argument("--out-close", type=Path, default=proc / "synthetic_close_allegro.npz")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--jitter", type=float, default=0.06, help="Fraction of joint span.")
    ap.add_argument("--contact-tol", type=float, default=0.004,
                    help="Allowed self-penetration depth (m). Closed hands touch themselves.")
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--with-close", action="store_true",
                    help="Also generate synthetic close (redundant: MultiDex grasps already "
                         "span the closed region; a free-space Allegro fist self-penetrates deeply).")
    args = ap.parse_args()

    # OPEN is the essential anchor: a grasp dataset (MultiDex) has NO open hands, only
    # closed grasps. CLOSE is redundant with the 37,774 MultiDex closed grasps and a
    # free-space fist deeply self-penetrates, so it is off by default.
    generate("open", open_target(), args.rows, args.jitter, args.seed, args.out_open, args.contact_tol)
    if args.with_close:
        generate("close", close_target(), args.rows, args.jitter, args.seed + 1, args.out_close, args.contact_tol)


if __name__ == "__main__":
    main()
