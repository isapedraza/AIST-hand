#!/usr/bin/env python3
"""
Generate synthetic open/close Leap hand poses to anchor the eigengrasp PCA basis.

BODex only contains final grasps (closed around objects), so the PCA basis would
not span the fully-open flat hand nor a clean empty-hand fist. We synthesize open +
close poses parametrically and collision-filter them against the leap_hand MJCF
(ncon==0 or shallow contact), then feed them to build_leap_eigengrasps.py via
--synthetic-qpos-npz.

Leap has NO menagerie keyframes.xml (unlike Shadow), so there is no hand-authored
"close hand" to anchor on. We do the direct analog of Shadow's close keyframe
parametrically: set flexion joints to a high fraction of their upper limit.

Joint order = canonical JOINTS16, imported from build_leap_eigengrasps to guarantee
a single source of truth:
    [0] if_mcp  [1] if_rot  [2] if_pip  [3] if_dip
    [4] mf_mcp  [5] mf_rot  [6] mf_pip  [7] mf_dip
    [8] rf_mcp  [9] rf_rot  [10] rf_pip  [11] rf_dip
    [12] th_cmc [13] th_axl [14] th_mcp [15] th_ipl

Usage:
    python generate_synthetic_open_close_leap.py \
        --out-open  .../synthetic_open_leap.npz \
        --out-close .../synthetic_close_leap.npz \
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
from build_leap_eigengrasps import JOINTS16, LOW16, HIGH16  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MJCF_PATH = REPO_ROOT / "third_party/mujoco_menagerie/leap_hand/right_hand.xml"

# Joint role indices in canonical order [index(0-3), middle(4-7), ring(8-11), thumb(12-15)].
ABD_IDX = [1, 5, 9]                       # finger abduction/spread (rot), neutral=0
FLEX_IDX = [0, 2, 3, 4, 6, 7, 8, 10, 11]  # finger MCP/PIP/DIP flexion, open=0
THUMB_CMC = 12                            # thumb base rotation (opposition)
THUMB_AXL = 13                            # thumb axial
THUMB_MCP = 14                            # thumb MCP flexion
THUMB_IPL = 15                            # thumb IP flexion


def open_target() -> np.ndarray:
    # flexion=0 (extended), abduction=0 (neutral), thumb neutral -> flat open hand.
    return np.zeros(16, dtype=np.float64)


# Two hand-tuned fist targets (joint tuner): fingers curled + abduction fan +
# thumb rotated OVER the fist (axl/cmc tuned so the thumb is not tucked to the
# side nor under the fingers). Using both enriches the closed region of the PCA.
CLOSE_TARGETS = np.array([
    [1.3380, 0.1200, 1.1310, 1.2252, 1.3380, 0.0000, 1.1310, 1.2252,
     1.3380, -0.1200, 1.1310, 1.2252, 1.1680, 1.0500, 0.0300, 1.6300],
    [1.4800, 0.1200, 1.1350, 1.2920, 1.4800, 0.0000, 1.1350, 1.2920,
     1.4800, -0.1200, 1.1350, 1.2920, 0.8180, 1.2500, 0.3800, 1.5300],
], dtype=np.float64)


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
    assert model.nq == 16, f"Leap MJCF nq expected 16, got {model.nq}"
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
    if len(accepted) == 0:
        print(f"ERROR: 0/{rows} {name} poses accepted. Target likely self-penetrates; "
              "soften the curl or raise --contact-tol. Skipping save.")
        return
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
        synthetic_pose_name=f"synthetic_{name}_leap",
        source="parametric_target_plus_jitter_mjcf_collision_filtered",
        mjcf=str(MJCF_PATH),
        seed=seed,
        jitter=jitter,
        target=target.astype(np.float32),
        base_qpos16=target.astype(np.float32),
        joint_names=np.array(JOINTS16),
        joint_low=LOW16.astype(np.float32),
        joint_high=HIGH16.astype(np.float32),
        qpos16=qpos16,
    )
    print(f"saved = {out}")


def generate_multi(name: str, targets: np.ndarray, rows: int, jitter: float,
                   seed: int, out: Path, contact_tol: float) -> None:
    """Generate jittered+collision-filtered samples around several targets, concat."""
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    assert model.nq == 16, f"Leap MJCF nq expected 16, got {model.nq}"

    per = rows // len(targets)
    all_accepted: list[np.ndarray] = []
    for ti, target in enumerate(targets):
        rng = np.random.default_rng(seed + ti)
        accepted: list[np.ndarray] = []
        generated = 0
        max_candidates = per * 400
        while len(accepted) < per and generated < max_candidates:
            batch = sample_around(target, 4096, jitter, rng)
            generated += batch.shape[0]
            for q in batch:
                if collision_ok(model, data, q.astype(np.float64), contact_tol):
                    accepted.append(q.astype(np.float32))
                    if len(accepted) >= per:
                        break
        acc = 100.0 * len(accepted) / max(1, generated)
        print(f"{name} target {ti}: accepted {len(accepted)}/{per} "
              f"(acceptance {acc:.1f}%)")
        all_accepted.extend(accepted[:per])

    if not all_accepted:
        print(f"ERROR: 0 {name} poses accepted across all targets. Skipping save.")
        return
    qpos16 = np.stack(all_accepted, axis=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        synthetic_pose_name=f"synthetic_{name}_leap",
        source="parametric_multi_target_plus_jitter_mjcf_collision_filtered",
        mjcf=str(MJCF_PATH),
        seed=seed,
        jitter=jitter,
        target=targets.astype(np.float32),
        base_qpos16=targets[0].astype(np.float32),
        joint_names=np.array(JOINTS16),
        joint_low=LOW16.astype(np.float32),
        joint_high=HIGH16.astype(np.float32),
        qpos16=qpos16,
    )
    print(f"{name}: total {qpos16.shape[0]} poses -> saved = {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    proc = REPO_ROOT / "robot/hands/leap_hand/datasets/processed"
    ap.add_argument("--out-open", type=Path, default=proc / "synthetic_open_leap.npz")
    ap.add_argument("--out-close", type=Path, default=proc / "synthetic_close_leap.npz")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--jitter", type=float, default=0.06, help="Fraction of joint span.")
    ap.add_argument("--contact-tol", type=float, default=0.004,
                    help="Allowed self-penetration depth (m). Closed hands touch themselves.")
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--no-close", action="store_true", help="Skip synthetic close generation.")
    args = ap.parse_args()

    generate("open", open_target(), args.rows, args.jitter, args.seed, args.out_open, args.contact_tol)
    if not args.no_close:
        generate_multi("close", CLOSE_TARGETS, args.rows, args.jitter, args.seed + 1, args.out_close, args.contact_tol)


if __name__ == "__main__":
    main()
