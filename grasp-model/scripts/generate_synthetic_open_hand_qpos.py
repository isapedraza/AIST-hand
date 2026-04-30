from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SHADOW_DIR = ROOT / "third_party" / "mujoco_menagerie" / "shadow_hand"
RIGHT_HAND = SHADOW_DIR / "right_hand.xml"
KEYFRAMES_XML = SHADOW_DIR / "keyframes.xml"
DEFAULT_OUT = ROOT / "grasp-model" / "data" / "processed" / "synthetic_open_hand_shadow_qpos.npz"

HAND_QPOS_DIM = 24
DEFAULT_ROWS = 29_670

QPOS24_NAMES = [
    "WRJ2",
    "WRJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
]
JOINTS22 = QPOS24_NAMES[2:]
IDX = {name: idx for idx, name in enumerate(QPOS24_NAMES)}

LOW22 = np.array(
    [
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        0.0,
        -0.349,
        -0.2618,
        0.0,
        0.0,
        -1.047,
        0.0,
        -0.209,
        -0.698,
        -0.2618,
    ],
    dtype=np.float64,
)
HIGH22 = np.array(
    [
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        0.7854,
        0.349,
        1.5708,
        1.5708,
        1.5708,
        1.047,
        1.2217,
        0.209,
        0.698,
        1.5708,
    ],
    dtype=np.float64,
)

THUMB_SPREAD_MIN = 0.0
THUMB_SPREAD_MAX = 0.6981
FAN_MIN = 0.0
FAN_MAX = 1.0
OPEN_RELAX_MIN = -0.08
OPEN_RELAX_MAX = 0.14

FAN_PATTERN_J = {
    "FFJ4": -0.34,
    "MFJ4": +0.00,
    "RFJ4": -0.23,
    "LFJ4": -0.34,
}


def load_keyframe(name: str) -> np.ndarray:
    root = ET.parse(KEYFRAMES_XML).getroot()
    for key in root.findall(".//key"):
        if key.attrib.get("name") != name:
            continue
        qpos = np.fromstring(key.attrib.get("qpos", ""), sep=" ", dtype=np.float64)
        if qpos.size != HAND_QPOS_DIM:
            raise ValueError(f"Expected {HAND_QPOS_DIM} qpos values for {name!r}, got {qpos.size}")
        return qpos
    raise KeyError(f"Missing keyframe {name!r} in {KEYFRAMES_XML}")


def open_qpos_from_params(base: np.ndarray, thumb_spread: float, fan: float, relax: float) -> np.ndarray:
    qpos = base.copy()
    qpos[IDX["THJ2"]] = np.clip(thumb_spread, THUMB_SPREAD_MIN, THUMB_SPREAD_MAX)

    fan = np.clip(fan, FAN_MIN, FAN_MAX)
    for joint, coeff in FAN_PATTERN_J.items():
        qpos[IDX[joint]] = coeff * fan

    relax = np.clip(relax, OPEN_RELAX_MIN, OPEN_RELAX_MAX)
    relax_pos = max(relax, 0.0)
    for prefix in ("FF", "MF", "RF", "LF"):
        qpos[IDX[f"{prefix}J3"]] = relax
        qpos[IDX[f"{prefix}J2"]] = 0.75 * relax_pos
        qpos[IDX[f"{prefix}J1"]] = 0.75 * relax_pos
    qpos[IDX["THJ1"]] = 0.5 * relax
    return qpos


def contact_stats(model: mujoco.MjModel, data: mujoco.MjData, qpos24: np.ndarray) -> tuple[int, float]:
    data.qpos[:HAND_QPOS_DIM] = qpos24
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        return 0, np.inf
    min_dist = min(float(data.contact[i].dist) for i in range(data.ncon))
    return int(data.ncon), min_dist


def sample_params(rng: np.random.Generator, n: int) -> np.ndarray:
    params = np.zeros((n, 3), dtype=np.float64)
    params[:, 0] = rng.uniform(THUMB_SPREAD_MIN, THUMB_SPREAD_MAX, size=n)
    params[:, 1] = rng.uniform(FAN_MIN, FAN_MAX, size=n)
    params[:, 2] = rng.uniform(OPEN_RELAX_MIN, OPEN_RELAX_MAX, size=n)
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--contact-tolerance-m", type=float, default=0.0002)
    parser.add_argument("--max-candidates", type=int, default=2_000_000)
    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError(f"--rows must be positive, got {args.rows}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")

    base_qpos24 = load_keyframe("open hand")
    model = mujoco.MjModel.from_xml_path(str(RIGHT_HAND))
    data = mujoco.MjData(model)

    baseline_ncon, baseline_min_dist = contact_stats(model, data, base_qpos24)
    min_allowed_dist = -args.contact_tolerance_m
    print(
        "baseline open hand: "
        f"ncon={baseline_ncon} min_contact_dist={baseline_min_dist} "
        f"min_allowed_dist={min_allowed_dist:+.6f}"
    )

    rng = np.random.default_rng(args.seed)
    accepted24: list[np.ndarray] = []
    accepted_params: list[np.ndarray] = []
    accepted_ncon: list[int] = []
    accepted_min_dist: list[float] = []
    generated = 0
    rejected_contact = 0

    while sum(chunk.shape[0] for chunk in accepted24) < args.rows:
        if generated >= args.max_candidates:
            raise RuntimeError(
                f"Only accepted {sum(chunk.shape[0] for chunk in accepted24)} rows after "
                f"{generated} candidates. Increase --max-candidates or relax filters."
            )

        n_batch = min(args.batch_size, args.max_candidates - generated)
        params_batch = sample_params(rng, n_batch)
        generated += n_batch

        keep_qpos: list[np.ndarray] = []
        keep_params: list[np.ndarray] = []
        for params in params_batch:
            qpos24 = open_qpos_from_params(base_qpos24, params[0], params[1], params[2])
            ncon, min_dist = contact_stats(model, data, qpos24)
            if min_dist < min_allowed_dist:
                rejected_contact += 1
                continue
            keep_qpos.append(qpos24.astype(np.float32))
            keep_params.append(params.astype(np.float32))
            accepted_ncon.append(ncon)
            accepted_min_dist.append(min_dist)
            if sum(chunk.shape[0] for chunk in accepted24) + len(keep_qpos) >= args.rows:
                break

        if keep_qpos:
            accepted24.append(np.stack(keep_qpos, axis=0))
            accepted_params.append(np.stack(keep_params, axis=0))

        accepted_total = sum(chunk.shape[0] for chunk in accepted24)
        print(
            f"progress accepted={accepted_total}/{args.rows} generated={generated} "
            f"rejected_contact={rejected_contact} acceptance={accepted_total / generated:.3f}"
        )

    qpos24 = np.concatenate(accepted24, axis=0)[: args.rows]
    params = np.concatenate(accepted_params, axis=0)[: args.rows]
    qpos22 = qpos24[:, 2:]
    sample_ncon = np.asarray(accepted_ncon[: args.rows], dtype=np.int32)
    sample_min_dist = np.asarray(accepted_min_dist[: args.rows], dtype=np.float32)
    delta = qpos22.astype(np.float64) - base_qpos24[None, 2:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        synthetic_pose_name="synthetic_open_hand",
        source="mujoco_menagerie_shadow_hand_keyframe_plus_parametric_fan_relax",
        source_keyframe="open hand",
        source_xml=str(RIGHT_HAND),
        keyframes_xml=str(KEYFRAMES_XML),
        seed=args.seed,
        target_rows=args.rows,
        accepted_rows=qpos22.shape[0],
        generated_candidates=generated,
        rejected_contact=rejected_contact,
        acceptance_rate=qpos22.shape[0] / generated,
        collision_rule="min_contact_dist_ge_neg_tolerance",
        contact_tolerance_m=args.contact_tolerance_m,
        baseline_ncon=baseline_ncon,
        baseline_min_contact_dist=baseline_min_dist,
        min_allowed_contact_dist=min_allowed_dist,
        sample_ncon=sample_ncon,
        sample_min_contact_dist=sample_min_dist,
        joint_names=np.asarray(JOINTS22),
        joint_low=LOW22.astype(np.float32),
        joint_high=HIGH22.astype(np.float32),
        base_qpos22=base_qpos24[2:].astype(np.float32),
        base_qpos24=base_qpos24.astype(np.float32),
        qpos22=qpos22.astype(np.float32),
        qpos24=qpos24.astype(np.float32),
        thumb_spread=params[:, 0].astype(np.float32),
        finger_fan=params[:, 1].astype(np.float32),
        open_relax=params[:, 2].astype(np.float32),
        thumb_spread_range=np.array([THUMB_SPREAD_MIN, THUMB_SPREAD_MAX], dtype=np.float32),
        finger_fan_range=np.array([FAN_MIN, FAN_MAX], dtype=np.float32),
        open_relax_range=np.array([OPEN_RELAX_MIN, OPEN_RELAX_MAX], dtype=np.float32),
        fan_pattern_names=np.asarray(list(FAN_PATTERN_J.keys())),
        fan_pattern_coeffs=np.asarray(list(FAN_PATTERN_J.values()), dtype=np.float32),
        relax_j2_j1_scale=0.75,
        relax_thj1_scale=0.5,
        delta_min=delta.min(axis=0).astype(np.float32),
        delta_max=delta.max(axis=0).astype(np.float32),
        delta_mean=delta.mean(axis=0).astype(np.float32),
    )

    print(f"saved={args.out}")
    print(
        f"accepted_rows={qpos22.shape[0]} generated_candidates={generated} "
        f"acceptance_rate={qpos22.shape[0] / generated:.3f}"
    )
    finite_dist = sample_min_dist[np.isfinite(sample_min_dist)]
    if finite_dist.size:
        print(
            f"sample_min_contact_dist: min={finite_dist.min():+.6f} "
            f"mean={finite_dist.mean():+.6f} max={finite_dist.max():+.6f}"
        )
    else:
        print("sample_min_contact_dist: all samples have ncon=0")
    print(
        f"thumb_spread=[{params[:, 0].min():.4f},{params[:, 0].max():.4f}] "
        f"finger_fan=[{params[:, 1].min():.4f},{params[:, 1].max():.4f}] "
        f"open_relax=[{params[:, 2].min():+.4f},{params[:, 2].max():+.4f}]"
    )


if __name__ == "__main__":
    main()
