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
DEFAULT_OUT = ROOT / "grasp-model" / "data" / "processed" / "synthetic_close_hand_shadow_qpos.npz"

HAND_QPOS_DIM = 24
DEFAULT_ROWS = 29_670

JOINTS22 = [
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

LONG_FINGERS = {
    "FF": (1, 2, 3),
    "MF": (5, 6, 7),
    "RF": (9, 10, 11),
    "LF": (14, 15, 16),
}
CURL_WEIGHTS = np.array([0.6, 1.0, 0.8], dtype=np.float64)
ABDUCTION_IDXS = np.array([0, 4, 8, 13, 19], dtype=np.int64)
LFJ5_IDX = 12
THUMB_IDXS = np.array([17, 18, 20, 21], dtype=np.int64)


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


def contact_stats(model: mujoco.MjModel, data: mujoco.MjData, qpos24: np.ndarray) -> tuple[int, float]:
    data.qpos[:HAND_QPOS_DIM] = qpos24
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        return 0, np.inf
    min_dist = min(float(data.contact[i].dist) for i in range(data.ncon))
    return int(data.ncon), min_dist


def generate_candidates(
    rng: np.random.Generator,
    base_q22: np.ndarray,
    n: int,
    curl_sigma: float,
    residual_sigma: float,
    thumb_sigma: float,
    abduction_sigma: float,
    lfj5_sigma: float,
    max_abs_delta: float,
) -> np.ndarray:
    samples = np.repeat(base_q22[None, :], n, axis=0)

    for sample in samples:
        delta = np.zeros_like(base_q22)

        for idxs in LONG_FINGERS.values():
            curl = rng.normal(0.0, curl_sigma)
            delta[list(idxs)] += CURL_WEIGHTS * curl
            delta[list(idxs)] += rng.normal(0.0, residual_sigma, size=3)

        delta[THUMB_IDXS] += rng.normal(0.0, thumb_sigma, size=THUMB_IDXS.size)
        delta[ABDUCTION_IDXS] += rng.normal(0.0, abduction_sigma, size=ABDUCTION_IDXS.size)
        delta[LFJ5_IDX] += rng.normal(0.0, lfj5_sigma)
        delta += rng.normal(0.0, residual_sigma * 0.5, size=base_q22.size)

        delta = np.clip(delta, -max_abs_delta, max_abs_delta)
        sample[:] = np.clip(base_q22 + delta, LOW22, HIGH22)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--contact-tolerance-m", type=float, default=0.0005)
    parser.add_argument("--curl-sigma", type=float, default=0.02)
    parser.add_argument("--residual-sigma", type=float, default=0.004)
    parser.add_argument("--thumb-sigma", type=float, default=0.015)
    parser.add_argument("--abduction-sigma", type=float, default=0.003)
    parser.add_argument("--lfj5-sigma", type=float, default=0.003)
    parser.add_argument("--max-abs-delta", type=float, default=0.04)
    parser.add_argument("--max-candidates", type=int, default=2_000_000)
    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError(f"--rows must be positive, got {args.rows}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")

    base_qpos24 = load_keyframe("close hand")
    base_q22 = base_qpos24[2:].copy()
    model = mujoco.MjModel.from_xml_path(str(RIGHT_HAND))
    data = mujoco.MjData(model)

    baseline_ncon, baseline_min_dist = contact_stats(model, data, base_qpos24)
    min_allowed_dist = baseline_min_dist - args.contact_tolerance_m
    print(
        "baseline close hand: "
        f"ncon={baseline_ncon} min_contact_dist={baseline_min_dist:+.6f} "
        f"min_allowed_dist={min_allowed_dist:+.6f}"
    )

    rng = np.random.default_rng(args.seed)
    accepted22: list[np.ndarray] = []
    accepted_ncon: list[int] = []
    accepted_min_dist: list[float] = []
    generated = 0
    rejected_contact = 0

    while sum(chunk.shape[0] for chunk in accepted22) < args.rows:
        if generated >= args.max_candidates:
            raise RuntimeError(
                f"Only accepted {sum(chunk.shape[0] for chunk in accepted22)} rows after "
                f"{generated} candidates. Increase --max-candidates or relax filters."
            )

        n_batch = min(args.batch_size, args.max_candidates - generated)
        candidates22 = generate_candidates(
            rng=rng,
            base_q22=base_q22,
            n=n_batch,
            curl_sigma=args.curl_sigma,
            residual_sigma=args.residual_sigma,
            thumb_sigma=args.thumb_sigma,
            abduction_sigma=args.abduction_sigma,
            lfj5_sigma=args.lfj5_sigma,
            max_abs_delta=args.max_abs_delta,
        )
        generated += n_batch

        keep_rows: list[np.ndarray] = []
        for q22 in candidates22:
            qpos24 = np.concatenate([base_qpos24[:2], q22])
            ncon, min_dist = contact_stats(model, data, qpos24)
            if min_dist < min_allowed_dist:
                rejected_contact += 1
                continue
            keep_rows.append(q22.astype(np.float32))
            accepted_ncon.append(ncon)
            accepted_min_dist.append(min_dist)
            if sum(chunk.shape[0] for chunk in accepted22) + len(keep_rows) >= args.rows:
                break

        if keep_rows:
            accepted22.append(np.stack(keep_rows, axis=0))

        accepted_total = sum(chunk.shape[0] for chunk in accepted22)
        print(
            f"progress accepted={accepted_total}/{args.rows} generated={generated} "
            f"rejected_contact={rejected_contact} acceptance={accepted_total / generated:.3f}"
        )

    qpos22 = np.concatenate(accepted22, axis=0)[: args.rows]
    sample_ncon = np.asarray(accepted_ncon[: args.rows], dtype=np.int32)
    sample_min_dist = np.asarray(accepted_min_dist[: args.rows], dtype=np.float32)
    qpos24 = np.concatenate(
        [np.repeat(base_qpos24[None, :2].astype(np.float32), args.rows, axis=0), qpos22],
        axis=1,
    )
    delta = qpos22.astype(np.float64) - base_q22[None, :]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        synthetic_pose_name="synthetic_close_hand",
        source="mujoco_menagerie_shadow_hand_keyframe",
        source_keyframe="close hand",
        source_xml=str(RIGHT_HAND),
        keyframes_xml=str(KEYFRAMES_XML),
        seed=args.seed,
        target_rows=args.rows,
        accepted_rows=qpos22.shape[0],
        generated_candidates=generated,
        rejected_contact=rejected_contact,
        acceptance_rate=qpos22.shape[0] / generated,
        contact_tolerance_m=args.contact_tolerance_m,
        baseline_ncon=baseline_ncon,
        baseline_min_contact_dist=baseline_min_dist,
        min_allowed_contact_dist=min_allowed_dist,
        sample_ncon=sample_ncon,
        sample_min_contact_dist=sample_min_dist,
        joint_names=np.asarray(JOINTS22),
        joint_low=LOW22.astype(np.float32),
        joint_high=HIGH22.astype(np.float32),
        base_qpos22=base_q22.astype(np.float32),
        base_qpos24=base_qpos24.astype(np.float32),
        qpos22=qpos22.astype(np.float32),
        qpos24=qpos24.astype(np.float32),
        delta_min=delta.min(axis=0).astype(np.float32),
        delta_max=delta.max(axis=0).astype(np.float32),
        delta_mean=delta.mean(axis=0).astype(np.float32),
        curl_sigma=args.curl_sigma,
        residual_sigma=args.residual_sigma,
        thumb_sigma=args.thumb_sigma,
        abduction_sigma=args.abduction_sigma,
        lfj5_sigma=args.lfj5_sigma,
        max_abs_delta=args.max_abs_delta,
    )

    print(f"saved={args.out}")
    print(
        f"accepted_rows={qpos22.shape[0]} generated_candidates={generated} "
        f"acceptance_rate={qpos22.shape[0] / generated:.3f}"
    )
    print(
        f"sample_min_contact_dist: min={sample_min_dist.min():+.6f} "
        f"mean={sample_min_dist.mean():+.6f} max={sample_min_dist.max():+.6f}"
    )


if __name__ == "__main__":
    main()
