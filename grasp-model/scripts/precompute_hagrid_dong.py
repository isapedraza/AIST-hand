#!/usr/bin/env python3
"""
Precompute Dong quaternion features for HaGRID open-hand/fist samples.

Reads HandLandmarks.csv from HandGesture2Emoji repo (HaGRID subset).
Outputs hagrid_dong.csv compatible with HumanLoader column format.

Classes used:
  2  = fist          -> closed_fist (grasp_type=29)
  7  = palm          -> open_hand   (grasp_type=28)
  10 = stop          -> open_hand   (grasp_type=28)
  11 = stop_inv      -> open_hand   (grasp_type=28)

Usage:
    python precompute_hagrid_dong.py \
        --input  /home/yareeez/HandGesture2Emoji/datasets/HandLandmarks.csv \
        --output /home/yareeez/AIST-hand/grasp-model/data/processed/hagrid_dong.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grasp-app" / "hand_preprocessing"))
from dong_kinematics import DongKinematics  # noqa: E402

# HaGRID class labels to include. The output grasp_type values intentionally
# live above the 28 HOGraspNet classes and are used only as metadata.
HAGRID_LABELS = {
    2: ("fist", 29, "closed_fist"),
    7: ("palm", 28, "open_hand"),
    10: ("stop", 28, "open_hand"),
    11: ("stop_inverted", 28, "open_hand"),
}
TARGET_CLASSES = list(HAGRID_LABELS)

# MediaPipe landmark indices for fingertips
TIP_IDX = {
    "THUMB_TIP":        4,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_TIP":12,
    "RING_FINGER_TIP":  16,
    "PINKY_TIP":        20,
}

# Landmark names in MediaPipe order (0-20)
MP_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",  "RING_FINGER_PIP",  "RING_FINGER_DIP",  "RING_FINGER_TIP",
    "PINKY_MCP",        "PINKY_PIP",        "PINKY_DIP",        "PINKY_TIP",
]

QUAT_COLS = [f"q{j}_{c}" for j in range(1, 21) for c in ("w", "x", "y", "z")]
XYZ_COLS  = [f"{n}_{ax}" for n in MP_NAMES for ax in ("x", "y", "z")]
TIP_COLS  = [f"{n}_{ax}" for n in TIP_IDX for ax in ("x", "y", "z")]


def _canonicalize_right(pts: np.ndarray, handedness: float) -> np.ndarray:
    """Reflect x for left hands to canonicalize to right hand."""
    if handedness == 1.0:
        pts = pts.copy()
        pts[:, 0] *= -1
    return pts


def _calibrate(dk: DongKinematics, samples: np.ndarray, n: int = 200) -> None:
    """Calibrate bone lengths using first n samples."""
    used = 0
    for pts in samples:
        if used >= n:
            break
        try:
            dk.calibrate(pts)
            used += 1
        except Exception:
            pass
    dk.force_freeze()
    print(f"Calibration done with {dk.calibration_count} frames.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/home/yareeez/HandGesture2Emoji/datasets/HandLandmarks.csv")
    parser.add_argument("--output", default="/home/yareeez/AIST-hand/grasp-model/data/processed/hagrid_dong.csv")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input, header=None)
    df_raw.columns = ["label", "handedness"] + [f"c{i}" for i in range(63)]

    # Filter target classes
    df = df_raw[df_raw["label"].isin(TARGET_CLASSES)].reset_index(drop=True)
    print(f"Samples after class filter: {len(df)}")
    print(df["label"].value_counts().sort_index())

    # Extract all landmarks [N, 21, 3]
    lm_vals = df[[f"c{i}" for i in range(63)]].values.astype(np.float64).reshape(-1, 21, 3)
    hand_vals = df["handedness"].values

    # Canonicalize to right hand
    lm_canon = np.stack([
        _canonicalize_right(lm_vals[i], hand_vals[i])
        for i in range(len(df))
    ])

    # Calibrate Dong with first 200 canonicalized samples (mix of all classes)
    dk = DongKinematics(calibration_frames=None)
    _calibrate(dk, lm_canon, n=200)

    # Process each sample
    rows = []
    failed = 0
    for i, (pts, row) in enumerate(zip(lm_canon, df.itertuples())):
        if i % 500 == 0:
            print(f"  {i}/{len(df)}")
        try:
            res = dk.process(pts)
        except Exception:
            failed += 1
            continue

        q = np.array([res["quaternions"][j] for j in res["joint_order"]])  # [20, 4]

        # Tips from landmark positions (normalized by hand_length = wrist-to-middle-tip)
        hand_length = np.linalg.norm(pts[12] - pts[0])
        if hand_length < 1e-6:
            failed += 1
            continue
        tips_norm = np.stack([pts[idx] / hand_length for idx in TIP_IDX.values()])  # [5, 3]

        source_label, grasp_type, anchor_label = HAGRID_LABELS[int(row.label)]
        r = {
            "source":     "hagrid",
            "source_label": source_label,
            "anchor_label": anchor_label,
            "subject_id": 9000 + int(row.label),  # reserved synthetic IDs
            "date_id":    0,
            "object_id":  grasp_type,
            "grasp_type": grasp_type,
            "trial_id":   i,            # unique per sample = no temporal pairing
            "cam":        0,
            "frame_id":   0,
        }

        # XYZ landmarks
        for j, name in enumerate(MP_NAMES):
            for k, ax in enumerate(("x", "y", "z")):
                r[f"{name}_{ax}"] = pts[j, k]

        # Quaternions
        for j in range(20):
            for k, c in enumerate(("w", "x", "y", "z")):
                r[f"q{j+1}_{c}"] = q[j, k]

        # Tip positions (normalized)
        for j, name in enumerate(TIP_IDX):
            for k, ax in enumerate(("x", "y", "z")):
                r[f"{name}_{ax}"] = tips_norm[j, k]

        rows.append(r)

    print(f"Processed: {len(rows)} | Failed: {failed}")
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output} | Shape: {out.shape}")


if __name__ == "__main__":
    main()
