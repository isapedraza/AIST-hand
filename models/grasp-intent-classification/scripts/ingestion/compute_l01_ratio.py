"""
compute_l01_ratio.py
--------------------
Estimate canonical thumb base ratio from a RAW HOGraspNet CSV.

Per-frame ratio:
  r_i = ||p1 - p0|| / ( ||p9-p0|| + ||p10-p9|| + ||p11-p10|| + ||p12-p11|| )

Robust aggregation:
  L1) median per block: (subject_id, date_id, object_id, grasp_type, trial_id, cam)
  L2) median per subject over L1 medians
  L3) global median over L2 subject medians

Output:
  - ratio_01 (recommended canonical ratio)
  - L01 for a given hand length (default 182 mm)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS_NEW = ["subject_id", "date_id", "object_id", "grasp_type", "trial_id", "cam"]
KEY_COLS_OLD = ["subject_id", "sequence_id", "cam", "grasp_type"]

XYZ_REQ = [
    "WRIST_x", "WRIST_y", "WRIST_z",  # kp0
    "THUMB_CMC_x", "THUMB_CMC_y", "THUMB_CMC_z",  # kp1
    "MIDDLE_FINGER_MCP_x", "MIDDLE_FINGER_MCP_y", "MIDDLE_FINGER_MCP_z",  # kp9
    "MIDDLE_FINGER_PIP_x", "MIDDLE_FINGER_PIP_y", "MIDDLE_FINGER_PIP_z",  # kp10
    "MIDDLE_FINGER_DIP_x", "MIDDLE_FINGER_DIP_y", "MIDDLE_FINGER_DIP_z",  # kp11
    "MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z",  # kp12
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute robust canonical ratio for link 0-1 from HOGraspNet RAW CSV."
    )
    p.add_argument(
        "--csv",
        default="data/raw/hograspnet_raw.csv",
        help="Input CSV path (default: data/raw/hograspnet_raw.csv)",
    )
    p.add_argument(
        "--hand-length-mm",
        type=float,
        default=182.0,
        help="Canonical hand length in mm (default: 182.0)",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="Optional JSON output path with computed metrics.",
    )
    return p.parse_args()


def _pick_key_cols(df_cols):
    if all(c in df_cols for c in KEY_COLS_NEW):
        return KEY_COLS_NEW
    if all(c in df_cols for c in KEY_COLS_OLD):
        return KEY_COLS_OLD
    raise KeyError(
        "CSV key columns not found. Expected either "
        f"{KEY_COLS_NEW} or {KEY_COLS_OLD}."
    )


def _norm(a, b):
    return np.linalg.norm(a - b, axis=1)


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read header first to choose grouping columns.
    head = pd.read_csv(csv_path, nrows=0)
    key_cols = _pick_key_cols(head.columns)
    usecols = key_cols + XYZ_REQ

    print(f"[INFO] Reading: {csv_path}")
    df = pd.read_csv(csv_path, usecols=usecols)
    print(f"[INFO] Rows: {len(df):,}")
    print(f"[INFO] Group keys: {key_cols}")

    p0 = df[["WRIST_x", "WRIST_y", "WRIST_z"]].to_numpy(np.float64)
    p1 = df[["THUMB_CMC_x", "THUMB_CMC_y", "THUMB_CMC_z"]].to_numpy(np.float64)
    p9 = df[["MIDDLE_FINGER_MCP_x", "MIDDLE_FINGER_MCP_y", "MIDDLE_FINGER_MCP_z"]].to_numpy(np.float64)
    p10 = df[["MIDDLE_FINGER_PIP_x", "MIDDLE_FINGER_PIP_y", "MIDDLE_FINGER_PIP_z"]].to_numpy(np.float64)
    p11 = df[["MIDDLE_FINGER_DIP_x", "MIDDLE_FINGER_DIP_y", "MIDDLE_FINGER_DIP_z"]].to_numpy(np.float64)
    p12 = df[["MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z"]].to_numpy(np.float64)

    l01 = _norm(p1, p0)
    lchain = _norm(p9, p0) + _norm(p10, p9) + _norm(p11, p10) + _norm(p12, p11)
    eps = 1e-12
    ratio = l01 / np.maximum(lchain, eps)

    agg = df[key_cols].copy()
    agg["ratio"] = ratio

    # L1: per block median
    l1 = agg.groupby(key_cols, as_index=False)["ratio"].median()
    # L2: per subject median
    l2 = l1.groupby(["subject_id"], as_index=False)["ratio"].median()
    # L3: global median over subjects
    ratio_final = float(l2["ratio"].median())

    l01_mm = float(args.hand_length_mm * ratio_final)
    l01_m = l01_mm / 1000.0

    print("\n[RESULT]")
    print(f"ratio_01 = {ratio_final:.15f}")
    print(f"ratio_01_percent = {ratio_final * 100:.6f}%")
    print(f"L01_mm (hand_length={args.hand_length_mm:.3f} mm) = {l01_mm:.9f}")
    print(f"L01_m  (hand_length={args.hand_length_mm:.3f} mm) = {l01_m:.12f}")
    print("\n[STATS]")
    print(f"L1 blocks: {len(l1):,}")
    print(f"L2 subjects: {len(l2):,}")
    print(f"frame_median_ratio: {float(np.median(ratio)):.15f}")
    print(f"frame_mean_ratio:   {float(np.mean(ratio)):.15f}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "csv": str(csv_path),
            "group_keys": key_cols,
            "rows": int(len(df)),
            "l1_blocks": int(len(l1)),
            "l2_subjects": int(len(l2)),
            "ratio_01": ratio_final,
            "ratio_01_percent": ratio_final * 100.0,
            "hand_length_mm": float(args.hand_length_mm),
            "l01_mm": l01_mm,
            "l01_m": l01_m,
            "frame_median_ratio": float(np.median(ratio)),
            "frame_mean_ratio": float(np.mean(ratio)),
        }
        out.write_text(json.dumps(result, indent=2))
        print(f"\n[INFO] JSON saved: {out}")


if __name__ == "__main__":
    main()
