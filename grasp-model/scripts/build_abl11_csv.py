"""
Build hograspnet_abl11.csv from hograspnet_raw.csv.

Output columns (per frame):
  - 7  meta:  subject_id, date_id, object_id, grasp_type, trial_id, cam, frame_id
  - 63 XYZ:   WRIST_x/y/z ... PINKY_TIP_x/y/z  (Dong wrist-local frame, NOT normalized)
  - 80 quats: q1_w/x/y/z  ... q20_w/x/y/z       (Dong joint quaternions)

XYZ are in the Dong wrist-local frame (Block 1, Dong Eq. 5-9/16).
Wrist is always the origin [0,0,0]. Divide by hand_length at training time.

Calibration: same as precompute_dong_features.py -- DongKinematics per subject,
frozen from first 50 valid frames (median bone lengths).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2] / "grasp-app" / "hand_preprocessing"))
from dong_kinematics import DongKinematics, _block_1_world_to_wrist_local

RAW_CSV = Path(__file__).parents[1] / "data" / "raw"       / "hograspnet_raw.csv"
OUT_CSV = Path(__file__).parents[1] / "data" / "processed" / "hograspnet_abl11.csv"

META_COLS = ["subject_id", "date_id", "object_id", "grasp_type", "trial_id", "cam", "frame_id"]

JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def main():
    print(f"Reading {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"  {len(df):,} rows, {df['subject_id'].nunique()} subjects")

    xyz_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    assert len(xyz_cols) == 63, f"Expected 63 XYZ cols, got {len(xyz_cols)}"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    results = []
    skipped = 0
    processed = 0
    first_write = True

    with tqdm(total=len(df), desc="Building abl11 CSV") as pbar:
        for subject_id, group in df.groupby("subject_id"):
            dk = DongKinematics(calibration_frames=None)

            # Calibrate on first 50 valid frames
            calib = 0
            for _, row in group.iterrows():
                if calib >= 50:
                    break
                pts = row[xyz_cols].values.astype(np.float64).reshape(21, 3)
                try:
                    dk.calibrate(pts)
                    calib += 1
                except ValueError:
                    pass
            dk.force_freeze()

            for _, row in group.iterrows():
                pts_w = row[xyz_cols].values.astype(np.float64).reshape(21, 3)

                try:
                    # Block 1: world -> Dong wrist-local frame
                    pts_local = _block_1_world_to_wrist_local(pts_w)["points_0"]  # [21,3]
                    # Quaternions
                    res = dk.process(pts_w)
                except ValueError:
                    skipped += 1
                    pbar.update(1)
                    continue

                out_row = {c: row[c] for c in META_COLS}

                # XYZ in Dong wrist-local (wrist = origin = [0,0,0])
                for i, joint in enumerate(JOINTS):
                    out_row[f"{joint}_x"] = float(pts_local[i, 0])
                    out_row[f"{joint}_y"] = float(pts_local[i, 1])
                    out_row[f"{joint}_z"] = float(pts_local[i, 2])

                # Quaternions q1..q20
                for j in res["joint_order"]:
                    q = res["quaternions"][j]
                    out_row[f"q{j}_w"] = float(q[0])
                    out_row[f"q{j}_x"] = float(q[1])
                    out_row[f"q{j}_y"] = float(q[2])
                    out_row[f"q{j}_z"] = float(q[3])

                results.append(out_row)
                pbar.update(1)

            # Flush subject batch to disk
            if results:
                pd.DataFrame(results).to_csv(OUT_CSV, mode="a", header=first_write, index=False)
                first_write = False
                processed += len(results)
                results = []

    print(f"\nDone. Processed: {processed:,}  Skipped (degenerate): {skipped:,}")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
