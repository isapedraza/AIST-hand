"""
Build hograspnet_abl13.csv by joining hograspnet_abl11.csv with the Euler
angle columns from hograspnet_dong.csv.

Output columns (per frame):
  -  7 meta:  subject_id, date_id, object_id, grasp_type, trial_id, cam, frame_id
  - 63 XYZ:   WRIST_x/y/z ... PINKY_TIP_x/y/z  (Dong wrist-local frame, NOT normalized)
  - 80 quats: q1_w/x/y/z  ... q20_w/x/y/z
  - 20 euler: beta1_deg, gamma1_deg, beta2_deg, beta3_deg, ...  (Dong Eq.13-36)

Euler columns added after quats, in the same order they appear in hograspnet_dong.csv.
"""

import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
ABL11_CSV = PROCESSED / "hograspnet_abl11.csv"
DONG_CSV  = PROCESSED / "hograspnet_dong.csv"
OUT_CSV   = PROCESSED / "hograspnet_abl13.csv"

META_COLS = ["subject_id", "date_id", "object_id", "grasp_type", "trial_id", "cam", "frame_id"]

EULER_COLS = [
    "beta1_deg", "gamma1_deg",
    "beta5_deg", "gamma5_deg",
    "beta9_deg", "gamma9_deg",
    "beta13_deg", "gamma13_deg",
    "beta17_deg", "gamma17_deg",
    "beta2_deg", "beta3_deg",
    "beta6_deg", "beta7_deg",
    "beta10_deg", "beta11_deg",
    "beta14_deg", "beta15_deg",
    "beta18_deg", "beta19_deg",
]


def main():
    print(f"Reading {ABL11_CSV}")
    abl11 = pd.read_csv(ABL11_CSV)
    print(f"  {len(abl11):,} rows, {len(abl11.columns)} cols")

    print(f"Reading {DONG_CSV}")
    dong = pd.read_csv(DONG_CSV, usecols=META_COLS + EULER_COLS)
    print(f"  {len(dong):,} rows, {len(dong.columns)} cols")

    print("Merging...")
    merged = abl11.merge(dong[META_COLS + EULER_COLS], on=META_COLS, how="inner")
    print(f"  {len(merged):,} rows after inner join (dropped {len(abl11) - len(merged):,})")
    print(f"  Total cols: {len(merged.columns)}")

    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
