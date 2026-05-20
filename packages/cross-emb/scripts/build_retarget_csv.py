"""
Merge hograspnet_dong.csv + hograspnet_raw.csv into hograspnet_retarget.csv.

Keeps:
  - 7 meta columns (subject_id, date_id, object_id, grasp_type, trial_id, cam, frame_id)
  - Dong quaternions q1-q20 (80 cols)
  - Raw XYZ landmarks WRIST + 20 joints (63 cols)

Drops:
  - wrist_alpha_deg, wrist_beta_deg, wrist_gamma_deg
  - beta/gamma scalar angle columns from Dong
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parents[1]
DONG_CSV = ROOT / "data" / "processed" / "hograspnet_dong.csv"
RAW_CSV  = ROOT / "data" / "raw"       / "hograspnet_raw.csv"
OUT_CSV  = ROOT / "data" / "processed" / "hograspnet_retarget.csv"

MERGE_KEYS = ["subject_id", "date_id", "object_id", "grasp_type", "trial_id", "cam", "frame_id"]

DROP_DONG = [
    "wrist_alpha_deg", "wrist_beta_deg", "wrist_gamma_deg",
    "beta1_deg", "gamma1_deg", "beta5_deg", "gamma5_deg",
    "beta9_deg", "gamma9_deg", "beta13_deg", "gamma13_deg",
    "beta17_deg", "gamma17_deg",
    "beta2_deg", "beta3_deg", "beta6_deg", "beta7_deg",
    "beta10_deg", "beta11_deg", "beta14_deg", "beta15_deg",
    "beta18_deg", "beta19_deg",
]


def main():
    print("Loading dong CSV...")
    dong = pd.read_csv(DONG_CSV)
    print(f"  {len(dong):,} rows, {len(dong.columns)} cols")

    print("Loading raw CSV...")
    raw = pd.read_csv(RAW_CSV)
    print(f"  {len(raw):,} rows, {len(raw.columns)} cols")

    # Drop unused columns from dong
    dong = dong.drop(columns=[c for c in DROP_DONG if c in dong.columns])
    print(f"  Dong after drop: {len(dong.columns)} cols")

    # Merge on keys
    print("Merging...")
    df = dong.merge(raw, on=MERGE_KEYS, how="inner")
    print(f"  Merged: {len(df):,} rows, {len(df.columns)} cols")

    print(f"Saving to {OUT_CSV}...")
    df.to_csv(OUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
