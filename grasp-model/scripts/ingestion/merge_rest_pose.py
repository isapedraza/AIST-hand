"""
Merge rest_pose_landmarks.csv into hograspnet.csv to produce hograspnet_r014.csv.

Run from grasp-model/ with the venv activated:

    python scripts/ingestion/merge_rest_pose.py

Inputs:
    data/raw/hograspnet.csv
    ../../grasp-app/rest_pose_landmarks.csv   (from capture_rest_pose.py)

Output:
    data/raw/hograspnet_r014.csv

Subject IDs in rest_pose CSV determine split membership:
    S11-S73  -> train
    S01-S10  -> val
    S74-S99  -> test
Capture each split separately using --subject_id (e.g. 50, 5, 80).
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
GRASP_MODEL  = SCRIPT_DIR.parents[1]
RAW_DIR      = GRASP_MODEL / "data" / "raw"
GRASP_APP    = GRASP_MODEL.parent / "grasp-app"

HOG_CSV      = RAW_DIR / "hograspnet.csv"
REST_CSV     = GRASP_APP / "rest_pose_landmarks.csv"
OUT_CSV      = RAW_DIR / "hograspnet_r014.csv"


def main():
    print(f"Loading HOGraspNet: {HOG_CSV}")
    df_hog  = pd.read_csv(HOG_CSV)
    print(f"  {len(df_hog):,} rows, classes: {sorted(df_hog['grasp_type'].unique())}")

    print(f"\nLoading rest pose: {REST_CSV}")
    df_rest = pd.read_csv(REST_CSV)
    print(f"  {len(df_rest):,} rows")
    print(f"  subject_ids: {sorted(df_rest['subject_id'].unique())}")
    print(f"  grasp_type values: {sorted(df_rest['grasp_type'].unique())}")

    # Validate rest pose columns match HOGraspNet columns
    missing = set(df_hog.columns) - set(df_rest.columns)
    extra   = set(df_rest.columns) - set(df_hog.columns)
    if missing:
        print(f"\nWARNING: columns in HOGraspNet but not in rest_pose (will be NaN): {missing}")
    if extra:
        print(f"WARNING: extra columns in rest_pose (will be dropped): {extra}")
        df_rest = df_rest[[c for c in df_hog.columns if c in df_rest.columns]]

    # Align columns to HOGraspNet order, fill missing with 0
    df_rest = df_rest.reindex(columns=df_hog.columns, fill_value=0)

    df_out = pd.concat([df_hog, df_rest], ignore_index=True)

    # Report split distribution of rest pose data
    from grasp_gcn.dataset.grasps import SPLIT_SUBJECTS
    for split, subs in SPLIT_SUBJECTS.items():
        n = df_rest[df_rest['subject_id'].isin(subs)].shape[0]
        print(f"  rest pose -> {split}: {n:,} rows")

    print(f"\nTotal rows: {len(df_out):,}")
    print(f"Classes: {sorted(df_out['grasp_type'].unique())}")
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(GRASP_MODEL / "src"))
    main()
