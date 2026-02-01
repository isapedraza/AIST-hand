import pandas as pd

for split in ["train", "val", "test"]:
    path = f"data/raw/grasps_sample_{split}.csv"
    df = pd.read_csv(path, nrows=1)
    print(f"\n📄 {split.upper()} — columns:")
    print(df.columns.tolist())
