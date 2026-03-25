"""
Quick MLP validation: is Rest (class 28) separable from the 28 HOGraspNet classes?

Uses flattened XYZ (63 features) from hograspnet_r014.csv.
Subsamples HOGraspNet to 5k/class to keep it fast.
Reports per-class F1 with focus on Rest.

Run from grasp-model/:
    python scripts/validate_rest_pose_mlp.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp_gcn.dataset.grasps import XYZ_COLS, SPLIT_SUBJECTS, GRASP_CLASS_NAMES

CSV = Path(__file__).resolve().parents[1] / "data/raw/hograspnet_r014.csv"
SAMPLES_PER_CLASS = 5_000   # cap HOGraspNet classes to keep training fast

print(f"Loading {CSV.name}...")
df = pd.read_csv(CSV, usecols=["subject_id", "grasp_type"] + XYZ_COLS)

train_df = df[df["subject_id"].isin(SPLIT_SUBJECTS["train"])]
test_df  = df[df["subject_id"].isin(SPLIT_SUBJECTS["test"])]

print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"Rest in train: {(train_df['grasp_type']==28).sum():,}")
print(f"Rest in test:  {(test_df['grasp_type']==28).sum():,}")

# Subsample majority classes so training is fast
parts = []
for cls, grp in train_df.groupby("grasp_type"):
    if len(grp) > SAMPLES_PER_CLASS:
        grp = resample(grp, n_samples=SAMPLES_PER_CLASS, random_state=42)
    parts.append(grp)
train_bal = pd.concat(parts).sample(frac=1, random_state=42)

X_train = train_bal[XYZ_COLS].values.astype(np.float32)
y_train = train_bal["grasp_type"].values

X_test  = test_df[XYZ_COLS].values.astype(np.float32)
y_test  = test_df["grasp_type"].values

print(f"\nTraining MLP on {len(X_train):,} samples ({len(np.unique(y_train))} classes)...")
t0 = time.time()
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    max_iter=30,
    random_state=42,
    verbose=True,
    early_stopping=True,
    n_iter_no_change=5,
)
clf.fit(X_train, y_train)
print(f"Done in {time.time()-t0:.1f}s")

print("\n--- Test set results ---")
class_names = {**GRASP_CLASS_NAMES, 28: "Rest"}
target_names = [class_names[i] for i in range(29)]
print(classification_report(y_test, clf.predict(X_test), target_names=target_names, zero_division=0))
