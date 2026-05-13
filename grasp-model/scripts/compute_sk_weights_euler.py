#!/usr/bin/env python3
"""
Compute per-segment weights for angle-based D_R (Run 22+).

Uses Dong Eq.24 angles from hograspnet_abl13.csv instead of quaternion distances.

D_R weights: w_j = (1/sigma_j) / sum(1/sigma)   [low-variance angles get high weight]

Per finger, 4 components (sum=1):
    mcp_flex : w for |beta_mcp_a  - beta_mcp_b|   (MCP flexion)
    mcp_abd  : w for |gamma_mcp_a - gamma_mcp_b|   (MCP abduction)
    pip      : w for |beta_pip_a  - beta_pip_b|    (PIP flexion)
    dip      : w for |beta_dip_a  - beta_dip_b|    (DIP flexion)

Angles are in degrees in abl13 -- converted to radians before sigma computation
so weights are consistent with robot NPZ (radians).

Usage:
    cd /home/yareeez/AIST-hand/grasp-model
    /home/yareeez/AIST-hand/.venv/bin/python scripts/compute_sk_weights_euler.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path("/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed/hograspnet_abl13.csv")
N_PAIRS  = 50_000
SEED     = 42

# S1 train split: subjects 11-73
TRAIN_LO, TRAIN_HI = 11, 73

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# abl13 column mapping: finger -> (beta_mcp, gamma_mcp, beta_pip, beta_dip)
ANGLE_COLS = {
    "thumb":  ("beta1_deg",  "gamma1_deg",  "beta2_deg",  "beta3_deg"),
    "index":  ("beta5_deg",  "gamma5_deg",  "beta6_deg",  "beta7_deg"),
    "middle": ("beta9_deg",  "gamma9_deg",  "beta10_deg", "beta11_deg"),
    "ring":   ("beta13_deg", "gamma13_deg", "beta14_deg", "beta15_deg"),
    "pinky":  ("beta17_deg", "gamma17_deg", "beta18_deg", "beta19_deg"),
}

DEG2RAD = np.pi / 180.0

print(f"Loading {CSV_PATH} (train split subjects {TRAIN_LO}-{TRAIN_HI}) ...")
df = pd.read_csv(CSV_PATH)
mask = (df["subject_id"] >= TRAIN_LO) & (df["subject_id"] <= TRAIN_HI)
df = df[mask].reset_index(drop=True)
N = len(df)
print(f"Frames: {N:,}")

rng = np.random.default_rng(SEED)
ia  = rng.integers(0, N, N_PAIRS)
ib  = rng.integers(0, N, N_PAIRS)

print("\n" + "="*60)
print("D_R euler weights  (1/sigma normalized)")
print("  sigma_j = std(|angle_a - angle_b|)  [radians]")
print("  components per finger: mcp_flex, mcp_abd, pip, dip")
print("="*60)

weights = {}
for finger in FINGERS:
    beta_mcp_col, gamma_mcp_col, beta_pip_col, beta_dip_col = ANGLE_COLS[finger]

    vals_mcp_flex = df[beta_mcp_col].values  * DEG2RAD
    vals_mcp_abd  = df[gamma_mcp_col].values * DEG2RAD
    vals_pip      = df[beta_pip_col].values  * DEG2RAD
    vals_dip      = df[beta_dip_col].values  * DEG2RAD

    diff_mcp_flex = np.abs(vals_mcp_flex[ia] - vals_mcp_flex[ib])
    diff_mcp_abd  = np.abs(vals_mcp_abd[ia]  - vals_mcp_abd[ib])
    diff_pip      = np.abs(vals_pip[ia]       - vals_pip[ib])
    diff_dip      = np.abs(vals_dip[ia]       - vals_dip[ib])

    sigma_mcp_flex = diff_mcp_flex.std()
    sigma_mcp_abd  = diff_mcp_abd.std()
    sigma_pip      = diff_pip.std()
    sigma_dip      = diff_dip.std()

    sigmas = [sigma_mcp_flex, sigma_mcp_abd, sigma_pip, sigma_dip]
    inv    = [1.0 / s for s in sigmas]
    total  = sum(inv)
    w      = [v / total for v in inv]

    weights[finger] = {
        "mcp_flex": w[0], "mcp_abd": w[1], "pip": w[2], "dip": w[3],
    }

    print(f"\n{finger}:")
    labels = ["mcp_flex", "mcp_abd", "pip", "dip"]
    for lbl, s, wi in zip(labels, sigmas, w):
        print(f"  {lbl:10s}: sigma={s:.4f} rad  w={wi:.4f}")

print("\n\nHardcode block for train_cross_emb.py (_sk_w_euler):")
print("_sk_w_euler = {")
for finger in FINGERS:
    wf = weights[finger]
    print(f'    "{finger}": dict(mcp_flex={wf["mcp_flex"]:.4f}, mcp_abd={wf["mcp_abd"]:.4f}, pip={wf["pip"]:.4f}, dip={wf["dip"]:.4f}),')
print("}")
