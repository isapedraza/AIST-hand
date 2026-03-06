"""
verify_cmc_abduction.py

Physical validation of the palmar abduction angle feature (θ_CMC) before Run 005.
Three checks, all grounded in Feix et al. (2016):

  1. Range:   all θ_CMC in [0, π/2]         (arcsin of absolute dot product)
  2. Global:  mean θ_CMC(abducted classes) > mean θ_CMC(adducted classes)
              (Feix Section IV: abducted/adducted is the row separator in the taxonomy matrix)
  3. Pairs:   specific high-contrast class pairs confirm the ordering per Feix Fig. 4
              a. Adducted Thumb (local 7)  < Tripod (local 27)
              b. Lateral (local 15)        < Palmar Pinch (local 20)

Feature definition (corrected from earlier candidate):
    thumb_dir   = normalize(THUMB_MCP - THUMB_CMC)                    # landmark 2 - 1
    palm_normal = normalize(cross(WRIST→INDEX_MCP, WRIST→PINKY_MCP))  # palm plane normal
    θ_CMC       = arcsin(|dot(thumb_dir, palm_normal)|)               # ∈ [0, π/2]

Why THUMB_CMC→THUMB_MCP and not WRIST→THUMB_CMC:
    THUMB_CMC (landmark 1) is the carpometacarpal joint, anatomically fixed relative to
    the wrist. After geometric normalization its position does not change with abduction.
    What changes is the direction the first metacarpal points from that joint (landmark 1→2).

Why out-of-plane component and not in-plane angle:
    Feix's row separator is palmar abduction (thumb rotating out of the palm plane to oppose
    fingertips), not radial abduction (spreading within the palm plane). The out-of-plane
    component of the metacarpal direction is the geometrically correct quantity.

Usage (from grasp-model/):
    python tests/verify_cmc_abduction.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CSV = ROOT / "data" / "raw" / "grasps_train.csv"
N_SAMPLES = 5000

# Joint indices in the 21-landmark skeleton (MediaPipe / HOGraspNet order)
IDX_WRIST       = 0
IDX_THUMB_CMC   = 1
IDX_THUMB_MCP   = 2
IDX_INDEX_MCP   = 5
IDX_PINKY_MCP   = 17

# Class categories from Feix taxonomy matrix (28 local indices, before collapse)
# Row 1 — Thumb Abducted (thumb pad can oppose fingertips)
ABDUCTED = {0, 1, 3, 6, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}
# Row 2 — Thumb Adducted (thumb alongside or across palm)
ADDUCTED = {2, 4, 5, 7, 8, 15, 16}

# Class names for the report
CLASS_NAMES = {
    0: "Large Diameter",      1: "Small Diameter",       2: "Index Finger Ext",
    3: "Extension Type",      4: "Parallel Extension",   5: "Palmar",
    6: "Medium Wrap",         7: "Adducted Thumb",       8: "Light Tool",
    9: "Distal",             10: "Ring",                11: "Power Disk",
    12: "Power Sphere",      13: "Sphere 4-Finger",     14: "Sphere 3-Finger",
    15: "Lateral",           16: "Stick",               17: "Adduction Grip",
    18: "Writing Tripod",    19: "Lateral Tripod",      20: "Palmar Pinch",
    21: "Tip Pinch",         22: "Inferior Pincer",     23: "Prismatic 3F",
    24: "Precision Disk",    25: "Precision Sphere",    26: "Quadpod",
    27: "Tripod",
}

# High-contrast pairs for Check 3: (adducted_class, abducted_class)
CONTRAST_PAIRS = [
    (7,  27, "Adducted Thumb", "Tripod"),
    (15, 20, "Lateral",        "Palmar Pinch"),
]


def compute_theta_cmc(vals):
    """
    Compute the palmar abduction angle θ_CMC for a single sample.

    vals: np.ndarray of shape [21, 3] — normalized 3D landmarks (WRIST at origin).
    Returns a scalar in [0, π/2].
    """
    thumb_dir   = vals[IDX_THUMB_MCP] - vals[IDX_THUMB_CMC]
    norm        = np.linalg.norm(thumb_dir)
    if norm < 1e-8:
        return 0.0
    thumb_dir  /= norm

    # WRIST is at origin after normalization, so WRIST→X = vals[X]
    index_mcp   = vals[IDX_INDEX_MCP]
    pinky_mcp   = vals[IDX_PINKY_MCP]
    palm_normal = np.cross(index_mcp, pinky_mcp)
    norm        = np.linalg.norm(palm_normal)
    if norm < 1e-8:
        return 0.0
    palm_normal /= norm

    dot_val = np.clip(np.abs(np.dot(thumb_dir, palm_normal)), 0.0, 1.0)
    return float(np.arcsin(dot_val))


def main():
    print(f"Loading {N_SAMPLES} samples from {CSV.name}...")
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()
    if 'grasp' in df.columns and 'grasp_type' not in df.columns:
        df.rename(columns={'grasp': 'grasp_type'}, inplace=True)

    df_sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=42).reset_index(drop=True)

    thetas  = []
    classes = []

    for i in range(len(df_sample)):
        row  = df_sample.iloc[i]
        vals = np.array(row.iloc[2:65], dtype=np.float64).reshape(21, 3)
        thetas.append(compute_theta_cmc(vals))
        classes.append(int(row['grasp_type']))

    thetas  = np.array(thetas)
    classes = np.array(classes)

    passed = True

    # ------------------------------------------------------------------
    # Check 1: Range [0, π/2]
    # arcsin(|x|) with x ∈ [-1, 1] always returns values in [0, π/2].
    # Values outside this range indicate a bug in compute_theta_cmc.
    # ------------------------------------------------------------------
    min_t = thetas.min()
    max_t = thetas.max()
    ok    = (min_t >= 0.0) and (max_t <= np.pi / 2 + 1e-6)
    print(f"\n[Check 1] Range [0, π/2]")
    print(f"  min={min_t:.4f}  max={max_t:.4f}  π/2={np.pi/2:.4f}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # ------------------------------------------------------------------
    # Check 2: mean θ_CMC(abducted) > mean θ_CMC(adducted)
    # Feix Section IV: abducted position allows thumb pad to oppose fingertips
    # (metacarpal points out of palm plane → large θ_CMC).
    # Adducted: metacarpal lies in palm plane → small θ_CMC.
    # ------------------------------------------------------------------
    mask_abd = np.isin(classes, list(ABDUCTED))
    mask_add = np.isin(classes, list(ADDUCTED))
    mean_abd = thetas[mask_abd].mean()
    mean_add = thetas[mask_add].mean()
    ok       = mean_abd > mean_add
    print(f"\n[Check 2] mean θ_CMC(abducted) > mean θ_CMC(adducted)")
    print(f"  abducted classes  ({mask_abd.sum():4d} samples): mean={mean_abd:.4f} rad ({np.degrees(mean_abd):.1f}°)")
    print(f"  adducted classes  ({mask_add.sum():4d} samples): mean={mean_add:.4f} rad ({np.degrees(mean_add):.1f}°)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # ------------------------------------------------------------------
    # Check 3: High-contrast class pairs
    # Each pair is (adducted_class, abducted_class) selected from Feix Fig. 4
    # as clear representatives of their row.
    # ------------------------------------------------------------------
    print(f"\n[Check 3] High-contrast class pairs")
    for cls_add, cls_abd, name_add, name_abd in CONTRAST_PAIRS:
        m_add = classes == cls_add
        m_abd = classes == cls_abd
        if m_add.sum() == 0 or m_abd.sum() == 0:
            print(f"  Skipped ({name_add} vs {name_abd}) — not enough samples")
            print(f"    {name_add}: {m_add.sum()}  {name_abd}: {m_abd.sum()}")
            continue
        t_add = thetas[m_add].mean()
        t_abd = thetas[m_abd].mean()
        ok_pair = t_add < t_abd
        print(f"  {name_add:20s} (adducted): {t_add:.4f} rad ({np.degrees(t_add):.1f}°)")
        print(f"  {name_abd:20s} (abducted): {t_abd:.4f} rad ({np.degrees(t_abd):.1f}°)")
        print(f"  {'PASS' if ok_pair else 'FAIL'}")
        if not ok_pair:
            passed = False

    # ------------------------------------------------------------------
    # Summary: per-class mean θ_CMC sorted ascending
    # Visual check that adducted classes cluster at the bottom.
    # ------------------------------------------------------------------
    print(f"\n[Summary] Per-class mean θ_CMC (ascending)")
    print(f"  {'Idx':>3}  {'Grasp':<22}  {'Row':<10}  {'mean θ':>8}  {'deg':>6}  {'n':>5}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*5}")
    per_class = []
    for c in sorted(CLASS_NAMES.keys()):
        mask = classes == c
        if mask.sum() == 0:
            continue
        row_label = "adducted" if c in ADDUCTED else "abducted"
        mean_t    = thetas[mask].mean()
        per_class.append((mean_t, c, row_label))
    for mean_t, c, row_label in sorted(per_class):
        print(f"  {c:>3}  {CLASS_NAMES[c]:<22}  {row_label:<10}  {mean_t:8.4f}  {np.degrees(mean_t):6.1f}  {(classes==c).sum():5d}")

    print(f"\n{'All checks passed.' if passed else 'One or more checks FAILED — review feature definition.'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
