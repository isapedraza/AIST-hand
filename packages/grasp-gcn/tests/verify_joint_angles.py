"""
verify_joint_angles.py

Physical validation of the joint flexion angle feature before Run 003.
Three checks, all grounded in literature:

  1. Range:    all angles in [0, pi]         (Jarque-Bou 2019, Table 4 sign convention)
  2. Variance: PIP variance > DIP variance   (Jarque-Bou 2019, Synergy 1 — PIP 2-5 dominant)
  3. Ordering: Large Diameter PIP < Index Finger Extension PIP
               (Stival 2019 — cylindrical grasps characterized by PIP flexion;
                index finger extension isolated from all other movements)

Usage (from grasp-model/):
    python tests/verify_joint_angles.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_gcn.transforms.tograph import ToGraph
from grasp_gcn.dataset.grasps import GraspsClass

CSV = ROOT / "data" / "raw" / "grasps_train.csv"
N_SAMPLES = 5000  # rows to sample — enough for stable variance estimates

JOINTS = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
]

# Node indices for PIP and DIP joints
PIP_NODES = [6, 10, 14, 18]   # INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP
DIP_NODES = [7, 11, 15, 19]   # INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP

# Local class indices (from hograspnet_to_csv.py FEIX_INDICES order)
CLASS_LARGE_DIAMETER       = 0   # Feix 1
CLASS_INDEX_FINGER_EXT     = 2   # Feix 17


def row_to_sample(row):
    vals = np.array(row.iloc[2:65], dtype=np.float32).reshape(21, 3)
    sample = {'grasp_type': int(row['grasp_type'])}
    for i, j in enumerate(JOINTS):
        sample[j] = vals[i]
    return sample


def main():
    print(f"Loading {N_SAMPLES} samples from {CSV.name}...")
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()
    if 'grasp' in df.columns and 'grasp_type' not in df.columns:
        df.rename(columns={'grasp': 'grasp_type'}, inplace=True)

    df_sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=42).reset_index(drop=True)

    tg = ToGraph(make_undirected=True, add_joint_angles=True)

    angles_all = []   # [N, 21] — angle feature per node per sample
    classes    = []

    for i in range(len(df_sample)):
        sample = row_to_sample(df_sample.iloc[i])
        data = tg(sample)
        angles_all.append(data.x[:, 3].numpy())
        classes.append(sample['grasp_type'])

    angles_all = np.array(angles_all)  # [N, 21]
    classes    = np.array(classes)

    passed = True

    # ------------------------------------------------------------------
    # Check 1: Range [0, pi]
    # Jarque-Bou 2019 Table 4: flexion angles have defined sign convention;
    # arccos of normalized dot product produces values in [0, pi].
    # ------------------------------------------------------------------
    min_angle = angles_all.min()
    max_angle = angles_all.max()
    ok = (min_angle >= 0.0) and (max_angle <= np.pi + 1e-6)
    print(f"\n[Check 1] Range [0, pi]")
    print(f"  min={min_angle:.4f}  max={max_angle:.4f}  pi={np.pi:.4f}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # ------------------------------------------------------------------
    # Check 2: PIP variance > DIP variance
    # Jarque-Bou 2019 Synergy 1: PIP 2-5 flexion is the dominant universal
    # synergy (present in 70/77 subjects). DIP does not appear in any of
    # the three primary synergies, indicating lower inter-grasp variance.
    # ------------------------------------------------------------------
    pip_var = angles_all[:, PIP_NODES].var(axis=0).mean()
    dip_var = angles_all[:, DIP_NODES].var(axis=0).mean()
    ok = pip_var > dip_var
    print(f"\n[Check 2] PIP variance > DIP variance")
    print(f"  mean PIP variance={pip_var:.6f}  mean DIP variance={dip_var:.6f}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # ------------------------------------------------------------------
    # Check 3: Per-class ordering at PIP joints
    # Stival 2019: cylindrical grasps (Large Diameter) characterized by
    # PIP flexion (small angle). Index Finger Extension is "clearly distant
    # from all other movements" — fingers 2-5 extended (angle close to pi).
    # ------------------------------------------------------------------
    mask_ld  = classes == CLASS_LARGE_DIAMETER
    mask_ife = classes == CLASS_INDEX_FINGER_EXT

    if mask_ld.sum() == 0 or mask_ife.sum() == 0:
        print(f"\n[Check 3] Skipped — not enough samples for both classes")
        print(f"  Large Diameter: {mask_ld.sum()}  Index Finger Ext: {mask_ife.sum()}")
    else:
        pip_ld  = angles_all[mask_ld][:, PIP_NODES].mean()
        pip_ife = angles_all[mask_ife][:, PIP_NODES].mean()
        ok = pip_ld < pip_ife
        print(f"\n[Check 3] Large Diameter PIP angle < Index Finger Extension PIP angle")
        print(f"  Large Diameter mean PIP angle    = {pip_ld:.4f} rad ({np.degrees(pip_ld):.1f} deg)")
        print(f"  Index Finger Ext mean PIP angle  = {pip_ife:.4f} rad ({np.degrees(pip_ife):.1f} deg)")
        print(f"  {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    print(f"\n{'All checks passed.' if passed else 'One or more checks FAILED — review ToGraph implementation.'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
