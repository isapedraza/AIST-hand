"""
Generates PyG graph cache for abl10 (Dong quaternions only, F=4, no XYZ).

Run from grasp-model/ with the venv activated:

    python scripts/build_cache_abl10.py [--dong-csv PATH]

Default dong CSV path: data/processed/hograspnet_dong.csv

Output .pt files (upload all 3 to Drive before running Colab):
    data/processed/hograspnet_dong_train_c28_dongq.pt
    data/processed/hograspnet_dong_val_c28_dongq.pt
    data/processed/hograspnet_dong_test_c28_dongq.pt
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_gcn.dataset.grasps import GraspsClass

parser = argparse.ArgumentParser()
parser.add_argument('--dong-csv', default='data/processed/hograspnet_dong.csv',
                    help='Path to hograspnet_dong.csv')
args = parser.parse_args()

dong_csv = Path(args.dong_csv).resolve()
if not dong_csv.exists():
    print(f"ERROR: dong CSV not found at {dong_csv}")
    sys.exit(1)

print(f"Dong CSV: {dong_csv} ({dong_csv.stat().st_size / 1e6:.0f} MB)")

for split in ('train', 'val', 'test'):
    print(f"\n{'='*50}")
    print(f"Processing split: {split}")
    t0 = time.time()
    ds = GraspsClass(
        root='data/',
        split=split,
        collapse=False,
        add_joint_angles=False,
        add_bone_vectors=False,
        add_velocity=False,
        add_mano_pose=False,
        add_global_swing=False,
        add_ahg_angles=False,
        add_ahg_distances=False,
        add_dong_quats=True,
        dong_csv_path=str(dong_csv),
    )
    elapsed = time.time() - t0
    print(f"{split}: {len(ds):,} graphs | F={ds.num_features} | done in {elapsed/60:.1f} min")

print("\nAll splits done.")
print("Upload data/processed/hograspnet_dong_*_c28_dongq.pt to Drive.")
