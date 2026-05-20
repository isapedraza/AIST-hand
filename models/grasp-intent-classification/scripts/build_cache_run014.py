"""
Generates the PyG cache for Run 014 (bone + global_swing + AHG, 29 classes with Rest).
Run from grasp-model/ with the venv activated:

    python scripts/build_cache_run014.py

Requires: data/raw/hograspnet_r014.csv  (run merge_rest_pose.py first)

Output: data/processed/hograspnet_r014_{train,val,test}_c28_cmc_bone_swing_ahga_ahgd.pt
  (c28 tag because collapse=False; actual class count is 29 -- derived from data)
Upload those 3 files to Drive before running Colab.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_gcn.dataset.grasps import GraspsClass

for split in ('train', 'val', 'test'):
    print(f"\n{'='*50}")
    print(f"Processing split: {split}")
    t0 = time.time()
    GraspsClass(
        root='data/',
        split=split,
        collapse=False,
        add_bone_vectors=True,
        add_velocity=False,
        add_mano_pose=False,
        add_global_swing=True,
        add_ahg_angles=True,
        add_ahg_distances=True,
        csv_filename='hograspnet_r014.csv',
    )
    elapsed = time.time() - t0
    print(f"{split} cache done in {elapsed/60:.1f} min")

print("\nAll splits done. Upload data/processed/hograspnet_r014_*_ahga_ahgd.pt to Drive.")
