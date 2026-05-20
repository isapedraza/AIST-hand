"""
Generates the PyG cache for abl03 (xyz + AHG only, no flex/cmc/bone/swing).
Run from grasp-model/ with the venv activated:

    python scripts/build_cache_abl03.py

Output: data/processed/hograspnet_{train,val,test}_c28_cmc_noflex_nocmc_ahga_ahgd.pt
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
        add_joint_angles=False,
        add_cmc_angle=False,
        add_bone_vectors=False,
        add_velocity=False,
        add_mano_pose=False,
        add_global_swing=False,
        add_ahg_angles=True,
        add_ahg_distances=True,
    )
    elapsed = time.time() - t0
    print(f"{split} cache done in {elapsed/60:.1f} min")

print("\nAll splits done. Upload data/processed/*_noflex_nocmc_ahga_ahgd.pt to Drive.")
