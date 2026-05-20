"""
Generates the PyG cache for Run 011 (bone + mano_pose, no velocity) locally.
Run from grasp-model/ with the venv activated:

    python scripts/build_cache_run011.py

Output: data/processed/hograspnet_{train,val,test}_c28_cmc_bone_pose.pt
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
        add_mano_pose=True,
    )
    elapsed = time.time() - t0
    print(f"{split} done in {elapsed/60:.1f} min")

print("\nAll splits done. Upload data/processed/*.pt to Drive.")
