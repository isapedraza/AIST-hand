"""
Generates the PyG cache for Run 012 (bone + global_swing, no velocity, no mano_pose).
Run from grasp-model/ with the venv activated:

    python scripts/build_cache_run012.py

Output: data/processed/hograspnet_{train,val,test}_c28_cmc_bone_swing.pt
Upload those 3 files to Drive before running Colab.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MANO_PKL = '/home/yareeez/hamer_ckpt/_DATA/data/mano_v1_2/models/MANO_RIGHT.pkl'

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
        mano_pkl=MANO_PKL,
    )
    elapsed = time.time() - t0
    print(f"{split} done in {elapsed/60:.1f} min")

print("\nAll splits done. Upload data/processed/*_swing.pt to Drive.")
