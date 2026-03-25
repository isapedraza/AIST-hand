"""
Builds PyG caches for the two ablation runs that precede R006 in the
additive feature chain:

  xyz only      -> hograspnet_{split}_c28_cmc_noflex_nocmc.pt
  xyz + flex    -> hograspnet_{split}_c28_cmc_nocmc.pt

Run from grasp-model/:
    python scripts/build_cache_ablation_xyz.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_gcn.dataset.grasps import GraspsClass

CONFIGS = [
    {
        "label": "xyz only (no flex, no CMC)",
        "kwargs": dict(add_joint_angles=False, add_cmc_angle=False),
    },
    {
        "label": "xyz + flex (no CMC)",
        "kwargs": dict(add_joint_angles=True, add_cmc_angle=False),
    },
]

for cfg in CONFIGS:
    print(f"\n{'='*55}")
    print(f"Config: {cfg['label']}")
    for split in ("train", "val", "test"):
        print(f"  Processing split: {split} ...")
        t0 = time.time()
        GraspsClass(
            root="data/",
            split=split,
            collapse=False,
            **cfg["kwargs"],
        )
        print(f"  Done in {time.time()-t0:.0f}s")

print("\nAll caches built.")
