"""
Generates PyG caches for the AHG interaction ablation runs (abl04-abl07).
Each run is xyz + AHG + one additional feature, to identify which feature
synergizes with AHG.

Run from grasp-model/ with the venv activated:

    python scripts/build_cache_ablation.py          # all 4 runs (~6h total)
    python scripts/build_cache_ablation.py abl06    # single run (~90 min)

GraspsClass writes to data/processed/ first, then each .pt file is moved to
CACHE_DEST immediately after generation to keep the internal disk free.

Estimated time: ~90 min per run (train ~60 min, val ~11 min, test ~19 min).
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_gcn.dataset.grasps import GraspsClass

PROCESSED_DIR = Path("data/processed")
CACHE_DEST    = Path("/media/yareeez/94649A33649A1856/DATOS DE TESIS/caches")

RUNS = {
    "abl04": dict(
        label            = "abl04: xyz + AHG + flex",
        add_joint_angles = True,
        add_cmc_angle    = False,
        add_bone_vectors = False,
        add_global_swing = False,
    ),
    "abl05": dict(
        label            = "abl05: xyz + AHG + cmc",
        add_joint_angles = False,
        add_cmc_angle    = True,
        add_bone_vectors = False,
        add_global_swing = False,
    ),
    "abl06": dict(
        label            = "abl06: xyz + AHG + bone",
        add_joint_angles = False,
        add_cmc_angle    = False,
        add_bone_vectors = True,
        add_global_swing = False,
    ),
    "abl07": dict(
        label            = "abl07: xyz + AHG + swing",
        add_joint_angles = False,
        add_cmc_angle    = False,
        add_bone_vectors = False,
        add_global_swing = True,
    ),
}

# Parse optional CLI argument
requested = sys.argv[1].lower() if len(sys.argv) > 1 else None
if requested and requested not in RUNS:
    print(f"Unknown run '{requested}'. Valid options: {', '.join(RUNS)}")
    sys.exit(1)

runs_to_build = {requested: RUNS[requested]} if requested else RUNS

CACHE_DEST.mkdir(parents=True, exist_ok=True)

t_global = time.time()
print(f"Building {len(runs_to_build)} cache(s): {', '.join(runs_to_build)}")
print(f"Estimated time: ~{90 * len(runs_to_build)} min total")
print(f"Output: {CACHE_DEST}\n")

for run_id, cfg in runs_to_build.items():
    marker = CACHE_DEST / f"{run_id}.done"
    if marker.exists():
        print(f"\n  [{run_id}] already done -- skipping (found {marker.name})")
        continue

    print(f"\n{'#'*60}")
    print(f"  {cfg['label']}")
    print(f"{'#'*60}")
    t_run = time.time()

    for split in ('train', 'val', 'test'):
        print(f"\n  [{split}] processing...")
        t0 = time.time()

        # Snapshot existing .pt files before generation
        before = set(PROCESSED_DIR.glob("*.pt"))

        GraspsClass(
            root='data/',
            split=split,
            collapse=False,
            add_joint_angles  = cfg['add_joint_angles'],
            add_cmc_angle     = cfg['add_cmc_angle'],
            add_bone_vectors  = cfg['add_bone_vectors'],
            add_velocity      = False,
            add_mano_pose     = False,
            add_global_swing  = cfg['add_global_swing'],
            add_ahg_angles    = True,
            add_ahg_distances = True,
        )

        elapsed = time.time() - t0
        print(f"  [{split}] done in {elapsed/60:.1f} min")

        # Move new .pt files to external drive
        after = set(PROCESSED_DIR.glob("*.pt"))
        new_files = after - before
        for src in new_files:
            dst = CACHE_DEST / src.name
            shutil.move(str(src), str(dst))
            size_mb = dst.stat().st_size / 1e6
            print(f"  Moved: {src.name} ({size_mb:.0f} MB) -> {CACHE_DEST}")

    marker.touch()
    print(f"\n  {run_id} complete in {(time.time()-t_run)/60:.1f} min")

print(f"\n{'='*60}")
print(f"All done in {(time.time()-t_global)/60:.1f} min total.")
print(f"Files saved to: {CACHE_DEST}")
print("Upload them to Drive before running Colab.")
