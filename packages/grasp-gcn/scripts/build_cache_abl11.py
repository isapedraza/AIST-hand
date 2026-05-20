"""
Generates PyG graph cache for abl11.

abl11 features per node:
    - Dong wrist-local XYZ, normalized on-the-fly (3)
    - Dong joint quaternions (4)

Run from grasp-model/ with the venv activated:

    python scripts/build_cache_abl11.py [--dong-csv PATH]

Default Dong CSV path: data/processed/hograspnet_abl11.csv

Output .pt files used by the abl11 notebook:
    data/processed/hograspnet_dong_train_c28_noflex_dongq_xyz_normxyz.pt
    data/processed/hograspnet_dong_val_c28_noflex_dongq_xyz_normxyz.pt
    data/processed/hograspnet_dong_test_c28_noflex_dongq_xyz_normxyz.pt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_gcn.dataset.grasps import GraspsClass


EXPECTED_OUTPUTS = {
    "train": "data/processed/hograspnet_dong_train_c28_noflex_dongq_xyz_normxyz.pt",
    "val": "data/processed/hograspnet_dong_val_c28_noflex_dongq_xyz_normxyz.pt",
    "test": "data/processed/hograspnet_dong_test_c28_noflex_dongq_xyz_normxyz.pt",
}


def build_split(split: str, dong_csv: Path) -> None:
    print(f"\n{'=' * 50}")
    print(f"Processing split: {split}")
    print(f"Expected cache: {EXPECTED_OUTPUTS[split]}")
    t0 = time.time()

    ds = GraspsClass(
        root="data/",
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
        add_xyz=True,
        normalize_xyz=True,
    )

    elapsed = time.time() - t0
    if ds.num_features != 7:
        raise RuntimeError(f"Expected F=7 for abl11, got F={ds.num_features}")

    print(f"{split}: {len(ds):,} graphs | F={ds.num_features} | done in {elapsed / 60:.1f} min")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dong-csv",
        default="data/processed/hograspnet_abl11.csv",
        help="Path to hograspnet_abl11.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing abl11 cache files before rebuilding.",
    )
    args = parser.parse_args()

    dong_csv = Path(args.dong_csv).resolve()
    if not dong_csv.exists():
        print(f"ERROR: abl11 CSV not found at {dong_csv}")
        return 1

    print(f"abl11 CSV: {dong_csv} ({dong_csv.stat().st_size / 1e6:.0f} MB)")
    print("Features/node: 7 (xyz_norm=3, dong_quats=4)")

    existing = [Path(path) for path in EXPECTED_OUTPUTS.values() if Path(path).exists()]
    if existing and not args.overwrite:
        print("\nExisting abl11 cache files found; leaving them untouched:")
        for path in existing:
            print(f"  {path} ({path.stat().st_size / 1e6:.0f} MB)")
        print("\nUse --overwrite to rebuild them.")
        return 0

    if args.overwrite:
        for path in existing:
            print(f"Deleting existing cache: {path}")
            path.unlink()

    for split in ("train", "val", "test"):
        build_split(split, dong_csv)

    print("\nAll abl11 caches done.")
    for path in EXPECTED_OUTPUTS.values():
        cache = Path(path)
        status = "OK" if cache.exists() else "MISSING"
        size_mb = cache.stat().st_size / 1e6 if cache.exists() else 0
        print(f"  {status:7s} {cache} ({size_mb:.0f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
