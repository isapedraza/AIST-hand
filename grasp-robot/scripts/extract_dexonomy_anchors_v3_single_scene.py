"""
Extract two canonical pose anchors per grasp class from the Dexonomy dataset.

For each of the 28 HOGraspNet grasp classes (where Dexonomy has coverage),
we find the single grasp with the smallest scene_scale (smallest object) and use:

  pose_open  = pregrasp_qpos of that grasp  (hand approaching smallest object)
  pose_close = squeeze_qpos  of that grasp  (max force on smallest object)

Using the smallest object ensures the hand reaches maximum closure for that grasp type,
and both anchors come from the same scene (consistent trajectory).
  qpos = (1 - apertura) * pose_open + apertura * pose_close

Output: grasp-robot/grasp_configs/shadow_hand_canonical.yaml

--- Dexonomy qpos layout (29 values, confirmed from DexLearn/base_dex.py) ---
[0:3]  hand root translation (x, y, z)  -- ignored
[3:7]  hand root quaternion (w, x, y, z) -- ignored
[7:29] 22 finger joints in MuJoCo order (same as Menagerie, no wrist):
  7:FFJ4  8:FFJ3  9:FFJ2  10:FFJ1
  11:MFJ4 12:MFJ3 13:MFJ2 14:MFJ1
  15:RFJ4 16:RFJ3 17:RFJ2 18:RFJ1
  19:LFJ5 20:LFJ4 21:LFJ3 22:LFJ2 23:LFJ1
  24:THJ5 25:THJ4 26:THJ3 27:THJ2 28:THJ1

--- MuJoCo Menagerie joint ordering (24 DOF) ---
0:rh_WRJ2  1:rh_WRJ1  (set to 0, no wrist in Dexonomy)
2:rh_FFJ4  3:rh_FFJ3  4:rh_FFJ2  5:rh_FFJ1
6:rh_MFJ4  7:rh_MFJ3  8:rh_MFJ2  9:rh_MFJ1
10:rh_RFJ4 11:rh_RFJ3 12:rh_RFJ2 13:rh_RFJ1
14:rh_LFJ5 15:rh_LFJ4 16:rh_LFJ3 17:rh_LFJ2 18:rh_LFJ1
19:rh_THJ5 20:rh_THJ4 21:rh_THJ3 22:rh_THJ2 23:rh_THJ1

Run from AIST-hand/:
    python grasp-robot/scripts/extract_dexonomy_anchors.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
GRASP_DIR  = Path("/media/yareeez/94649A33649A1856/dexonomy/Dexonomy_GRASP_shadow/succ_collect")
OUT_YAML   = ROOT / "grasp-robot" / "grasp_configs" / "shadow_hand_canonical.yaml"

MAX_FILES_PER_CLASS = 150   # ~3 300 poses per class (each file has ~22 poses)

# ---------------------------------------------------------------------------
# Feix ID (Dexonomy folder prefix) -> HOGraspNet class ID
# Classes not present in HOGraspNet 28 map to None.
# ---------------------------------------------------------------------------
FEIX_TO_HOG: dict[int, int | None] = {
    1:  0,    # Large Diameter
    2:  1,    # Small Diameter
    3:  6,    # Medium Wrap
    4:  7,    # Adducted Thumb
    5:  8,    # Light Tool
    6:  None, # Prismatic 4 Finger  (not in HOGraspNet)
    7:  23,   # Prismatic 3 Finger
    8:  None, # Prismatic 2 Finger  (not in HOGraspNet)
    9:  20,   # Palmar Pinch
    10: 11,   # Power Disk
    11: 12,   # Power Sphere
    12: 24,   # Precision Disk
    13: 25,   # Precision Sphere
    14: 27,   # Tripod
    15: None, # Fixed Hook          (not in HOGraspNet)
    16: 15,   # Lateral
    17: 2,    # Index Finger Extension
    18: 3,    # Extension Type
    20: 18,   # Writing Tripod
    22: 4,    # Parallel Extension
    23: 17,   # Adduction Grip
    24: 21,   # Tip Pinch
    25: 19,   # Lateral Tripod
    26: 13,   # Sphere 4-Finger
    27: 26,   # Quadpod
    28: 14,   # Sphere 3-Finger
    29: 16,   # Stick
    30: 5,    # Palmar
    31: 10,   # Ring
    32: None, # Ventral             (not in HOGraspNet)
    33: 22,   # Inferior Pincer
}

HOG_CLASS_NAMES = {
    0: "Large Diameter",        1: "Small Diameter",
    2: "Index Finger Extension",3: "Extension Type",
    4: "Parallel Extension",    5: "Palmar",
    6: "Medium Wrap",           7: "Adducted Thumb",
    8: "Light Tool",            9: "Distal",
    10: "Ring",                 11: "Power Disk",
    12: "Power Sphere",         13: "Sphere 4-Finger",
    14: "Sphere 3-Finger",      15: "Lateral",
    16: "Stick",                17: "Adduction Grip",
    18: "Writing Tripod",       19: "Lateral Tripod",
    20: "Palmar Pinch",         21: "Tip Pinch",
    22: "Inferior Pincer",      23: "Prismatic 3 Finger",
    24: "Precision Disk",       25: "Precision Sphere",
    26: "Quadpod",              27: "Tripod",
}

MUJOCO_JOINT_NAMES = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

# qpos layout (confirmed from DexLearn/base_dex.py):
#   [0:3]  = hand root translation (ignore)
#   [3:7]  = hand root quaternion  (ignore)
#   [7:29] = 22 finger joints, same order as MuJoCo Menagerie (no wrist)
#
# MuJoCo Menagerie has 24 joints: WRJ2, WRJ1 first, then the same 22.
# WRJ2 and WRJ1 are set to 0 (Dexonomy hand floats freely, no wrist joints).
DEX_FINGER_SLICE = slice(7, 29)   # qpos[7:29] -> 22 joints
MJC_WR_OFFSET    = 2              # Menagerie joints 0,1 = WRJ2,WRJ1 (set to 0)


def dex29_to_mjc24(qpos29: np.ndarray) -> np.ndarray:
    """Convert [29] Dexonomy qpos to [24] MuJoCo Menagerie qpos.
    qpos29[0:7] = free joint (pos+quat), ignored.
    qpos29[7:29] = 22 finger joints, same order as Menagerie joints 2-23.
    Menagerie joints 0-1 (WRJ2, WRJ1) = 0.
    """
    out = np.zeros(24, dtype=np.float64)
    out[MJC_WR_OFFSET:] = qpos29[DEX_FINGER_SLICE]
    return out



def extract_poses() -> dict[int, np.ndarray]:
    """
    Read up to MAX_FILES_PER_CLASS .npy files per class from the extracted
    Dexonomy directory. Much faster than streaming the tar.gz.

    Returns {feix_id: [N, 29] stacked array}.
    """
    if not GRASP_DIR.exists():
        raise FileNotFoundError(f"Extracted directory not found: {GRASP_DIR}")

    feix_pattern = re.compile(r"^(\d+)_")
    class_dirs = sorted(GRASP_DIR.iterdir())

    n_needed = sum(1 for v in FEIX_TO_HOG.values() if v is not None)
    print(f"Leyendo de {GRASP_DIR}")
    print(f"  {len(class_dirs)} carpetas de clase encontradas\n")

    # {feix_id: {"pregrasp": [29], "squeeze": [29], "scale": float}}
    class_poses: dict[int, dict] = {}

    for class_dir in class_dirs:
        m = feix_pattern.match(class_dir.name)
        if m is None:
            continue
        feix_id = int(m.group(1))
        if feix_id not in FEIX_TO_HOG or FEIX_TO_HOG[feix_id] is None:
            continue

        npy_files = list(class_dir.rglob("*.npy"))[:MAX_FILES_PER_CLASS]

        best_scale   = np.inf
        best_pregrasp = None
        best_squeeze  = None

        for npy_path in npy_files:
            try:
                data     = np.load(npy_path, allow_pickle=True).item()
                pregrasp = data["pregrasp_qpos"].astype(np.float64)
                squeeze  = data["squeeze_qpos"].astype(np.float64)
                scale    = data["scene_scale"].astype(np.float64)
                if pregrasp.ndim != 2 or pregrasp.shape[1] != 29:
                    continue
                # row with smallest scale in this file
                idx = int(np.argmin(scale))
                if scale[idx] < best_scale:
                    best_scale    = scale[idx]
                    best_pregrasp = pregrasp[idx]
                    best_squeeze  = squeeze[idx]
            except Exception:
                continue

        if best_pregrasp is not None:
            class_poses[feix_id] = {
                "pregrasp": best_pregrasp,
                "squeeze":  best_squeeze,
                "scale":    best_scale,
            }
            hog_id = FEIX_TO_HOG[feix_id]
            print(f"  [{hog_id:2d}] {HOG_CLASS_NAMES.get(hog_id, '?'):<26s}  "
                  f"scale={best_scale:.3f}", flush=True)

    return class_poses


def _fmt(arr: np.ndarray) -> list[float]:
    return [round(float(v), 5) for v in arr]


def main() -> None:
    if not GRASP_DIR.exists():
        print(f"ERROR: {GRASP_DIR} not found.")
        sys.exit(1)

    poses_by_feix = extract_poses()

    print(f"\nComputing anchors for {len(poses_by_feix)} classes ...")

    doc: dict = {
        "_meta": {
            "source": "Dexonomy (RSS 2025) – Shadow Hand IsaacGym grasps",
            "joint_ordering": "MuJoCo Menagerie (24 DOF): "
                              + ", ".join(MUJOCO_JOINT_NAMES),
            "apertura_convention": (
                "qpos = (1 - apertura) * pose_open + apertura * pose_close  "
                "[apertura in 0..1, 0=most open, 1=most closed]"
            ),
            "anchor_method": (
                "pose_open = pregrasp_qpos of the grasp with smallest scene_scale; "
                "pose_close = squeeze_qpos of the same grasp (same scene, consistent trajectory)"
            ),
        }
    }

    missing_classes = []
    for feix_id in sorted(FEIX_TO_HOG.keys()):
        hog_id = FEIX_TO_HOG[feix_id]
        if hog_id is None:
            continue
        class_name = HOG_CLASS_NAMES.get(hog_id, f"class_{hog_id}")

        if feix_id not in poses_by_feix:
            print(f"  WARNING: no data for Feix {feix_id} ({class_name})")
            missing_classes.append(hog_id)
            continue

        poses = poses_by_feix[feix_id]
        pose_open_29  = poses["pregrasp"]
        pose_close_29 = poses["squeeze"]

        open_24  = dex29_to_mjc24(pose_open_29)
        close_24 = dex29_to_mjc24(pose_close_29)

        doc[hog_id] = {
            "class_name":  class_name,
            "feix_id":     feix_id,
            "scene_scale": round(float(poses["scale"]), 4),
            "pose_open":   _fmt(open_24),
            "pose_close":  _fmt(close_24),
        }
        print(f"  [{hog_id:2d}] {class_name:<26s}  scale={poses['scale']:.3f}")

    if missing_classes:
        print(f"\nNo Dexonomy coverage for HOGraspNet classes: {sorted(missing_classes)}")
        print("  (class 9 'Distal' has no Feix equivalent in Dexonomy)")

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=True, allow_unicode=True)
    print(f"\nSaved: {OUT_YAML}")


if __name__ == "__main__":
    main()
