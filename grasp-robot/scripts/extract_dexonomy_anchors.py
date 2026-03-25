"""
Extract two canonical pose anchors per grasp class from the Dexonomy dataset.

For each of the 28 HOGraspNet grasp classes (where Dexonomy has coverage),
we read all available grasp_qpos from the Dexonomy SHADOW tar and compute:

  pose_open  = P10 of total-flexion score  (most-open successful grasp)
  pose_close = P90 of total-flexion score  (most-closed successful grasp)

These become the two anchors for apertura ∈ [0, 1] interpolation:
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

import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
GRASP_TAR = Path("/media/yareeez/94649A33649A1856/dexonomy/Dexonomy_GRASP_shadow.tar.gz")
OUT_YAML  = ROOT / "grasp-robot" / "grasp_configs" / "shadow_hand_canonical.yaml"

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


def flexion_score(qpos29: np.ndarray) -> float:
    """
    Proxy for total hand closure: sum of main flexion joints.
    FFJ3/J2/J1 = dex[8,9,10], MFJ3/J2/J1 = dex[12,13,14],
    RFJ3/J2/J1 = dex[16,17,18], LFJ3/J2/J1 = dex[21,22,23], THJ1 = dex[28].
    """
    flex_indices = [8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 28]
    return float(np.sum(qpos29[flex_indices]))


def extract_poses() -> dict[int, list[np.ndarray]]:
    """
    Stream through the Dexonomy tar once, collecting up to MAX_FILES_PER_CLASS
    grasp_qpos arrays per class.

    Returns {feix_id: [N, 29] stacked array}.
    """
    class_poses: dict[int, list[np.ndarray]] = {}
    class_counts: dict[int, int] = {}

    feix_pattern = re.compile(r"succ_collect/(\d+)_")

    total = 0
    members_seen = 0
    print(f"Streaming {GRASP_TAR} ...")
    print("  (abriendo archivo comprimido, puede tardar unos segundos...)", flush=True)
    with tarfile.open(GRASP_TAR, "r:gz") as tf:
        for member in tf:
            members_seen += 1
            if members_seen % 10000 == 0:
                collected = sum(v >= MAX_FILES_PER_CLASS for v in class_counts.values())
                print(f"  {members_seen:7d} archivos leidos | {len(class_poses)} clases | {collected}/{len([v for v in FEIX_TO_HOG.values() if v is not None])} completas", flush=True)

            if not member.name.endswith(".npy"):
                continue

            m = feix_pattern.search(member.name)
            if m is None:
                continue
            feix_id = int(m.group(1))

            if feix_id not in FEIX_TO_HOG:
                continue
            if FEIX_TO_HOG[feix_id] is None:
                continue  # class not in HOGraspNet

            count = class_counts.get(feix_id, 0)
            if count >= MAX_FILES_PER_CLASS:
                continue  # already have enough for this class

            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = np.load(io.BytesIO(f.read()), allow_pickle=True).item()
                qpos = data["grasp_qpos"]   # [N, 29]
                if qpos.ndim != 2 or qpos.shape[1] != 29:
                    continue
            except Exception:
                continue

            if feix_id not in class_poses:
                class_poses[feix_id] = []
            class_poses[feix_id].append(qpos.astype(np.float64))
            class_counts[feix_id] = count + 1
            total += len(qpos)

            if total % 5000 == 0:
                collected = sum(v >= MAX_FILES_PER_CLASS for v in class_counts.values())
                print(f"  {total:7d} poses collected | "
                      f"{len(class_poses)} classes seen | "
                      f"{collected}/{len(FEIX_TO_HOG) - sum(v is None for v in FEIX_TO_HOG.values())} complete")

            # Early exit: all mappable classes are full
            n_needed = sum(1 for v in FEIX_TO_HOG.values() if v is not None)
            if sum(v >= MAX_FILES_PER_CLASS for v in class_counts.values()) >= n_needed:
                print("  All classes filled — stopping early.")
                break

    return {fid: np.vstack(arrs) for fid, arrs in class_poses.items()}


def compute_anchors(all_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given [N, 29] qpos, return (pose_open [29], pose_close [29]).
    pose_open  = median of poses at P10 of flexion score
    pose_close = median of poses at P90 of flexion score
    """
    scores = np.array([flexion_score(q) for q in all_qpos])
    p10 = np.percentile(scores, 25)
    p90 = np.percentile(scores, 75)
    open_mask  = scores <= p10
    close_mask = scores >= p90
    return (
        np.median(all_qpos[open_mask],  axis=0),
        np.median(all_qpos[close_mask], axis=0),
    )


def _fmt(arr: np.ndarray) -> list[float]:
    return [round(float(v), 5) for v in arr]


def main() -> None:
    if not GRASP_TAR.exists():
        print(f"ERROR: {GRASP_TAR} not found.")
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
                "pose_open = median of P10 percentile (flexion score); "
                "pose_close = median of P90 percentile"
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

        all_qpos = poses_by_feix[feix_id]   # [N, 29]
        pose_open_29, pose_close_29 = compute_anchors(all_qpos)

        open_24  = dex29_to_mjc24(pose_open_29)
        close_24 = dex29_to_mjc24(pose_close_29)

        doc[hog_id] = {
            "class_name": class_name,
            "feix_id": feix_id,
            "n_poses_sampled": int(len(all_qpos)),
            "pose_open":  _fmt(open_24),
            "pose_close": _fmt(close_24),
        }
        print(f"  [{hog_id:2d}] {class_name:<26s}  n={len(all_qpos):5d}  "
              f"flex P10={np.percentile([flexion_score(q) for q in all_qpos], 10):.3f}  "
              f"P90={np.percentile([flexion_score(q) for q in all_qpos], 90):.3f}")

    if missing_classes:
        print(f"\nNo Dexonomy coverage for HOGraspNet classes: {sorted(missing_classes)}")
        print("  (class 9 'Distal' has no Feix equivalent in Dexonomy)")

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=True, allow_unicode=True)
    print(f"\nSaved: {OUT_YAML}")


if __name__ == "__main__":
    main()
