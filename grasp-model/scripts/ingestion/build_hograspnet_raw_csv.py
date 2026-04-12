"""
build_hograspnet_raw_csv.py
---------------------------
Genera un único CSV RAW desde los zips de anotaciones de HOGraspNet.

Formato de salida (alineado a hograspnet_hamer_s1 pero sin MANO_pose):
  subject_id  : int (1-99)
  sequence_id : str (ej. "231023_S01_obj_01_grasp_22/trial_0")
  cam         : str (mas|sub1|sub2|sub3)
  frame_id    : int
  grasp_type  : int (índice local 0-27, orden HOGraspNet config.py)
  contact_sum : float (suma de los 778 valores del campo contact)
  WRIST_x ... PINKY_TIP_z : float (63 columnas XYZ RAW, sin normalización)

Notas:
  - XYZ se toma tal cual del JSON: anno["hand"]["3D_pose_per_cam"].
  - NO se aplica root-relative, escala, mirror ni post-procesamiento.
  - El split S1 y filtros de contacto NO se aplican aquí.

Uso:
  python scripts/ingestion/build_hograspnet_raw_csv.py \
      --zips_dir ~/HOGraspNet/data/zipped/annotations \
      --out      data/raw/hograspnet_raw.csv
"""

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np

# ── Orden canónico HOGraspNet (de config.py del dataloader oficial) ───────────
FEIX_INDICES = [
    1, 2, 17, 18, 22, 30, 3, 4, 5, 19, 31, 10, 11, 26, 28,
    16, 29, 23, 20, 25, 9, 24, 33, 7, 12, 13, 27, 14,
]
FEIX_TO_LOCAL = {feix: local for local, feix in enumerate(FEIX_INDICES)}

# ── Joints (orden MediaPipe / HOGraspNet) ─────────────────────────────────────
JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HEADER = (
    ["subject_id", "sequence_id", "cam", "frame_id", "grasp_type", "contact_sum"]
    + [f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y", "z")]
)


def iter_rows_from_zip(zip_path: Path, subject_id: int):
    """Yield rows parsed from a single HOGraspNet subject zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_names = sorted(n for n in zf.namelist() if n.endswith(".json"))

        for jname in json_names:
            # Expected:
            # date_S01_obj_01_grasp_22/trial_0/annotation/sub3/sub3_83.json
            parts = jname.split("/")
            if len(parts) < 5:
                continue

            seq_id = parts[0] + "/" + parts[1]
            cam = parts[3]

            frame_match = re.search(r"_(\d+)\.json$", parts[-1])
            frame_id = int(frame_match.group(1)) if frame_match else -1

            grasp_match = re.search(r"grasp_(\d+)", parts[0])
            if not grasp_match:
                continue
            feix_id = int(grasp_match.group(1))
            if feix_id not in FEIX_TO_LOCAL:
                continue
            local_class = FEIX_TO_LOCAL[feix_id]

            try:
                raw = zf.read(jname)
                anno = json.loads(raw.decode("utf-8-sig"))
            except Exception:
                continue

            try:
                pts = np.array(anno["hand"]["3D_pose_per_cam"], dtype=np.float32)
            except (KeyError, ValueError):
                continue
            if pts.shape != (21, 3):
                continue

            contact = anno.get("contact", [])
            contact_sum = float(np.sum(contact)) if contact else 0.0

            row = [subject_id, seq_id, cam, frame_id, local_class, round(contact_sum, 6)]
            row.extend(pts.flatten().tolist())
            yield row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera hograspnet_raw.csv (XYZ raw sin normalización) desde zips HOGraspNet."
    )
    parser.add_argument(
        "--zips_dir",
        required=True,
        help="Directorio con los zips de anotaciones de HOGraspNet",
    )
    parser.add_argument(
        "--out",
        default="data/raw/hograspnet_raw.csv",
        help="Ruta del CSV de salida (default: data/raw/hograspnet_raw.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zips_dir = Path(args.zips_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zip_pattern = re.compile(r"GraspNet_S(\d+)_\d+_Labeling_data\.zip")
    all_zips = sorted(zips_dir.glob("GraspNet_S*_Labeling_data.zip"))
    if not all_zips:
        raise FileNotFoundError(f"No se encontraron zips en {zips_dir}")

    total = 0
    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(HEADER)

        for zip_path in all_zips:
            match = zip_pattern.match(zip_path.name)
            if not match:
                continue
            subject_id = int(match.group(1))
            print(f"  S{subject_id:02d} — {zip_path.name} ...", end=" ", flush=True)

            n_rows = 0
            for row in iter_rows_from_zip(zip_path, subject_id):
                writer.writerow(row)
                n_rows += 1

            total += n_rows
            print(f"{n_rows} frames")

    print(f"\nTotal: {total} frames guardados en {out_path}")


if __name__ == "__main__":
    main()
