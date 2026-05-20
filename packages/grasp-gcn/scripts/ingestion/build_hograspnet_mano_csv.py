"""
build_hograspnet_mano_csv.py
----------------------------
Igual que build_hograspnet_csv.py pero añade los parámetros MANO por frame.

Columnas de salida:
  subject_id  : int (1-99)
  sequence_id : str (ej. "231023_S01_obj_01_grasp_22/trial_0")
  cam         : str (mas|sub1|sub2|sub3)
  grasp_type  : int (índice local 0-27, orden HOGraspNet config.py)
  contact_sum : float (suma de los 778 valores del campo contact)
  WRIST_x ... PINKY_TIP_z : float (63 columnas XYZ normalizadas)
  MANO_pose_00 ... MANO_pose_44 : float (45 parámetros de pose axis-angle)

Parámetros MANO (fuente: HOGraspNet docs/data_structure.md, Mesh[0]):
  mano_pose  (1*45): rotaciones axis-angle de los 15 joints de dedos
                     (excluye la rotación global del wrist, que va en mano_trans)
Referencia: Romero et al. (2017). Embodied Hands: Modeling and Capturing Hands
  and Bodies Together. SIGGRAPH Asia.

Normalización geométrica (por frame, idéntica a build_hograspnet_csv.py):
  1. Root-relative: restar WRIST (keypoint 0) a los 21 puntos.
  2. Escala: factor S = 0.1 / dist(WRIST, INDEX_FINGER_MCP) → distancia objetivo 10 cm.
     Referencia: Santos et al. (2025).
  Los parámetros MANO NO se normalizan -- son ángulos / coeficientes de forma,
  ya invariantes a posición y escala por definición del modelo.

El split S1 y el filtro de contacto NO se aplican aquí.

Uso:
  python scripts/ingestion/build_hograspnet_mano_csv.py \\
      --zips_dir ~/HOGraspNet/data/zipped/annotations \\
      --out      data/raw/hograspnet_mano.csv
"""

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np

# ── Orden canónico HOGraspNet (de config.py del dataloader oficial) ────────────
FEIX_INDICES = [1, 2, 17, 18, 22, 30, 3, 4, 5, 19, 31, 10, 11, 26, 28,
                16, 29, 23, 20, 25, 9, 24, 33, 7, 12, 13, 27, 14]
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

WRIST_IDX     = 0
INDEX_MCP_IDX = 5      # referencia de escala — Santos et al. (2025)
TARGET_DIST   = 0.1    # metros (10 cm)

HEADER = (
    ["subject_id", "sequence_id", "cam", "frame_id", "grasp_type", "contact_sum"]
    + [f"{j}_{ax}" for j in JOINTS for ax in ("x", "y", "z")]
    + [f"MANO_pose_{i:02d}" for i in range(45)]
)


# ── Normalización geométrica ──────────────────────────────────────────────────

def normalize_geometric(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts[WRIST_IDX]
    d = np.linalg.norm(pts[INDEX_MCP_IDX])
    if d > 1e-6:
        pts = pts * (TARGET_DIST / d)
    return pts


# ── Procesado de un zip ───────────────────────────────────────────────────────

def process_zip(zip_path: Path, subject_id: int) -> list:
    rows = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]

        for jname in json_names:
            parts = jname.split("/")
            if len(parts) < 5:
                continue

            seq_id = parts[0] + "/" + parts[1]
            cam    = parts[3]

            mf = re.search(r"_(\d+)\.json$", parts[-1])
            frame_id = int(mf.group(1)) if mf else -1

            m = re.search(r"grasp_(\d+)", parts[0])
            if not m:
                continue
            feix_id = int(m.group(1))
            if feix_id not in FEIX_TO_LOCAL:
                continue
            local_class = FEIX_TO_LOCAL[feix_id]

            try:
                raw  = zf.read(jname)
                anno = json.loads(raw.decode("utf-8-sig"))
            except Exception:
                continue

            # 3D pose
            try:
                pts = np.array(anno["hand"]["3D_pose_per_cam"], dtype=np.float32)
            except (KeyError, ValueError):
                continue
            if pts.shape != (21, 3):
                continue

            # MANO pose y betas
            try:
                mesh = anno["Mesh"][0]
                mano_pose = np.array(mesh["mano_pose"][0], dtype=np.float32)   # (45,)
            except (KeyError, IndexError, ValueError):
                continue
            if mano_pose.shape != (45,):
                continue

            # Contact sum
            contact     = anno.get("contact", [])
            contact_sum = float(np.sum(contact)) if contact else 0.0

            # Normalización geométrica (solo XYZ)
            pts = normalize_geometric(pts)

            row = [subject_id, seq_id, cam, frame_id, local_class, round(contact_sum, 6)]
            row.extend(pts.flatten().tolist())
            row.extend(mano_pose.tolist())
            rows.append(row)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Genera hograspnet_mano.csv con parámetros MANO desde los zips."
    )
    p.add_argument("--zips_dir", required=True,
                   help="Directorio con los zips de anotaciones de HOGraspNet")
    p.add_argument("--out", default="data/raw/hograspnet_mano.csv",
                   help="Ruta del CSV de salida (default: data/raw/hograspnet_mano.csv)")
    return p.parse_args()


def main():
    args     = parse_args()
    zips_dir = Path(args.zips_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zip_pattern = re.compile(r"GraspNet_S(\d+)_\d+_Labeling_data\.zip")
    all_zips    = sorted(zips_dir.glob("GraspNet_S*_Labeling_data.zip"))

    if not all_zips:
        raise FileNotFoundError(f"No se encontraron zips en {zips_dir}")

    total = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for zip_path in all_zips:
            m = zip_pattern.match(zip_path.name)
            if not m:
                continue
            subject_id = int(m.group(1))
            print(f"  S{subject_id:02d} — {zip_path.name} ...", end=" ", flush=True)
            rows = process_zip(zip_path, subject_id)
            writer.writerows(rows)
            total += len(rows)
            print(f"{len(rows)} frames")

    print(f"\nTotal: {total} frames guardados en {out_path}")


if __name__ == "__main__":
    main()
