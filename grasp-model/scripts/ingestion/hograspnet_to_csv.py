"""
hograspnet_to_csv.py
--------------------
Convierte las anotaciones de HOGraspNet (zips de JSONs) al formato CSV
que usa el pipeline de entrenamiento (GraspsClass / ToGraph).

Normalización geométrica aplicada por muestra:
  1. Centrar en muñeca: restar WRIST a todos los joints
  2. Escalar por tamaño de mano: dividir por dist(WRIST, MIDDLE_FINGER_MCP)

Uso:
  python scripts/ingestion/hograspnet_to_csv.py \
      --zips_dir ~/HOGraspNet/data/zipped/annotations \
      --out_dir  data/raw \
      --seed 230

Opciones:
  --cam        Cámara a usar: mas|sub1|sub2|sub3|all  (default: all)
  --subjects   Rango de sujetos: "1-99" o "1,2,5"    (default: all)
  --val_frac   Fracción de validación                 (default: 0.1)
  --test_frac  Fracción de test                       (default: 0.1)
"""

import argparse
import csv
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Taxonomía ────────────────────────────────────────────────────────────────
# 28 índices Feix presentes en HOGraspNet → índice local 0-27
FEIX_INDICES = [1, 2, 17, 18, 22, 30, 3, 4, 5, 19, 31, 10, 11, 26, 28,
                16, 29, 23, 20, 25, 9, 24, 33, 7, 12, 13, 27, 14]
FEIX_TO_LOCAL = {feix: local for local, feix in enumerate(FEIX_INDICES)}

# ── Joints (orden MediaPipe / HOGraspNet idéntico) ────────────────────────────
JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9  # referencia de escala

HEADER = ["object", "grasp_type", "handedness", "mirrored"] + [
    f"{j}_{ax}" for j in JOINTS for ax in ("x", "y", "z")
]

CAMS_ALL = ["mas", "sub1", "sub2", "sub3"]


# ── Normalización geométrica ─────────────────────────────────────────────────

def normalize_geometric(pts: np.ndarray) -> np.ndarray:
    """
    pts: (21, 3) en coordenadas métricas.
    Retorna (21, 3) centrado en muñeca y escalado por dist(wrist, middle_mcp).
    """
    wrist = pts[WRIST_IDX].copy()
    pts = pts - wrist  # centrar

    scale = np.linalg.norm(pts[MIDDLE_MCP_IDX])
    if scale < 1e-6:
        return pts  # muestra degenerada, devolver centrada sin escalar
    return pts / scale


# ── Parser de argumentos ─────────────────────────────────────────────────────

def parse_subjects(s: str):
    """'all' → list(1..99) | '1-20' → [1..20] | '1,3,5' → [1,3,5]"""
    if s == "all":
        return list(range(1, 100))
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zips_dir", required=True,
                   help="Directorio con los zips de anotaciones de HOGraspNet")
    p.add_argument("--out_dir", default="data/raw",
                   help="Directorio de salida para los CSVs")
    p.add_argument("--cam", default="all",
                   choices=["mas", "sub1", "sub2", "sub3", "all"])
    p.add_argument("--subjects", default="all")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=230)
    return p.parse_args()


# ── Extracción de un zip ──────────────────────────────────────────────────────

def extract_rows_from_zip(zip_path: Path, cams: list) -> list:
    """
    Lee un zip de un sujeto y devuelve lista de filas CSV.
    Descomprime en memoria (sin tocar disco).
    """
    rows = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        json_names = [n for n in names if n.endswith(".json")]

        for jname in json_names:
            # Path: 231023_S01_obj_01_grasp_22/trial_0/annotation/sub1/sub1_83.json
            parts = jname.split("/")
            if len(parts) < 4:
                continue

            seq_name = parts[0]          # 231023_S01_obj_01_grasp_22
            cam_id = parts[3]            # mas / sub1 / sub2 / sub3

            if cam_id not in cams:
                continue

            # Extraer grasp_idx del nombre de secuencia
            m = re.search(r"grasp_(\d+)", seq_name)
            if not m:
                continue
            feix_idx = int(m.group(1))
            if feix_idx not in FEIX_TO_LOCAL:
                continue
            local_class = FEIX_TO_LOCAL[feix_idx]

            # Leer JSON en memoria
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

            # Normalización geométrica
            pts = normalize_geometric(pts)

            # Construir fila CSV
            row = [jname, local_class, "Right", 0]
            for i in range(21):
                row.extend([float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])])
            rows.append(row)

    return rows


# ── Split estratificado ───────────────────────────────────────────────────────

def stratified_split(rows: list, val_frac: float, test_frac: float, seed: int):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r[1]].append(r)

    train, val, test = [], [], []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        n = len(cls_rows)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test.extend(cls_rows[:n_test])
        val.extend(cls_rows[n_test:n_test + n_val])
        train.extend(cls_rows[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    zips_dir = Path(args.zips_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = CAMS_ALL if args.cam == "all" else [args.cam]
    subjects = parse_subjects(args.subjects)

    zip_pattern = re.compile(r"GraspNet_S(\d+)_\d+_Labeling_data\.zip")
    all_zips = sorted(zips_dir.glob("GraspNet_S*_Labeling_data.zip"))

    all_rows = []
    for zip_path in all_zips:
        m = zip_pattern.match(zip_path.name)
        if not m:
            continue
        subj_id = int(m.group(1))
        if subj_id not in subjects:
            continue

        print(f"Procesando S{subj_id:02d} — {zip_path.name} ...", end=" ", flush=True)
        rows = extract_rows_from_zip(zip_path, cams)
        all_rows.extend(rows)
        print(f"{len(rows)} muestras")

    print(f"\nTotal: {len(all_rows)} muestras | Dividiendo ...")
    train, val, test = stratified_split(all_rows, args.val_frac, args.test_frac, args.seed)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    splits = {
        "grasps_train.csv": train,
        "grasps_val.csv": val,
        "grasps_test.csv": test,
    }
    for fname, rows in splits.items():
        out_path = out_dir / fname
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"  -> {out_path}  ({len(rows)} filas)")

    print("\nListo.")


if __name__ == "__main__":
    main()
