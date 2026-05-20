# ====== build_dataset_mp.py (versión tipo Nadia, con debug) ======
import os
import csv
import glob
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp

# ====== CONFIG ======
DATA_ROOT = Path("data/raw/images/grasps")
OUT_DIR   = Path("data/raw")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 230

CLASS_TO_ID = {
    "Large diameter": 0,
    "Parallel extension": 1,
    "Precision sphere": 2,
}

JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

HEADER = ["object", "grasp_type", "handedness", "mirrored"] + [
    f"{j}_{ax}" for j in JOINTS for ax in ("x", "y", "z")
]

# ====== MediaPipe setup (estático, 1 mano) ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ================================================================
# ---------------------- FUNCIONES PRINCIPALES -------------------
# ================================================================

def detect_landmarks_norm(img_rgb, img_path):
    """Devuelve (21,3) normalizado y etiqueta de mano, con depuración."""
    print(f"[DEBUG] Detectando en: {img_path}")
    print(f"  -> shape={img_rgb.shape}, dtype={img_rgb.dtype}, rango=({img_rgb.min()}, {img_rgb.max()})")

    try:
        res = hands.process(img_rgb)
    except Exception as e:
        print(f"[ERROR] MediaPipe falló en {img_path}: {e}")
        return None, None

    if not getattr(res, "multi_hand_landmarks", None):
        print(f"[INFO] ❌ Sin mano detectada en {img_path}")
        return None, None

    hand = res.multi_hand_landmarks[0]
    lm = np.array([(p.x, p.y, p.z) for p in hand.landmark], dtype=np.float32)
    handed_label = None
    if getattr(res, "multi_handedness", None):
        hinfo = res.multi_handedness[0].classification[0]
        handed_label = hinfo.label
        print(f"[INFO] ✅ Mano detectada: {handed_label}")
    else:
        print("[WARN] No se pudo obtener handedness")

    print(f"  -> Primer punto (WRIST): {lm[0]}")
    return lm, handed_label


def mirror_x01(landmarks_norm):
    """Espeja horizontalmente (x' = 1 - x)."""
    out = landmarks_norm.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def collect_samples():
    """Recorre DATA_ROOT y arma filas CSV según HEADER."""
    print(f"[DEBUG] Iniciando colecta desde: {DATA_ROOT.resolve()}")
    per_class_rows = defaultdict(list)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    for class_name, class_id in CLASS_TO_ID.items():
        class_dir = DATA_ROOT / class_name
        if not class_dir.is_dir():
            print(f"[WARN] No existe carpeta de clase: {class_dir}")
            continue

        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(str(class_dir / ext)))
        image_paths.sort()

        print(f"[DEBUG] Clase {class_name} ({class_id}) -> {len(image_paths)} imágenes encontradas")

        for i, img_path in enumerate(image_paths):
            print(f"[DEBUG] ({i+1}/{len(image_paths)}) Leyendo: {img_path}")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] No se pudo leer: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            lm_norm01, handed = detect_landmarks_norm(img_rgb, img_path)
            if lm_norm01 is None:
                continue

            mirrored = 0
            if handed == "Left":
                lm_norm01 = mirror_x01(lm_norm01)
                handed = "Right"
                mirrored = 1
                print(f"[DEBUG] ↔️ Espejado aplicado en {img_path}")

            # ====== USAR COORDENADAS ABSOLUTAS (NO CENTRADAS) ======
            lm_final = lm_norm01
            print(f"[DEBUG] Coordenadas finales WRIST: {lm_final[0]}")

            row = [Path(img_path).name, int(class_id), (handed or "Unknown"), int(mirrored)]
            for j in range(21):
                row.extend([float(lm_final[j, 0]), float(lm_final[j, 1]), float(lm_final[j, 2])])

            per_class_rows[class_id].append(row)

    print(f"[DEBUG] Finalizada colecta. Clases con muestras:")
    for k, v in per_class_rows.items():
        print(f"  - {k}: {len(v)} muestras")

    return per_class_rows


def stratified_split_and_write(per_class_rows):
    random.seed(SEED)
    all_rows = {"train": [], "val": [], "test": []}

    for _, rows in per_class_rows.items():
        random.shuffle(rows)
        n = len(rows)
        n_train = int(SPLITS["train"] * n)
        n_val = int(SPLITS["val"] * n)
        parts = {
            "train": rows[:n_train],
            "val": rows[n_train:n_train + n_val],
            "test": rows[n_train + n_val:]
        }
        for split, part in parts.items():
            all_rows[split].extend(part)

    for split in all_rows:
        random.shuffle(all_rows[split])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "grasps_sample_train.csv",
        "val": "grasps_sample_val.csv",
        "test": "grasps_sample_test.csv",
    }

    for split, rows in all_rows.items():
        out_csv = OUT_DIR / names[split]
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"[OK] {split}: {len(rows)} filas -> {out_csv}")

    print(f"[OK] Split meta guardado en {OUT_DIR / 'split_meta.json'}")


def main():
    print("[DEBUG] === INICIO DE PROCESAMIENTO ===")
    per_class_rows = collect_samples()
    total = sum(len(v) for v in per_class_rows.values())
    print(f"[DEBUG] Total muestras detectadas: {total}")
    if total == 0:
        print("[ERROR] No se generaron muestras. Revisa rutas/clases.")
        return
    stratified_split_and_write(per_class_rows)
    print("[DEBUG] === FIN ===")


if __name__ == "__main__":
    main()
