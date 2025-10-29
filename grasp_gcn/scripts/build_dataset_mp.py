# build_dataset_mp.py
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
DATA_ROOT = Path("grasps")   # carpeta con subcarpetas por clase
OUT_DIR   = Path(".")
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

HEADER = (
    ["object", "grasp_type", "handedness", "mirrored"] +
    [f"{j}_{ax}" for j in JOINTS for ax in ("x", "y", "z")]
)

# ====== MediaPipe setup (estático, 1 mano) ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def detect_landmarks_norm(img_rgb):
    """
    Devuelve:
      - landmarks_norm: (21,3) en coords NORMALIZADAS [0,1] para x,y; z relativa MP
      - handed_label: "Left"/"Right" o None
    """
    res = hands.process(img_rgb)
    if not getattr(res, "multi_hand_landmarks", None):
        return None, None
    hand = res.multi_hand_landmarks[0]
    lm = np.array([(p.x, p.y, p.z) for p in hand.landmark], dtype=np.float32)  # (21,3)

    handed_label = None
    if getattr(res, "multi_handedness", None):
        hinfo = res.multi_handedness[0].classification[0]
        handed_label = hinfo.label  # "Left" o "Right"

    return lm, handed_label


def mirror_x01(landmarks_norm):
    """Espejo horizontal en rango [0,1]: x' = 1 - x."""
    out = landmarks_norm.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def center_scale_by_palm(lm):
    """
    Invariancia de traslación/escala:
      - Centro: WRIST (id 0)
      - Escala: || WRIST -> MIDDLE_MCP (id 9) || usando (x,y)
    Devuelve lm_cs (21,3) centrado y escalado. z se divide por la misma norma.
    """
    lm = lm.astype(np.float32)
    wrist = lm[0]
    middle_mcp = lm[9]
    v = middle_mcp - wrist
    scale = np.linalg.norm(v[:2]) + 1e-9
    lm_cs = (lm - wrist) / scale
    return lm_cs


def collect_samples():
    """
    Recorre DATA_ROOT y arma filas para CSV en el HEADER definido.
    Features por nodo = (x,y,z) normalizados, luego centrados y escalados por palma.
    Todas las manos se dejan como 'Right' (se espeja si viene 'Left').
    """
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
        image_paths.sort()  # orden estable antes del shuffle

        for img_path in image_paths:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] No se pudo leer: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            lm_norm01, handed = detect_landmarks_norm(img_rgb)
            if lm_norm01 is None:
                print(f"[INFO] Sin mano: {img_path}")
                continue

            mirrored = 0
            if handed == "Left":
                lm_norm01 = mirror_x01(lm_norm01)
                handed = "Right"
                mirrored = 1

            # Invariancia: centro y escala por palma (sobre coords normalizadas)
            lm_cs = center_scale_by_palm(lm_norm01)  # (21,3)

            # Construir fila
            row = [Path(img_path).name, int(class_id), (handed or "Unknown"), int(mirrored)]
            for j in range(21):
                row.extend([float(lm_cs[j, 0]), float(lm_cs[j, 1]), float(lm_cs[j, 2])])

            per_class_rows[class_id].append(row)

    return per_class_rows


def stratified_split_and_write(per_class_rows):
    random.seed(SEED)
    all_rows = {"train": [], "val": [], "test": []}

    # estratificado por clase
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

    # mezcla global por split (opcional)
    for split in all_rows:
        random.shuffle(all_rows[split])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "grasps_sample_train.csv",
        "val": "grasps_sample_val.csv",
        "test": "grasps_sample_test.csv"
    }

    for split, rows in all_rows.items():
        out_csv = OUT_DIR / names[split]
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"[OK] {split}: {len(rows)} filas -> {out_csv}")

    # ---- Guardar manifiesto del split ----
    def digest_of_list(lst):
        m = hashlib.sha256()
        for r in lst:
            m.update((",".join(map(str, r)) + "\n").encode("utf-8"))
        return m.hexdigest()

    meta = {
        "seed": SEED,
        "splits_sizes": {k: len(v) for k, v in all_rows.items()},
        "classes": CLASS_TO_ID,
        "train_digest": digest_of_list(all_rows["train"]),
        "val_digest": digest_of_list(all_rows["val"]),
        "test_digest": digest_of_list(all_rows["test"]),
        "header": HEADER,
    }
    (OUT_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] split_meta.json -> {OUT_DIR / 'split_meta.json'}")


def main():
    per_class_rows = collect_samples()
    total = sum(len(v) for v in per_class_rows.values())
    if total == 0:
        print("[ERROR] No se generaron muestras. Revisa rutas/clases.")
        return
    stratified_split_and_write(per_class_rows)


if __name__ == "__main__":
    main()
