"""
Reannotate HOGraspNet S1 frames with HaMeR.

Lee hograspnet_mano.csv, filtra S1, para cada frame:
  1. Carga la imagen RGB de source_data
  2. Detecta la mano con MediaPipe (bbox + handedness)
  3. Manda el crop al servidor HaMeR
  4. Normaliza los keypoints (root-relative + scale)
  5. Escribe un nuevo CSV con XYZ y MANO_pose de HaMeR

Output: hograspnet_hamer_s1.csv (mismo formato que hograspnet_mano.csv)

Uso:
  python reannotate_hamer_s1.py --url https://xxxx.trycloudflare.com
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests

# ── Rutas ─────────────────────────────────────────────────────────────────────
SOURCE_DATA = Path("/media/yareeez/94649A33649A1856/HOGraspNet/source_data")
CSV_IN      = Path("/home/yareeez/AIST-hand/grasp-model/data/raw/hograspnet_mano.csv")
CSV_OUT     = Path("/home/yareeez/AIST-hand/grasp-model/data/raw/hograspnet_hamer_s1.csv")
LOG_FAILED  = Path("/home/yareeez/AIST-hand/grasp-model/data/raw/hograspnet_hamer_s1_failed.txt")

# ── Constantes ─────────────────────────────────────────────────────────────────
JOINTS = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
]
XYZ_COLS  = [f'{j}_{ax}' for j in JOINTS for ax in ('x', 'y', 'z')]
MANO_COLS = [f'MANO_pose_{i:02d}' for i in range(45)]

WRIST_IDX     = 0
INDEX_MCP_IDX = 5
TARGET_DIST   = 0.1
CROP_SIZE     = 256
PADDING       = 0.3
JPEG_QUALITY  = 85
REQUEST_TIMEOUT = 10.0


def normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts.copy()
    pts -= pts[WRIST_IDX]
    d = np.linalg.norm(pts[INDEX_MCP_IDX])
    if d > 1e-6:
        pts *= TARGET_DIST / d
    return pts


def get_bbox(landmarks, padding=PADDING):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    return (
        max(0.0, min(xs) - padding),
        max(0.0, min(ys) - padding),
        min(1.0, max(xs) + padding),
        min(1.0, max(ys) + padding),
    )


def crop_hand(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    crop = frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


def send_to_hamer(url, crop, is_right):
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    try:
        resp = requests.post(
            f"{url}/infer",
            files={"crop": ("crop.jpg", buf.tobytes(), "image/jpeg")},
            data={"is_right": int(is_right)},
            timeout=REQUEST_TIMEOUT,
        )
        body = resp.json()
        kp   = body.get("keypoints")
        pose = body.get("mano_pose")
        if kp is None or pose is None:
            return None, None
        return np.array(kp, dtype=np.float32), np.array(pose, dtype=np.float32)
    except Exception as e:
        print(f"  [HaMeR error] {e}")
        return None, None


def img_path(sequence_id, cam, frame_id):
    # sequence_id en CSV: "230905_S01_obj_16_grasp_14/trial_0"
    parts    = sequence_id.split("/")
    seq_base = parts[0]
    trial    = parts[1]
    return SOURCE_DATA / seq_base / trial / "rgb" / cam / f"{cam}_{frame_id}.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL del servidor HaMeR en Colab")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    print(f"Leyendo CSV: {CSV_IN}")
    df = pd.read_csv(CSV_IN)
    s1 = df[df["subject_id"] == 1].reset_index(drop=True)
    print(f"Frames S1: {len(s1):,}")

    # Resumir si ya existe output parcial
    done_keys = set()
    if CSV_OUT.exists():
        done_df = pd.read_csv(CSV_OUT, usecols=["sequence_id", "cam", "frame_id"])
        for _, row in done_df.iterrows():
            done_keys.add((row["sequence_id"], row["cam"], row["frame_id"]))
        print(f"Reanudando -- ya procesados: {len(done_keys):,}")

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )

    out_cols = list(df.columns)  # conserva todas las columnas originales
    write_header = not CSV_OUT.exists()

    n_ok = 0
    n_fail = 0
    failed_log = open(LOG_FAILED, "a")
    import time
    t_start = time.time()
    total_frames = len(s1)
    already_done = len(done_keys)
    pending = total_frames - already_done

    with open(CSV_OUT, "a", newline="") as f_out:
        import csv
        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        if write_header:
            writer.writeheader()

        for i, row in s1.iterrows():
            seq_id   = row["sequence_id"]
            cam      = row["cam"]
            frame_id = int(row["frame_id"])
            key      = (seq_id, cam, frame_id)

            if key in done_keys:
                continue

            path = img_path(seq_id, cam, frame_id)
            if not path.exists():
                print(f"  [missing] {path}")
                n_fail += 1
                failed_log.write(f"missing|{path}\n")
                failed_log.flush()
                continue

            frame = cv2.imread(str(path))
            if frame is None:
                n_fail += 1
                failed_log.write(f"read_error|{path}\n")
                failed_log.flush()
                continue

            # MediaPipe
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if not result.multi_hand_landmarks:
                n_fail += 1
                failed_log.write(f"no_hand|{path}\n")
                failed_log.flush()
                continue

            hand = result.multi_hand_landmarks[0]
            if result.multi_handedness:
                label    = result.multi_handedness[0].classification[0].label
                is_right = (label == "Left")  # selfie mode flip
            else:
                is_right = True

            crop = crop_hand(frame, get_bbox(hand))
            if crop is None:
                n_fail += 1
                failed_log.write(f"crop_empty|{path}\n")
                failed_log.flush()
                continue

            kp, pose = send_to_hamer(url, crop, is_right)
            if kp is None:
                n_fail += 1
                failed_log.write(f"hamer_fail|{path}\n")
                failed_log.flush()
                continue

            kp_norm = normalize(kp)
            # mirror left hand
            if not is_right:
                kp_norm[:, 0] *= -1.0

            # Construir fila de salida
            out_row = row.to_dict()
            for j, joint in enumerate(JOINTS):
                out_row[f"{joint}_x"] = float(kp_norm[j, 0])
                out_row[f"{joint}_y"] = float(kp_norm[j, 1])
                out_row[f"{joint}_z"] = float(kp_norm[j, 2])
            for k in range(45):
                out_row[f"MANO_pose_{k:02d}"] = float(pose[k])

            writer.writerow(out_row)
            f_out.flush()
            n_ok += 1

            if (n_ok + n_fail) % 100 == 0:
                elapsed = time.time() - t_start
                done = n_ok + n_fail
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (pending - done) / rate if rate > 0 else 0
                print(f"  [{already_done + done:,}/{total_frames:,}] ok={n_ok} fail={n_fail} | {rate:.1f} frames/s | ETA {remaining/60:.1f} min")

    hands.close()
    failed_log.close()
    print(f"\nListo. ok={n_ok} | fail={n_fail} | output: {CSV_OUT}")


if __name__ == "__main__":
    main()
