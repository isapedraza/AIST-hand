"""
Paso 2 (Dong-lite): capturar keypoints crudos de MediaPipe.

Hace solo esto:
1) abre camara,
2) detecta mano,
3) al presionar SPACE imprime los 21 keypoints (x, y, z).

No aplica normalizacion, no calcula angulos, no usa profundidad RGB-D.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

# Reduce ruido de logs (TensorFlow/MediaPipe/protobuf).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

import cv2
import mediapipe as mp


def _open_capture(camera_index: int | str):
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if os.name == "posix" else [cv2.CAP_ANY]
    last_cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
        except TypeError:
            cap = cv2.VideoCapture(camera_index)
        if cap and cap.isOpened():
            return cap
        last_cap = cap
        if cap is not None:
            cap.release()
    return last_cap


def _camera_source(raw: str) -> int | str:
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura 21 keypoints crudos de MediaPipe.")
    parser.add_argument("--camera", type=str, default="0", help="Indice de camara (0,1,2,...) o URL.")
    parser.add_argument("--mirror", action="store_true", help="Espejar vista de camara.")
    parser.add_argument("--save-json", type=Path, default=None, help="Guardar ultima captura en JSON.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Guardar capturas en CSV (append).")
    return parser.parse_args()


def _print_capture(capture_id: int, points: list[dict]) -> None:
    print(f"\n=== Captura {capture_id} ===", flush=True)
    for p in points:
        print(
            f"[{p['id']:02d}] "
            f"x={p['x']:+.6f}  y={p['y']:+.6f}  z={p['z']:+.6f}"
        , flush=True)


def _print_source_diag(capture_id: int, hand_landmarks, points: list[dict]) -> None:
    has_2d = hand_landmarks is not None
    has_world = len(points) == 21
    print(
        f"[SOURCE] capture={capture_id} "
        f"using=WORLD "
        f"has_2d={has_2d} has_world={has_world} n_points={len(points)}"
    , flush=True)
    if has_2d:
        lm2d = hand_landmarks.landmark[0]
        w0 = points[0]
        print(
            "[kp0 compare] "
            f"2D_norm=(x={lm2d.x:+.6f}, y={lm2d.y:+.6f}, z={lm2d.z:+.6f}) "
            f"WORLD=(x={w0['x']:+.6f}, y={w0['y']:+.6f}, z={w0['z']:+.6f})"
        , flush=True)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"JSON guardado en: {path}", flush=True)


def _csv_header() -> list[str]:
    cols = ["capture_id"]
    for i in range(21):
        cols.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"])
    return cols


def _save_csv(path: Path, capture_id: int, points: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    row = [capture_id]
    for p in points:
        row.extend([p["x"], p["y"], p["z"]])

    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(_csv_header())
        w.writerow(row)
    print(f"CSV append en: {path}", flush=True)


def main() -> int:
    # -------------------- FASE 1: setup de captura --------------------
    args = _parse_args()
    camera_source = _camera_source(args.camera)

    cap = _open_capture(camera_source)
    if not cap or not cap.isOpened():
        print(f"No se pudo abrir la camara: {camera_source!r}", flush=True)
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    capture_id = 0
    last_payload = None

    window_name = "Dong Step 2 - Raw 21 Keypoints"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print(
        "Iniciado. Usa SPACE para capturar y ver salida en terminal. Q/ESC para salir.",
        flush=True,
    )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            # -------------------- FASE 1: lectura de frame --------------------
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            # -------------------- FASE 1: landmarks 2D + world --------------------
            hand_landmarks = None
            hand_world_landmarks = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Numeracion 0..20 sobre landmarks 2D.
                h, w = frame_bgr.shape[:2]
                for idx, lm in enumerate(hand_landmarks.landmark):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.putText(
                        frame_bgr,
                        str(idx),
                        (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            if result.multi_hand_world_landmarks:
                hand_world_landmarks = result.multi_hand_world_landmarks[0]

            world_ok = hand_world_landmarks is not None
            msg = f"SPACE: capturar 21 keypoints | WORLD={'YES' if world_ok else 'NO'} | Q/ESC: salir"
            if hand_landmarks is None:
                msg = "No hand detected | " + msg
            elif hand_world_landmarks is None:
                msg = "Hand 2D ok, world_landmarks no disponibles | " + msg
            cv2.putText(
                frame_bgr,
                msg,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:  # SPACE
                # -------------------- FASE 1: captura cruda de 21 world keypoints --------------------
                if hand_world_landmarks is None:
                    print("\nNo hay world_landmarks en este frame.", flush=True)
                    continue

                capture_id += 1
                points = []
                for idx, lm in enumerate(hand_world_landmarks.landmark):
                    points.append(
                        {
                            "id": idx,
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                        }
                    )

                _print_source_diag(capture_id, hand_landmarks, points)
                _print_capture(capture_id, points)
                last_payload = {"capture_id": capture_id, "points": points}

                if args.save_csv is not None:
                    _save_csv(args.save_csv, capture_id, points)

    cap.release()
    cv2.destroyAllWindows()

    if args.save_json is not None:
        if last_payload is None:
            print("No se guardo JSON porque no hubo capturas.", flush=True)
        else:
            _save_json(args.save_json, last_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
