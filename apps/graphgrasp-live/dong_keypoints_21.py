"""
Activa y visualiza los 21 keypoints de mano con MediaPipe.

Este script solo:
1) abre la camara,
2) detecta la mano,
3) dibuja los 21 landmarks (con su indice 0..20).

No aplica normalizacion, no usa modelo GCN y no guarda datos.
"""

from __future__ import annotations

import argparse
import os

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualiza 21 keypoints de mano (MediaPipe).")
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Indice de camara (0,1,2,...) o URL de stream.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Espejar vista de camara.",
    )
    return parser.parse_args()


def _camera_source(raw: str) -> int | str:
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def main() -> None:
    args = _parse_args()
    camera_source = _camera_source(args.camera)

    cap = _open_capture(camera_source)
    if not cap or not cap.isOpened():
        print(f"No se pudo abrir la camara: {camera_source!r}")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window_name = "Dong - 21 Keypoints"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            hand_count = 0
            if result.multi_hand_landmarks:
                hand_count = len(result.multi_hand_landmarks)
                h, w = frame_bgr.shape[:2]
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )
                    # Etiqueta cada landmark con su indice [0..20].
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

            status = f"Hands: {hand_count} | Keypoints por mano: 21 | Q: salir"
            cv2.putText(
                frame_bgr,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
