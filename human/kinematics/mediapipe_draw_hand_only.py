from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Reduce ruido de logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

PALM_ROOT_IDS = (1, 5, 9, 13, 17)
FINGER_LINKS = (
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


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
    parser = argparse.ArgumentParser(
        description="MediaPipe Hands minimal: solo detecta y dibuja la mano."
    )
    parser.add_argument("--camera", type=str, default="0", help="Indice de camara o URL.")
    parser.add_argument("--image", type=str, default=None, help="Ruta de imagen para prueba unica.")
    parser.add_argument("--max-hands", type=int, default=1, help="Numero maximo de manos.")
    parser.add_argument("--det-conf", type=float, default=0.5, help="Min detection confidence.")
    parser.add_argument("--trk-conf", type=float, default=0.5, help="Min tracking confidence.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida para imagen procesada (modo --image).",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Mostrar ventana en modo --image ademas de guardar archivo.",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Ruta TXT de salida para coordenadas y distancias (modo --image).",
    )
    return parser.parse_args()


def _landmarks_2d_array(hand_landmarks) -> np.ndarray:
    return np.asarray([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=float)


def _landmarks_world_array(hand_world_landmarks) -> np.ndarray:
    return np.asarray([(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark], dtype=float)


def _compute_lengths(points: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for a, b in pairs:
        out[(a, b)] = float(np.linalg.norm(points[b] - points[a]))
    return out


def _write_report_txt(
    txt_path: Path,
    image_shape: tuple[int, int, int],
    result,
) -> None:
    h, w = image_shape[:2]
    hands_2d = list(result.multi_hand_landmarks or [])
    hands_world = list(result.multi_hand_world_landmarks or [])
    handed = list(result.multi_handedness or [])

    lines: list[str] = []
    lines.append("MediaPipe Hands report")
    lines.append(f"image_size_px: w={w} h={h}")
    lines.append(f"hands_2d_count: {len(hands_2d)}")
    lines.append(f"hands_world_count: {len(hands_world)}")
    lines.append("")

    if not hands_2d:
        lines.append("No hand detected.")
    else:
        for idx, hand2d in enumerate(hands_2d):
            lines.append(f"=== hand[{idx}] ===")

            label = "UNKNOWN"
            score = None
            if idx < len(handed):
                try:
                    cls = handed[idx].classification[0]
                    label = str(getattr(cls, "label", "UNKNOWN"))
                    score = float(getattr(cls, "score", np.nan))
                except Exception:
                    pass
            if score is None or np.isnan(score):
                lines.append(f"handedness: {label}")
            else:
                lines.append(f"handedness: {label} score={score:.6f}")

            p2d = _landmarks_2d_array(hand2d)
            lines.append("[coords_2d_norm] kp x y z")
            for k, p in enumerate(p2d):
                lines.append(f"[{k:02d}] x={p[0]:+.6f} y={p[1]:+.6f} z={p[2]:+.6f}")

            lines.append("[coords_2d_px] kp x_px y_px z_norm")
            for k, p in enumerate(p2d):
                x_px = p[0] * w
                y_px = p[1] * h
                lines.append(f"[{k:02d}] x={x_px:+.2f} y={y_px:+.2f} z={p[2]:+.6f}")

            pairs_palm = tuple((0, i) for i in PALM_ROOT_IDS)
            palm_2d = _compute_lengths(p2d[:, :2], pairs_palm)
            link_2d = _compute_lengths(p2d[:, :2], FINGER_LINKS)
            lines.append("[dist_2d_norm] palm roots (0->i)")
            for (_, i), v in palm_2d.items():
                lines.append(f"l0,{i}={v:+.6f}")
            lines.append("[dist_2d_norm] finger links")
            for (a, b), v in link_2d.items():
                lines.append(f"l{a},{b}={v:+.6f}")

            if idx < len(hands_world):
                pw = _landmarks_world_array(hands_world[idx])
                lines.append("[coords_world_m] kp x y z")
                for k, p in enumerate(pw):
                    lines.append(f"[{k:02d}] x={p[0]:+.6f} y={p[1]:+.6f} z={p[2]:+.6f}")

                palm_w = _compute_lengths(pw, pairs_palm)
                link_w = _compute_lengths(pw, FINGER_LINKS)
                lines.append("[dist_world_m] palm roots (0->i)")
                for (_, i), v in palm_w.items():
                    lines.append(f"l0,{i}={v:+.6f} m")
                lines.append("[dist_world_m] finger links")
                for (a, b), v in link_w.items():
                    lines.append(f"l{a},{b}={v:+.6f} m")
            else:
                lines.append("No world landmarks for this hand.")

            lines.append("")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"No se pudo leer la imagen: {args.image}", flush=True)
            return 1
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max(1, int(args.max_hands)),
            min_detection_confidence=float(args.det_conf),
            min_tracking_confidence=float(args.trk_conf),
        ) as hands:
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            else:
                print("No hand detected.", flush=True)

        if args.output:
            out_path = Path(args.output)
        else:
            src = Path(args.image)
            out_path = src.with_name(f"{src.stem}_drawn{src.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), image)
        if not ok:
            print(f"No se pudo guardar la imagen de salida: {out_path}", flush=True)
            return 1
        print(f"Imagen guardada en: {out_path}", flush=True)

        if args.output_txt:
            txt_path = Path(args.output_txt)
        else:
            txt_path = out_path.with_name(f"{out_path.stem}_landmarks.txt")
        _write_report_txt(txt_path=txt_path, image_shape=image.shape, result=result)
        print(f"TXT guardado en: {txt_path}", flush=True)

        if args.show_image:
            cv2.namedWindow("MediaPipe Hands - Draw Only", cv2.WINDOW_NORMAL)
            cv2.imshow("MediaPipe Hands - Draw Only", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    source = _camera_source(args.camera)
    cap = _open_capture(source)
    if not cap or not cap.isOpened():
        print(f"No se pudo abrir la camara: {source!r}", flush=True)
        return 1

    win = "MediaPipe Hands - Draw Only"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max(1, int(args.max_hands)),
        min_detection_confidence=float(args.det_conf),
        min_tracking_confidence=float(args.trk_conf),
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                frame,
                "Q/ESC: salir",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
