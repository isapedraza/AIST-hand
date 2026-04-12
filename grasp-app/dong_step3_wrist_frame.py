"""
Paso 3 (Dong-lite): wrist frame desde world landmarks.

Basado en Dong & Payandeh (2025), seccion 4.1:
- Eq. (5): Y0 = normalize(d13 - d5)
- Eq. (6): Z0 = normalize(normalize(d9 - d0) x Y0)
- Eq. (7): X0 = Y0 x Z0
- Eq. (8): beta, gamma, alpha a partir de X0, Y0, Z0

Controles:
- SPACE: captura frame actual y calcula wrist frame
- Q / ESC: salir
"""

from __future__ import annotations

import argparse
import os
import warnings

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


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Vector degenerado (norm={n:.3e})")
    return v / n


def _world_points(hand_world_landmarks) -> np.ndarray:
    return np.asarray([(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark], dtype=float)


def _compute_wrist_frame(points: np.ndarray) -> dict:
    # Landmarks usados por Dong:
    # 0: wrist, 5: index base, 9: middle base, 13: ring base
    d0 = points[0]
    d5 = points[5]
    d9 = points[9]
    d13 = points[13]

    # Eq. (5)
    y0 = _normalize(d13 - d5)
    # Eq. (6)
    z0 = _normalize(np.cross(_normalize(d9 - d0), y0))
    # Eq. (7)
    x0 = _normalize(np.cross(y0, z0))

    # R0w con columnas [X0 Y0 Z0]
    r0w = np.column_stack((x0, y0, z0))

    # Eq. (8)
    beta = float(np.arcsin(np.clip(-x0[2], -1.0, 1.0)))
    gamma = float(np.arctan2(x0[1], x0[0]))
    alpha = float(np.arctan2(y0[2], z0[2]))

    return {
        "d0": d0,
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "r0w": r0w,
        "alpha_rad": alpha,
        "beta_rad": beta,
        "gamma_rad": gamma,
        "alpha_deg": float(np.degrees(alpha)),
        "beta_deg": float(np.degrees(beta)),
        "gamma_deg": float(np.degrees(gamma)),
        "det_r0w": float(np.linalg.det(r0w)),
        "orth_err": float(np.linalg.norm(r0w.T @ r0w - np.eye(3))),
    }


def _print_vec(name: str, v: np.ndarray) -> None:
    print(f"{name} = [{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}]", flush=True)


def _print_result(capture_id: int, out: dict) -> None:
    print(f"\n=== Step3 Wrist Frame | Captura {capture_id} ===", flush=True)
    _print_vec("d0w (kp0)", out["d0"])
    _print_vec("X0", out["x0"])
    _print_vec("Y0", out["y0"])
    _print_vec("Z0", out["z0"])
    print("R0w =", flush=True)
    r = out["r0w"]
    for i in range(3):
        print(f"  [{r[i,0]:+.6f}, {r[i,1]:+.6f}, {r[i,2]:+.6f}]", flush=True)
    print(
        f"alpha={out['alpha_rad']:+.6f} rad ({out['alpha_deg']:+.2f} deg) | "
        f"beta={out['beta_rad']:+.6f} rad ({out['beta_deg']:+.2f} deg) | "
        f"gamma={out['gamma_rad']:+.6f} rad ({out['gamma_deg']:+.2f} deg)",
        flush=True,
    )
    print(
        f"diag: det(R0w)={out['det_r0w']:+.6f} | ||R^T R - I||={out['orth_err']:.3e}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 3 Dong-lite: wrist frame")
    parser.add_argument("--camera", type=str, default="0", help="Indice de camara (0,1,2,...) o URL.")
    parser.add_argument("--mirror", action="store_true", help="Espejar vista de camara.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = _camera_source(args.camera)

    cap = _open_capture(source)
    if not cap or not cap.isOpened():
        print(f"No se pudo abrir la camara: {source!r}", flush=True)
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    capture_id = 0
    win = "Dong Step 3 - Wrist Frame"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("Iniciado. SPACE: calcular wrist frame | Q/ESC: salir.", flush=True)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_2d = None
            hand_world = None
            if result.multi_hand_landmarks:
                hand_2d = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_2d, mp_hands.HAND_CONNECTIONS)
                h, w = frame.shape[:2]
                for idx, lm in enumerate(hand_2d.landmark):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.putText(
                        frame,
                        str(idx),
                        (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            if result.multi_hand_world_landmarks:
                hand_world = result.multi_hand_world_landmarks[0]

            msg = f"SPACE: step3 wrist frame | WORLD={'YES' if hand_world is not None else 'NO'} | Q/ESC"
            if hand_2d is None:
                msg = "No hand detected | " + msg
            cv2.putText(
                frame,
                msg,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:  # SPACE
                if hand_world is None:
                    print("\nNo hay world_landmarks en este frame.", flush=True)
                    continue
                try:
                    capture_id += 1
                    points = _world_points(hand_world)
                    print(
                        f"[SOURCE] capture={capture_id} using=WORLD "
                        f"has_2d={hand_2d is not None} has_world=True n_points={len(points)}",
                        flush=True,
                    )
                    out = _compute_wrist_frame(points)
                    _print_result(capture_id, out)
                except ValueError as exc:
                    print(f"\nFrame degenerado: {exc}", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
