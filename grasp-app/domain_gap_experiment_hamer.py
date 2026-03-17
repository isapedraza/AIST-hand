"""
Domain Gap Experiment (HaMeR): air vs object grasping.

Mismo protocolo que domain_gap_experiment.py pero usando HaMeRBackend
en lugar de MediaPipeBackend. Permite comparacion directa entre estimadores.

Output:
  experiment_hamer/<session_id>/
    captures.csv
    frames/
      {class_id}_{class_name}_{condition}_f{frame:04d}.jpg

Usage:
  python domain_gap_experiment_hamer.py --url https://xxxx.trycloudflare.com
  python domain_gap_experiment_hamer.py --url https://xxxx.trycloudflare.com --feix-images /path/to/feix
  python domain_gap_experiment_hamer.py --url https://xxxx.trycloudflare.com --takes 3 --capture-seconds 5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import torch

from grasp_gcn import ToGraph, get_network
from grasp_gcn.dataset.grasps import GRASP_CLASS_NAMES

from inference_runtime import parse_model_output, to_probs
from model_variant import resolve_model_spec
from perception.hamer_backend import HaMeRBackend


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

CAPTURE_SECONDS = 20
HAMER_TIMEOUT = 15.0  # segundos sin respuesta antes de mostrar aviso de reconexion
COUNTDOWN_SECONDS = 3
TAKES = 1

CONDITIONS = ["air", "object"]

FEIX_IMAGE_MAP = {
    0:  "1_Large_Diameter.jpg",
    1:  "2_Small_Diameter.jpg",
    2:  "17_Index_Finger_Extension.jpg",
    3:  "18_Extension_Type.jpg",
    4:  "22_Parallel_Extension.jpg",
    5:  "30_Palmar.jpg",
    6:  "3_Medium_Wrap.jpg",
    7:  "4_Adducted_Thumb.jpg",
    8:  "5_Light_Tool.jpg",
    9:  "19_Distal_Type.jpg",
    10: "31_Ring.jpg",
    11: "10_Power_Disk.jpg",
    12: "11_Power_Sphere.jpg",
    13: "26_Sphere_4_Finger.jpg",
    14: "28_Sphere_3_Finger.jpg",
    15: "16_Lateral.jpg",
    16: "29_Stick.jpg",
    17: "23_Adduction_Grip.jpg",
    18: "20_Writing_Tripod.jpg",
    19: "25_Lateral_Tripod.jpg",
    20: "9_Palmar_Pinch.jpg",
    21: "24_Tip_Pinch.jpg",
    22: "33_Inferior_Pincer.jpg",
    23: "7_Prismatic_3_Finger.jpg",
    24: "12_Precision_Disk.jpg",
    25: "13_Precision_Sphere.jpg",
    26: "27_Quadpod.jpg",
    27: "14_Tripod.jpg",
}

CSV_FIELDS = [
    "session_id",
    "phase_idx",
    "class_id_target",
    "class_name_target",
    "condition",
    "take",
    "frame",
    "class_id_pred",
    "class_name_pred",
    "confidence",
]

WINDOW_NAME = "GraphGrasp - Domain Gap (HaMeR)"


# ---------------------------------------------------------------------------
# Reference image loader
# ---------------------------------------------------------------------------

def _load_feix_image(class_id: int, feix_dir: Path | None) -> np.ndarray | None:
    if feix_dir is None:
        return None
    fname = FEIX_IMAGE_MAP.get(class_id)
    if fname is None:
        return None
    path = feix_dir / fname
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    target_h = 220
    h, w = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

def _render_ui(
    backend: HaMeRBackend,
    feix_img: np.ndarray | None,
    class_name: str,
    condition: str,
    take: int,
    total_takes: int,
    phase_idx: int,
    total_phases: int,
    state: str,
    countdown_remaining: float = 0.0,
    captured_frames: int = 0,
    total_capture_frames: int = 0,
    status_text: str = "",
) -> int:
    """Render composite UI. Returns cv2.waitKey(1) result."""

    CAM_W, CAM_H = 480, 360
    REF_W = 300
    INFO_H = 200
    PANEL_W = max(REF_W, CAM_W)

    # --- Camera frame ---
    frame = backend._frame_bgr.copy() if backend._frame_bgr is not None else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    if backend.mirror_display:
        frame = cv2.flip(frame, 1)

    # Draw MediaPipe landmarks (used for bbox detection)
    if backend._last_result and backend._last_result.multi_hand_landmarks:
        hand_lm = backend._last_result.multi_hand_landmarks[0]
        if backend.mirror_display:
            hand_lm_draw = type(hand_lm)()
            hand_lm_draw.CopyFrom(hand_lm)
            for lm in hand_lm_draw.landmark:
                lm.x = 1.0 - lm.x
        else:
            hand_lm_draw = hand_lm
        backend._mp_draw.draw_landmarks(frame, hand_lm_draw, backend._mp_hands.HAND_CONNECTIONS)

    # Draw bbox
    if backend._last_bbox is not None:
        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = backend._last_bbox
        bx1 = int((1.0 - x2) * w_f) if backend.mirror_display else int(x1 * w_f)
        bx2 = int((1.0 - x1) * w_f) if backend.mirror_display else int(x2 * w_f)
        color = (0, 255, 0) if backend._last_handedness == "Right" else (255, 0, 0)
        cv2.rectangle(frame, (bx1, int(y1 * h_f)), (bx2, int(y2 * h_f)), color, 2)

    frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

    # --- State overlay ---
    if state == "countdown":
        text = f"{int(countdown_remaining) + 1}"
        cv2.putText(frame, text, (CAM_W // 2 - 30, CAM_H // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 80, 255), 6, cv2.LINE_AA)
    elif state == "capturing":
        progress = captured_frames / max(total_capture_frames, 1)
        bar_w = int(CAM_W * progress)
        cv2.rectangle(frame, (0, CAM_H - 12), (bar_w, CAM_H), (0, 220, 80), -1)
        cv2.putText(frame, "CAPTURING", (10, CAM_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2, cv2.LINE_AA)

    # --- Right panel ---
    panel = np.zeros((CAM_H + INFO_H, PANEL_W, 3), dtype=np.uint8)

    if feix_img is not None:
        fh, fw = feix_img.shape[:2]
        x0 = (PANEL_W - fw) // 2
        y0 = 10
        panel[y0:y0 + fh, x0:x0 + fw] = feix_img
        ref_bottom = y0 + fh + 8
    else:
        ref_bottom = 10

    condition_color = (80, 180, 255) if condition == "air" else (80, 255, 160)
    condition_label = "AIR" if condition == "air" else "OBJECT"

    lines = [
        ("GraphGrasp - HaMeR Domain Gap", (100, 200, 255), 0.65, 2),
        (f"Phase {phase_idx + 1}/{total_phases}", (180, 180, 180), 0.55, 1),
        (f"Target: {class_name}", (235, 235, 235), 0.65, 2),
        (f"Condition: {condition_label}", condition_color, 0.65, 2),
        (f"Take: {take}/{total_takes}", (180, 180, 180), 0.55, 1),
        ("", (0, 0, 0), 0.4, 1),
        (status_text, (255, 220, 80), 0.6, 1),
        ("", (0, 0, 0), 0.4, 1),
        ("SPACE = start/next   Q = quit", (120, 120, 120), 0.45, 1),
    ]

    y = ref_bottom + 20
    for text, color, scale, thick in lines:
        if text:
            cv2.putText(panel, text, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += int(scale * 38) + 4

    frame_padded = np.zeros((CAM_H + INFO_H, CAM_W, 3), dtype=np.uint8)
    frame_padded[:CAM_H] = frame
    canvas = np.hstack([frame_padded, panel])
    cv2.imshow(WINDOW_NAME, canvas)
    return cv2.waitKey(1) & 0xFF


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    url: str,
    feix_dir: Path | None,
    takes: int,
    capture_seconds: float,
    output_dir: Path,
    camera_source: int | str = 0,
    mirror: bool = True,
    only_classes: list[str] | None = None,
):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True)

    csv_path = session_dir / "captures.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    print(f"[experiment] Session: {session_id}")
    print(f"[experiment] Output:  {session_dir}")
    print(f"[experiment] HaMeR URL: {url}")

    # --- Load model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = resolve_model_spec()
    num_classes = spec["num_classes"]
    class_names = spec["class_names"]

    from grasp_gcn import get_network
    model = get_network("GCN_CAM_8_8_16_16_32", 4, num_classes, use_cmc_angle=True).to(device)
    model.load_state_dict(torch.load(spec["model_path"], map_location=device, weights_only=True))
    model.eval()
    print(f"[experiment] Model: {spec['variant']} | {num_classes} classes")

    # --- Setup perception ---
    backend = HaMeRBackend(
        url=url,
        camera_index=camera_source,
        mirror_display=mirror,
        mirror_left_hand=mirror,
    )
    if backend.startup_error():
        print(backend.startup_error())
        return

    to_graph = ToGraph(features="xyz", make_undirected=True, add_joint_angles=True, add_cmc_angle=True)

    # --- Build phase list ---
    class_ids = list(range(num_classes))
    if only_classes:
        only_lower = {c.lower() for c in only_classes}
        class_ids = [cid for cid in class_ids
                     if class_names.get(cid, GRASP_CLASS_NAMES.get(cid, "")).lower() in only_lower]
        if not class_ids:
            print(f"[experiment] ERROR: ninguna clase coincide con --classes {only_classes}")
            return
        print(f"[experiment] Filtrando a {len(class_ids)} clases: {[class_names.get(c) for c in class_ids]}")
    random.shuffle(class_ids)
    phases = [(cid, cond) for cid in class_ids for cond in CONDITIONS]
    total_phases = len(phases)

    fps_estimate = 30.0
    total_capture_frames = int(capture_seconds * fps_estimate)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    phase_idx = 0
    running = True

    while running and phase_idx < total_phases:
        class_id_target, condition = phases[phase_idx]
        class_name_target = class_names.get(class_id_target, GRASP_CLASS_NAMES.get(class_id_target, str(class_id_target)))
        feix_img = _load_feix_image(class_id_target, feix_dir)

        for take in range(1, takes + 1):
            if not running:
                break

            # --- PREPARE ---
            condition_label = "AIR (no object)" if condition == "air" else "OBJECT (hold an object)"
            status = f"Prepare: {condition_label} — SPACE=iniciar  U=nueva URL  Q=salir"

            while running:
                backend.get_landmarks()
                key = _render_ui(
                    backend, feix_img, class_name_target, condition,
                    take, takes, phase_idx, total_phases,
                    state="prepare", status_text=status,
                )
                if key == ord("q"):
                    running = False
                    break
                if key == ord("u"):
                    print("\n[experiment] Pega la nueva URL de Cloudflare:")
                    new_url = input("  Nueva URL: ").strip()
                    if new_url:
                        backend.url = new_url.rstrip("/")
                        print(f"[experiment] URL actualizada: {backend.url}")
                if key == ord(" "):
                    break

            if not running:
                break

            # --- COUNTDOWN ---
            countdown_start = time.time()
            while running:
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SECONDS - elapsed
                if remaining <= 0:
                    break
                backend.get_landmarks()
                _render_ui(
                    backend, feix_img, class_name_target, condition,
                    take, takes, phase_idx, total_phases,
                    state="countdown", countdown_remaining=remaining,
                    status_text="Get ready...",
                )

            if not running:
                break

            # --- CAPTURE (with retry) ---
            while running:
                frame_count = 0
                capture_start = time.time()
                pending_rows = []
                pending_frames = []
                last_sample_id = None
                last_hamer_time = time.time()
                reconnect_requested = False

                while running:
                    elapsed = time.time() - capture_start
                    if elapsed >= capture_seconds:
                        break

                    landmarks = backend.get_landmarks()
                    current_sample_id = id(backend._latest_sample)

                    hamer_silent = (time.time() - last_hamer_time) > HAMER_TIMEOUT
                    if hamer_silent:
                        status = "HAMER DOWN -- U=nueva URL  R=reintentar  Q=salir"
                    elif pending_rows:
                        status = f"Pred: {pending_rows[-1]['class_name_pred']} ({pending_rows[-1]['confidence']:.2f})"
                    else:
                        status = "Waiting for HaMeR..."

                    elapsed_frames = int(elapsed * 30)
                    key = _render_ui(
                        backend, feix_img, class_name_target, condition,
                        take, takes, phase_idx, total_phases,
                        state="capturing",
                        captured_frames=elapsed_frames,
                        total_capture_frames=total_capture_frames,
                        status_text=status,
                    )
                    if key == ord("q"):
                        running = False
                        break
                    if key == ord("u") and hamer_silent:
                        reconnect_requested = True
                        break

                    # Only record when a new HaMeR response has arrived
                    if current_sample_id == last_sample_id:
                        continue
                    last_sample_id = current_sample_id
                    last_hamer_time = time.time()

                    class_id_pred = -1
                    class_name_pred = "no_hand"
                    confidence = 0.0

                    if landmarks is not None:
                        try:
                            data = to_graph(landmarks).to(device)
                            with torch.no_grad():
                                output = model(data)
                            head_a, _ = parse_model_output(output)
                            probs = to_probs(head_a)
                            class_id_pred = int(probs.argmax().item())
                            confidence = float(probs.max().item())
                            class_name_pred = class_names.get(class_id_pred, GRASP_CLASS_NAMES.get(class_id_pred, str(class_id_pred)))

                        except Exception as e:
                            print(f"[experiment] Inference error: {e}")

                    frame_bgr = backend._frame_bgr
                    if frame_bgr is not None:
                        fname = f"{class_id_target:02d}_{class_name_target.replace(' ', '_')}_{condition}_t{take}_f{frame_count:04d}.jpg"
                        pending_frames.append((fname, frame_bgr.copy()))

                    pending_rows.append({
                        "session_id": session_id,
                        "phase_idx": phase_idx,
                        "class_id_target": class_id_target,
                        "class_name_target": class_name_target,
                        "condition": condition,
                        "take": take,
                        "frame": frame_count,
                        "class_id_pred": class_id_pred,
                        "class_name_pred": class_name_pred,
                        "confidence": round(confidence, 4),
                    })

                    frame_count += 1

                # --- RECONNECT ---
                if reconnect_requested:
                    print("\n[experiment] HaMeR desconectado. Pega la nueva URL de Cloudflare:")
                    new_url = input("  Nueva URL: ").strip()
                    if new_url:
                        backend.url = new_url.rstrip("/")
                        print(f"[experiment] URL actualizada: {backend.url}")
                    continue  # retry la fase

                # --- CONFIRM ---
                while running:
                    backend.get_landmarks()
                    key = _render_ui(
                        backend, feix_img, class_name_target, condition,
                        take, takes, phase_idx, total_phases,
                        state="prepare",
                        status_text="SPACE = save  |  R = retry  |  Q = quit",
                    )
                    if key == ord("q"):
                        running = False
                        break
                    if key == ord("r"):
                        break
                    if key == ord(" "):
                        for fname, img in pending_frames:
                            cv2.imwrite(str(frames_dir / fname), img)
                        for row in pending_rows:
                            writer.writerow(row)
                        csv_file.flush()
                        break

                if not running:
                    break
                if key == ord(" "):
                    break

            csv_file.flush()

        phase_idx += 1

    csv_file.close()
    backend.release()
    cv2.destroyAllWindows()
    print(f"[experiment] Done. CSV: {csv_path}")
    print(f"[experiment] Frames: {frames_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Domain gap experiment con HaMeR: air vs object")
    parser.add_argument("--url", type=str, required=True, help="URL de cloudflared del servidor Colab")
    parser.add_argument(
        "--feix-images",
        type=Path,
        default=Path("/home/yareeez/Downloads/ilovepdf_images-extracted(1)"),
        help="Directorio con imagenes de referencia Feix",
    )
    parser.add_argument("--takes", type=int, default=TAKES)
    parser.add_argument("--capture-seconds", type=float, default=CAPTURE_SECONDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiment_hamer")
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--classes", nargs="+", default=None, metavar="CLASS",
                        help="Grabar solo estas clases (nombres exactos, ej: 'Light Tool' 'Medium Wrap')")
    args = parser.parse_args()

    feix_dir = args.feix_images if args.feix_images.exists() else None
    if feix_dir is None:
        print(f"[experiment] Feix dir not found: {args.feix_images} — running without reference images")

    try:
        camera_source = int(args.camera)
    except ValueError:
        camera_source = args.camera

    run_experiment(
        url=args.url,
        feix_dir=feix_dir,
        takes=args.takes,
        capture_seconds=args.capture_seconds,
        output_dir=args.output_dir,
        camera_source=camera_source,
        mirror=not args.no_mirror,
        only_classes=args.classes,
    )


if __name__ == "__main__":
    main()
