"""
Capture rest pose landmarks using HaMeR for R014 training.

Records N seconds of open/rest hand pose and saves landmarks to CSV
in the same format as hograspnet_mano.csv (compatible with GraspsClass).

Usage:
    python capture_rest_pose.py --url https://xxxx.trycloudflare.com --camera 1
    python capture_rest_pose.py --url https://xxxx.trycloudflare.com --camera 1 --seconds 10 --takes 3

Output:
    grasp-app/rest_pose_landmarks.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

from perception.hamer_backend import HaMeRBackend

# Class ID 28 = Rest (new class for R014)
REST_CLASS_ID = 28
REST_CLASS_NAME = "Rest"

# Guided poses shown per take (cycles if takes > len)
POSE_GUIDE = [
    "PALMA al frente, dedos SEPARADOS",
    "PALMA al frente, dedos JUNTOS",
    "PALMA ARRIBA (como bandeja), dedos extendidos",
    "Mano RELAJADA -- curl natural, orientacion neutral",
    "Mano rotada 45 grados lateral",
    "DORSO al frente, dedos SEPARADOS",
    "DORSO al frente, dedos JUNTOS",
    "DORSO ARRIBA, dedos extendidos",
    "Mano RELAJADA -- dorso visible",
    "LATERAL -- pulgar hacia la camara",
]

JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

CSV_COLS = (
    ["subject_id", "sequence_id", "cam", "frame_id", "grasp_type", "contact_sum"]
    + [f"{j}_{ax}" for j in JOINTS for ax in ("x", "y", "z")]
)

OUTPUT_CSV = Path(__file__).resolve().parent / "rest_pose_landmarks.csv"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     type=str, required=True)
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--seconds", type=int, default=8,  help="Capture duration per take")
    parser.add_argument("--takes",   type=int, default=10, help="Number of takes")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--subject_id", type=int, default=50,
                        help="Subject ID controls split: 11-73=train, 1-10=val, 74-99=test")
    parser.add_argument("--pose_offset", type=int, default=0,
                        help="Start pose guide from this index (e.g. 5 to start at dorsal poses)")
    return parser.parse_args()


def _draw_ui(frame, text_lines, color=(200, 200, 200)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 35
    for line, col, scale, thick in text_lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        y += 32


def main():
    args = _parse_args()

    backend = HaMeRBackend(url=args.url, camera_index=args.camera)
    if backend.startup_error():
        print(backend.startup_error())
        return

    WINDOW = "Rest Pose Capture"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    # Load existing CSV or create new one
    existing_rows = 0
    write_header = not OUTPUT_CSV.exists()
    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()
    else:
        existing_rows = sum(1 for _ in open(OUTPUT_CSV)) - 1  # subtract header

    frame_counter = existing_rows
    total_captured = 0

    print(f"Output: {OUTPUT_CSV}")
    print(f"Existing rows: {existing_rows}")
    print(f"Takes: {args.takes} x {args.seconds}s")
    print("SPACE = start take | Q = quit")

    take = 0
    while take < args.takes and backend.is_ready():
        # --- Wait for SPACE ---
        pose_hint = POSE_GUIDE[(take + args.pose_offset) % len(POSE_GUIDE)]
        while backend.is_ready():
            landmarks = backend.get_landmarks()
            frame = backend._frame_bgr.copy() if backend._frame_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            hand_detected = landmarks is not None
            _draw_ui(frame, [
                (f"Take {take+1}/{args.takes}: {pose_hint}", (100, 200, 255), 0.55, 2),
                (f"Hand: {'DETECTED' if hand_detected else 'not detected'}", (80, 220, 80) if hand_detected else (80, 80, 200), 0.6, 1),
                ("SPACE = start capture   Q = quit", (180, 180, 180), 0.55, 1),
            ])
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                break
        else:
            break
        if key == ord('q'):
            break

        # --- Countdown ---
        for c in range(args.countdown, 0, -1):
            t_start = time.time()
            while time.time() - t_start < 1.0:
                backend.get_landmarks()
                frame = backend._frame_bgr.copy() if backend._frame_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                _draw_ui(frame, [
                    (f"Preparate... {c}", (100, 200, 255), 1.2, 3),
                    (pose_hint, (180, 180, 180), 0.5, 1),
                ])
                cv2.imshow(WINDOW, frame)
                cv2.waitKey(1)

        # --- Capture ---
        t_end = time.time() + args.seconds
        captured_this_take = 0
        while time.time() < t_end and backend.is_ready():
            landmarks = backend.get_landmarks()
            remaining = t_end - time.time()
            frame = backend._frame_bgr.copy() if backend._frame_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)

            if landmarks is not None:
                row = {
                    "subject_id":  args.subject_id,
                    "sequence_id": f"rest_pose_s{args.subject_id:02d}_take{take+1}",
                    "cam":         "sub1",
                    "frame_id":    frame_counter,
                    "grasp_type":  REST_CLASS_ID,
                    "contact_sum": 0.0,
                }
                for joint in JOINTS:
                    xyz = landmarks[joint]
                    row[f"{joint}_x"] = float(xyz[0])
                    row[f"{joint}_y"] = float(xyz[1])
                    row[f"{joint}_z"] = float(xyz[2])
                writer.writerow(row)
                frame_counter += 1
                captured_this_take += 1
                total_captured += 1

            _draw_ui(frame, [
                (f"REC take {take+1}/{args.takes} — {remaining:.1f}s | {pose_hint}", (80, 80, 220), 0.5, 2),
                (f"Frames: {captured_this_take} this take | {total_captured} total", (80, 220, 80), 0.6, 1),
            ])
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        csv_file.flush()
        print(f"Take {take+1} done: {captured_this_take} frames captured")
        take += 1

    csv_file.close()
    backend.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total frames: {total_captured} -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
