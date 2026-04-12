"""Capture one MediaPipe frame and report Dong-style long-finger angles.

Controls:
  SPACE: capture current frame and print angles in terminal
  Q: exit
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import cv2
import mediapipe as mp
import numpy as np

from grasp_gcn.transforms.kinematics import compute_dong_angles, summarize_open_palm_likeness


def _landmarks_to_numpy(hand_world_landmarks) -> np.ndarray:
    return np.asarray([(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark], dtype=float)


def _print_angles(angles: dict, open_palm: dict) -> None:
    print("\n=== Dong Angles (long fingers) ===")
    for finger in ("index", "middle", "ring", "pinky"):
        a = angles[finger]
        print(
            f"{finger:6s} | "
            f"MCP_flex={a['MCP_flex_beta']:>6.2f}  "
            f"MCP_abd={a['MCP_abd_gamma']:>6.2f}  "
            f"PIP={a['PIP_beta']:>6.2f}  "
            f"DIP={a['DIP_beta']:>6.2f}"
        )
    print("--- open palm diagnostic (ROM-normalized) ---")
    print(
        f"threshold={open_palm['open_ratio_threshold']:.2f}, "
        f"score={open_palm['open_score']:.2f}, "
        f"looks_open_palm={open_palm['looks_open_palm']}"
    )
    for finger in ("index", "middle", "ring", "pinky"):
        ratios = open_palm["per_finger_ratios"][finger]
        flag = open_palm["finger_open_flags"][finger]
        print(
            f"{finger:6s} | open={flag} | "
            f"MCP_flex_ratio={ratios['MCP_flex_ratio']:.3f}  "
            f"PIP_ratio={ratios['PIP_ratio']:.3f}  "
            f"DIP_ratio={ratios['DIP_ratio']:.3f}  "
            f"MCP_abd_ratio={ratios['MCP_abd_ratio']:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture MediaPipe frame and compute Dong angles")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the last captured payload as JSON",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    print("Press SPACE to capture one frame. Press Q to quit.")
    last_payload = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            overlay = "No hand detected"
            world_points = None

            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            if results.multi_hand_world_landmarks:
                world_points = _landmarks_to_numpy(results.multi_hand_world_landmarks[0])
                overlay = "Hand detected - SPACE: capture angles"

            cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Q: quit", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.imshow("Dong Frame Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == 32:  # SPACE
                if world_points is None:
                    print("\nNo hand available in current frame.")
                    continue
                angles = compute_dong_angles(world_points, round_digits=2)
                open_palm = summarize_open_palm_likeness(angles)
                _print_angles(angles, open_palm)

                last_payload = {
                    "angles": angles,
                    "open_palm_diagnostic": open_palm,
                    "world_points": world_points.tolist(),
                }

    cap.release()
    cv2.destroyAllWindows()

    if args.save_json and last_payload is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(last_payload, indent=2))
        print(f"Saved capture to {args.save_json}")
    elif args.save_json:
        print("No capture saved because no frame was captured with SPACE.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
