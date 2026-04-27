"""
MediaPipe live source for retargeting pipeline.

Captures world landmarks from webcam, applies Dong kinematics,
and exposes quaternion tensors [1, 20, 4] ready for Retargeter.

Usage:
    source = MediaPipeSource(camera=0, calib_seconds=3.0)
    while source.is_running():
        quats = source.next_frame()   # [1, 20, 4] or None
        if quats is None:
            continue
        qpos = retargeter(quats)
    source.release()
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

# dong_kinematics lives in grasp-app, outside this package
_HAND_PREPROCESSING = Path(__file__).resolve().parents[5] / "grasp-app" / "hand_preprocessing"
if str(_HAND_PREPROCESSING) not in sys.path:
    sys.path.insert(0, str(_HAND_PREPROCESSING))

from dong_kinematics import DongKinematics, canonicalize_to_right_hand  # noqa: E402


class MediaPipeSource:
    """
    Live camera source that produces Dong quaternion tensors.

    Calibration runs automatically during the first `calib_seconds` of
    valid frames. next_frame() returns None until calibration is done.
    """

    def __init__(self, camera: int | str = 0, calib_seconds: float = 3.0):
        self._dk = DongKinematics(calibration_frames=None)
        self._calib_seconds = calib_seconds
        self._calib_done = calib_seconds <= 0.0
        self._calib_t0: float | None = None

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if hasattr(cv2, "CAP_V4L2") else [cv2.CAP_ANY]
        self._cap = None
        for backend in backends:
            cap = cv2.VideoCapture(camera, backend)
            if cap and cap.isOpened():
                self._cap = cap
                break
            if cap:
                cap.release()

        self._running = self._cap is not None and self._cap.isOpened()
        self._win = "MediaPipe Source"
        cv2.namedWindow(self._win, cv2.WINDOW_AUTOSIZE)

    def is_running(self) -> bool:
        return self._running

    def next_frame(self) -> torch.Tensor | None:
        """
        Read one camera frame and return quaternion tensor [1, 20, 4],
        or None if no hand detected or still calibrating.
        """
        if not self._running:
            return None

        ok, frame = self._cap.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)   # webcam selfie mode: mirror before MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        # Draw landmarks on display frame
        display = frame.copy()
        if result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display,
                result.multi_hand_landmarks[0],
                self._mp_hands.HAND_CONNECTIONS,
            )

        # Overlay status
        if not self._calib_done:
            elapsed = 0.0 if self._calib_t0 is None else time.time() - self._calib_t0
            remaining = max(0.0, self._calib_seconds - elapsed)
            msg = f"Calibrating... {remaining:.1f}s | samples={self._dk.calibration_count}"
        else:
            msg = "Running"
        cv2.putText(display, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
        cv2.imshow(self._win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self._running = False
            return None

        if not result.multi_hand_world_landmarks:
            return None

        # Extract and canonicalize world landmarks
        raw_pts = np.array(
            [(lm.x, lm.y, lm.z) for lm in result.multi_hand_world_landmarks[0].landmark],
            dtype=np.float64,
        )
        hand_label = None
        if result.multi_handedness:
            hand_label = result.multi_handedness[0].classification[0].label
        points_w, _, _ = canonicalize_to_right_hand(raw_pts, hand_label)

        # Calibration phase
        if not self._calib_done:
            if self._calib_t0 is None:
                self._calib_t0 = time.time()
            try:
                self._dk.calibrate(points_w)
            except ValueError:
                return None
            elapsed = time.time() - self._calib_t0
            if elapsed >= self._calib_seconds:
                self._dk.force_freeze()
                self._calib_done = True
            return None

        # Inference phase
        try:
            res = self._dk.process(points_w)
        except ValueError:
            return None

        quats = np.array(
            [res["quaternions"][j] for j in res["joint_order"]],
            dtype=np.float32,
        )  # [20, 4]
        return torch.from_numpy(quats).unsqueeze(0)  # [1, 20, 4]

    def release(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
        self._hands.close()
        cv2.destroyAllWindows()
