"""
WiLoR live source for the retargeting pipeline.

Same Dong path as MediaPipeSource/HaMeRSource, but the 21 XYZ landmarks come from
WiLoR (remote Colab inference via WiLoRBackend). WiLoR reconstructs MANO with a
lightweight ViT, much faster on GPU than HaMeR's ViT-H, and returns keypoints
already in OpenPose/MediaPipe 21-joint order -- a drop-in for Dong.

Handedness (note: OPPOSITE of HaMeRSource). WiLoR returns keypoints in the
PHYSICAL hand frame: for a left hand it negates x of pred_keypoints_3d, so a left
hand arrives left-handed. Dong's wrist-local frame (Eq. 5-9) is chirality
sensitive, so we must reflect left hands to right-canonical here, driven by the
server's `is_right` flag, using canonicalize_to_right_hand (x-reflects when the
label is "Left"). For comparison, the HaMeR server returned left already
right-canonical, so HaMeRSource disables that reflection.

Why no Dong change is needed: Dong rebuilds its wrist-local frame from anatomy, so
the pipeline is invariant to WiLoR's global axis convention and to absolute scale;
only chirality matters, which the is_right-driven canonicalization handles.

Usage:
    source = WiLoRSource(url="https://xxxx.trycloudflare.com", camera=0)
    while source.is_running():
        quats = source.next_frame()   # [1, 20, 4] or None
        if quats is None:
            continue
        qpos = retargeter(quats)
    source.release()
"""

from __future__ import annotations

import time

import numpy as np
import torch

try:
    from human.kinematics.dong_kinematics import DongKinematics, canonicalize_to_right_hand
except ImportError as e:
    raise ImportError(
        "human package not found. Install it with: pip install -e /path/to/AIST-hand/human"
    ) from e

# Joint order produced by WiLoRBackend.get_landmarks() (standard MediaPipe 21).
_JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Empirical chirality fix (same knob as HaMeRSource). Identity by default; Dong is
# invariant to rigid rotation, so this is only needed if a reflection between
# WiLoR's native coord system and the expected one ever shows up.
AXIS_FLIP = np.array([1.0, 1.0, 1.0], dtype=np.float64)


class WiLoRSource:
    """
    Live source that produces Dong quaternion tensors from WiLoR keypoints.

    Wraps WiLoRBackend (remote inference + its own camera/window). Calibration
    runs automatically during the first `calib_seconds` of valid frames;
    next_frame() returns None until calibration is done.
    """

    def __init__(
        self,
        url: str,
        camera: int | str = 0,
        calib_seconds: float = 3.0,
    ):
        # Deferred import so MediaPipe/HaMeR-only usage never requires WiLoR deps.
        from human.perception.wilor_backend import WiLoRBackend

        self._backend = WiLoRBackend(url=url, camera_index=camera)
        self._dk = DongKinematics(calibration_frames=None)
        self._calib_seconds = calib_seconds
        self._calib_done = calib_seconds <= 0.0
        self._calib_t0: float | None = None
        self._running = True

    def is_running(self) -> bool:
        return self._running and self._backend.is_ready()

    def _extract_points(self, sample: dict) -> np.ndarray | None:
        """Stack the 21 joint XYZ from a WiLoR sample dict, in Dong order."""
        try:
            pts = np.array([sample[name] for name in _JOINTS], dtype=np.float64)
        except (KeyError, TypeError):
            return None
        if pts.shape != (21, 3):
            return None
        return pts * AXIS_FLIP

    def next_frame(self) -> torch.Tensor | None:
        """
        Pull the latest WiLoR keypoints and return quaternion tensor [1, 20, 4],
        or None if no keypoints yet or still calibrating.
        """
        if not self._running:
            return None

        sample = self._backend.get_landmarks()

        if self._calib_done:
            status = "Running (WiLoR)"
        else:
            elapsed = 0.0 if self._calib_t0 is None else time.time() - self._calib_t0
            remaining = max(0.0, self._calib_seconds - elapsed)
            status = f"Calibrating... {remaining:.1f}s | samples={self._dk.calibration_count}"
        if not self._backend.render(status_text=status):
            self._running = False
            return None

        if sample is None:
            return None
        points = self._extract_points(sample)
        if points is None:
            return None

        # WiLoR returns physical-frame keypoints: reflect left hands to
        # right-canonical for Dong (opposite of the HaMeR path).
        hand_label = "Right" if int(sample.get("is_right", 1)) else "Left"
        points_w, _, _ = canonicalize_to_right_hand(points, hand_label)

        # Calibration phase.
        if not self._calib_done:
            if self._calib_t0 is None:
                self._calib_t0 = time.time()
            try:
                self._dk.calibrate(points_w)
            except ValueError:
                return None
            if time.time() - self._calib_t0 >= self._calib_seconds:
                self._dk.force_freeze()
                self._calib_done = True
            return None

        # Inference phase.
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
        self._backend.release()
