"""
HaMeR live source for the retargeting pipeline.

Same Dong path as MediaPipeSource, but the 21 XYZ landmarks come from HaMeR
(remote Colab inference via HaMeRBackend) instead of MediaPipe world landmarks.
HaMeR's ViT regresses MANO theta and exports XYZ = FK(theta), giving cleaner MCP
flexion from the frontal view -- the perception diagnostic: does better frontal
perception create MCP closure where MediaPipe does not? (Confirmed: yes.)

Why no Dong change is needed: Dong rebuilds its own wrist-local frame from the
hand anatomy (Eq. 5-9, built from wrist + index/middle/ring MCPs), so the angle
pipeline is invariant to HaMeR's global axis convention and to absolute scale.
Only chirality matters, and HaMeRBackend already normalizes left->right. If a
single-axis reflection ever shows up empirically, flip it via AXIS_FLIP below.

Usage:
    source = HaMeRSource(url="https://xxxx.trycloudflare.com", camera=0)
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

# Joint order produced by HaMeRBackend.get_landmarks() (standard MediaPipe 21).
_JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Empirical chirality fix. Dong is invariant to rigid rotation, but a reflection
# between HaMeR's native coord system and the expected one would mirror the hand.
# Identity by default (confirmed live: no flip needed); set one entry to -1.0 if
# the rendered hand comes out mirrored (e.g. AXIS_FLIP = [1.0, 1.0, -1.0]).
AXIS_FLIP = np.array([1.0, 1.0, 1.0], dtype=np.float64)


class HaMeRSource:
    """
    Live source that produces Dong quaternion tensors from HaMeR keypoints.

    Wraps HaMeRBackend (remote inference + its own camera/window). Calibration
    runs automatically during the first `calib_seconds` of valid frames;
    next_frame() returns None until calibration is done.
    """

    def __init__(
        self,
        url: str,
        camera: int | str = 0,
        calib_seconds: float = 3.0,
    ):
        # Deferred import so MediaPipe-only usage never requires HaMeR deps.
        from human.perception.hamer_backend import HaMeRBackend

        # mirror_left_hand=False: HaMeR internally flips left hands to be predicted
        # in right-canonical space (vitdet flip=right==0) and returns the 3D
        # keypoints WITHOUT un-flipping (hamer_module returns pred_keypoints_3d
        # as-is). So a left hand already arrives right-canonical; the client-side
        # x-reflection would double-flip it into a left-handed coordinate system
        # and break Dong's chirality-sensitive frame (Eq 5-9). Right hand is
        # unaffected (the reflection only ever applied to handedness=="Left").
        self._backend = HaMeRBackend(url=url, camera_index=camera, mirror_left_hand=False)
        self._dk = DongKinematics(calibration_frames=None)
        self._calib_seconds = calib_seconds
        self._calib_done = calib_seconds <= 0.0
        self._calib_t0: float | None = None
        self._running = True

    def is_running(self) -> bool:
        return self._running and self._backend.is_ready()

    def _extract_points(self, sample: dict) -> np.ndarray | None:
        """Stack the 21 joint XYZ from a HaMeR sample dict, in Dong order."""
        try:
            pts = np.array([sample[name] for name in _JOINTS], dtype=np.float64)
        except (KeyError, TypeError):
            return None
        if pts.shape != (21, 3):
            return None
        return pts * AXIS_FLIP

    def next_frame(self) -> torch.Tensor | None:
        """
        Pull the latest HaMeR keypoints and return quaternion tensor [1, 20, 4],
        or None if no keypoints yet or still calibrating.
        """
        if not self._running:
            return None

        # Reads a frame, fires the async HaMeR request, returns last keypoints.
        sample = self._backend.get_landmarks()

        # Camera window + quit handling.
        status = "Calibrating..." if not self._calib_done else "Running (HaMeR)"
        if not self._backend.render(status_text=status):
            self._running = False
            return None

        if sample is None:
            return None
        points = self._extract_points(sample)
        if points is None:
            return None

        # HaMeR already mirrors left->right, so the canonical label is "Right"
        # (no double reflection).
        points_w, _, _ = canonicalize_to_right_hand(points, "Right")

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
