from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


FINGER_CHAINS = [
    ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    ("RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP"),
    ("PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"),
]
OPEN_HAND_HOLD_FRAMES = 8


def to_probs(head_a: torch.Tensor) -> torch.Tensor:
    lse = torch.logsumexp(head_a, dim=1)
    if torch.allclose(lse, torch.zeros_like(lse), atol=1e-4):
        return head_a.exp()
    return torch.softmax(head_a, dim=1)


def parse_model_output(output: Any):
    if isinstance(output, torch.Tensor):
        return output, None

    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]

    if isinstance(output, dict):
        head_a = output.get("head_a") or output.get("logits") or output.get("log_probs")
        head_b = output.get("head_b") or output.get("synergy") or output.get("synergy_coeffs")
        if head_a is None:
            raise ValueError("Model dict output missing Head A logits/log_probs.")
        return head_a, head_b

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return 0.0
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return math.acos(cosang)


def is_open_hand(landmarks: dict) -> bool:
    """Heuristic override for obvious idle/open-hand states.

    The grasp model has no explicit 'open hand' class, so runtime can force an
    idle state when the four long fingers are nearly straight and their tips are
    well separated from the wrist/palm.
    """

    try:
        wrist = np.asarray(landmarks["WRIST"], dtype=np.float32)
        index_mcp = np.asarray(landmarks["INDEX_FINGER_MCP"], dtype=np.float32)
        thumb_tip = np.asarray(landmarks["THUMB_TIP"], dtype=np.float32)
    except KeyError:
        return False

    finger_extensions = []
    fingertip_distances = []
    for mcp_name, pip_name, dip_name, tip_name in FINGER_CHAINS:
        mcp = np.asarray(landmarks[mcp_name], dtype=np.float32)
        pip = np.asarray(landmarks[pip_name], dtype=np.float32)
        dip = np.asarray(landmarks[dip_name], dtype=np.float32)
        tip = np.asarray(landmarks[tip_name], dtype=np.float32)

        pip_angle = _angle(mcp, pip, dip)
        dip_angle = _angle(pip, dip, tip)
        finger_extensions.append((pip_angle + dip_angle) * 0.5)
        fingertip_distances.append(float(np.linalg.norm(tip - wrist)))

    mean_extension = float(np.mean(finger_extensions))
    min_extension = float(np.min(finger_extensions))
    mean_tip_distance = float(np.mean(fingertip_distances))
    thumb_open = float(np.linalg.norm(thumb_tip - index_mcp))

    return (
        min_extension > 2.35
        and mean_extension > 2.50
        and mean_tip_distance > 0.17
        and thumb_open > 0.07
    )


class OpenHandLatch:
    """Temporal hysteresis for the open-hand override."""

    def __init__(self, hold_frames: int = OPEN_HAND_HOLD_FRAMES):
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        self.hold_frames = hold_frames
        self._remaining = 0

    def update(self, landmarks: dict | None) -> bool:
        if landmarks is not None and is_open_hand(landmarks):
            self._remaining = self.hold_frames
            return True
        if self._remaining > 0:
            self._remaining -= 1
            return True
        return False

    def reset(self) -> None:
        self._remaining = 0
