#!/usr/bin/env python3
"""
Human Dong Loader.

Loads hograspnet_abl11.csv into RAM and provides random batches
of human Dong quaternions and normalized fingertip positions.

Output contract matches what stage3_assemble expects in human_batch:
    quats      [B, 20, 4]  Dong quaternions (q1-q20, wxyz, w>=0)
    labels     list[str]   20 joint labels (fixed)
    tips       [B, 5, 3]   fingertip positions normalized by hand_length
    tip_labels list[str]   ["thumb","index","middle","ring","pinky"]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


# S1 subject split of HOGraspNet (by subject_id number)
_SPLIT_SUBJECTS: dict[str, tuple[int, int]] = {
    "train": (11, 73),
    "val":   (1,  10),
    "test":  (74, 99),
}

# Dong joint IDs -> label mapping (IDs 1-20, 4 per finger)
# TIP joints (4,8,12,16,20) are always identity but included for consistency
DONG_LABELS: list[str] = [
    "thumb_mcp",  "thumb_pip",  "thumb_dip",  "thumb_tip",   # 1-4
    "index_mcp",  "index_pip",  "index_dip",  "index_tip",   # 5-8
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",  # 9-12
    "ring_mcp",   "ring_pip",   "ring_dip",   "ring_tip",    # 13-16
    "pinky_mcp",  "pinky_pip",  "pinky_dip",  "pinky_tip",   # 17-20
]

TIP_LABELS: list[str] = ["thumb", "index", "middle", "ring", "pinky"]

# CSV column groups
_QUAT_COLS: list[str] = [
    f"q{j}_{c}" for j in range(1, 21) for c in ("w", "x", "y", "z")
]

_TIP_COLS: list[str] = [
    "THUMB_TIP_x",        "THUMB_TIP_y",        "THUMB_TIP_z",
    "INDEX_FINGER_TIP_x", "INDEX_FINGER_TIP_y", "INDEX_FINGER_TIP_z",
    "MIDDLE_FINGER_TIP_x","MIDDLE_FINGER_TIP_y","MIDDLE_FINGER_TIP_z",
    "RING_FINGER_TIP_x",  "RING_FINGER_TIP_y",  "RING_FINGER_TIP_z",
    "PINKY_TIP_x",        "PINKY_TIP_y",        "PINKY_TIP_z",
]

# Middle finger tip columns for hand_length computation
_MIDDLE_TIP_COLS: list[str] = [
    "MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z"
]


class HumanDongLoader:
    """
    Loads hograspnet_abl11.csv into RAM and samples random batches.

    Args:
        csv_path  : path to hograspnet_abl11.csv
        split     : "train", "val", or "test" (S1 subject split)
        device    : torch device for output tensors
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        device: str = "cpu",
    ) -> None:
        if split not in _SPLIT_SUBJECTS:
            raise ValueError(f"split must be one of {list(_SPLIT_SUBJECTS)}, got '{split}'")

        self.device = torch.device(device)
        self.labels: list[str] = DONG_LABELS
        self.tip_labels: list[str] = TIP_LABELS

        print(f"[HumanDongLoader] Loading {csv_path} (split={split}) ...")
        df = pd.read_csv(csv_path)

        # Filter by subject split
        lo, hi = _SPLIT_SUBJECTS[split]
        mask = (df["subject_id"] >= lo) & (df["subject_id"] <= hi)
        df = df[mask].reset_index(drop=True)
        print(f"[HumanDongLoader] {len(df):,} frames after split filter.")

        # Compute hand_length per subject (median of wrist->middle_tip distance)
        # Tips are already root-relative (wrist=origin), so ||middle_tip|| = distance
        middle_tip = df[_MIDDLE_TIP_COLS].values  # [N, 3]
        hand_length_per_frame = np.linalg.norm(middle_tip, axis=1)
        subject_hl = pd.Series(hand_length_per_frame, index=df.index).groupby(df["subject_id"]).median()
        hl_per_frame = df["subject_id"].map(subject_hl).values.astype(np.float32)

        # Quaternions: [N, 20, 4]
        quats_np = df[_QUAT_COLS].values.astype(np.float32)  # [N, 80]
        quats_np = quats_np.reshape(-1, 20, 4)

        # Tips: [N, 5, 3] normalized by subject hand_length
        tips_np = df[_TIP_COLS].values.astype(np.float32)    # [N, 15]
        tips_np = tips_np.reshape(-1, 5, 3)
        hl = hl_per_frame[:, None, None]  # [N, 1, 1]
        tips_np = tips_np / hl

        self._quats = torch.from_numpy(quats_np).to(self.device)  # [N, 20, 4]
        self._tips  = torch.from_numpy(tips_np).to(self.device)   # [N, 5, 3]
        self._N = len(df)
        print(f"[HumanDongLoader] Ready. quats={tuple(self._quats.shape)}, tips={tuple(self._tips.shape)}")

    def get_batch(self, B: int, seed: int | None = None) -> dict:
        """
        Sample B random frames.

        Returns:
            quats      [B, 20, 4]
            labels     list[str]  (fixed, len 20)
            tips       [B, 5, 3]
            tip_labels list[str]  (fixed, len 5)
        """
        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.randint(0, self._N, (B,), device=self.device)
        return {
            "quats":      self._quats[idx],
            "labels":     self.labels,
            "tips":       self._tips[idx],
            "tip_labels": self.tip_labels,
        }
