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

Temporal pairs (get_batch_temporal):
    quats_t    [B, 20, 4]  frame t
    quats_t1   [B, 20, 4]  frame t+1 (same trial, consecutive frame_id)
    tips_t     [B, 5, 3]
    tips_t1    [B, 5, 3]
    labels, tip_labels (fixed)
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

# Trial identity columns — frames with identical values are in the same sequence
_TRIAL_COLS: list[str] = ["subject_id", "date_id", "object_id", "trial_id", "cam"]

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

_MIDDLE_TIP_COLS: list[str] = [
    "MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z"
]


class HumanLoader:
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

        print(f"[HumanLoader] Loading {csv_path} (split={split}) ...")
        df = pd.read_csv(csv_path)

        # Filter by subject split
        lo, hi = _SPLIT_SUBJECTS[split]
        mask = (df["subject_id"] >= lo) & (df["subject_id"] <= hi)
        df = df[mask].copy()

        # Sort by trial identity + frame_id to ensure consecutive indices = consecutive frames
        df = df.sort_values(_TRIAL_COLS + ["frame_id"]).reset_index(drop=True)
        print(f"[HumanLoader] {len(df):,} frames after split filter.")

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

        # Build next_idx: for each frame i, next_idx[i] = i+1 if same trial, else -1
        # Two consecutive rows are in the same trial iff all _TRIAL_COLS match
        trial_keys = df[_TRIAL_COLS].values  # [N, 5]
        same_trial = np.all(trial_keys[:-1] == trial_keys[1:], axis=1)  # [N-1] bool
        next_idx = np.where(same_trial, np.arange(1, self._N), -1)      # [N-1]
        next_idx = np.append(next_idx, -1)                               # last frame: always -1

        # valid_indices: frames that have a valid t+1 in the same trial
        valid_mask = next_idx != -1
        self._next_idx = torch.from_numpy(next_idx).to(self.device)
        self._valid_idx = torch.from_numpy(np.where(valid_mask)[0].astype(np.int64)).to(self.device)

        n_valid = self._valid_idx.shape[0]
        print(f"[HumanLoader] Ready. quats={tuple(self._quats.shape)}, "
              f"valid temporal pairs={n_valid:,} ({100*n_valid/self._N:.1f}%)")

    def get_batch(self, B: int, seed: int | None = None) -> dict:
        """
        Sample B random frames (no temporal pairing).

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

    def get_batch_temporal(self, B: int, seed: int | None = None) -> dict:
        """
        Sample B consecutive frame pairs (t, t+1) from the same trial.

        Returns:
            quats_t    [B, 20, 4]  frame t
            quats_t1   [B, 20, 4]  frame t+1
            tips_t     [B, 5, 3]   frame t  (normalized)
            tips_t1    [B, 5, 3]   frame t+1 (normalized)
            labels     list[str]   (fixed, len 20)
            tip_labels list[str]   (fixed, len 5)
        """
        if seed is not None:
            torch.manual_seed(seed)
        # Sample B positions from valid_idx (frames that have a t+1 in the same trial)
        pos = torch.randint(0, self._valid_idx.shape[0], (B,), device=self.device)
        idx_t  = self._valid_idx[pos]
        idx_t1 = self._next_idx[idx_t]
        return {
            "quats_t":    self._quats[idx_t],
            "quats_t1":   self._quats[idx_t1],
            "tips_t":     self._tips[idx_t],
            "tips_t1":    self._tips[idx_t1],
            "labels":     self.labels,
            "tip_labels": self.tip_labels,
        }
