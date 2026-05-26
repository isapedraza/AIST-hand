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

_R6_COLS: list[str] = [
    f"q{j}_r6_{c}"
    for j in range(1, 21)
    for c in ("c1x", "c1y", "c1z", "c2x", "c2y", "c2z")
]


def _validate_human_rot_repr(human_rot_repr: str) -> str:
    if human_rot_repr not in {"quat", "r6"}:
        raise ValueError(f"human_rot_repr must be quat or r6, got {human_rot_repr!r}")
    return human_rot_repr


def _pose_cols_and_dim(human_rot_repr: str) -> tuple[list[str], int]:
    human_rot_repr = _validate_human_rot_repr(human_rot_repr)
    if human_rot_repr == "quat":
        return _QUAT_COLS, 4
    return _R6_COLS, 6


def _require_cols(df: pd.DataFrame, cols: list[str], csv_path: str | Path, human_rot_repr: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing {human_rot_repr} columns. First missing: {missing[:5]}"
        )


def _pose_batch(pose: torch.Tensor, labels: list[str], tips: torch.Tensor, tip_labels: list[str], chain: torch.Tensor, human_rot_repr: str) -> dict:
    out = {
        "pose": pose,
        "labels": labels,
        "tips": tips,
        "tip_labels": tip_labels,
        "chain": chain,
    }
    if human_rot_repr == "quat":
        out["quats"] = pose
    return out


def _pose_temporal_batch(
    pose_t: torch.Tensor,
    pose_t1: torch.Tensor,
    labels: list[str],
    tips_t: torch.Tensor,
    tips_t1: torch.Tensor,
    tip_labels: list[str],
    chain_t: torch.Tensor,
    chain_t1: torch.Tensor,
    human_rot_repr: str,
) -> dict:
    out = {
        "pose_t": pose_t,
        "pose_t1": pose_t1,
        "tips_t": tips_t,
        "tips_t1": tips_t1,
        "labels": labels,
        "tip_labels": tip_labels,
        "chain_t": chain_t,
        "chain_t1": chain_t1,
    }
    if human_rot_repr == "quat":
        out["quats_t"] = pose_t
        out["quats_t1"] = pose_t1
    return out


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

# Full chain positions per finger: [MCP, PIP, DIP, TIP] x [x, y, z]
# Ordered to match TIP_LABELS: thumb, index, middle, ring, pinky
# Thumb: human anatomy has only 3 phalanges (no DIP). To make formula
# `chain[:,f,3]-chain[:,f,2]` (last-segment direction) work uniformly for
# all fingers, thumb slot 2 (called "DIP") is filled with THUMB_IP (the
# last joint before TIP). Therefore slot 1 == slot 2 == THUMB_IP for thumb,
# and slot 3 - slot 2 = TIP - IP = real last segment vector.
# Robot side already uses analogous convention (thdistal -> thtip).
_CHAIN_COLS: list[str] = [
    "THUMB_MCP_x",         "THUMB_MCP_y",         "THUMB_MCP_z",
    "THUMB_IP_x",          "THUMB_IP_y",           "THUMB_IP_z",
    "THUMB_IP_x",          "THUMB_IP_y",           "THUMB_IP_z",
    "THUMB_TIP_x",         "THUMB_TIP_y",          "THUMB_TIP_z",
    "INDEX_FINGER_MCP_x",  "INDEX_FINGER_MCP_y",   "INDEX_FINGER_MCP_z",
    "INDEX_FINGER_PIP_x",  "INDEX_FINGER_PIP_y",   "INDEX_FINGER_PIP_z",
    "INDEX_FINGER_DIP_x",  "INDEX_FINGER_DIP_y",   "INDEX_FINGER_DIP_z",
    "INDEX_FINGER_TIP_x",  "INDEX_FINGER_TIP_y",   "INDEX_FINGER_TIP_z",
    "MIDDLE_FINGER_MCP_x", "MIDDLE_FINGER_MCP_y",  "MIDDLE_FINGER_MCP_z",
    "MIDDLE_FINGER_PIP_x", "MIDDLE_FINGER_PIP_y",  "MIDDLE_FINGER_PIP_z",
    "MIDDLE_FINGER_DIP_x", "MIDDLE_FINGER_DIP_y",  "MIDDLE_FINGER_DIP_z",
    "MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y",  "MIDDLE_FINGER_TIP_z",
    "RING_FINGER_MCP_x",   "RING_FINGER_MCP_y",    "RING_FINGER_MCP_z",
    "RING_FINGER_PIP_x",   "RING_FINGER_PIP_y",    "RING_FINGER_PIP_z",
    "RING_FINGER_DIP_x",   "RING_FINGER_DIP_y",    "RING_FINGER_DIP_z",
    "RING_FINGER_TIP_x",   "RING_FINGER_TIP_y",    "RING_FINGER_TIP_z",
    "PINKY_MCP_x",         "PINKY_MCP_y",          "PINKY_MCP_z",
    "PINKY_PIP_x",         "PINKY_PIP_y",          "PINKY_PIP_z",
    "PINKY_DIP_x",         "PINKY_DIP_y",          "PINKY_DIP_z",
    "PINKY_TIP_x",         "PINKY_TIP_y",          "PINKY_TIP_z",
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
        human_rot_repr: str = "quat",
    ) -> None:
        if split not in _SPLIT_SUBJECTS:
            raise ValueError(f"split must be one of {list(_SPLIT_SUBJECTS)}, got '{split}'")

        self.device = torch.device(device)
        self.human_rot_repr = _validate_human_rot_repr(human_rot_repr)
        self.pose_cols, self.pose_dim = _pose_cols_and_dim(self.human_rot_repr)
        self.labels: list[str] = DONG_LABELS
        self.tip_labels: list[str] = TIP_LABELS

        print(f"[HumanLoader] Loading {csv_path} (split={split}, rot_repr={self.human_rot_repr}) ...")
        df = pd.read_csv(csv_path)

        # Filter by subject split
        lo, hi = _SPLIT_SUBJECTS[split]
        mask = (df["subject_id"] >= lo) & (df["subject_id"] <= hi)
        df = df[mask].copy()

        # Sort by trial identity + frame_id to ensure consecutive indices = consecutive frames
        df = df.sort_values(_TRIAL_COLS + ["frame_id"]).reset_index(drop=True)
        print(f"[HumanLoader] {len(df):,} frames after split filter.")

        # Compute hand_length per subject as sum of middle finger segment lengths.
        # Segments: WRIST(origin)→MCP + MCP→PIP + PIP→DIP + DIP→TIP.
        # Pose-invariant: segment lengths = bone lengths, unaffected by flexion.
        mcp = df[["MIDDLE_FINGER_MCP_x", "MIDDLE_FINGER_MCP_y", "MIDDLE_FINGER_MCP_z"]].values
        pip = df[["MIDDLE_FINGER_PIP_x", "MIDDLE_FINGER_PIP_y", "MIDDLE_FINGER_PIP_z"]].values
        dip = df[["MIDDLE_FINGER_DIP_x", "MIDDLE_FINGER_DIP_y", "MIDDLE_FINGER_DIP_z"]].values
        tip_m = df[["MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z"]].values
        seg_lengths = (
            np.linalg.norm(mcp, axis=1)
            + np.linalg.norm(pip - mcp, axis=1)
            + np.linalg.norm(dip - pip, axis=1)
            + np.linalg.norm(tip_m - dip, axis=1)
        )
        subject_hl = pd.Series(seg_lengths, index=df.index).groupby(df["subject_id"]).median()
        hl_per_frame = df["subject_id"].map(subject_hl).values.astype(np.float32)

        # Human rotation pose: [N, 20, F], F=4 for quat or F=6 for R6.
        _require_cols(df, self.pose_cols, csv_path, self.human_rot_repr)
        pose_np = df[self.pose_cols].values.astype(np.float32)
        pose_np = pose_np.reshape(-1, 20, self.pose_dim)

        hl = hl_per_frame[:, None, None]  # [N, 1, 1]

        # Tips: [N, 5, 3] normalized by subject hand_length
        tips_np = df[_TIP_COLS].values.astype(np.float32)    # [N, 15]
        tips_np = tips_np.reshape(-1, 5, 3)
        tips_np = tips_np / hl

        # Chain positions: [N, 5, 4, 3] — MCP/PIP/DIP/TIP per finger, normalized
        chain_np = df[_CHAIN_COLS].values.astype(np.float32)  # [N, 60]
        chain_np = chain_np.reshape(-1, 5, 4, 3)
        chain_np = chain_np / hl[:, :, None, :]

        self._pose  = torch.from_numpy(pose_np).to(self.device)    # [N, 20, F]
        self._tips  = torch.from_numpy(tips_np).to(self.device)    # [N, 5, 3]
        self._chain = torch.from_numpy(chain_np).to(self.device)   # [N, 5, 4, 3]
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
        print(f"[HumanLoader] Ready. pose={tuple(self._pose.shape)} ({self.human_rot_repr}), "
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
        return _pose_batch(
            self._pose[idx],
            self.labels,
            self._tips[idx],
            self.tip_labels,
            self._chain[idx],
            self.human_rot_repr,
        )

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
        return _pose_temporal_batch(
            self._pose[idx_t],
            self._pose[idx_t1],
            self.labels,
            self._tips[idx_t],
            self._tips[idx_t1],
            self.tip_labels,
            self._chain[idx_t],
            self._chain[idx_t1],
            self.human_rot_repr,
        )


class StaticHumanAnchorLoader:
    """
    Loads static human anchor poses, such as HaGRID open-hand/fist samples.

    These rows are not temporal sequences. get_batch_temporal returns t1=t so
    the temporal loss sees zero human fingertip velocity for anchor poses.
    TIP_COLS are expected to be already normalized by hand length.
    """

    def __init__(
        self,
        csv_path: str | Path,
        device: str = "cpu",
        human_rot_repr: str = "quat",
    ) -> None:
        self.device = torch.device(device)
        self.human_rot_repr = _validate_human_rot_repr(human_rot_repr)
        self.pose_cols, self.pose_dim = _pose_cols_and_dim(self.human_rot_repr)
        self.labels: list[str] = DONG_LABELS
        self.tip_labels: list[str] = TIP_LABELS

        print(f"[StaticHumanAnchorLoader] Loading {csv_path} (rot_repr={self.human_rot_repr}) ...")
        df = pd.read_csv(csv_path)
        if "grasp_type" not in df.columns:
            raise ValueError(f"{csv_path} must include a grasp_type column")

        _require_cols(df, self.pose_cols, csv_path, self.human_rot_repr)
        pose_np = df[self.pose_cols].values.astype(np.float32).reshape(-1, 20, self.pose_dim)
        labels_np = df["grasp_type"].values.astype(np.int64)

        # Normalize per sample by middle finger segment sum (same definition as HumanLoader).
        # HaGRID has no fixed subjects, so per-sample normalization is correct.
        mcp = df[["MIDDLE_FINGER_MCP_x", "MIDDLE_FINGER_MCP_y", "MIDDLE_FINGER_MCP_z"]].values
        pip = df[["MIDDLE_FINGER_PIP_x", "MIDDLE_FINGER_PIP_y", "MIDDLE_FINGER_PIP_z"]].values
        dip = df[["MIDDLE_FINGER_DIP_x", "MIDDLE_FINGER_DIP_y", "MIDDLE_FINGER_DIP_z"]].values
        tip_m = df[["MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z"]].values
        hl = (
            np.linalg.norm(mcp, axis=1)
            + np.linalg.norm(pip - mcp, axis=1)
            + np.linalg.norm(dip - pip, axis=1)
            + np.linalg.norm(tip_m - dip, axis=1)
        ).astype(np.float32)  # [N]
        hl = hl[:, None, None]  # [N, 1, 1] for broadcasting

        tips_np = df[_TIP_COLS].values.astype(np.float32).reshape(-1, 5, 3) / hl

        if all(c in df.columns for c in _CHAIN_COLS):
            chain_np = df[_CHAIN_COLS].values.astype(np.float32).reshape(-1, 5, 4, 3) / hl[:, :, None, :]
        else:
            chain_np = np.repeat(tips_np[:, :, None, :], 4, axis=2)

        self._pose = torch.from_numpy(pose_np).to(self.device)
        self._tips = torch.from_numpy(tips_np).to(self.device)
        self._chain = torch.from_numpy(chain_np).to(self.device)
        self._grasp_type = torch.from_numpy(labels_np).to(self.device)
        self._N = len(df)
        self._classes = sorted(int(v) for v in np.unique(labels_np))
        self._class_indices = {
            c: torch.from_numpy(np.where(labels_np == c)[0].astype(np.int64)).to(self.device)
            for c in self._classes
        }
        class_counts = {c: int((labels_np == c).sum()) for c in self._classes}
        print(f"[StaticHumanAnchorLoader] Ready. pose={tuple(self._pose.shape)} ({self.human_rot_repr}), classes={class_counts}")

    def _sample_balanced_indices(self, B: int) -> torch.Tensor:
        if B <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        chunks = []
        n_classes = len(self._classes)
        base = B // n_classes
        rem = B % n_classes
        for i, c in enumerate(self._classes):
            n = base + (1 if i < rem else 0)
            pool = self._class_indices[c]
            pos = torch.randint(0, pool.shape[0], (n,), device=self.device)
            chunks.append(pool[pos])
        return torch.cat(chunks, dim=0)[torch.randperm(B, device=self.device)]

    def get_batch(self, B: int) -> dict:
        idx = self._sample_balanced_indices(B)
        out = _pose_batch(
            self._pose[idx],
            self.labels,
            self._tips[idx],
            self.tip_labels,
            self._chain[idx],
            self.human_rot_repr,
        )
        out["grasp_type"] = self._grasp_type[idx]
        return out

    def get_batch_temporal(self, B: int) -> dict:
        batch = self.get_batch(B)
        out = _pose_temporal_batch(
            batch["pose"],
            batch["pose"],
            batch["labels"],
            batch["tips"],
            batch["tips"],
            batch["tip_labels"],
            batch["chain"],
            batch["chain"],
            self.human_rot_repr,
        )
        out["grasp_type"] = batch["grasp_type"]
        return out
