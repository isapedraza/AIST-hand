#!/usr/bin/env python3
"""
CrossEmbodimentSampler — Stage 3 assembly for cross-embodiment training.

Wraps HumanLoader and RobotLoader into a single object.
Each call to get_batch(B) / get_batch_temporal(B) returns a flat dict
with everything the training loop needs for one step.

Output contract:
    q_r            [B, J]      raw robot joint angles             -> E_r
    quats_h        [B, Nh, 4]  human Dong quats frame t           -> E_h
    quats_h_sub    [B, K, 4]   human quats, common joints only    -> D_R human
    quats_r_sub    [B, K, 4]   robot quats, common joints only    -> D_R robot
    tips_h_sub     [B, Fc, 3]  human fingertips, common fingers   -> D_ee human
    tips_r_sub     [B, Fc, 3]  robot fingertips, common fingers   -> D_ee robot
    common_labels  list[str]   joint labels in subspace (len K)
    common_fingers list[str]   finger names in subspace (len Fc)

Extra keys for get_batch_temporal only:
    quats_h_t1     [B, Nh, 4]  human quats frame t+1              -> L_temporal / v_H^hand
    tips_h_t1      [B, Fh, 3]  human fingertips frame t+1         -> v_H^hand velocity
"""

from __future__ import annotations

from pathlib import Path

import torch

from human_loader import HumanLoader, StaticHumanAnchorLoader
from robot_loader import RobotLoader


def filter_to_subspace(
    quats_h: torch.Tensor,
    labels_h: list[str],
    quats_r: torch.Tensor,
    labels_r: list[str],
    tips_h: torch.Tensor,
    tip_labels_h: list[str],
    tips_r: torch.Tensor,
    tip_labels_r: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    """
    Filter quaternions and fingertip positions to the common subspace.

    Joint labels have the form "<finger>_<joint>" (e.g. "thumb_mcp").
    Tip labels have the form "<finger>" (e.g. "thumb").
    Common fingers are derived from the common joint labels.

    Args:
        quats_h       : [B, Nh, 4]  human Dong quaternions
        labels_h      : list[str]   len Nh
        quats_r       : [B, Nr, 4]  robot Dong quaternions
        labels_r      : list[str]   len Nr
        tips_h        : [B, Fh, 3]  human fingertip positions (normalized)
        tip_labels_h  : list[str]   len Fh
        tips_r        : [B, Fr, 3]  robot fingertip positions (normalized)
        tip_labels_r  : list[str]   len Fr

    Returns:
        quats_h_sub     : [B, K, 4]
        quats_r_sub     : [B, K, 4]
        tips_h_sub      : [B, Fc, 3]
        tips_r_sub      : [B, Fc, 3]
        common_labels   : list[str] len K
        common_fingers  : list[str] len Fc
    """
    common = [l for l in labels_r if l in labels_h]
    if not common:
        raise ValueError(f"No common joint labels between human {labels_h} and robot {labels_r}")
    idx_h = [labels_h.index(l) for l in common]
    idx_r = [labels_r.index(l) for l in common]

    common_fingers_from_joints = list(dict.fromkeys(l.split("_")[0] for l in common))
    common_fingers = [f for f in common_fingers_from_joints
                      if f in tip_labels_h and f in tip_labels_r]
    if not common_fingers:
        raise ValueError(f"No common fingers for tips: human={tip_labels_h} robot={tip_labels_r}")
    tidx_h = [tip_labels_h.index(f) for f in common_fingers]
    tidx_r = [tip_labels_r.index(f) for f in common_fingers]

    return (
        quats_h[:, idx_h, :],
        quats_r[:, idx_r, :],
        tips_h[:, tidx_h, :],
        tips_r[:, tidx_r, :],
        common,
        common_fingers,
    )


class CrossEmbodimentSampler:
    """
    Single entry point for the training loop.

    Args:
        csv_path         : path to hograspnet_abl11.csv
        urdf_path        : path to robot URDF
        hand_config_path : path to hand YAML (hand_configs/*.yaml)
        split            : "train", "val", or "test"
        device           : torch device
        valid_poses_path : path to valid_robot_poses.npz (mode=VALID_NPZ).
                           If None, uses random uniform sampling (mode=RANDOM_UNIFORM).
                           If set but file missing, crashes immediately.
        extra_human_csv  : optional static human anchor CSV, mixed into train batches.
        extra_human_ratio: fraction of human batch reserved for static anchors.
    """

    def __init__(
        self,
        csv_path: str | Path,
        urdf_path: str | Path,
        hand_config_path: str | Path,
        split: str = "train",
        device: str = "cpu",
        valid_poses_path: str | Path | None = None,
        extra_human_csv: str | Path | None = None,
        extra_human_ratio: float = 0.0,
    ) -> None:
        self.hand_config_path = Path(hand_config_path)
        self.split = split
        self.extra_human_ratio = float(extra_human_ratio)
        self.human_loader = HumanLoader(csv_path, split=split, device=device)
        self.extra_human_loader = None
        if extra_human_csv is not None and self.extra_human_ratio > 0:
            if split != "train":
                print(f"[CrossEmbodimentSampler] Ignoring extra_human_csv for split={split}.")
            else:
                self.extra_human_loader = StaticHumanAnchorLoader(extra_human_csv, device=device)
                print(
                    "[CrossEmbodimentSampler] Extra human anchors enabled: "
                    f"ratio={self.extra_human_ratio:.3f}"
                )
        self.robot_rnd = RobotLoader(urdf_path, device=device, valid_poses_path=valid_poses_path)

    def _batch_counts(self, B: int) -> tuple[int, int]:
        if self.extra_human_loader is None:
            return B, 0
        B_extra = min(B, max(1, int(round(B * self.extra_human_ratio))))
        return B - B_extra, B_extra

    def get_batch(self, B: int, seed: int | None = None) -> dict:
        """Sample B random human frames + B random robot poses. No temporal pairing."""
        B_base, B_extra = self._batch_counts(B)
        hb = self.human_loader.get_batch(B_base, seed=seed)
        extra_counts = {}
        if B_extra:
            eb = self.extra_human_loader.get_batch(B_extra)
            hb["quats"] = torch.cat([hb["quats"], eb["quats"]], dim=0)
            hb["tips"] = torch.cat([hb["tips"], eb["tips"]], dim=0)
            extra_counts = {
                int(k.item()): int(v.item())
                for k, v in zip(*torch.unique(eb["grasp_type"], return_counts=True))
            }
        return self._assemble(
            quats_h=hb["quats"],
            tips_h=hb["tips"],
            labels=hb["labels"],
            tip_labels=hb["tip_labels"],
            B=B,
            seed=seed,
            extra_human_count=B_extra,
            extra_human_by_class=extra_counts,
        )

    def get_batch_temporal(self, B: int, seed: int | None = None) -> dict:
        """
        Sample B consecutive human frame pairs (t, t+1) + B random robot poses.
        Adds quats_h_t1 and tips_h_t1 to the output dict.
        """
        B_base, B_extra = self._batch_counts(B)
        hb = self.human_loader.get_batch_temporal(B_base, seed=seed)
        extra_counts = {}
        if B_extra:
            eb = self.extra_human_loader.get_batch_temporal(B_extra)
            hb["quats_t"] = torch.cat([hb["quats_t"], eb["quats_t"]], dim=0)
            hb["quats_t1"] = torch.cat([hb["quats_t1"], eb["quats_t1"]], dim=0)
            hb["tips_t"] = torch.cat([hb["tips_t"], eb["tips_t"]], dim=0)
            hb["tips_t1"] = torch.cat([hb["tips_t1"], eb["tips_t1"]], dim=0)
            extra_counts = {
                int(k.item()): int(v.item())
                for k, v in zip(*torch.unique(eb["grasp_type"], return_counts=True))
            }
        out = self._assemble(
            quats_h=hb["quats_t"],
            tips_h=hb["tips_t"],
            labels=hb["labels"],
            tip_labels=hb["tip_labels"],
            B=B,
            seed=seed,
            extra_human_count=B_extra,
            extra_human_by_class=extra_counts,
        )
        out["quats_h_t1"] = hb["quats_t1"]  # [B, Nh, 4] -> L_temporal
        out["tips_h_t1"]  = hb["tips_t1"]   # [B, Fh, 3] -> v_H^hand velocity
        return out

    def _assemble(
        self,
        quats_h: torch.Tensor,
        tips_h: torch.Tensor,
        labels: list[str],
        tip_labels: list[str],
        B: int,
        seed: int | None,
        extra_human_count: int = 0,
        extra_human_by_class: dict[int, int] | None = None,
    ) -> dict:
        q_r, quats_r, labels_r, meta_r = self.robot_rnd.sample_dong(
            B, self.hand_config_path, seed=seed
        )

        quats_h_sub, quats_r_sub, tips_h_sub, tips_r_sub, common_labels, common_fingers = filter_to_subspace(
            quats_h, labels,
            quats_r, labels_r,
            tips_h, tip_labels,
            meta_r["tips"], meta_r["tip_labels"],
        )

        return {
            "q_r":            q_r,
            "quats_h":        quats_h,
            "quats_h_sub":    quats_h_sub,
            "quats_r_sub":    quats_r_sub,
            "tips_h_sub":     tips_h_sub,
            "tips_r_sub":     tips_r_sub,
            "common_labels":  common_labels,
            "common_fingers": common_fingers,
            "extra_human_count": extra_human_count,
            "extra_human_by_class": extra_human_by_class or {},
        }
