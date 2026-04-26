#!/usr/bin/env python3
"""
CrossEmbodimentSampler — Stage 3 assembly for cross-embodiment training.

Wraps HumanDongLoader and URDFRandomizer into a single object.
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

from human_loader import HumanDongLoader
from robot_loader import URDFRandomizer


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
    """

    def __init__(
        self,
        csv_path: str | Path,
        urdf_path: str | Path,
        hand_config_path: str | Path,
        split: str = "train",
        device: str = "cpu",
    ) -> None:
        self.hand_config_path = Path(hand_config_path)
        self.human_loader = HumanDongLoader(csv_path, split=split, device=device)
        self.robot_rnd = URDFRandomizer(urdf_path, device=device)

    def get_batch(self, B: int, seed: int | None = None) -> dict:
        """Sample B random human frames + B random robot poses. No temporal pairing."""
        hb = self.human_loader.get_batch(B, seed=seed)
        return self._assemble(
            quats_h=hb["quats"],
            tips_h=hb["tips"],
            labels=hb["labels"],
            tip_labels=hb["tip_labels"],
            B=B,
            seed=seed,
        )

    def get_batch_temporal(self, B: int, seed: int | None = None) -> dict:
        """
        Sample B consecutive human frame pairs (t, t+1) + B random robot poses.
        Adds quats_h_t1 and tips_h_t1 to the output dict.
        """
        hb = self.human_loader.get_batch_temporal(B, seed=seed)
        out = self._assemble(
            quats_h=hb["quats_t"],
            tips_h=hb["tips_t"],
            labels=hb["labels"],
            tip_labels=hb["tip_labels"],
            B=B,
            seed=seed,
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
        }
