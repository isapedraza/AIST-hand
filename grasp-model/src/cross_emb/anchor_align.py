"""Run 37: supervised anchor alignment loss (self-contained, Run 20 base).

Root cause of the persistent "MCP won't close" bug (measured 2026-06-03 on the
Run 20 checkpoint): the human and robot latent clouds are not co-located
(modality gap). A static offset experiment proved that translating the
human-closed latent onto the robot-closed region recovers full closure
(MCP 0.55 -> 1.5 rad). This loss is the trainable form of that offset.

We do not need the contrastive to DISCOVER the human<->robot correspondence:
we already have it, labeled, on both sides (HaGRID closed_fist/open_hand and
the synthetic closed/open Shadow poses). It supervises co-location of the
labeled extremes directly:

    L_anchor = || mean E_h(human_closed)  - mean E_X(E_r(robot_closed)) ||
             + || mean E_h(human_open)    - mean E_X(E_r(robot_open))   ||

Mean (centroid) matching is the minimal intervention: it co-locates the clouds
at the two extremes (exactly the offset that worked) without forcing every
sample to a point, leaving the contrastive free to order what lies between.

Self-contained: reads the human anchors straight from hagrid_dong.csv (no
StaticHumanAnchorLoader dependency, which does not exist on the Run 20 base)
and the robot anchors from the synthetic qpos NPZ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# HaGRID grasp_type codes in hagrid_dong.csv (verified: 28 -> open_hand,
# 29 -> closed_fist).
_OPEN_CLASS = 28
_CLOSE_CLASS = 29

# Dong quaternion columns q1..q20, each wxyz, in joint order (4 per finger).
_QUAT_COLS = [f"q{j}_{c}" for j in range(1, 21) for c in ("w", "x", "y", "z")]


class AnchorAligner:
    def __init__(
        self,
        human_csv: str | Path,
        robot_close_npz: str | Path,
        robot_open_npz: str | Path,
        device: str = "cpu",
        zero_wrj: bool = True,
    ) -> None:
        self.device = torch.device(device)

        df = pd.read_csv(human_csv, usecols=["grasp_type", *_QUAT_COLS])
        present = set(int(v) for v in df["grasp_type"].unique())
        if _CLOSE_CLASS not in present or _OPEN_CLASS not in present:
            raise ValueError(
                f"Anchor CSV must contain grasp_type {_OPEN_CLASS} (open) and "
                f"{_CLOSE_CLASS} (close). Found {sorted(present)}."
            )

        def _human(cls: int) -> torch.Tensor:
            rows = df.loc[df["grasp_type"] == cls, _QUAT_COLS].to_numpy(np.float32)
            t = torch.from_numpy(rows).to(self.device)
            return t.view(-1, 20, 4).contiguous()  # [N, 20, 4]

        self.qh_close = _human(_CLOSE_CLASS)   # [Nc,20,4]
        self.qh_open = _human(_OPEN_CLASS)     # [No,20,4]

        def _robot(p: str | Path) -> torch.Tensor:
            d = np.load(Path(p).expanduser())
            q = torch.from_numpy(d["qpos24"].astype(np.float32)).to(self.device)
            if zero_wrj:
                q = q.clone()
                q[:, 0:2] = 0.0
            return q.contiguous()

        self.qr_close = _robot(robot_close_npz)   # [Mc,24]
        self.qr_open = _robot(robot_open_npz)      # [Mo,24]
        print(
            f"[AnchorAligner] human close/open = {len(self.qh_close)}/{len(self.qh_open)}  "
            f"robot close/open = {len(self.qr_close)}/{len(self.qr_open)}"
        )

    def _samp(self, t: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.randint(0, len(t), (n,), device=t.device)
        return t[idx]

    def loss(self, E_h, E_r, E_X, n: int = 512) -> torch.Tensor:
        """Centroid-match human and robot latents at the two labeled extremes."""
        zh_c = E_h(self._samp(self.qh_close, n))
        zh_o = E_h(self._samp(self.qh_open, n))
        zr_c = E_X(E_r(self._samp(self.qr_close, n)))
        zr_o = E_X(E_r(self._samp(self.qr_open, n)))
        return (zh_c.mean(0) - zr_c.mean(0)).norm() + (zh_o.mean(0) - zr_o.mean(0)).norm()
