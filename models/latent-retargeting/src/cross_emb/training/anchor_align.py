"""Run 37: supervised anchor alignment loss.

Root cause of the persistent "MCP won't close" bug (measured 2026-06-03): the
human and robot latent clouds are not co-located (modality gap), and neither
contrastive oracle co-locates the CLOSED region -- Xin S_k is inverted
cross-embodiment (rates human-fist closer to robot-open) and D_R's closed
signal is razor-thin. A static offset experiment proved that translating the
human-closed latent onto the robot-closed region recovers full closure
(MCP 0.55 -> 1.5 rad).

We do not need the contrastive to DISCOVER the human<->robot correspondence:
we already have it, labeled, on both sides (HaGRID closed_fist/open_hand and
the synthetic closed/open Shadow poses). This loss supervises co-location of
the labeled extremes directly:

    L_anchor = || mean E_h(human_closed)  - mean E_X(E_r(robot_closed)) ||
             + || mean E_h(human_open)    - mean E_X(E_r(robot_open))   ||

Mean (centroid) matching is the minimal intervention -- it co-locates the
clouds at the two extremes (exactly the offset that worked) without forcing
every sample to a point, leaving the contrastive free to order what lies
between.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cross_emb.loaders.human_loader import StaticHumanAnchorLoader

# HaGRID grasp_type codes in hagrid_dong.csv (verified: anchor_label
# open_hand -> 28, closed_fist -> 29).
_OPEN_CLASS = 28
_CLOSE_CLASS = 29


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

        hl = StaticHumanAnchorLoader(human_csv, device=device)
        if _CLOSE_CLASS not in hl._class_indices or _OPEN_CLASS not in hl._class_indices:
            raise ValueError(
                f"Anchor CSV must contain grasp_type {_OPEN_CLASS} (open) and "
                f"{_CLOSE_CLASS} (close). Found {sorted(hl._class_indices)}."
            )
        self.qh_close = hl._quats[hl._class_indices[_CLOSE_CLASS]].contiguous()  # [Nc,20,4]
        self.qh_open  = hl._quats[hl._class_indices[_OPEN_CLASS]].contiguous()   # [No,20,4]

        def _load_robot(p: str | Path) -> torch.Tensor:
            d = np.load(Path(p).expanduser())
            q = torch.from_numpy(d["qpos24"].astype(np.float32)).to(self.device)
            if zero_wrj:
                q = q.clone()
                q[:, 0:2] = 0.0
            return q.contiguous()

        self.qr_close = _load_robot(robot_close_npz)   # [Mc,24]
        self.qr_open  = _load_robot(robot_open_npz)    # [Mo,24]
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
