from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.human_modules import HumanEncoder_E_h, HumanEncoder_E_h_single, CAMLayer
from ..nn.shared_modules import _mlp
from ..nn.robot_modules  import RobotDecoder_D_r


class _LegacyHumanEncoder(nn.Module):
    """3-subspace CAM-GNN encoder (thumb / precision / support)."""
    N_JOINTS = 21
    _THUMB_NODES     = [1, 2, 3, 4]
    _PRECISION_NODES = [5, 6, 7, 8, 9, 10, 11, 12]
    _SUPPORT_NODES   = [13, 14, 15, 16, 17, 18, 19, 20]

    def __init__(self, in_dim=4, hidden_dim=32, z_dim=64):
        super().__init__()
        self.cam    = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS).uniform_(-1, 1))
        self.layer1 = CAMLayer(in_dim,     hidden_dim, self.N_JOINTS)
        self.layer2 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.layer3 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.proj_thumb     = nn.Linear(hidden_dim, z_dim)
        self.proj_precision = nn.Linear(hidden_dim, z_dim)
        self.proj_support   = nn.Linear(hidden_dim, z_dim)
        self.out_act = nn.Tanh()

    def forward(self, quats):
        B = quats.shape[0]
        wrist = torch.zeros(B, 1, 4, dtype=quats.dtype, device=quats.device)
        wrist[:, 0, 0] = 1.0
        quats = torch.cat([wrist, quats], dim=1)
        B, N, in_f = quats.shape
        x = quats.reshape(B * N, in_f)
        x = self.layer1(x, self.cam)
        x = self.layer2(x, self.cam)
        x = self.layer3(x, self.cam)
        x = x.view(B, N, -1)
        z_t = self.out_act(self.proj_thumb(    x[:, self._THUMB_NODES,     :].max(dim=1).values))
        z_p = self.out_act(self.proj_precision(x[:, self._PRECISION_NODES, :].max(dim=1).values))
        z_s = self.out_act(self.proj_support(  x[:, self._SUPPORT_NODES,   :].max(dim=1).values))
        return torch.cat([z_t, z_p, z_s], dim=-1)


class _SharedDecoder_compat(nn.Module):
    """D_X with configurable input dim (handles both 3*z_dim and 5*z_dim checkpoints)."""
    def __init__(self, in_dim: int, shared_dim: int = 1024):
        super().__init__()
        self.net = _mlp([in_dim, 256, 256, 256, 256, 256, 256, 256, shared_dim])

    def forward(self, z):
        return self.net(z)

# Shadow Hand joint limits (24 DOF)
_LOWER = np.array([
    -0.524, -0.698,
    -0.349, -0.262, 0.000, 0.000,
    -0.349, -0.262, 0.000, 0.000,
    -0.349, -0.262, 0.000, 0.000,
     0.000, -0.349, -0.262, 0.000, 0.000,
    -1.047,  0.000, -0.209, -0.698, -0.262,
], dtype=np.float32)

_UPPER = np.array([
     0.175,  0.489,
     0.349,  1.571, 1.571, 1.571,
     0.349,  1.571, 1.571, 1.571,
     0.349,  1.571, 1.571, 1.571,
     0.785,  0.349,  1.571, 1.571, 1.571,
     1.047,  1.222,  0.209,  0.698,  1.571,
], dtype=np.float32)


class Retargeter:
    """
    Human quats -> Shadow Hand qpos.

    Input : torch.Tensor [B, 20, 4]  Dong quaternions (joints 1-20, no wrist)
    Output: np.ndarray   [B, 24]     joint positions clipped to Shadow Hand limits
    """

    def __init__(self, ckpt_path: str | Path):
        ck  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        n_j = ck["D_r"]["fc.weight"].shape[0]
        dx_in_dim = ck["D_X"]["net.0.weight"].shape[1]

        if "proj_hand.0.weight" in ck["E_h"]:
            # Run 25+ single-latent encoder
            z_dim_total = ck["E_h"]["proj_hand.0.weight"].shape[0]
            self.E_h = HumanEncoder_E_h_single(in_dim=4, hidden_dim=32, z_dim_total=z_dim_total).eval()
        elif "proj_precision.weight" in ck["E_h"]:
            # Legacy 3-subspace encoder (thumb / precision / support)
            z_dim    = ck["E_h"]["proj_thumb.weight"].shape[0]
            self.E_h = _LegacyHumanEncoder(in_dim=4, hidden_dim=32, z_dim=z_dim).eval()
        else:
            # 5-subspace encoder (thumb / index / middle / ring / pinky)
            z_dim    = ck["E_h"]["proj_thumb.weight"].shape[0]
            self.E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=z_dim).eval()

        self.D_X = _SharedDecoder_compat(in_dim=dx_in_dim, shared_dim=1024).eval()
        self.D_r = RobotDecoder_D_r(n_joints=n_j, shared_dim=1024).eval()

        self.E_h.load_state_dict(ck["E_h"])
        self.D_X.load_state_dict(ck["D_X"])
        self.D_r.load_state_dict(ck["D_r"])

    @torch.no_grad()
    def __call__(self, quats: torch.Tensor) -> np.ndarray:
        q = self.D_r(self.D_X(self.E_h(quats))).numpy()
        q[:, 0:2] = 0.0  # WRJ2, WRJ1 — not encoded in wrist-frame human signal
        return np.clip(q, _LOWER, _UPPER)
