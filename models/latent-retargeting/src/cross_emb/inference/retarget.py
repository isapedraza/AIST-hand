from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.human_modules import HumanEncoder_E_h, CAMLayer
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
_SHADOW_LOWER = np.array([
    -0.524, -0.698,
    -0.349, -0.262, 0.000, 0.000,
    -0.349, -0.262, 0.000, 0.000,
    -0.349, -0.262, 0.000, 0.000,
     0.000, -0.349, -0.262, 0.000, 0.000,
    -1.047,  0.000, -0.209, -0.698, -0.262,
], dtype=np.float32)

_SHADOW_UPPER = np.array([
     0.175,  0.489,
     0.349,  1.571, 1.571, 1.571,
     0.349,  1.571, 1.571, 1.571,
     0.349,  1.571, 1.571, 1.571,
     0.785,  0.349,  1.571, 1.571, 1.571,
     1.047,  1.222,  0.209,  0.698,  1.571,
], dtype=np.float32)

# Allegro Hand joint limits (16 DOF) — wonik_allegro right_hand.xml jnt_range.
# Order: ff[j0..j3], mf[j0..j3], rf[j0..j3], th[j0..j3]. No wrist DOFs (n_pad=0).
_ALLEGRO_LOWER = np.array([
    -0.470, -0.196, -0.174, -0.227,
    -0.470, -0.196, -0.174, -0.227,
    -0.470, -0.196, -0.174, -0.227,
     0.263, -0.105, -0.189, -0.162,
], dtype=np.float32)

_ALLEGRO_UPPER = np.array([
     0.470,  1.610,  1.709,  1.618,
     0.470,  1.610,  1.709,  1.618,
     0.470,  1.610,  1.709,  1.618,
     1.396,  1.163,  1.644,  1.719,
], dtype=np.float32)

# Per-robot output spec: n_pad = leading wrist DOFs zeroed (not in wrist-frame
# human signal), lower/upper = qpos clip limits matching the MuJoCo model.
_OUTPUT_SPECS = {
    "shadow":  {"n_pad": 2, "lower": _SHADOW_LOWER,  "upper": _SHADOW_UPPER},
    "allegro": {"n_pad": 0, "lower": _ALLEGRO_LOWER, "upper": _ALLEGRO_UPPER},
}


class Retargeter:
    """
    Human pose -> robot qpos (robot auto-detected from checkpoint).

    Input : torch.Tensor [B, 20, F]  Dong pose (joints 1-20, no wrist)
                                     F=4 for quat, F=6 for R6 (auto-detected from ckpt)
    Output: np.ndarray   [B, J]      qpos clipped to the robot's joint limits
                                     (Shadow J=24, Allegro J=16). self.robot_name
                                     names the active robot.
    """

    def __init__(self, ckpt_path: str | Path):
        ck  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Robot id: multi-robot ckpt stores per-robot weights under "robots";
        # legacy single-robot ckpts only have top-level E_r/D_r (assume shadow).
        robots = ck.get("robots")
        self.robot_name = next(iter(robots)) if robots else "shadow"
        if self.robot_name not in _OUTPUT_SPECS:
            raise ValueError(f"Unsupported robot '{self.robot_name}' (have {list(_OUTPUT_SPECS)})")
        spec = _OUTPUT_SPECS[self.robot_name]
        self._n_pad = spec["n_pad"]
        self._lower = spec["lower"]
        self._upper = spec["upper"]

        d_r_sd = robots[self.robot_name]["D_r"] if robots else ck["D_r"]
        n_j = d_r_sd["fc.weight"].shape[0]
        dx_in_dim = ck["D_X"]["net.0.weight"].shape[1]

        cfg = ck.get("config", {})
        self.human_rot_repr = cfg.get("human_rot_repr", "quat") if isinstance(cfg, dict) else "quat"
        human_in_dim = 6 if self.human_rot_repr == "r6" else 4

        if "proj_precision.weight" in ck["E_h"]:
            z_dim    = ck["E_h"]["proj_thumb.weight"].shape[0]
            self.E_h = _LegacyHumanEncoder(in_dim=4, hidden_dim=32, z_dim=z_dim).eval()
        else:
            z_dim    = ck["E_h"]["proj_thumb.weight"].shape[0]
            self.E_h = HumanEncoder_E_h(in_dim=human_in_dim, hidden_dim=32, z_dim=z_dim).eval()

        self.D_X = _SharedDecoder_compat(in_dim=dx_in_dim, shared_dim=1024).eval()
        self.D_r = RobotDecoder_D_r(n_joints=n_j, shared_dim=1024).eval()

        self.E_h.load_state_dict(ck["E_h"])
        self.D_X.load_state_dict(ck["D_X"])
        self.D_r.load_state_dict(d_r_sd)

    @torch.no_grad()
    def __call__(self, pose: torch.Tensor) -> np.ndarray:
        q = self.D_r(self.D_X(self.E_h(pose))).numpy()
        if self._n_pad:
            q[:, :self._n_pad] = 0.0  # wrist DOFs — not encoded in wrist-frame human signal
        return np.clip(q, self._lower, self._upper)
