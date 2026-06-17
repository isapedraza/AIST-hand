from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.human_modules import HumanEncoder_E_h, TemporalHumanEncoder_E_h, CAMLayer
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

# Leap Hand joint limits (16 DOF) — leap_hand/right_hand.xml jnt_range.
# Order: if[mcp,rot,pip,dip], mf[...], rf[...], th[cmc,axl,mcp,ipl]. No wrist (n_pad=0).
_LEAP_LOWER = np.array([
    -0.314, -1.047, -0.506, -0.366,
    -0.314, -1.047, -0.506, -0.366,
    -0.314, -1.047, -0.506, -0.366,
    -0.349, -0.349, -0.470, -1.340,
], dtype=np.float32)

_LEAP_UPPER = np.array([
     2.230,  1.047,  1.885,  2.042,
     2.230,  1.047,  1.885,  2.042,
     2.230,  1.047,  1.885,  2.042,
     2.094,  2.094,  2.443,  1.880,
], dtype=np.float32)

# Barrett Hand joint limits (8 DOF) — barrett.mjcf jnt_range.
# Order: f1[prox,med,dist], f2[prox,med,dist], f3[med,dist]. No wrist (n_pad=0).
_BARRETT_LOWER = np.array([
    -3.140, -2.440, -0.785,
     0.000, -2.440, -0.785,
    -2.440, -0.785,
], dtype=np.float32)

_BARRETT_UPPER = np.array([
     0.000,  0.000,  0.000,
     3.140,  0.000,  0.000,
     0.000,  0.000,
], dtype=np.float32)

# Per-robot output spec: n_pad = leading wrist DOFs zeroed (not in wrist-frame
# human signal), lower/upper = qpos clip limits matching the MuJoCo model.
_OUTPUT_SPECS = {
    "shadow":  {"n_pad": 2, "lower": _SHADOW_LOWER,  "upper": _SHADOW_UPPER},
    "allegro": {"n_pad": 0, "lower": _ALLEGRO_LOWER, "upper": _ALLEGRO_UPPER},
    "leap":    {"n_pad": 0, "lower": _LEAP_LOWER,    "upper": _LEAP_UPPER},
    "barrett": {"n_pad": 0, "lower": _BARRETT_LOWER, "upper": _BARRETT_UPPER},
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

    def __init__(self, ckpt_path: str | Path, robot_name: str | None = None):
        ck  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Robot id: multi-robot ckpt stores per-robot weights under "robots";
        # legacy single-robot ckpts only have top-level E_r/D_r (assume shadow).
        # robot_name selects which robot of a multi-robot ckpt to decode; if
        # omitted, the first robot is used.
        robots = ck.get("robots")
        if robot_name is not None:
            if robots is None or robot_name not in robots:
                raise ValueError(f"robot '{robot_name}' not in checkpoint (have {list(robots) if robots else 'legacy single'})")
            self.robot_name = robot_name
        else:
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
        self.human_encoder = cfg.get("human_encoder", "spatial") if isinstance(cfg, dict) else "spatial"
        self.temporal_window = int(cfg.get("temporal_window", 8)) if isinstance(cfg, dict) else 8
        human_in_dim = 6 if self.human_rot_repr == "r6" else 4
        layer1_node_dim = ck["E_h"].get("layer1.linear.weight", torch.empty(0, 0)).shape[1] // 2
        if "proj_precision.weight" not in ck["E_h"] and layer1_node_dim not in (human_in_dim, 0):
            self.human_encoder = "temporal_cam"
            self.temporal_window = layer1_node_dim // human_in_dim
        self._temporal_buffer: list[torch.Tensor] = []
        self._last_pose: torch.Tensor | None = None
        self._last_qpos: np.ndarray | None = None

        if "proj_precision.weight" in ck["E_h"]:
            z_dim    = ck["E_h"]["proj_thumb.weight"].shape[0]
            self.human_encoder = "spatial"
            self.E_h = _LegacyHumanEncoder(in_dim=4, hidden_dim=32, z_dim=z_dim).eval()
        elif self.human_encoder == "temporal_cam":
            z_dim    = ck["E_h"]["proj_thumb.weight"].shape[0]
            self.E_h = TemporalHumanEncoder_E_h(
                in_dim=human_in_dim, hidden_dim=32, z_dim=z_dim,
                temporal_window=self.temporal_window,
            ).eval()
        else:
            z_dim    = ck["E_h"]["proj_thumb.weight"].shape[0]
            self.E_h = HumanEncoder_E_h(in_dim=human_in_dim, hidden_dim=32, z_dim=z_dim).eval()

        self.D_X = _SharedDecoder_compat(in_dim=dx_in_dim, shared_dim=1024).eval()
        self.D_r = RobotDecoder_D_r(n_joints=n_j, shared_dim=1024).eval()

        self.E_h.load_state_dict(ck["E_h"])
        self.D_X.load_state_dict(ck["D_X"])
        self.D_r.load_state_dict(d_r_sd)

    def _temporal_input(self, pose: torch.Tensor) -> torch.Tensor | None:
        if pose.ndim == 4:
            return pose
        if pose.ndim != 3:
            raise ValueError(f"pose must have shape [B,20,F] or [B,T,20,F], got {tuple(pose.shape)}")
        if pose.shape[0] != 1:
            return pose.unsqueeze(1).expand(-1, self.temporal_window, -1, -1)

        pose_cpu = pose.detach().cpu()
        if self._last_pose is not None and torch.equal(pose_cpu, self._last_pose):
            return None

        sample = pose.detach().clone()
        if not self._temporal_buffer:
            self._temporal_buffer = [sample.clone() for _ in range(self.temporal_window)]
        else:
            self._temporal_buffer.append(sample)
            self._temporal_buffer = self._temporal_buffer[-self.temporal_window:]
        self._last_pose = pose_cpu.clone()
        return torch.cat(self._temporal_buffer, dim=0).unsqueeze(0)

    @torch.no_grad()
    def __call__(self, pose: torch.Tensor) -> np.ndarray:
        if self.human_encoder == "temporal_cam":
            pose_in = self._temporal_input(pose)
            if pose_in is None and self._last_qpos is not None:
                return self._last_qpos.copy()
            if pose_in is None:
                pose_in = pose.unsqueeze(1).expand(-1, self.temporal_window, -1, -1)
        else:
            pose_in = pose

        q = self.D_r(self.D_X(self.E_h(pose_in))).numpy()
        if self._n_pad:
            q[:, :self._n_pad] = 0.0  # wrist DOFs — not encoded in wrist-frame human signal
        q = np.clip(q, self._lower, self._upper)
        self._last_qpos = q.copy()
        return q
