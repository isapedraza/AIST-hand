from pathlib import Path

import numpy as np
import torch

from ..human_modules import HumanEncoder_E_h
from ..shared_modules import SharedDecoder_D_X
from ..robot_modules  import RobotDecoder_D_r

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
        ck    = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        z_dim = ck["E_h"]["proj_thumb.weight"].shape[0]  # per-subspace dim
        n_j   = ck["D_r"]["fc.weight"].shape[0]

        self.E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=z_dim).eval()
        self.D_X = SharedDecoder_D_X(z_dim=z_dim, shared_dim=1024).eval()  # internally uses 3*z_dim
        self.D_r = RobotDecoder_D_r(n_joints=n_j, shared_dim=1024).eval()

        self.E_h.load_state_dict(ck["E_h"])
        self.D_X.load_state_dict(ck["D_X"])
        self.D_r.load_state_dict(ck["D_r"])

    @torch.no_grad()
    def __call__(self, quats: torch.Tensor) -> np.ndarray:
        q = self.D_r(self.D_X(self.E_h(quats))).numpy()
        return np.clip(q, _LOWER, _UPPER)
