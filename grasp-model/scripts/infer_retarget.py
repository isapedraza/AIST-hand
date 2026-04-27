"""
Human-to-robot retargeting inference.

Pipeline: quats_h [B, 20, 4] -> E_h -> D_X -> D_r -> q_robot [B, 24]

Usage:
    python infer_retarget.py --ckpt checkpoints/stage1_cam_5000.pt

Prints retargeted joint angles for a random batch of human poses.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/cross_emb"))

from human_modules import HumanEncoder_E_h
from shared_modules import SharedDecoder_D_X
from robot_modules import RobotDecoder_D_r


def load_models(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    n_joints = ck["D_r"]["fc.weight"].shape[0]
    z_dim = ck["E_h"]["proj.weight"].shape[0]

    E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=z_dim).to(device)
    D_X = SharedDecoder_D_X(z_dim=z_dim, shared_dim=1024).to(device)
    D_r = RobotDecoder_D_r(n_joints=n_joints, shared_dim=1024).to(device)

    E_h.load_state_dict(ck["E_h"])
    D_X.load_state_dict(ck["D_X"])
    D_r.load_state_dict(ck["D_r"])

    for m in (E_h, D_X, D_r):
        m.eval()

    print(f"Loaded step={ck['step']}, z_dim={z_dim}, n_joints={n_joints}")
    return E_h, D_X, D_r


@torch.no_grad()
def retarget(quats_h: torch.Tensor, E_h, D_X, D_r) -> torch.Tensor:
    """
    Args:
        quats_h: [B, 20, 4] Dong quaternions (wxyz, w>=0)
    Returns:
        q_robot: [B, n_joints] scalar joint angles
    """
    z_h = E_h(quats_h)          # [B, z_dim]
    emb = D_X(z_h)              # [B, 1024]
    return D_r(emb)             # [B, n_joints]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/stage1_cam_5000.pt")
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path

    E_h, D_X, D_r = load_models(str(ckpt_path), args.device)

    quats_h = torch.zeros(args.B, 20, 4)
    quats_h[:, :, 0] = 1.0  # identity pose (all joints extended)

    q_robot = retarget(quats_h.to(args.device), E_h, D_X, D_r)
    print(f"quats_h shape : {quats_h.shape}")
    print(f"q_robot shape : {q_robot.shape}")
    print(f"q_robot[0]    : {q_robot[0].tolist()}")
