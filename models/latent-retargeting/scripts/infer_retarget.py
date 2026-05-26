"""Human-to-robot retargeting inference smoke script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_emb.inference.retarget import Retargeter


def identity_pose(B: int, in_dim: int) -> torch.Tensor:
    pose = torch.zeros(B, 20, in_dim)
    if in_dim == 4:
        pose[:, :, 0] = 1.0
    elif in_dim == 6:
        pose[:, :, 0] = 1.0
        pose[:, :, 4] = 1.0
    else:
        raise ValueError(f"Unsupported input dim: {in_dim}")
    return pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/stage1_latest.pt")
    parser.add_argument("--B", type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path

    retargeter = Retargeter(ckpt_path)
    pose_h = identity_pose(args.B, retargeter.human_in_dim)
    q_robot = retargeter(pose_h)

    print(f"rot_repr      : {retargeter.human_rot_repr}")
    print(f"pose_h shape  : {tuple(pose_h.shape)}")
    print(f"q_robot shape : {q_robot.shape}")
    print(f"q_robot[0]    : {q_robot[0].tolist()}")
