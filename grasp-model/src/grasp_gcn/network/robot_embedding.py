import torch
import torch.nn as nn


class RobotEmbedding(nn.Module):
    """
    E_r: projects robot joint space to shared embedding dim.
    D_r: projects shared embedding back to robot joint space.

    Both are single linear layers (Yan & Lee 2026).
    Robot-specific: one instance per robot.

    Args:
        n_joints   : number of actuable joints for this robot
        shared_dim : shared embedding dimensionality (must match D_X output)
    """

    def __init__(self, n_joints: int, shared_dim: int = 256):
        super().__init__()
        self.encoder = nn.Linear(n_joints, shared_dim)   # E_r
        self.decoder = nn.Linear(shared_dim, n_joints)   # D_r

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_joints] -> [B, shared_dim]"""
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, shared_dim] -> [B, n_joints]"""
        return self.decoder(x)


# Shadow Hand: 24 actuable joints (WRJ1, WRJ2 + 22 finger joints)
SHADOW_HAND_JOINTS = 24


def make_shadow_hand_embedding(shared_dim: int = 256) -> RobotEmbedding:
    return RobotEmbedding(n_joints=SHADOW_HAND_JOINTS, shared_dim=shared_dim)
