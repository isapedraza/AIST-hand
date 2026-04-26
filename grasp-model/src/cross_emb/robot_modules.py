import torch.nn as nn


class RobotEncoder_E_r(nn.Module):
    """
    E_r: robot-specific projection from joint space to shared embedding.

    Single linear layer — no activation (Yan et al. 2026).
    One instance per robot; swap for a different robot.

    Args:
        n_joints   : actuable joints for this robot (infer from q_r.shape[1])
        shared_dim : shared embedding dim (must match SharedEncoder_E_X input)
    """

    def __init__(self, n_joints: int, shared_dim: int = 1024):
        super().__init__()
        self.fc = nn.Linear(n_joints, shared_dim)

    def forward(self, x):
        """x: [B, n_joints] -> [B, shared_dim]"""
        return self.fc(x)


class RobotDecoder_D_r(nn.Module):
    """
    D_r: robot-specific projection from shared embedding back to joint space.

    Single linear layer — no activation (Yan et al. 2026).
    One instance per robot; swap for a different robot.

    Args:
        n_joints   : actuable joints for this robot (infer from q_r.shape[1])
        shared_dim : shared embedding dim (must match SharedDecoder_D_X output)
    """

    def __init__(self, n_joints: int, shared_dim: int = 1024):
        super().__init__()
        self.fc = nn.Linear(shared_dim, n_joints)

    def forward(self, x):
        """x: [B, shared_dim] -> [B, n_joints]"""
        return self.fc(x)
