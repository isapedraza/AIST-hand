import torch
import torch.nn as nn

# Shadow Hand joint indices in q_r (24 DOF, from robot.yaml)
FINGER_JOINT_INDICES = {
    "thumb":  [19, 20, 21, 22, 23],  # THJ5-1
    "index":  [2,  3,  4,  5],       # FFJ4-1
    "middle": [6,  7,  8,  9],       # MFJ4-1
    "ring":   [10, 11, 12, 13],      # RFJ4-1
    "pinky":  [14, 15, 16, 17, 18],  # LFJ5-1
}


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


class RobotDecoder_D_r_residual(nn.Module):
    """Run 36 residual decoder for hybrid latent (Idea I-v2).

    D_base(z_global) -> base pose (all 24 joints).
    D_k(z_finger_k) -> per-finger residual (finger's joints only).
    q_r = base_pose + sum(residuals).

    z layout must be [z_global | z_thumb | z_index | z_middle | z_ring | z_pinky].
    No structural path exists for the optimizer to ignore z_global.
    """

    def __init__(self, z_dim_global: int = 64, z_dim_finger: int = 64, n_joints: int = 24):
        super().__init__()
        self.z_dim_global = z_dim_global
        self.z_dim_finger = z_dim_finger
        self.D_base   = nn.Linear(z_dim_global, n_joints)
        self.D_thumb  = nn.Linear(z_dim_finger, len(FINGER_JOINT_INDICES["thumb"]))
        self.D_index  = nn.Linear(z_dim_finger, len(FINGER_JOINT_INDICES["index"]))
        self.D_middle = nn.Linear(z_dim_finger, len(FINGER_JOINT_INDICES["middle"]))
        self.D_ring   = nn.Linear(z_dim_finger, len(FINGER_JOINT_INDICES["ring"]))
        self.D_pinky  = nn.Linear(z_dim_finger, len(FINGER_JOINT_INDICES["pinky"]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, z_dim_global + 5*z_dim_finger] -> q_r: [B, n_joints]"""
        G, F = self.z_dim_global, self.z_dim_finger
        z_global = z[:, :G]
        z_thumb  = z[:, G:G+F]
        z_index  = z[:, G+F:G+2*F]
        z_middle = z[:, G+2*F:G+3*F]
        z_ring   = z[:, G+3*F:G+4*F]
        z_pinky  = z[:, G+4*F:G+5*F]
        q = self.D_base(z_global)
        delta = torch.zeros_like(q)
        delta[:, FINGER_JOINT_INDICES["thumb"]]  = self.D_thumb(z_thumb)
        delta[:, FINGER_JOINT_INDICES["index"]]  = self.D_index(z_index)
        delta[:, FINGER_JOINT_INDICES["middle"]] = self.D_middle(z_middle)
        delta[:, FINGER_JOINT_INDICES["ring"]]   = self.D_ring(z_ring)
        delta[:, FINGER_JOINT_INDICES["pinky"]]  = self.D_pinky(z_pinky)
        return q + delta
