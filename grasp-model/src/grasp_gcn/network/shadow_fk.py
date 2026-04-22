import torch
import pytorch_kinematics as pk
from pathlib import Path

# Joint order matches robot_embedding.py (24 joints)
JOINT_NAMES = [
    "WRJ2", "WRJ1",
    "FFJ4", "FFJ3", "FFJ2", "FFJ1",
    "MFJ4", "MFJ3", "MFJ2", "MFJ1",
    "RFJ4", "RFJ3", "RFJ2", "RFJ1",
    "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
]

# Joint limits extracted from shadow_hand_right.urdf [lower, upper] in radians
JOINT_LIMITS = torch.tensor([
    [-0.5236,  0.1745],   # WRJ2
    [-0.6981,  0.4887],   # WRJ1
    [-0.3491,  0.3491],   # FFJ4
    [-0.2618,  1.5708],   # FFJ3
    [ 0.0000,  1.5708],   # FFJ2
    [ 0.0000,  1.5708],   # FFJ1
    [-0.3491,  0.3491],   # MFJ4
    [-0.2618,  1.5708],   # MFJ3
    [ 0.0000,  1.5708],   # MFJ2
    [ 0.0000,  1.5708],   # MFJ1
    [-0.3491,  0.3491],   # RFJ4
    [-0.2618,  1.5708],   # RFJ3
    [ 0.0000,  1.5708],   # RFJ2
    [ 0.0000,  1.5708],   # RFJ1
    [ 0.0000,  0.7854],   # LFJ5
    [-0.3491,  0.3491],   # LFJ4
    [-0.2618,  1.5708],   # LFJ3
    [ 0.0000,  1.5708],   # LFJ2
    [ 0.0000,  1.5708],   # LFJ1
    [-1.0472,  1.0472],   # THJ5
    [ 0.0000,  1.2217],   # THJ4
    [-0.2094,  0.2094],   # THJ3
    [-0.6981,  0.6981],   # THJ2
    [-0.2618,  1.5708],   # THJ1
])  # [24, 2]

FINGERTIP_LINKS = ["fftip", "mftip", "rftip", "lftip", "thtip"]
WRIST_LINK = "palm"

# Child link for each actuable joint (for extracting joint rotation from FK)
JOINT_CHILD_LINKS = [
    "wrist",        # WRJ2
    "palm",         # WRJ1
    "ffknuckle",    # FFJ4
    "ffproximal",   # FFJ3
    "ffmiddle",     # FFJ2
    "ffdistal",     # FFJ1
    "mfknuckle",    # MFJ4
    "mfproximal",   # MFJ3
    "mfmiddle",     # MFJ2
    "mfdistal",     # MFJ1
    "rfknuckle",    # RFJ4
    "rfproximal",   # RFJ3
    "rfmiddle",     # RFJ2
    "rfdistal",     # RFJ1
    "lfmetacarpal", # LFJ5
    "lfknuckle",    # LFJ4
    "lfproximal",   # LFJ3
    "lfmiddle",     # LFJ2
    "lfdistal",     # LFJ1
    "thbase",       # THJ5
    "thproximal",   # THJ4
    "thhub",        # THJ3
    "thmiddle",     # THJ2
    "thdistal",     # THJ1
]

_DEFAULT_URDF = Path(__file__).parents[5] / "dex-urdf" / "robots" / "hands" / "shadow_hand" / "shadow_hand_right.urdf"


class ShadowFK:
    """
    Forward kinematics for Shadow Hand right.

    Args:
        urdf_path : path to shadow_hand_right.urdf (default: dex-urdf in repo)
        device    : torch device

    Usage:
        fk = ShadowFK(device=device)
        qpos = fk.sample(B)                     # [B, 24] uniform in limits
        tips, quats = fk.forward(qpos)           # [B,5,3], [B,24,4]
        tips_norm = fk.normalize_tips(tips, qpos) # wrist-relative, /hand_length
    """

    def __init__(self, urdf_path: str = None, device: torch.device = None):
        self.device = device or torch.device("cpu")
        path = Path(urdf_path) if urdf_path else _DEFAULT_URDF
        with open(path, "rb") as f:
            urdf_bytes = f.read()
        self._chain = pk.build_chain_from_urdf(urdf_bytes)
        self._limits = JOINT_LIMITS.to(self.device)  # [24, 2]

    def sample(self, B: int) -> torch.Tensor:
        """Sample B random qpos uniformly within joint limits. Returns [B, 24]."""
        lo = self._limits[:, 0]
        hi = self._limits[:, 1]
        u = torch.rand(B, 24, device=self.device)
        return lo + u * (hi - lo)

    def forward(self, qpos: torch.Tensor):
        """
        Run FK on qpos [B, 24].

        Returns:
            tips  : fingertip positions [B, 5, 3] in palm frame (meters)
            quats : joint rotation quaternions [B, 24, 4] wxyz
        """
        B = qpos.shape[0]
        th = {name: qpos[:, i] for i, name in enumerate(JOINT_NAMES)}
        ret = self._chain.forward_kinematics(th)

        # Fingertip positions in world frame [B, 5, 3]
        tips_world = torch.stack(
            [ret[link].get_matrix()[:, :3, 3] for link in FINGERTIP_LINKS], dim=1
        )  # [B, 5, 3]

        # Wrist (palm) position in world frame [B, 3]
        wrist_pos = ret[WRIST_LINK].get_matrix()[:, :3, 3]  # [B, 3]

        # Fingertips relative to wrist
        tips = tips_world - wrist_pos.unsqueeze(1)  # [B, 5, 3]

        # Palm rotation for wrist-relative joint quats (matches Dong frame convention)
        R_palm = ret[WRIST_LINK].get_matrix()[:, :3, :3]  # [B, 3, 3]
        R_palm_inv = R_palm.transpose(-1, -2)              # [B, 3, 3]

        # Joint rotations relative to palm → comparable to Dong wrist-local quats
        quats = torch.stack(
            [pk.matrix_to_quaternion(
                torch.bmm(R_palm_inv, ret[link].get_matrix()[:, :3, :3])
             ) for link in JOINT_CHILD_LINKS], dim=1
        )  # [B, 24, 4]  wxyz, palm-relative

        return tips.to(self.device), quats.to(self.device)

    @property
    def hand_length(self) -> float:
        """Distance from palm to mftip at zero pose (meters). Cached after first call."""
        if not hasattr(self, "_hand_length"):
            zero = torch.zeros(1, 24, device=self.device)
            tips_zero, _ = self.forward(zero)
            self._hand_length = tips_zero[0, 1].norm().item()
        return self._hand_length

    def normalize_tips(self, tips: torch.Tensor) -> torch.Tensor:
        """
        Normalize fingertip positions by hand_length (mftip at zero pose).

        tips : [B, 5, 3] wrist-relative
        Returns [B, 5, 3] dimensionless
        """
        return tips / self.hand_length
