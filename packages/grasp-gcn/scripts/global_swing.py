"""
Global swing representation from MANO FK equations (Romero et al., 2017).

From the MANO forward kinematics:

    b_world[i] = R_global[parent[i]] @ b_rest[i]

where:
    b_rest[i]  = J_rest[i]  - J_rest[parent[i]]   (rest-pose bone vector)
    b_world[i] = j_world[i] - j_world[parent[i]]   (observed bone vector)

Solving for R_global[parent[i]]:

    R_global[parent[i]] = swing_rotation(b_rest[i], b_world[i])

The swing rotation is the unique zero-twist rotation that maps b_rest to b_world.
Zero-twist is justified because XYZ positions cannot encode twist around a bone axis.

For each of the 15 non-root MANO joints, this gives a 3D axis-angle vector.
Total: 45 values, convention-independent (same XYZ -> same representation).

Usage:
    gs = GlobalSwing('/path/to/MANO_RIGHT.pkl')
    feat = gs(xyz_21)  # (21, 3) OpenPose order -> (45,) axis-angle
"""

import io
import pickle
import numpy as np


# ── MANO pkl loader (no chumpy dependency) ────────────────────────────────────

class _Dummy:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __setstate__(self, state): pass

class _IgnoreChumpy(pickle.Unpickler):
    def find_class(self, module, name):
        if 'chumpy' in module or 'scipy.sparse' in module:
            return _Dummy
        return super().find_class(module, name)


# ── Math ──────────────────────────────────────────────────────────────────────

def _swing(a, b):
    """
    Zero-twist rotation R such that R @ a = b (up to scale).
    Directly from MANO FK: R_global[parent[i]] maps b_rest[i] to b_world[i].
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    axis = np.cross(a, b)
    sin_t = np.linalg.norm(axis)
    cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sin_t < 1e-8:
        if cos_t > 0:
            return np.eye(3)
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        ax = np.cross(a, perp)
        ax /= np.linalg.norm(ax)
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    axis /= sin_t
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)


def _rot_to_aa(R):
    """Rotation matrix -> axis-angle (3,)."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return angle * axis


# ── Joint mapping ─────────────────────────────────────────────────────────────

# OpenPose index -> MANO joint index (-1 = fingertip, no MANO joint)
_OP_TO_MANO = [
    0, 13, 14, 15, -1,   # WRIST, THUMB_CMC/MCP/IP/TIP
    1,  2,  3, -1,       # INDEX_MCP/PIP/DIP/TIP
    4,  5,  6, -1,       # MIDDLE_MCP/PIP/DIP/TIP
   10, 11, 12, -1,       # RING_MCP/PIP/DIP/TIP
    7,  8,  9, -1,       # PINKY_MCP/PIP/DIP/TIP
]

# MANO kintree parents (from kintree_table[0], root=-1)
# [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP,
#  MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP,
#  PINKY_MCP,  PINKY_PIP,  PINKY_DIP,
#  RING_MCP,   RING_PIP,   RING_DIP,
#  THUMB_CMC,  THUMB_MCP,  THUMB_IP]
_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]


# ── GlobalSwing ───────────────────────────────────────────────────────────────

class GlobalSwing:
    """
    Computes the global swing representation from XYZ joint positions.

    For each MANO bone i (joints 1..15), applies MANO FK equation:
        b_world[i] = R_global[parent[i]] @ b_rest[i]
    -> R_global[parent[i]] = swing_rotation(b_rest[i], b_world[i])
    -> axis_angle(R_global[parent[i]]) = output[i-1]

    Args:
        mano_pkl: path to MANO_RIGHT.pkl
    """

    def __init__(self, mano_pkl: str):
        with open(mano_pkl, 'rb') as f:
            raw = f.read()
        mano = _IgnoreChumpy(io.BytesIO(raw), encoding='latin1').load()

        J_raw = np.array(mano['J'], dtype=float)  # (16, 3)

        # Normalize: WRIST at origin, WRIST-INDEX_MCP = 0.1
        # MANO joint 1 = INDEX_MCP
        origin = J_raw[0].copy()
        scale  = np.linalg.norm(J_raw[1] - J_raw[0])
        self.J_rest = (J_raw - origin) * 0.1 / scale

    def __call__(self, xyz_21: np.ndarray) -> np.ndarray:
        """
        Args:
            xyz_21: (21, 3) joint positions in OpenPose order,
                    root-relative, WRIST-INDEX_MCP = 0.1
        Returns:
            feat: (45,) axis-angle of R_global[parent[i]] for i=1..15
        """
        # OpenPose -> MANO order
        j_world = np.zeros((16, 3))
        for op_i, mano_i in enumerate(_OP_TO_MANO):
            if mano_i >= 0:
                j_world[mano_i] = xyz_21[op_i]

        # For each bone i (1..15): swing(b_rest[i], b_world[i])
        feat = np.zeros(45)
        for i in range(1, 16):
            p = _PARENTS[i]
            b_rest  = self.J_rest[i] - self.J_rest[p]
            b_world = j_world[i]     - j_world[p]
            R = _swing(b_rest, b_world)
            feat[(i - 1) * 3: i * 3] = _rot_to_aa(R)

        return feat
