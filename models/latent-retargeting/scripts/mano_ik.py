"""
Analytical IK from MANO FK equations (Romero et al., 2017).

From the MANO forward kinematics:
    b_world[i] = R_global[parent[i]] @ b_rest[i]

where b_rest[i] = J_rest[i] - J_rest[parent[i]]
      b_world[i] = j_world[i] - j_world[parent[i]]

R_global[parent[i]] is the rotation that maps b_rest[i] -> b_world[i].
With twist=0 (anatomically justified: finger joints are hinge joints),
this rotation is unique (swing rotation).

Then local rotation:
    R_local[i] = R_global[parent[i]].T @ R_global[i]
    pose[(i-1)*3 : i*3] = axis_angle(R_local[i])

Usage:
    ik = ManoIK('/path/to/MANO_RIGHT.pkl')
    pose = ik(xyz_21)  # (21,3) OpenPose order -> (45,) axis-angle
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


# ── Rotation utilities ────────────────────────────────────────────────────────

def swing_rotation(a, b):
    """
    Rotation matrix R with zero twist such that R @ a = b (up to scale).
    Follows from twist=0 assumption: the only rotation applied is the
    minimum-angle rotation that aligns the bone direction.
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    axis = np.cross(a, b)
    sin_t = np.linalg.norm(axis)
    cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sin_t < 1e-8:
        if cos_t > 0:
            return np.eye(3)
        # 180 deg: pick arbitrary perpendicular axis
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        ax = np.cross(a, perp)
        ax /= np.linalg.norm(ax)
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    axis /= sin_t
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)


def rot_to_aa(R):
    """Rotation matrix -> axis-angle (3,)."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return angle * axis


def kabsch(P, Q):
    """
    R such that Q ≈ R @ P for each row pair, minimizing sum of squared distances.
    Used for WRIST: 5 children give 5 equations, solved jointly.
    P: (N, 3) rest bone vectors
    Q: (N, 3) world bone vectors
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


# ── ManoIK ────────────────────────────────────────────────────────────────────

# OpenPose joint index -> MANO joint index (-1 = tip, no MANO joint)
_OP_TO_MANO = [0, 13, 14, 15, -1,   # WRIST, THUMB_CMC/MCP/IP/TIP
               1,  2,  3, -1,         # INDEX_MCP/PIP/DIP/TIP
               4,  5,  6, -1,         # MIDDLE_MCP/PIP/DIP/TIP
               10, 11, 12, -1,        # RING_MCP/PIP/DIP/TIP
               7,  8,  9, -1]         # PINKY_MCP/PIP/DIP/TIP

# WRIST children in MANO order (MCP/CMC joints)
_WRIST_CHILDREN = [1, 4, 7, 10, 13]

# Non-leaf joints and their single child (MANO joint indices)
_MANO_CHILD = {1: 2, 2: 3,    # INDEX chain
               4: 5, 5: 6,    # MIDDLE chain
               7: 8, 8: 9,    # PINKY chain
               10: 11, 11: 12, # RING chain
               13: 14, 14: 15} # THUMB chain

# Leaf MANO joints -> OpenPose TIP index
_LEAF_TO_OP_TIP = {3: 8,   # INDEX_DIP  -> INDEX_TIP
                   6: 12,  # MIDDLE_DIP -> MIDDLE_TIP
                   9: 20,  # PINKY_DIP  -> PINKY_TIP
                   12: 16, # RING_DIP   -> RING_TIP
                   15: 4}  # THUMB_IP   -> THUMB_TIP


class ManoIK:
    """
    Analytical IK following MANO FK equations.

    Args:
        mano_pkl: path to MANO_RIGHT.pkl
    """

    def __init__(self, mano_pkl: str):
        with open(mano_pkl, 'rb') as f:
            raw = f.read()
        mano = _IgnoreChumpy(io.BytesIO(raw), encoding='latin1').load()

        J_rest_raw = np.array(mano['J'], dtype=float)       # (16, 3)
        v_template  = np.array(mano['v_template'], dtype=float)  # (N, 3)
        parents_raw = np.array(mano['kintree_table'][0]).astype(int)

        # Normalize: WRIST at origin, WRIST-INDEX_MCP = 0.1
        # MANO joint 1 = INDEX_MCP
        origin = J_rest_raw[0].copy()
        scale  = np.linalg.norm(J_rest_raw[1] - J_rest_raw[0])
        self.J_rest = (J_rest_raw - origin) * 0.1 / scale

        v_norm = (v_template - origin) * 0.1 / scale

        # TIP rest positions from template vertices (same indices as manolayer)
        # Order matches _LEAF_TO_OP_TIP: INDEX, MIDDLE, PINKY, RING, THUMB
        tip_verts = {3: 317,   # INDEX_TIP vertex
                     6: 444,   # MIDDLE_TIP vertex
                     9: 673,   # PINKY_TIP vertex
                     12: 556,  # RING_TIP vertex
                     15: 745}  # THUMB_TIP vertex
        self.tip_rest = {leaf: v_norm[vidx] for leaf, vidx in tip_verts.items()}

        self.parents = parents_raw.copy()
        self.parents[0] = -1

    def __call__(self, xyz_21: np.ndarray) -> np.ndarray:
        """
        Args:
            xyz_21: (21, 3) joint positions in OpenPose order,
                    root-relative, WRIST-INDEX_MCP = 0.1
        Returns:
            pose: (45,) axis-angle, same convention as MANO pose params
        """
        # Convert OpenPose -> MANO joint order
        j_world = np.zeros((16, 3))
        for op_i, mano_i in enumerate(_OP_TO_MANO):
            if mano_i >= 0:
                j_world[mano_i] = xyz_21[op_i]

        # ── Step 1: R_global[0] (WRIST) via Kabsch over 5 children ───────────
        # From MANO FK: b_world[c] = R_global[0] @ b_rest[c]  for c in children
        b_rest_wrist  = np.array([self.J_rest[c] - self.J_rest[0] for c in _WRIST_CHILDREN])
        b_world_wrist = np.array([j_world[c]     - j_world[0]     for c in _WRIST_CHILDREN])
        R_global = [np.eye(3)] * 16
        R_global[0] = kabsch(b_rest_wrist, b_world_wrist)

        # ── Step 2: R_global[i] for joints 1..15 ─────────────────────────────
        # R_global[i] is determined by the OUTGOING bone from i to its child:
        #   b_world[child] = R_global[i] @ b_rest[child]
        # This follows from MANO FK applied one level deeper.
        for i in range(1, 16):
            if i in _MANO_CHILD:
                child = _MANO_CHILD[i]
                b_rest_out  = self.J_rest[child] - self.J_rest[i]
                b_world_out = j_world[child]     - j_world[i]
            else:
                # Leaf joint: use TIP as virtual child
                b_rest_out  = self.tip_rest[i] - self.J_rest[i]
                b_world_out = xyz_21[_LEAF_TO_OP_TIP[i]] - j_world[i]
            R_global[i] = swing_rotation(b_rest_out, b_world_out)

        # ── Step 3: local rotations ───────────────────────────────────────────
        # From MANO FK: R_global[i] = R_global[parent[i]] @ R_local[i]
        # => R_local[i] = R_global[parent[i]].T @ R_global[i]
        pose = np.zeros(45)
        for i in range(1, 16):
            p = self.parents[i]
            R_local = R_global[p].T @ R_global[i]
            pose[(i - 1) * 3: i * 3] = rot_to_aa(R_local)

        return pose
