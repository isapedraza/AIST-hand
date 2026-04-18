import logging
import time
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from typing import Optional

log = logging.getLogger(__name__)


# ── Global swing helpers ──────────────────────────────────────────────────────
# Derived from MANO FK (Romero et al., 2017):
#   b_world[i] = R_global[parent[i]] @ b_rest[i]
# => R_global[parent[i]] = swing_rotation(b_rest[i], b_world[i])
# => global_swing[i] = rot_to_aa(R_global[parent[i]])

def _swing(a, b):
    """Zero-twist rotation R such that R @ a = b (up to scale)."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    axis = np.cross(a, b)
    sin_t = np.linalg.norm(axis)
    cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sin_t < 1e-8:
        if cos_t > 0:
            return np.eye(3)
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        ax = np.cross(a, perp); ax /= np.linalg.norm(ax)
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    axis /= sin_t
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)

def _rot_to_aa(R):
    """Rotation matrix -> axis-angle (3,)."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2.0 * np.sin(angle))
    return angle * axis

# OpenPose index -> MANO joint index (-1 = tip/no MANO joint)
_OP_TO_MANO = [0,13,14,15,-1, 1,2,3,-1, 4,5,6,-1, 10,11,12,-1, 7,8,9,-1]
# MANO kintree parents (root = -1)
_MANO_PARENTS = [-1, 0,1,2, 0,4,5, 0,7,8, 0,10,11, 0,13,14]

# MANO rest-pose joint positions, wrist at origin (from MANO_RIGHT.pkl, betas=0).
# Only directions matter -- _swing normalizes bone vectors internally.
# Source: J_raw = mano['J']; J_REST = J_raw - J_raw[0]
_J_REST = np.array([
    [ 0.        ,  0.        ,  0.        ],
    [-0.08809725, -0.00520036,  0.02068599],
    [-0.12077615, -0.001191  ,  0.02290306],
    [-0.14293206, -0.00248942,  0.02278894],
    [-0.09466044, -0.00147896, -0.00335754],
    [-0.12584311,  0.00038237, -0.00895205],
    [-0.14874775, -0.00086974, -0.01289656],
    [-0.06878697, -0.00994033, -0.04320934],
    [-0.08580138, -0.0098785 , -0.05570812],
    [-0.10166828, -0.01056966, -0.06604002],
    [-0.08173555, -0.00395742, -0.02667319],
    [-0.11004983, -0.00189041, -0.03177173],
    [-0.13357034, -0.00357853, -0.03940555],
    [-0.02408971, -0.01552233,  0.02581285],
    [-0.04372295, -0.01463105,  0.0495124 ],
    [-0.06594069, -0.02006402,  0.06403652],
], dtype=np.float64)

# For each of the 21 MediaPipe nodes, the start index into the 45-element MANO
# pose param vector (axis-angle, 15 non-root finger joints x 3).
# Derived from mano_to_openpose = [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20].
# pose_start = (mano_joint - 1) * 3  for mano_joint in 1..15.
# -1 = no MANO joint (root WRIST or fingertip) -> features are [0, 0, 0].
_MANO_POSE_SLICE = [
    -1,          # 0  WRIST        (root)
    36, 39, 42,  # 1-3  THUMB_CMC, MCP, IP   (MANO joints 13,14,15)
    -1,          # 4  THUMB_TIP    (tip)
     0,  3,  6,  # 5-7  INDEX_MCP, PIP, DIP  (MANO joints 1,2,3)
    -1,          # 8  INDEX_TIP    (tip)
     9, 12, 15,  # 9-11 MIDDLE_MCP, PIP, DIP (MANO joints 4,5,6)
    -1,          # 12 MIDDLE_TIP   (tip)
    27, 30, 33,  # 13-15 RING_MCP, PIP, DIP  (MANO joints 10,11,12)
    -1,          # 16 RING_TIP     (tip)
    18, 21, 24,  # 17-19 PINKY_MCP, PIP, DIP (MANO joints 7,8,9)
    -1,          # 20 PINKY_TIP    (tip)
]

# ── AHG features (Aiman & Ahmad, Wiley 2024) ─────────────────────────────────
# 10 critical joints: 5 fingertips + 5 finger bases (MediaPipe indexing).
# For each node j, 10 angles and/or 10 distances are computed relative to these.
_CRITICAL_JOINTS = [
    4, 8, 12, 16, 20,   # fingertips:   THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP
    1, 5,  9, 13, 17,   # finger bases: THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP
]

class ToGraph:
    """
    Convierte un sample de MediaPipe a PyG Data:
      - x: [21, F] en orden fijo (WRIST, THUMB_*, INDEX_*, MIDDLE_*, RING_*, PINKY_*)
      - edge_index: esqueleto de mano
      - y: grasp_type (LongTensor [1])
      - mask: [21, 1] -> 1.0 si el joint es válido, 0.0 si faltante
      - joint_id: [21] = 0..20
      - handedness (opcional): 0=Right, 1=Left, -1=Unknown

    Params:
      features: 'xy' o 'xyz'
      make_undirected: si True, convierte edge_index a no dirigido
      use_confidence: si True, usa '<JOINT>_conf' del sample para enmascarar
      conf_threshold: umbral de confianza para marcar nodos válidos
      add_velocity: si True, añade v_i = pos(t) - pos(t-1) como 3 features por nodo.
        En training: se espera key 'velocity' en el sample (computada por GraspsClass).
        En deploy: se mantiene _prev_positions entre llamadas; llamar reset_velocity()
        cuando se pierda tracking.
    """

    def __init__(self,
                 features: str = 'xyz',
                 make_undirected: bool = True,
                 use_confidence: bool = False,
                 conf_threshold: float = 0.5,
                 add_joint_angles: bool = False,
                 add_bone_vectors: bool = False,
                 add_velocity: bool = False,
                 add_mano_pose: bool = False,
                 add_global_swing: bool = False,
                 add_ahg_angles: bool = False,
                 add_ahg_distances: bool = False,
                 add_dong_quats: bool = False):
        assert features in ('xy', 'xyz', 'none')
        self.features = features
        self.make_undirected = make_undirected
        self.use_confidence = use_confidence
        self.conf_threshold = conf_threshold
        self.add_joint_angles = add_joint_angles
        self.add_bone_vectors = add_bone_vectors
        self.add_velocity = add_velocity
        self.add_mano_pose = add_mano_pose
        self.add_global_swing = add_global_swing
        self.add_ahg_angles = add_ahg_angles
        self.add_ahg_distances = add_ahg_distances
        self.add_dong_quats = add_dong_quats
        self._prev_positions: Optional[np.ndarray] = None  # [21, 3], deploy-time buffer
        self._prev_timestamp: Optional[float] = None       # wall-clock time of last frame
        self.F_xyz = 0 if features == 'none' else (2 if features == 'xy' else 3)
        self.F = (self.F_xyz
                  + (1 if add_joint_angles else 0)
                  + (3 if add_bone_vectors else 0)
                  + (3 if add_velocity else 0)
                  + (3 if add_mano_pose else 0)
                  + (3 if add_global_swing else 0)
                  + (10 if add_ahg_angles else 0)
                  + (10 if add_ahg_distances else 0)
                  + (4 if add_dong_quats else 0))  # total features per node

        # Parent joint index for each of the 21 joints (-1 = no parent).
        # Order: WRIST(0), THUMB_CMC(1)..THUMB_TIP(4),
        #        INDEX_MCP(5)..INDEX_TIP(8), MIDDLE_MCP(9)..MIDDLE_TIP(12),
        #        RING_MCP(13)..RING_TIP(16), PINKY_MCP(17)..PINKY_TIP(20)
        self._parent_of = [
            -1,           # 0  WRIST
             0, 1, 2, 3,  # 1-4  THUMB_CMC..TIP
             0, 5, 6, 7,  # 5-8  INDEX_MCP..TIP
             0, 9,10,11,  # 9-12 MIDDLE_MCP..TIP
             0,13,14,15,  # 13-16 RING_MCP..TIP
             0,17,18,19,  # 17-20 PINKY_MCP..TIP
        ]
        # First child index for each joint (-1 = tip or WRIST with multiple children).
        self._child_of = [
            -1,             # 0  WRIST
             2, 3, 4, -1,   # 1-4  THUMB
             6, 7, 8, -1,   # 5-8  INDEX
            10,11,12, -1,   # 9-12 MIDDLE
            14,15,16, -1,   # 13-16 RING
            18,19,20, -1,   # 17-20 PINKY
        ]

        # Orden fijo de 21 joints
        self.joints = [
            'WRIST',
            'THUMB_CMC','THUMB_MCP','THUMB_IP','THUMB_TIP',
            'INDEX_FINGER_MCP','INDEX_FINGER_PIP','INDEX_FINGER_DIP','INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP','MIDDLE_FINGER_PIP','MIDDLE_FINGER_DIP','MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP','RING_FINGER_PIP','RING_FINGER_DIP','RING_FINGER_TIP',
            'PINKY_MCP','PINKY_PIP','PINKY_DIP','PINKY_TIP'
        ]

        # Esqueleto (tus conexiones manuales)
        self.m_edge_origins = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9,
                               5, 9, 10, 10, 11, 11, 12, 13, 9, 13, 14, 14, 15, 15,
                               16, 17, 13, 17, 18, 17, 18, 19, 19, 20]
        self.m_edge_ends   = [1, 5, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 5, 7, 6, 8, 7, 5,
                               9, 10, 9, 11, 10, 12, 11, 9, 13, 14, 13, 15, 14, 16,
                               15, 13, 17, 18, 17, 0, 19, 18, 20, 19]

        # Campos de confianza opcionales
        self.joint_conf_name = {j: f"{j}_conf" for j in self.joints}

    # -------------------- NUEVO: parser robusto de handedness --------------------
    def _to_int_handedness(self, value) -> int:
        """Mapea 'Right'/'Left'/'Unknown' (o variantes) a 0/1/-1. Tolera enteros/strings."""
        if value is None:
            return -1
        if isinstance(value, (int, np.integer)):
            v = int(value)
            if v in (0, 1, -1):
                return v
        s = str(value).strip().lower()
        if s in ("right", "r", "0"):
            return 0
        if s in ("left", "l", "1"):
            return 1
        if s in ("unknown", "u", "-1"):
            return -1
        # Intento de conversión numérica genérica
        try:
            v = int(s)
            return v if v in (0, 1, -1) else -1
        except Exception:
            return -1
    # -----------------------------------------------------------------------------

    def __call__(self, sample: dict) -> Data:
        rows = []
        mask_vals = []

        if self.F_xyz > 0:
            for j in self.joints:
                raw = sample.get(j, None)
                if raw is None:
                    v = np.zeros(self.F_xyz, dtype=float)
                    valid = False
                else:
                    arr = np.array(raw, dtype=float)
                    if arr.shape[0] < self.F_xyz:
                        arr = np.pad(arr, (0, self.F_xyz - arr.shape[0]), mode='constant', constant_values=0.0)
                    elif arr.shape[0] > self.F_xyz:
                        arr = arr[:self.F_xyz]
                    valid = np.isfinite(arr).all()
                    v = np.nan_to_num(arr, nan=0.0)

                if self.use_confidence:
                    conf = float(sample.get(self.joint_conf_name[j], 1.0))
                    valid = valid and (conf >= self.conf_threshold)

                rows.append(v)
                mask_vals.append(1.0 if valid else 0.0)
        else:
            # features='none': no xyz base — dong_quats fills all features
            rows = [np.zeros(0, dtype=float) for _ in self.joints]
            mask_vals = [1.0] * 21

        # Joint flexion angles (optional)
        if self.add_joint_angles:
            positions = np.vstack(rows)  # [21, 3] — always xyz regardless of F
            angles = []
            for i in range(21):
                p = self._parent_of[i]
                c = self._child_of[i]
                if p == -1 or c == -1:
                    angles.append(0.0)
                else:
                    v_in  = positions[i] - positions[p]
                    v_out = positions[c] - positions[i]
                    n_in  = np.linalg.norm(v_in)
                    n_out = np.linalg.norm(v_out)
                    if n_in < 1e-8 or n_out < 1e-8:
                        angles.append(0.0)
                    else:
                        cos_a = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
                        angles.append(float(np.arccos(cos_a)))
            rows = [np.append(r, a) for r, a in zip(rows, angles)]

        # Bone vectors: position_i - position_parent(i), raw (not normalized).
        # WRIST has no parent -> [0, 0, 0].
        # Scale is already controlled by geometric normalization (dist WRIST->INDEX_MCP=0.1),
        # so bone length is meaningful and should not be discarded by unit-normalization.
        if self.add_bone_vectors:
            positions = np.vstack(rows)[:, :3]  # [21, 3] — always xyz
            for i in range(21):
                p = self._parent_of[i]
                bone = np.zeros(3) if p == -1 else positions[i] - positions[p]
                rows[i] = np.append(rows[i], bone)

        # Velocity: v_i = pos(t) - pos(t-1), 3 features per node.
        # Source priority:
        #   1. 'velocity' key in sample (set by GraspsClass during training)
        #   2. _prev_positions buffer (deploy-time, maintained between calls)
        #   3. zeros (first frame of a sequence or buffer not initialized)
        if self.add_velocity:
            positions = np.vstack(rows)[:, :3]  # [21, 3] — always xyz
            if 'velocity' in sample:
                vel = np.array(sample['velocity'], dtype=np.float32)  # [21, 3]
            elif self._prev_positions is not None:
                now = time.time()
                dt = now - self._prev_timestamp if self._prev_timestamp is not None else 0.1
                dt = max(dt, 1e-3)  # guard against zero division
                vel = (positions - self._prev_positions) / dt
                self._prev_timestamp = now
            else:
                vel = np.zeros((21, 3), dtype=np.float32)
                self._prev_timestamp = time.time()
            self._prev_positions = positions.copy()
            for i in range(21):
                rows[i] = np.append(rows[i], vel[i])

        # MANO pose params: 3 axis-angle values per node (zeros for WRIST and tips).
        # Source: sample['mano_pose'] -- (45,) array from hograspnet_mano.csv.
        # Mapping: _MANO_POSE_SLICE[i] gives the start index into the 45-element vector.
        if self.add_mano_pose:
            pose = np.array(sample.get('mano_pose', np.zeros(45)), dtype=np.float32)
            for i in range(21):
                start = _MANO_POSE_SLICE[i]
                params = pose[start:start + 3] if start >= 0 else np.zeros(3, dtype=np.float32)
                rows[i] = np.append(rows[i], params)

        # Global swing: R_global[parent[i]] = swing(b_rest[i], b_world[i])  for i=1..15
        # From MANO FK (Romero et al., 2017): b_world[i] = R_global[parent[i]] @ b_rest[i]
        # Computed on-the-fly from XYZ. Same node mapping as mano_pose (_MANO_POSE_SLICE).
        if self.add_global_swing:
            positions = np.vstack(rows)[:, :3]  # [21, 3]
            j_world = np.zeros((16, 3))
            for op_i, mano_i in enumerate(_OP_TO_MANO):
                if mano_i >= 0:
                    j_world[mano_i] = positions[op_i]
            swing_feat = np.zeros(45, dtype=np.float32)
            for i in range(1, 16):
                p = _MANO_PARENTS[i]
                b_rest  = _J_REST[i] - _J_REST[p]
                b_world = j_world[i]       - j_world[p]
                swing_feat[(i - 1) * 3: i * 3] = _rot_to_aa(_swing(b_rest, b_world))
            for i in range(21):
                start = _MANO_POSE_SLICE[i]
                params = swing_feat[start:start + 3] if start >= 0 else np.zeros(3, dtype=np.float32)
                rows[i] = np.append(rows[i], params)

        # AHG angles: for each node j, 10 angles at wrist between j and each critical joint.
        # theta = arccos(dot(wrist->j, wrist->c) / (|wrist->j| * |wrist->c|))
        # 3D adaptation of Aiman & Ahmad (Wiley CAV 2024), Eq. 1.
        # Wrist node (j=0): v_i=[0,0,0] -> angle=0.0 for all critical joints.
        if self.add_ahg_angles:
            positions = np.vstack(rows)[:, :3]  # [21, 3]
            wrist = positions[0]
            for i in range(21):
                v_i = positions[i] - wrist
                norm_i = np.linalg.norm(v_i)
                angles = []
                for c in _CRITICAL_JOINTS:
                    v_c = positions[c] - wrist
                    norm_c = np.linalg.norm(v_c)
                    if norm_i < 1e-8 or norm_c < 1e-8:
                        angles.append(0.0)
                    else:
                        cos_a = np.clip(np.dot(v_i, v_c) / (norm_i * norm_c), -1.0, 1.0)
                        angles.append(float(np.arccos(cos_a)))
                rows[i] = np.append(rows[i], angles)

        # AHG distances: for each node j, 10 Euclidean distances to each critical joint.
        # d = ||pos_j - pos_c||   (3D Euclidean)
        # Aiman & Ahmad (Wiley CAV 2024), Eq. 2 extended to 3D.
        if self.add_ahg_distances:
            positions = np.vstack(rows)[:, :3]  # [21, 3]
            for i in range(21):
                dists = [float(np.linalg.norm(positions[i] - positions[c]))
                         for c in _CRITICAL_JOINTS]
                rows[i] = np.append(rows[i], dists)

        # Dong quaternions: [w, x, y, z] per node from wrist-local frame.
        # WRIST (joint 0) is the reference frame -> identity quaternion [1, 0, 0, 0].
        # Joints 1-20 from sample['dong_quats'] [21, 4] precomputed by DongKinematics.
        if self.add_dong_quats:
            dong_quats = sample.get('dong_quats', None)
            if dong_quats is None:
                dong_quats = np.zeros((21, 4), dtype=np.float32)
                dong_quats[:, 0] = 1.0  # identity for all if missing
            for i in range(21):
                rows[i] = np.append(rows[i], dong_quats[i])

        # Nodo features y máscara
        x = torch.tensor(np.vstack(rows), dtype=torch.float32)               # [21, F]
        mask = torch.tensor(mask_vals, dtype=torch.float32).unsqueeze(1)    # [21, 1]

        # Estructura de grafo
        edge_index = torch.tensor([self.m_edge_origins, self.m_edge_ends], dtype=torch.long)
        if self.make_undirected:
            edge_index = to_undirected(edge_index)

        # Etiqueta
        if 'grasp_type' not in sample:
            raise KeyError("El sample no contiene 'grasp_type'.")
        y = torch.tensor([int(sample['grasp_type'])], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            mask=mask,
            joint_id=torch.arange(x.size(0), dtype=torch.long)
        )

        # (Opcional) handedness como atributo de grafo (0=Right, 1=Left, -1=Unknown)
        if 'handedness' in sample:
            data.handedness = torch.tensor(
                [self._to_int_handedness(sample.get('handedness'))],
                dtype=torch.long
            )

        return data

    def reset_velocity(self):
        """Reset the previous-frame buffer. Call when hand tracking is lost."""
        self._prev_positions = None
        self._prev_timestamp = None

    def __repr__(self):
        return (f"ToGraph(features={self.features}, undirected={self.make_undirected}, "
                f"use_conf={self.use_confidence}, conf_thr={self.conf_threshold})")
