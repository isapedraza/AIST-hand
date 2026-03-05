import logging
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from typing import Optional

log = logging.getLogger(__name__)

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
    """

    def __init__(self,
                 features: str = 'xyz',
                 make_undirected: bool = True,
                 use_confidence: bool = False,
                 conf_threshold: float = 0.5,
                 add_joint_angles: bool = False):
        assert features in ('xy', 'xyz')
        self.features = features
        self.make_undirected = make_undirected
        self.use_confidence = use_confidence
        self.conf_threshold = conf_threshold
        self.add_joint_angles = add_joint_angles
        self.F_xyz = 2 if features == 'xy' else 3   # dims used for xyz padding
        self.F = self.F_xyz + (1 if add_joint_angles else 0)  # total features per node

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

        for j in self.joints:
            raw = sample.get(j, None)
            if raw is None:
                v = np.zeros(self.F, dtype=float)
                valid = False
            else:
                arr = np.array(raw, dtype=float)
                # Ajustar a F dims (recortar o rellenar)
                if arr.shape[0] < self.F_xyz:
                    arr = np.pad(arr, (0, self.F_xyz - arr.shape[0]), mode='constant', constant_values=0.0)
                elif arr.shape[0] > self.F_xyz:
                    arr = arr[:self.F_xyz]
                valid = np.isfinite(arr).all()
                v = np.nan_to_num(arr, nan=0.0)

            # Umbral de confianza opcional
            if self.use_confidence:
                conf = float(sample.get(self.joint_conf_name[j], 1.0))
                valid = valid and (conf >= self.conf_threshold)

            rows.append(v)
            mask_vals.append(1.0 if valid else 0.0)

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

    def __repr__(self):
        return (f"ToGraph(features={self.features}, undirected={self.make_undirected}, "
                f"use_conf={self.use_confidence}, conf_thr={self.conf_threshold})")
