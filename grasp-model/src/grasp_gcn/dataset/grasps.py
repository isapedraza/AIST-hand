import os
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from ..transforms.tograph import ToGraph

log = logging.getLogger(__name__)

# Collapse the 28 HOGraspNet local indices to 16 functional classes,
# one per Feix cell (same opposition type + VF assignment + thumb position).
# Grasps within a cell share the same hand topology; only the held object varies.
# Mapping verified against Feix et al. (2016), Fig. 4.
FEIX_COLLAPSE = {
    0:  0,   # Large Diameter      -> Power, Palm, VF 2-5, Abducted
    1:  0,   # Small Diameter      -> Power, Palm, VF 2-5, Abducted
    6:  0,   # Medium Wrap         -> Power, Palm, VF 2-5, Abducted
    11: 0,   # Power Disk          -> Power, Palm, VF 2-5, Abducted
    12: 0,   # Power Sphere        -> Power, Palm, VF 2-5, Abducted
    5:  1,   # Palmar              -> Power, Palm, VF 2-5, Adducted
    7:  1,   # Adducted Thumb      -> Power, Palm, VF 2-5, Adducted
    8:  1,   # Light Tool          -> Power, Palm, VF 2-5, Adducted
    3:  2,   # Extension Type      -> Power, Pad, VF 2-4, Abducted
    13: 2,   # Sphere 4-Finger     -> Power, Pad, VF 2-4, Abducted
    15: 3,   # Lateral             -> Intermediate, Side, VF 2, Adducted
    16: 3,   # Stick               -> Intermediate, Side, VF 2, Adducted
    20: 4,   # Palmar Pinch        -> Precision, Pad, VF 2, Abducted
    21: 4,   # Tip Pinch           -> Precision, Pad, VF 2, Abducted
    22: 4,   # Inferior Pincer     -> Precision, Pad, VF 2, Abducted
    23: 5,   # Prismatic 3-Finger  -> Precision, Pad, VF 2-4, Abducted
    26: 5,   # Quadpod             -> Precision, Pad, VF 2-4, Abducted
    24: 6,   # Precision Disk      -> Precision, Pad, VF 2-5, Abducted
    25: 6,   # Precision Sphere    -> Precision, Pad, VF 2-5, Abducted
    2:  7,   # Index Finger Extension  -> singleton
    4:  8,   # Parallel Extension      -> singleton
    9:  9,   # Distal                  -> singleton
    10: 10,  # Ring                    -> singleton
    14: 11,  # Sphere 3-Finger         -> singleton
    17: 12,  # Adduction Grip          -> singleton
    18: 13,  # Writing Tripod          -> singleton
    19: 14,  # Lateral Tripod          -> singleton
    27: 15,  # Tripod                  -> singleton
}

# Human-readable names for the 16 functional classes.
FEIX_CLASS_NAMES = {
    0:  "Power, Palm, VF 2-5, Abducted",
    1:  "Power, Palm, VF 2-5, Adducted",
    2:  "Power, Pad, VF 2-4, Abducted",
    3:  "Intermediate, Side, VF 2, Adducted",
    4:  "Precision, Pad, VF 2, Abducted",
    5:  "Precision, Pad, VF 2-4, Abducted",
    6:  "Precision, Pad, VF 2-5, Abducted",
    7:  "Index Finger Extension",
    8:  "Parallel Extension",
    9:  "Distal",
    10: "Ring",
    11: "Sphere 3-Finger",
    12: "Adduction Grip",
    13: "Writing Tripod",
    14: "Lateral Tripod",
    15: "Tripod",
}

class GraspsClass(InMemoryDataset):
    """
    Dataset de agarres para GCN.
    - Lee CSVs con coordenadas 3D de los 21 landmarks (MediaPipe order).
    - Convierte cada muestra en un grafo (ToGraph).
    - Normalización z-score sin data leakage:
        * Calcula mean/std SOLO en train y guarda processed/train_stats.npz.
        * Aplica esas mismas stats a val/test; si no existen, lanza error.
    - Ignora columnas extra ('handedness', 'mirrored') si existen.

    Convenciones de rutas:
    - PyG asume: raw_dir = <root>/raw   y   processed_dir = <root>/processed
    """

    def __init__(self, root, split="train", normalize=True, csvs=None,
                 transform=None, pre_transform=None, stats=None,
                 collapse=False):
        self.split = split
        self.csvs = csvs
        self.normalize = normalize
        self.stats = stats
        self.collapse = collapse
        self.k = 1

        super(GraspsClass, self).__init__(root, transform, pre_transform)

        # Carga el almacenamiento procesado
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

        # ----- Metadatos internos (no sobrescribir propiedades de PyG) -----
        self._num_features = self._data.num_features
        self._num_classes = int(self._data.y.max().item() + 1) if self._data.y.numel() > 0 else 0

        # Class weights
        y_np = self._data.y.view(-1).cpu().numpy()
        n_classes = int(y_np.max()) + 1 if y_np.size > 0 else 0
        counts = np.bincount(y_np, minlength=n_classes).astype(np.float32) if n_classes > 0 else np.array([])
        self.class_weights = counts / counts.sum() if counts.size > 0 else counts

    # -------------------------------------------------------------
    @property
    def raw_file_names(self):
        if self.split == "train":
            return ['grasps_train.csv']
        elif self.split == "val":
            return ['grasps_val.csv']
        elif self.split == "test":
            return ['grasps_test.csv']
        elif self.split is None and self.csvs is not None:
            out = []
            raw_dir_abs = os.path.abspath(self.raw_dir)
            for c in (self.csvs if isinstance(self.csvs, (list, tuple)) else [self.csvs]):
                p = os.path.abspath(c)
                try:
                    if os.path.commonpath([p, raw_dir_abs]) == raw_dir_abs:
                        out.append(os.path.relpath(p, raw_dir_abs))
                    else:
                        out.append(p)
                except ValueError:
                    out.append(p)
            return out
        else:
            raise ValueError(f"Unknown split: {self.split}")

    # -------------------------------------------------------------
    @property
    def processed_file_names(self):
        suffix = f"_{self.k}" + ("_c16" if self.collapse else "") + "_cmc"
        if self.split == "train":
            return [f"grasps_train{suffix}.pt"]
        elif self.split == "val":
            return [f"grasps_val{suffix}.pt"]
        elif self.split == "test":
            return [f"grasps_test{suffix}.pt"]
        elif self.split is None and self.csvs is not None:
            names = (self.csvs if isinstance(self.csvs, (list, tuple)) else [self.csvs])
            joined = "_".join([os.path.splitext(os.path.basename(c))[0] for c in names])
            return [f"grasps_{joined}{suffix}.pt"]
        else:
            raise ValueError(f"Unknown split: {self.split}")

    # -------------------------------------------------------------
    def process(self):
        transform_tograph_ = ToGraph(features='xyz', make_undirected=True, add_joint_angles=True, add_cmc_angle=True)
        data_list_ = []

        for csv_path in self.raw_paths:
            log.info(f"Reading CSV file {csv_path}")
            grasps_ = pd.read_csv(csv_path)
            grasps_.columns = grasps_.columns.str.strip()
            if 'grasp' in grasps_.columns and 'grasp_type' not in grasps_.columns:
                grasps_.rename(columns={'grasp': 'grasp_type'}, inplace=True)

            print(f"Data from {csv_path} loaded successfully. First few rows:")
            print(grasps_.head())

            for i in range(len(grasps_)):
                sample_ = self._sample_from_csv(grasps_, i)
                sample_ = transform_tograph_(sample_)
                if self.pre_transform is not None:
                    sample_ = self.pre_transform(sample_)
                data_list_.append(sample_)

        # ---- Colapso de clases Feix (28 -> 16 clases funcionales) ----
        if self.collapse:
            for d in data_list_:
                old_label = int(d.y.item())
                d.y = torch.tensor([FEIX_COLLAPSE[old_label]], dtype=torch.long)

        # ---- Normalización (z-score) sin leakage (modo estricto CS230) ----
        if self.normalize and len(data_list_) > 0:
            try:
                if self.stats is None:
                    if self.split == "train":
                        raw_np = np.vstack([d.x.cpu().numpy() for d in data_list_])
                        mean = raw_np.mean(axis=0)
                        std = raw_np.std(axis=0)
                        std[std < 1e-8] = 1e-8
                        os.makedirs(self.processed_dir, exist_ok=True)
                        np.savez(os.path.join(self.processed_dir, "train_stats.npz"),
                                 mean=mean, std=std)
                    else:
                        stats_path = os.path.join(self.processed_dir, "train_stats.npz")
                        if not os.path.exists(stats_path):
                            raise RuntimeError(
                                "train_stats.npz no encontrado. "
                                "Genera el split 'train' primero (para guardar stats) "
                                "o pasa 'stats=(mean,std)' explícitamente."
                            )
                        z = np.load(stats_path)
                        mean = np.asarray(z["mean"], dtype=np.float32)
                        std = np.asarray(z["std"], dtype=np.float32)
                        std[std < 1e-8] = 1e-8
                else:
                    mean, std = self.stats
                    mean = np.asarray(mean, dtype=np.float32)
                    std = np.asarray(std, dtype=np.float32)
                    std[std < 1e-8] = 1e-8

                for d in data_list_:
                    x = d.x.cpu().numpy()
                    d.x = torch.from_numpy((x - mean) / std).float()
            except Exception as e:
                print(f"[WARN] Normalization skipped due to error: {e}")

        data_ = self.collate(data_list_)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save(data_, self.processed_paths[0])

    # -------------------------------------------------------------
    def _sample_from_csv(self, grasps, idx):
        """Extrae una muestra del CSV."""
        row = grasps.iloc[idx]
        object_ = row["object"]
        grasp_type_ = int(row["grasp_type"])

        # Extrae 21x3 valores (MediaPipe joints) — columnas 2..64
        vals = np.copy(row.iloc[2:65]).astype(np.float32, copy=False).reshape(21, 3)
        (WRIST_,
         THUMB_CMC_, THUMB_MCP_, THUMB_IP_, THUMB_TIP_,
         INDEX_FINGER_MCP_, INDEX_FINGER_PIP_, INDEX_FINGER_DIP_, INDEX_FINGER_TIP_,
         MIDDLE_FINGER_MCP_, MIDDLE_FINGER_PIP_, MIDDLE_FINGER_DIP_, MIDDLE_FINGER_TIP_,
         RING_FINGER_MCP_, RING_FINGER_PIP_, RING_FINGER_DIP_, RING_FINGER_TIP_,
         PINKY_MCP_, PINKY_PIP_, PINKY_DIP_, PINKY_TIP_) = [vals[i] for i in range(21)]

        sample_dict = {
            'object': object_,
            'grasp_type': grasp_type_,
            'WRIST': WRIST_,
            'THUMB_CMC': THUMB_CMC_,
            'THUMB_MCP': THUMB_MCP_,
            'THUMB_IP': THUMB_IP_,
            'THUMB_TIP': THUMB_TIP_,
            'INDEX_FINGER_MCP': INDEX_FINGER_MCP_,
            'INDEX_FINGER_PIP': INDEX_FINGER_PIP_,
            'INDEX_FINGER_DIP': INDEX_FINGER_DIP_,
            'INDEX_FINGER_TIP': INDEX_FINGER_TIP_,
            'MIDDLE_FINGER_MCP': MIDDLE_FINGER_MCP_,
            'MIDDLE_FINGER_PIP': MIDDLE_FINGER_PIP_,
            'MIDDLE_FINGER_DIP': MIDDLE_FINGER_DIP_,
            'MIDDLE_FINGER_TIP': MIDDLE_FINGER_TIP_,
            'RING_FINGER_MCP': RING_FINGER_MCP_,
            'RING_FINGER_PIP': RING_FINGER_PIP_,
            'RING_FINGER_DIP': RING_FINGER_DIP_,
            'RING_FINGER_TIP': RING_FINGER_TIP_,
            'PINKY_MCP': PINKY_MCP_,
            'PINKY_PIP': PINKY_PIP_,
            'PINKY_DIP': PINKY_DIP_,
            'PINKY_TIP': PINKY_TIP_,
        }

        return sample_dict

    # -------------------------------------------------------------
    # === Propiedades para compatibilidad moderna ===
    @property
    def num_features(self):
        return getattr(self, "_num_features", self._data.num_features)

    @property
    def num_classes(self):
        if hasattr(self, "_num_classes"):
            return self._num_classes
        return int(self._data.y.max().item() + 1) if self._data.y.numel() > 0 else 0

    @property
    def num_node_features(self):
        return self._data.num_features
