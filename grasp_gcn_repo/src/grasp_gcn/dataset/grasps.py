import os
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from ..transforms.tograph import ToGraph

log = logging.getLogger(__name__)

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
    - Para split=None con csvs: cada CSV puede ser
        * nombre dentro de <root>/raw  (p.ej. "mini.csv"), o
        * ruta absoluta (p.ej. "/home/.../data/raw/mini.csv").
      Las rutas dentro de <root>/raw se convierten a relativas para evitar duplicar prefijos.
    """

    def __init__(self, root, split="train", normalize=True, csvs=None,
                 transform=None, pre_transform=None, stats=None):
        self.split = split
        self.csvs = csvs
        self.normalize = normalize
        self.stats = stats
        self.k = 0

        super(GraspsClass, self).__init__(root, transform, pre_transform)

        # Compatibilidad PyTorch >= 2.6 (weights_only=True por defecto)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

        # Class weights sin warning (no tocar InMemoryDataset.data directamente)
        y = self._data.y.view(-1).cpu().numpy()
        n_classes = int(y.max()) + 1 if y.size > 0 else 0
        counts = np.bincount(y, minlength=n_classes).astype(np.float32) if n_classes > 0 else np.array([])
        self.class_weights = counts / counts.sum() if counts.size > 0 else counts

    # -------------------------------------------------------------
    @property
    def raw_file_names(self):
        if self.split == "train":
            return ['grasps_sample_train.csv']
        elif self.split == "val":
            return ['grasps_sample_val.csv']
        elif self.split == "test":
            return ['grasps_sample_test.csv']
        elif self.split is None and self.csvs is not None:
            # Evitar "data/raw/data/raw/..." si ya viene prefijado
            out = []
            raw_dir_abs = os.path.abspath(self.raw_dir)  # típico: <root>/raw
            for c in self.csvs:
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
        if self.split == "train":
            return [f"grasps_train_{self.k}.pt"]
        elif self.split == "val":
            return [f"grasps_val_{self.k}.pt"]
        elif self.split == "test":
            return [f"grasps_test_{self.k}.pt"]
        elif self.split is None and self.csvs is not None:
            joined = "_".join([os.path.splitext(os.path.basename(c))[0] for c in self.csvs])
            return [f"grasps_{joined}.pt"]
        else:
            raise ValueError(f"Unknown split: {self.split}")

    # -------------------------------------------------------------
    def process(self):
        transform_tograph_ = ToGraph()
        data_list_ = []

        for csv_path in self.raw_paths:
            log.info(f"Reading CSV file {csv_path}")
            grasps_ = pd.read_csv(csv_path)
            print(f"Data from {csv_path} loaded successfully. First few rows:")
            print(grasps_.head())

            for i in range(len(grasps_)):
                sample_ = self._sample_from_csv(grasps_, i)
                sample_ = transform_tograph_(sample_)
                if self.pre_transform is not None:
                    sample_ = self.pre_transform(sample_)
                data_list_.append(sample_)

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
        """Extrae una muestra del CSV, ignorando columnas extra (handedness/mirrored)."""
        row = grasps.iloc[idx]
        object_ = row.iloc[0]
        grasp_type_ = row.iloc[1]

        # Detecta columnas adicionales
        has_handed = ("handedness" in grasps.columns)
        has_mirror = ("mirrored" in grasps.columns)
        base = 2 + int(has_handed) + int(has_mirror)  # offset dinámico

        # Extrae 21x3 valores y los reordena
        vals = np.copy(row.iloc[base:base + 63]).astype(np.float32, copy=False).reshape(21, 3)
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
            'PINKY_TIP': PINKY_TIP_
        }

        if has_handed:
            sample_dict['handedness'] = row['handedness']
        if has_mirror:
            sample_dict['mirrored'] = int(row['mirrored'])

        return sample_dict
