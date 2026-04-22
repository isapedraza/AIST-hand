import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Hand skeleton edges (same as ToGraph in tograph.py)
_EDGE_SRC = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9,
             5, 9, 10, 10, 11, 11, 12, 13, 9, 13, 14, 14, 15, 15,
             16, 17, 13, 17, 18, 17, 18, 19, 19, 20]
_EDGE_DST = [1, 5, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 5, 7, 6, 8, 7, 5,
             9, 10, 9, 11, 10, 12, 11, 9, 13, 14, 13, 15, 14, 16,
             15, 13, 17, 18, 17, 0, 19, 18, 20, 19]
_EDGE_INDEX = to_undirected(
    torch.tensor([_EDGE_SRC, _EDGE_DST], dtype=torch.long)
)

# XYZ column order: WRIST(0), thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
_XYZ_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
_XYZ_COLS = [f"{lm}_{ax}" for lm in _XYZ_LANDMARKS for ax in ("x", "y", "z")]

# Dong quat columns q1-q20
_QUAT_COLS = [f"q{j}_{c}" for j in range(1, 21) for c in ("w", "x", "y", "z")]

# Sequence grouping keys (identifies a single video)
_SEQ_KEYS = ["subject_id", "date_id", "object_id", "grasp_type", "trial_id", "cam"]


class RetargetDataset(Dataset):
    """
    Dataset for cross-embodiment retargeting training.

    Each item returns:
        graph  : PyG Data, x=[21,4] Dong quats (wrist=identity, nodes 1-20=q1-q20)
        xyz    : [21, 3] raw XYZ landmarks in world frame
        quats  : [20, 4] Dong quats q1-q20 (for D_R in loss_contrastive)

    Args:
        csv_path        : path to hograspnet_retarget.csv
        return_temporal : if True, __getitem__ returns a pair (t, t+1) from same sequence
    """

    def __init__(self, csv_path: str, return_temporal: bool = False):
        self.return_temporal = return_temporal
        self._edge_index = _EDGE_INDEX.clone()

        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  {len(df):,} rows")

        # Extract tensors
        self._xyz   = torch.tensor(df[_XYZ_COLS].values, dtype=torch.float32).view(-1, 21, 3)
        self._quats = torch.tensor(df[_QUAT_COLS].values, dtype=torch.float32).view(-1, 20, 4)

        if return_temporal:
            self._build_temporal_pairs(df)
        else:
            self._indices = torch.arange(len(df))

    def _build_temporal_pairs(self, df: pd.DataFrame, seq_keys: list = None):
        """Build list of (t, t+1) index pairs within same sequence.

        seq_keys: if provided, only include pairs from these sequence keys.
        """
        df = df.reset_index(drop=True)
        pairs = []
        for key, grp in df.groupby(_SEQ_KEYS):
            if seq_keys is not None and key not in seq_keys:
                continue
            idx = grp.sort_values("frame_id").index.tolist()
            for i in range(len(idx) - 1):
                pairs.append((idx[i], idx[i + 1]))
        self._pairs = pairs

    @classmethod
    def train_test_split(cls, csv_path: str, test_frac: float = 0.2, seed: int = 42):
        """
        Return (train_dataset, test_dataset) split at sequence level.

        Sequences are split 80/20 so no sequence appears in both sets.
        Both datasets use return_temporal=True.
        """
        import random
        rng = random.Random(seed)

        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  {len(df):,} rows")

        all_seqs = list(df.groupby(_SEQ_KEYS).groups.keys())
        rng.shuffle(all_seqs)
        n_test = max(1, int(len(all_seqs) * test_frac))
        test_seqs  = set(all_seqs[-n_test:])
        train_seqs = set(all_seqs[:-n_test])

        xyz   = torch.tensor(df[_XYZ_COLS].values, dtype=torch.float32).view(-1, 21, 3)
        quats = torch.tensor(df[_QUAT_COLS].values, dtype=torch.float32).view(-1, 20, 4)

        def _make(seq_set):
            obj = object.__new__(cls)
            obj.return_temporal = True
            obj._edge_index = _EDGE_INDEX.clone()
            obj._xyz   = xyz
            obj._quats = quats
            obj._build_temporal_pairs(df, seq_keys=seq_set)
            return obj

        train_ds = _make(train_seqs)
        test_ds  = _make(test_seqs)
        print(f"  Train: {len(train_ds):,} pairs | Test: {len(test_ds):,} pairs")
        return train_ds, test_ds

    def __len__(self) -> int:
        if self.return_temporal:
            return len(self._pairs)
        return len(self._indices)

    def _make_graph(self, i: int) -> Data:
        quats_20 = self._quats[i]   # [20, 4]
        wrist = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity
        x = torch.cat([wrist, quats_20], dim=0)        # [21, 4]
        return Data(x=x, edge_index=self._edge_index)

    def __getitem__(self, idx):
        if self.return_temporal:
            i, j = self._pairs[idx]
            return {
                "graph_t":  self._make_graph(i),
                "graph_t1": self._make_graph(j),
                "xyz_t":    self._xyz[i],
                "xyz_t1":   self._xyz[j],
                "quats_t":  self._quats[i],
                "quats_t1": self._quats[j],
            }
        i = int(self._indices[idx])
        return {
            "graph": self._make_graph(i),
            "xyz":   self._xyz[i],
            "quats": self._quats[i],
        }
