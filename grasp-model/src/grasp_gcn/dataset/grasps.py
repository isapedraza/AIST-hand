import os
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from ..transforms.tograph import ToGraph

log = logging.getLogger(__name__)

# S1 split -- official HOGraspNet protocol (subject-level, no leakage)
TRAIN_SUBJECTS = list(range(11, 74))   # S11-S73
VAL_SUBJECTS   = list(range(1, 11))    # S01-S10
TEST_SUBJECTS  = list(range(74, 100))  # S74-S99

SPLIT_SUBJECTS = {
    'train': TRAIN_SUBJECTS,
    'val':   VAL_SUBJECTS,
    'test':  TEST_SUBJECTS,
}

# Joint names in MediaPipe / HOGraspNet order (21 landmarks)
JOINT_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
]

# Collapse 28 HOGraspNet classes -> 16 Feix functional cells.
# Grasps within a cell differ mainly in contact type / object shape, not hand topology.
# Mapping verified against Feix et al. (2016), Fig. 4.
FEIX_COLLAPSE = {
    0:  0,   # Large Diameter          -> Power, Palm, VF 2-5, Abducted
    1:  0,   # Small Diameter          -> Power, Palm, VF 2-5, Abducted
    6:  0,   # Medium Wrap             -> Power, Palm, VF 2-5, Abducted
    11: 0,   # Power Disk              -> Power, Palm, VF 2-5, Abducted
    12: 0,   # Power Sphere            -> Power, Palm, VF 2-5, Abducted
    5:  1,   # Palmar                  -> Power, Palm, VF 2-5, Adducted
    7:  1,   # Adducted Thumb          -> Power, Palm, VF 2-5, Adducted
    8:  1,   # Light Tool              -> Power, Palm, VF 2-5, Adducted
    3:  2,   # Extension Type          -> Power, Pad, VF 2-4, Abducted
    13: 2,   # Sphere 4-Finger         -> Power, Pad, VF 2-4, Abducted
    15: 3,   # Lateral                 -> Intermediate, Side, VF 2, Adducted
    16: 3,   # Stick                   -> Intermediate, Side, VF 2, Adducted
    20: 4,   # Palmar Pinch            -> Precision, Pad, VF 2, Abducted
    21: 4,   # Tip Pinch               -> Precision, Pad, VF 2, Abducted
    22: 4,   # Inferior Pincer         -> Precision, Pad, VF 2, Abducted
    23: 5,   # Prismatic 3-Finger      -> Precision, Pad, VF 2-4, Abducted
    26: 5,   # Quadpod                 -> Precision, Pad, VF 2-4, Abducted
    24: 6,   # Precision Disk          -> Precision, Pad, VF 2-5, Abducted
    25: 6,   # Precision Sphere        -> Precision, Pad, VF 2-5, Abducted
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
    GCN grasp dataset built from hograspnet.csv.

    Reads the single unified CSV and splits by subject ID following the
    official HOGraspNet S1 protocol (subject-level, no frame leakage):
      - train: S11-S73
      - val:   S01-S10
      - test:  S74-S99

    CSV column layout (from build_hograspnet_csv.py):
      subject_id | sequence_id | cam | grasp_type | contact_sum | [63 XYZ]

    Geometric normalisation (root-relative + scale) is pre-applied in the CSV.
    No z-score is applied here -- it was eliminated from the pipeline because
    geometric normalisation already removes the domain gap between HOGraspNet
    (metric) and MediaPipe ([0,1]).

    Args:
        root:     PyG root directory. CSV must be at <root>/raw/hograspnet.csv.
        split:    'train', 'val', or 'test'.
        collapse: If True, remap 28 HOGraspNet classes to 16 Feix functional
                  cells via FEIX_COLLAPSE. If False, keep all 28 classes.
    """

    def __init__(self, root, split='train', collapse=False,
                 transform=None, pre_transform=None):
        assert split in SPLIT_SUBJECTS, f"split must be one of {list(SPLIT_SUBJECTS)}"
        self.split = split
        self.collapse = collapse
        super().__init__(root, transform, pre_transform)

        try:
            self.data, self.slices = torch.load(
                self.processed_paths[0], weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

        self._num_features = self._data.num_features
        self._num_classes = (
            int(self._data.y.max().item() + 1)
            if self._data.y.numel() > 0 else 0
        )

        y_np = self._data.y.view(-1).cpu().numpy()
        n_cls = int(y_np.max()) + 1 if y_np.size > 0 else 0
        counts = (np.bincount(y_np, minlength=n_cls).astype(np.float32)
                  if n_cls > 0 else np.array([]))
        self.class_weights = counts / counts.sum() if counts.size > 0 else counts

    # ------------------------------------------------------------------
    @property
    def raw_file_names(self):
        return ['hograspnet.csv']

    @property
    def processed_file_names(self):
        cls_tag = 'c16' if self.collapse else 'c28'
        return [f'hograspnet_{self.split}_{cls_tag}_cmc.pt']

    # ------------------------------------------------------------------
    def process(self):
        tograph = ToGraph(
            features='xyz',
            make_undirected=True,
            add_joint_angles=True,
            add_cmc_angle=True,
        )

        csv_path = self.raw_paths[0]
        log.info(f"Reading {csv_path}")
        df = pd.read_csv(csv_path)

        subjects = SPLIT_SUBJECTS[self.split]
        df = df[df['subject_id'].isin(subjects)].reset_index(drop=True)
        log.info(
            f"Split '{self.split}': {len(df):,} frames "
            f"from {df['subject_id'].nunique()} subjects"
        )
        print(
            f"[GraspsClass] {self.split}: {len(df):,} frames "
            f"from {df['subject_id'].nunique()} subjects"
        )

        data_list = []
        for i in range(len(df)):
            sample = self._sample_from_row(df.iloc[i])
            graph = tograph(sample)
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            data_list.append(graph)

        if self.collapse:
            for d in data_list:
                old = int(d.y.item())
                d.y = torch.tensor([FEIX_COLLAPSE[old]], dtype=torch.long)

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    # ------------------------------------------------------------------
    def _sample_from_row(self, row):
        """
        Build a ToGraph-compatible dict from a single CSV row.

        CSV layout: subject_id(0) | sequence_id(1) | cam(2) |
                    grasp_type(3) | contact_sum(4) | XYZ[5:68]
        """
        grasp_type = int(row['grasp_type'])
        # 63 XYZ values starting at column index 5
        vals = row.iloc[5:68].values.astype(np.float32).reshape(21, 3)

        sample = {'grasp_type': grasp_type}
        for j, name in enumerate(JOINT_NAMES):
            sample[name] = vals[j]
        return sample

    # ------------------------------------------------------------------
    @property
    def num_features(self):
        return getattr(self, '_num_features', self._data.num_features)

    @property
    def num_classes(self):
        if hasattr(self, '_num_classes'):
            return self._num_classes
        return (int(self._data.y.max().item() + 1)
                if self._data.y.numel() > 0 else 0)

    @property
    def num_node_features(self):
        return self._data.num_features
