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

# Named XYZ columns in the same order as JOINT_NAMES (works for both CSVs)
XYZ_COLS = [
    f'{j}_{ax}'
    for j in [
        'WRIST',
        'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
        'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
        'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
    ]
    for ax in ('x', 'y', 'z')
]

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

# Data-driven taxonomy v1: 28 -> 17 classes
# Derived from GCN Run 006 confusion analysis (decide_collapses.py, gcn_thresh=0.15)
# Groups formed via union-find on 13 collapse pairs.
TAXONOMY_V1_COLLAPSE = {
    0:  3,   # Large_Diameter      -> Power_Wrap cluster
    1:  5,   # Small_Diameter      -> singleton
    2:  6,   # Index_Finger_Ext    -> singleton
    3:  7,   # Extension_Type      -> singleton
    4:  8,   # Parallel_Ext        -> singleton
    5:  9,   # Palmar              -> singleton
    6:  10,  # Medium_Wrap         -> singleton
    7:  11,  # Adducted_Thumb      -> singleton
    8:  2,   # Light_Tool          -> Lateral cluster
    9:  12,  # Distal              -> singleton
    10: 1,   # Ring                -> Pinch cluster
    11: 3,   # Power_Disk          -> Power_Wrap cluster
    12: 13,  # Power_Sphere        -> singleton
    13: 14,  # Sphere_4_Finger     -> singleton
    14: 0,   # Sphere_3_Finger     -> Tripod cluster
    15: 2,   # Lateral             -> Lateral cluster
    16: 2,   # Stick               -> Lateral cluster
    17: 15,  # Adduction_Grip      -> singleton
    18: 0,   # Writing_Tripod      -> Tripod cluster
    19: 0,   # Lateral_Tripod      -> Tripod cluster
    20: 1,   # Palmar_Pinch        -> Pinch cluster
    21: 1,   # Tip_Pinch           -> Pinch cluster
    22: 1,   # Inferior_Pincer     -> Pinch cluster
    23: 0,   # Prismatic_3F        -> Tripod cluster
    24: 4,   # Precision_Disk      -> Precision cluster
    25: 4,   # Precision_Sphere    -> Precision cluster
    26: 16,  # Quadpod             -> singleton
    27: 0,   # Tripod              -> Tripod cluster
}

TAXONOMY_V1_CLASS_NAMES = {
    0:  "Tripod_cluster",       # Sphere_3F, Writing_Tripod, Lateral_Tripod, Prismatic_3F, Tripod
    1:  "Pinch_cluster",        # Ring, Palmar_Pinch, Tip_Pinch, Inferior_Pincer
    2:  "Lateral_cluster",      # Light_Tool, Lateral, Stick
    3:  "Power_Wrap_cluster",   # Large_Diameter, Power_Disk
    4:  "Precision_cluster",    # Precision_Disk, Precision_Sphere
    5:  "Small_Diameter",
    6:  "Index_Finger_Ext",
    7:  "Extension_Type",
    8:  "Parallel_Ext",
    9:  "Palmar",
    10: "Medium_Wrap",
    11: "Adducted_Thumb",
    12: "Distal",
    13: "Power_Sphere",
    14: "Sphere_4_Finger",
    15: "Adduction_Grip",
    16: "Quadpod",
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

GRASP_CLASS_NAMES = {
    0:  "Large Diameter",
    1:  "Small Diameter",
    2:  "Index Finger Extension",
    3:  "Extension Type",
    4:  "Parallel Extension",
    5:  "Palmar",
    6:  "Medium Wrap",
    7:  "Adducted Thumb",
    8:  "Light Tool",
    9:  "Distal",
    10: "Ring",
    11: "Power Disk",
    12: "Power Sphere",
    13: "Sphere 4-Finger",
    14: "Sphere 3-Finger",
    15: "Lateral",
    16: "Stick",
    17: "Adduction Grip",
    18: "Writing Tripod",
    19: "Lateral Tripod",
    20: "Palmar Pinch",
    21: "Tip Pinch",
    22: "Inferior Pincer",
    23: "Prismatic 3 Finger",
    24: "Precision Disk",
    25: "Precision Sphere",
    26: "Quadpod",
    27: "Tripod",
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
                 add_bone_vectors=False,
                 add_velocity=False,
                 transform=None, pre_transform=None):
        assert split in SPLIT_SUBJECTS, f"split must be one of {list(SPLIT_SUBJECTS)}"
        # collapse: False (28 cls) | True or 'feix' (16 cls) | 'taxonomy_v1' (17 cls)
        assert collapse in (False, True, 'feix', 'taxonomy_v1'), \
            "collapse must be False, True, 'feix', or 'taxonomy_v1'"
        self.split = split
        self.collapse = collapse
        self.add_bone_vectors = add_bone_vectors
        self.add_velocity = add_velocity
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
        # velocity requires frame_id column, only present in hograspnet_mano.csv
        return ['hograspnet_mano.csv'] if self.add_velocity else ['hograspnet.csv']

    @property
    def processed_file_names(self):
        if self.collapse in (True, 'feix'):
            cls_tag = 'c16'
        elif self.collapse == 'taxonomy_v1':
            cls_tag = 'c17'
        else:
            cls_tag = 'c28'
        bone_tag = '_bone' if self.add_bone_vectors else ''
        vel_tag  = '_vel'  if self.add_velocity     else ''
        return [f'hograspnet_{self.split}_{cls_tag}_cmc{bone_tag}{vel_tag}.pt']

    # ------------------------------------------------------------------
    def process(self):
        tograph = ToGraph(
            features='xyz',
            make_undirected=True,
            add_joint_angles=True,
            add_cmc_angle=True,
            add_bone_vectors=self.add_bone_vectors,
            add_velocity=self.add_velocity,
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
        if self.add_velocity:
            # Sequence-aware loop: compute v_i = pos(t) - pos(t-1) within each
            # (sequence_id, cam) group, ordered by frame_id.
            # First frame of each group gets v = [0, 0, 0].
            df = df.sort_values(['sequence_id', 'cam', 'frame_id'])
            for _, group in df.groupby(['sequence_id', 'cam'], sort=False):
                prev_xyz = None
                for _, row in group.iterrows():
                    sample = self._sample_from_row(row)
                    curr_xyz = row[XYZ_COLS].values.astype(np.float32).reshape(21, 3)
                    # Velocity normalized by dt so units are consistent across
                    # frame rates.  HOGraspNet annotations are at 10 fps -> dt=0.1s.
                    # Deploy normalizes by real elapsed time -> both in units/second.
                    _DT_TRAIN = 0.1  # 10 fps HOGraspNet annotation rate
                    sample['velocity'] = (
                        (curr_xyz - prev_xyz) / _DT_TRAIN if prev_xyz is not None
                        else np.zeros((21, 3), dtype=np.float32)
                    )
                    prev_xyz = curr_xyz
                    graph = tograph(sample)
                    if self.pre_transform is not None:
                        graph = self.pre_transform(graph)
                    data_list.append(graph)
        else:
            for i in range(len(df)):
                sample = self._sample_from_row(df.iloc[i])
                graph = tograph(sample)
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                data_list.append(graph)

        if self.collapse in (True, 'feix'):
            for d in data_list:
                old = int(d.y.item())
                d.y = torch.tensor([FEIX_COLLAPSE[old]], dtype=torch.long)
        elif self.collapse == 'taxonomy_v1':
            for d in data_list:
                old = int(d.y.item())
                d.y = torch.tensor([TAXONOMY_V1_COLLAPSE[old]], dtype=torch.long)

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    # ------------------------------------------------------------------
    def _sample_from_row(self, row):
        """
        Build a ToGraph-compatible dict from a single CSV row.
        Uses named columns so it works with both hograspnet.csv and
        hograspnet_mano.csv (which has an extra frame_id column).
        """
        grasp_type = int(row['grasp_type'])
        vals = row[XYZ_COLS].values.astype(np.float32).reshape(21, 3)

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
