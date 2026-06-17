from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "models/latent-retargeting/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross_emb.inference.retarget import Retargeter
from cross_emb.loaders.human_loader import HumanLoader
from cross_emb.nn.human_modules import TemporalHumanEncoder_E_h


TIP_COLS = [
    "THUMB_TIP_x", "THUMB_TIP_y", "THUMB_TIP_z",
    "INDEX_FINGER_TIP_x", "INDEX_FINGER_TIP_y", "INDEX_FINGER_TIP_z",
    "MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z",
    "RING_FINGER_TIP_x", "RING_FINGER_TIP_y", "RING_FINGER_TIP_z",
    "PINKY_TIP_x", "PINKY_TIP_y", "PINKY_TIP_z",
]
CHAIN_COLS = [
    "THUMB_MCP_x", "THUMB_MCP_y", "THUMB_MCP_z",
    "THUMB_IP_x", "THUMB_IP_y", "THUMB_IP_z",
    "THUMB_TIP_x", "THUMB_TIP_y", "THUMB_TIP_z",
    "THUMB_TIP_x", "THUMB_TIP_y", "THUMB_TIP_z",
    "INDEX_FINGER_MCP_x", "INDEX_FINGER_MCP_y", "INDEX_FINGER_MCP_z",
    "INDEX_FINGER_PIP_x", "INDEX_FINGER_PIP_y", "INDEX_FINGER_PIP_z",
    "INDEX_FINGER_DIP_x", "INDEX_FINGER_DIP_y", "INDEX_FINGER_DIP_z",
    "INDEX_FINGER_TIP_x", "INDEX_FINGER_TIP_y", "INDEX_FINGER_TIP_z",
    "MIDDLE_FINGER_MCP_x", "MIDDLE_FINGER_MCP_y", "MIDDLE_FINGER_MCP_z",
    "MIDDLE_FINGER_PIP_x", "MIDDLE_FINGER_PIP_y", "MIDDLE_FINGER_PIP_z",
    "MIDDLE_FINGER_DIP_x", "MIDDLE_FINGER_DIP_y", "MIDDLE_FINGER_DIP_z",
    "MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y", "MIDDLE_FINGER_TIP_z",
    "RING_FINGER_MCP_x", "RING_FINGER_MCP_y", "RING_FINGER_MCP_z",
    "RING_FINGER_PIP_x", "RING_FINGER_PIP_y", "RING_FINGER_PIP_z",
    "RING_FINGER_DIP_x", "RING_FINGER_DIP_y", "RING_FINGER_DIP_z",
    "RING_FINGER_TIP_x", "RING_FINGER_TIP_y", "RING_FINGER_TIP_z",
    "PINKY_MCP_x", "PINKY_MCP_y", "PINKY_MCP_z",
    "PINKY_PIP_x", "PINKY_PIP_y", "PINKY_PIP_z",
    "PINKY_DIP_x", "PINKY_DIP_y", "PINKY_DIP_z",
    "PINKY_TIP_x", "PINKY_TIP_y", "PINKY_TIP_z",
]


def _row(frame_id: int, grasp_type: int) -> dict:
    row = {
        "subject_id": 11,
        "date_id": 1,
        "object_id": 2,
        "trial_id": 3,
        "cam": 0,
        "grasp_type": grasp_type,
        "frame_id": frame_id,
    }
    for j in range(1, 21):
        row[f"q{j}_w"] = float(frame_id)
        row[f"q{j}_x"] = 0.0
        row[f"q{j}_y"] = 0.0
        row[f"q{j}_z"] = 0.0
    coords = {
        "MIDDLE_FINGER_MCP_x": 1.0,
        "MIDDLE_FINGER_MCP_y": 0.0,
        "MIDDLE_FINGER_MCP_z": 0.0,
        "MIDDLE_FINGER_PIP_x": 2.0,
        "MIDDLE_FINGER_PIP_y": 0.0,
        "MIDDLE_FINGER_PIP_z": 0.0,
        "MIDDLE_FINGER_DIP_x": 3.0,
        "MIDDLE_FINGER_DIP_y": 0.0,
        "MIDDLE_FINGER_DIP_z": 0.0,
        "MIDDLE_FINGER_TIP_x": 4.0,
        "MIDDLE_FINGER_TIP_y": 0.0,
        "MIDDLE_FINGER_TIP_z": 0.0,
    }
    row.update(coords)
    for col in set(TIP_COLS + CHAIN_COLS):
        row.setdefault(col, 0.1)
    row.update(coords)
    return row


def test_human_loader_temporal_windows_do_not_cross_grasp_type(tmp_path: Path) -> None:
    csv = tmp_path / "human.csv"
    rows = [_row(i, 1) for i in range(4)] + [_row(i, 2) for i in range(4, 8)]
    pd.DataFrame(rows).to_csv(csv, index=False)

    loader = HumanLoader(csv, split="train", temporal_window=3)

    assert loader._valid_idx.cpu().tolist() == [2, 6]
    assert loader._window_idx.cpu().tolist() == [[0, 1, 2], [4, 5, 6]]
    assert loader._window_t1_idx.cpu().tolist() == [[1, 2, 3], [5, 6, 7]]
    batch = loader.get_batch_temporal(2, seed=0)
    assert batch["pose_window"].shape == (2, 3, 20, 4)
    assert batch["pose_t1_window"].shape == (2, 3, 20, 4)


def test_temporal_human_encoder_forward_quat_and_r6() -> None:
    for feature_dim in (4, 6):
        enc = TemporalHumanEncoder_E_h(in_dim=feature_dim, hidden_dim=8, z_dim=5, temporal_window=8)
        out = enc(torch.randn(2, 8, 20, feature_dim))
        assert out.shape == (2, 25)


def test_retargeter_temporal_buffer_skips_duplicate_samples() -> None:
    ret = Retargeter.__new__(Retargeter)
    ret.temporal_window = 3
    ret._temporal_buffer = []
    ret._last_pose = None
    ret._last_qpos = None

    a = torch.zeros(1, 20, 4)
    b = torch.ones(1, 20, 4)

    first = Retargeter._temporal_input(ret, a)
    assert first.shape == (1, 3, 20, 4)
    assert len(ret._temporal_buffer) == 3

    duplicate = Retargeter._temporal_input(ret, a.clone())
    assert duplicate is None
    assert len(ret._temporal_buffer) == 3

    second = Retargeter._temporal_input(ret, b)
    assert second.shape == (1, 3, 20, 4)
    assert torch.equal(second[:, -1], b)
