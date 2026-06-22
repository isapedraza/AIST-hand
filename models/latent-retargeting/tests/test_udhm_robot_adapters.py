from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from cross_emb.udhm_backbone_refactor.core.udhm import SLOT_IDX
from cross_emb.udhm_backbone_refactor.robot.robot_to_udhm import robot_to_udhm
from cross_emb.udhm_backbone_refactor.robot.validate_adapter import validate_adapter_file


ROOT = Path(__file__).resolve().parents[3]
ADAPTERS = ROOT / "models/latent-retargeting/src/cross_emb/udhm_backbone_refactor/robot/adapters"


@pytest.mark.parametrize(
    ("name", "mapped", "movable"),
    [("shadow", 22, 24), ("leap", 16, 16), ("allegro", 16, 16)],
)
def test_adapter_is_structurally_complete(name: str, mapped: int, movable: int) -> None:
    result = validate_adapter_file(ADAPTERS / f"{name}_udhm_adapter.yaml")
    assert result["mapped_joints"] == mapped
    assert result["movable_joints"] == movable


@pytest.mark.parametrize("name", ["shadow", "leap", "allegro"])
def test_adapter_drives_exactly_its_declared_slots(name: str) -> None:
    adapter = yaml.safe_load((ADAPTERS / f"{name}_udhm_adapter.yaml").read_text())
    order = list(adapter["joints"])
    qpos = torch.ones(1, len(order))
    udhm = robot_to_udhm(qpos, adapter, order)
    expected_slots = {entry["slot"] for entry in adapter["joints"].values()}
    nonzero_slots = {slot for slot, index in SLOT_IDX.items() if udhm[0, index] != 0}
    assert nonzero_slots == expected_slots


def test_thumb_semantics_are_explicit() -> None:
    leap = yaml.safe_load((ADAPTERS / "leap_udhm_adapter.yaml").read_text())["joints"]
    allegro = yaml.safe_load((ADAPTERS / "allegro_udhm_adapter.yaml").read_text())["joints"]
    expected = {
        "thumb_cmc_flex",
        "thumb_cmc_spread",
        "thumb_mcp_flex",
        "thumb_ip_flex",
    }
    assert {leap[str(index)]["slot"] for index in range(12, 16)} == expected
    assert {allegro[f"joint_{index}.0"]["slot"] for index in range(12, 16)} == expected
