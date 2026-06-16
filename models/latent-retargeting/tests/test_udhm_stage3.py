"""Tests for udhm_run_stage3 (stage-2 rotations -> UDHM 22-slot named angles).

Run: python models/latent-retargeting/tests/test_udhm_stage3.py
 or: pytest models/latent-retargeting/tests/test_udhm_stage3.py

Validates the decomposition against Dong Eq. 24 geometry analytically:
  - pure abduction by a  -> gamma = a, beta = 0
  - pure flexion  by f   -> beta  = f, gamma = 0
  - PIP/DIP hinge Ry(b)  -> flex  = b
and guards UDHM22_SLOTS against the canonical contract yaml.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import pytorch_kinematics as pk
import yaml

from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS, udhm_run_stage3, _SLOT_IDX

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "robot/hand-configs/udhm_canonical_22dof.yaml"
)
# 1e-3: the _ACOS_EPS clamp floors arccos near +-1 at ~1.4e-4 (harmless at angle 0).
_TOL = 1e-3


def _quat_from_frame(x, y, z) -> torch.Tensor:
    """Build wxyz quat from an MCP frame with columns [Xi, Yi, Zi] (mirrors stage-2)."""
    R = torch.tensor([x, y, z], dtype=torch.float64).T.unsqueeze(0)  # [1,3,3] cols
    q = pk.matrix_to_quaternion(R)
    return torch.where(q[:, :1] < 0, -q, q).unsqueeze(1)  # [1,1,4] wxyz, w>=0


def _mcp_frame(abd: float, flex: float):
    """Xi/Yi/Zi for abduction (around palm normal Z) then flexion (toward palm)."""
    ca, sa, cf, sf = math.cos(abd), math.sin(abd), math.cos(flex), math.sin(flex)
    # abduct in palm plane, then tilt down by flex (around the lateral axis)
    Xi = [ca * cf, sa * cf, -sf]
    Z0 = [0.0, 0.0, 1.0]
    # Yi = normalize(Z0 x Xi)
    yx = Z0[1] * Xi[2] - Z0[2] * Xi[1]
    yy = Z0[2] * Xi[0] - Z0[0] * Xi[2]
    yz = Z0[0] * Xi[1] - Z0[1] * Xi[0]
    n = math.sqrt(yx * yx + yy * yy + yz * yz)
    Yi = [yx / n, yy / n, yz / n]
    # Zi = Xi x Yi
    Zi = [
        Xi[1] * Yi[2] - Xi[2] * Yi[1],
        Xi[2] * Yi[0] - Xi[0] * Yi[2],
        Xi[0] * Yi[1] - Xi[1] * Yi[0],
    ]
    return Xi, Yi, Zi


def test_slots_match_contract():
    cfg = yaml.safe_load(_CONTRACT.read_text())
    names = [s["name"] for s in sorted(cfg["slots"], key=lambda s: s["i"])]
    assert names == UDHM22_SLOTS, "UDHM22_SLOTS drifted from the contract yaml"
    assert len(UDHM22_SLOTS) == 22


def test_mcp_neutral_is_zero():
    q = _quat_from_frame([1, 0, 0], [0, 1, 0], [0, 0, 1])  # identity frame
    v = udhm_run_stage3(q, ["index_mcp"])
    assert abs(v[0, _SLOT_IDX["index_mcp_flex"]].item()) < _TOL
    assert abs(v[0, _SLOT_IDX["index_mcp_abd"]].item()) < _TOL


def test_mcp_pure_abduction():
    a = 0.4
    q = _quat_from_frame(*_mcp_frame(abd=a, flex=0.0))
    v = udhm_run_stage3(q, ["index_mcp"])
    assert abs(v[0, _SLOT_IDX["index_mcp_abd"]].item() * math.pi - a) < _TOL
    assert abs(v[0, _SLOT_IDX["index_mcp_flex"]].item()) < _TOL


def test_mcp_pure_flexion():
    f = 0.7
    q = _quat_from_frame(*_mcp_frame(abd=0.0, flex=f))
    v = udhm_run_stage3(q, ["index_mcp"])
    assert abs(v[0, _SLOT_IDX["index_mcp_flex"]].item() * math.pi - f) < _TOL
    assert abs(v[0, _SLOT_IDX["index_mcp_abd"]].item()) < _TOL


def test_hinge_flexion():
    b = 0.9
    # Ry(b) quaternion = (cos b/2, 0, sin b/2, 0)
    q = torch.tensor([[[math.cos(b / 2), 0.0, math.sin(b / 2), 0.0]]], dtype=torch.float64)
    v = udhm_run_stage3(q, ["index_pip"])
    assert abs(v[0, _SLOT_IDX["index_pip_flex"]].item() * math.pi - b) < _TOL


def test_thumb_mapping():
    # Dong "thumb_mcp" (the CMC) -> thumb_cmc_flex + thumb_cmc_spread.
    q = _quat_from_frame(*_mcp_frame(abd=0.3, flex=0.5))
    v = udhm_run_stage3(q, ["thumb_mcp"])
    assert abs(v[0, _SLOT_IDX["thumb_cmc_spread"]].item() * math.pi - 0.3) < _TOL
    assert abs(v[0, _SLOT_IDX["thumb_cmc_flex"]].item() * math.pi - 0.5) < _TOL


def test_shape_zero_pads_and_tip_ignored():
    labels = [
        "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
    ]
    q = torch.zeros(5, len(labels), 4, dtype=torch.float64)
    q[..., 0] = 1.0  # identity quats
    v = udhm_run_stage3(q, labels)
    assert v.shape == (5, 22)
    # Dong-unmodeled slots stay exactly zero
    assert torch.all(v[:, _SLOT_IDX["thumb_mcp_abd"]] == 0.0)
    assert torch.all(v[:, _SLOT_IDX["pinky_twist"]] == 0.0)
    # pinky never provided -> its slots stay zero
    assert torch.all(v[:, _SLOT_IDX["pinky_mcp_flex"]] == 0.0)


def test_rot6_matches_quat():
    from cross_emb.rotations import quat_wxyz_to_rot6d
    q = _quat_from_frame(*_mcp_frame(abd=0.25, flex=0.6))
    r6 = quat_wxyz_to_rot6d(q)
    vq = udhm_run_stage3(q, ["index_mcp"])
    vr = udhm_run_stage3(r6, ["index_mcp"])
    assert torch.allclose(vq, vr, atol=_TOL)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
