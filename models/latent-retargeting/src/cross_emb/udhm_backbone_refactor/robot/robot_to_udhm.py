"""robot qpos -> UDHM[22], direct (no Dong).

The robot ALREADY has joint angles (qpos). So the common representation is just:
place each joint's angle, with its sign, into its UDHM slot, normalized by pi.
Missing slots stay 0. This is what fixes "Dong breaks on robot topology": we never
run Dong on the robot -- qpos IS the angle.

Signs + slot come from the adapter YAML (robot/adapters/{robot}_udhm_adapter.yaml),
built by primitives.compute_signs (4 fingers) + annotate.py (thumb / Barrett).
"""
from __future__ import annotations

import math

import torch

from cross_emb.udhm_backbone_refactor.core.udhm import SLOT_IDX, UDHM22_SLOTS


def robot_to_udhm(qpos: torch.Tensor, adapter: dict, joint_order: list[str]) -> torch.Tensor:
    """[B, J] qpos -> [B, 22] UDHM angles (radians / pi).

    qpos:        robot joint angles, columns in `joint_order`.
    adapter:     parsed adapter YAML; adapter["joints"][name] = {slot, sign}.
    joint_order: list mapping qpos column index -> joint name (e.g. RobotLoader.chain_joint_names).
    """
    out = qpos.new_zeros(qpos.shape[0], len(UDHM22_SLOTS))
    col_of = {name: i for i, name in enumerate(joint_order)}
    for name, e in adapter["joints"].items():
        if name not in col_of or e.get("sign") is None:
            continue
        out[:, SLOT_IDX[e["slot"]]] = float(e["sign"]) * qpos[:, col_of[name]] / math.pi
    return out
