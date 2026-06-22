"""UDHM canonical contract: 22-slot named-angle vector.

This module owns the slot ordering and the UDHMContainer type.
All conversion logic (human or robot) lives in their respective subpackages
and imports UDHM22_SLOTS / SLOT_IDX from here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

UDHM22_SLOTS: list[str] = [
    "thumb_cmc_flex", "thumb_cmc_spread", "thumb_mcp_flex", "thumb_mcp_abd", "thumb_ip_flex",
    "index_mcp_abd",  "index_mcp_flex",  "index_pip_flex",  "index_dip_flex",
    "middle_mcp_abd", "middle_mcp_flex", "middle_pip_flex", "middle_dip_flex",
    "ring_mcp_abd",   "ring_mcp_flex",   "ring_pip_flex",   "ring_dip_flex",
    "pinky_mcp_abd",  "pinky_mcp_flex",  "pinky_pip_flex",  "pinky_dip_flex", "pinky_twist",
]

SLOT_IDX: dict[str, int] = {name: i for i, name in enumerate(UDHM22_SLOTS)}


@dataclass
class UDHMContainer:
    """Output of human_to_udhm and robot_to_udhm.

    angles: [B, 22] signed angles normalized by pi. Zero = straight adducted hand.
    mask:   [22] bool. True = slot is populated for this embodiment.
            Robots with fewer fingers leave missing slots at 0 AND mask=False.
    """
    angles: Tensor
    mask: Tensor
