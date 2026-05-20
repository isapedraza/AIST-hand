"""
Canonical hand segment lengths from Engelhardt et al. (2021), Table 1.

Ratios are relative to hand length.
This module outputs lengths in meters to match Dong-style processing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Segment = Tuple[int, int]


# Engelhardt Table 1 (percent of hand length -> ratio)
ENGELHARDT_RATIOS: Dict[str, Dict[str, float]] = {
    "thumb": {"MC": 0.2338, "PP": 0.1576, "DP": 0.1152},
    "index": {"MC": 0.3482, "PP": 0.2027, "MP": 0.1175, "DP": 0.0882},
    "middle": {"MC": 0.3341, "PP": 0.2254, "MP": 0.1431, "DP": 0.0930},
    "ring": {"MC": 0.2948, "PP": 0.2105, "MP": 0.1362, "DP": 0.0954},
    "little": {"MC": 0.2711, "PP": 0.1672, "MP": 0.0956, "DP": 0.0843},
}


# MediaPipe 21-keypoint hand tree (20 links)
MEDIAPIPE_TREE_EDGES: List[Segment] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

# Current working default for MediaPipe link (0,1) = WRIST->THUMB_CMC.
# Keep it fixed for now, but expose ratio mode for future tuning.
DEFAULT_WRIST_TO_THUMB_CMC_M = 0.045
DEFAULT_WRIST_TO_THUMB_CMC_RATIO = DEFAULT_WRIST_TO_THUMB_CMC_M / 0.182


def compute_anatomical_lengths_m(hand_length_m: float = 0.182) -> Dict[str, Dict[str, float]]:
    """
    Return canonical anatomical segment lengths in meters for each finger.

    Args:
        hand_length_m: canonical hand length in meters (e.g., 0.182 for 182 mm).
    """
    if hand_length_m <= 0.0:
        raise ValueError("hand_length_m must be > 0")
    return {
        finger: {seg: hand_length_m * ratio for seg, ratio in segs.items()}
        for finger, segs in ENGELHARDT_RATIOS.items()
    }


def compute_mediapipe_link_lengths_m(
    hand_length_m: float = 0.182,
    wrist_to_thumb_cmc_m: float | None = DEFAULT_WRIST_TO_THUMB_CMC_M,
    wrist_to_thumb_cmc_ratio: float | None = None,
) -> Dict[Segment, float]:
    """
    Build canonical lengths for the 20 MediaPipe links.

    Important:
    - Engelhardt does not define MediaPipe link (0,1) = WRIST->THUMB_CMC.
    - For now, default uses a fixed metric value (0.045 m).
    - Later, you can switch to ratio mode using:
      wrist_to_thumb_cmc_ratio * hand_length_m
    """
    if wrist_to_thumb_cmc_m is not None and wrist_to_thumb_cmc_ratio is not None:
        raise ValueError(
            "Use only one mode for (0,1): wrist_to_thumb_cmc_m OR wrist_to_thumb_cmc_ratio"
        )

    lengths = compute_anatomical_lengths_m(hand_length_m)
    mp: Dict[Segment, float] = {}

    # Thumb
    if wrist_to_thumb_cmc_ratio is not None:
        if wrist_to_thumb_cmc_ratio <= 0.0:
            raise ValueError("wrist_to_thumb_cmc_ratio must be > 0 when provided")
        wrist_to_thumb_cmc_m = hand_length_m * wrist_to_thumb_cmc_ratio
    if wrist_to_thumb_cmc_m is not None:
        if wrist_to_thumb_cmc_m <= 0.0:
            raise ValueError("wrist_to_thumb_cmc_m must be > 0 when provided")
        mp[(0, 1)] = wrist_to_thumb_cmc_m
    mp[(1, 2)] = lengths["thumb"]["MC"]
    mp[(2, 3)] = lengths["thumb"]["PP"]
    mp[(3, 4)] = lengths["thumb"]["DP"]

    # Index
    mp[(0, 5)] = lengths["index"]["MC"]
    mp[(5, 6)] = lengths["index"]["PP"]
    mp[(6, 7)] = lengths["index"]["MP"]
    mp[(7, 8)] = lengths["index"]["DP"]

    # Middle
    mp[(0, 9)] = lengths["middle"]["MC"]
    mp[(9, 10)] = lengths["middle"]["PP"]
    mp[(10, 11)] = lengths["middle"]["MP"]
    mp[(11, 12)] = lengths["middle"]["DP"]

    # Ring
    mp[(0, 13)] = lengths["ring"]["MC"]
    mp[(13, 14)] = lengths["ring"]["PP"]
    mp[(14, 15)] = lengths["ring"]["MP"]
    mp[(15, 16)] = lengths["ring"]["DP"]

    # Little
    mp[(0, 17)] = lengths["little"]["MC"]
    mp[(17, 18)] = lengths["little"]["PP"]
    mp[(18, 19)] = lengths["little"]["MP"]
    mp[(19, 20)] = lengths["little"]["DP"]

    return mp


def ordered_lengths_for_tree(
    link_lengths_m: Dict[Segment, float],
    edges: List[Segment] | None = None,
) -> List[float]:
    """
    Return lengths ordered by `edges` (defaults to MEDIAPIPE_TREE_EDGES).
    """
    order = MEDIAPIPE_TREE_EDGES if edges is None else edges
    missing = [e for e in order if e not in link_lengths_m]
    if missing:
        raise KeyError(
            "Missing lengths for edges: "
            + ", ".join(f"({a},{b})" for a, b in missing)
        )
    return [link_lengths_m[e] for e in order]
