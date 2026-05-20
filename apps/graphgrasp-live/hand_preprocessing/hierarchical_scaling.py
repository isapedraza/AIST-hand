"""
Hierarchical hand scaling in wrist-local frame.

Pipeline:
1) Extract unit directions for each MediaPipe tree edge.
2) Inject canonical link lengths (meters).
3) Reconstruct points from wrist to fingertips.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from .engelhardt_lengths import MEDIAPIPE_TREE_EDGES, Segment
except ImportError:
    from engelhardt_lengths import MEDIAPIPE_TREE_EDGES, Segment


def _validate_points(points_local: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_local, dtype=float)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected shape (21,3), got {pts.shape}")
    return pts


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Degenerate segment direction (norm={n:.3e})")
    return v / n


def compute_link_lengths(points_local: np.ndarray, edges: Iterable[Segment] = MEDIAPIPE_TREE_EDGES) -> Dict[Segment, float]:
    """
    Measure current link lengths from a 21x3 point set.
    """
    pts = _validate_points(points_local)
    lengths: Dict[Segment, float] = {}
    for p, c in edges:
        lengths[(p, c)] = float(np.linalg.norm(pts[c] - pts[p]))
    return lengths


def extract_unit_directions(
    points_local: np.ndarray,
    edges: Iterable[Segment] = MEDIAPIPE_TREE_EDGES,
    root_index: int = 0,
) -> Dict[Segment, np.ndarray]:
    """
    Compute per-link unit vectors from wrist-local or any 21x3 hand points.

    Points are first translated so root_index sits at origin.
    """
    pts = _validate_points(points_local)
    centered = pts - pts[root_index]
    dirs: Dict[Segment, np.ndarray] = {}
    for p, c in edges:
        dirs[(p, c)] = _unit(centered[c] - centered[p])
    return dirs


def reconstruct_from_directions(
    unit_dirs: Dict[Segment, np.ndarray],
    link_lengths_m: Dict[Segment, float],
    edges: Iterable[Segment] = MEDIAPIPE_TREE_EDGES,
    root_index: int = 0,
    root_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_points: int = 21,
) -> np.ndarray:
    """
    Rebuild points hierarchically using direction * canonical length for each link.
    """
    points = np.zeros((n_points, 3), dtype=float)
    points[root_index] = np.asarray(root_position, dtype=float)

    for p, c in edges:
        if (p, c) not in unit_dirs:
            raise KeyError(f"Missing unit direction for edge {(p, c)}")
        if (p, c) not in link_lengths_m:
            raise KeyError(f"Missing canonical length for edge {(p, c)}")
        length = float(link_lengths_m[(p, c)])
        if length <= 0.0:
            raise ValueError(f"Invalid length for edge {(p, c)}: {length}")
        points[c] = points[p] + unit_dirs[(p, c)] * length

    return points


def hierarchical_scale_keypoints(
    points_local: np.ndarray,
    link_lengths_m: Dict[Segment, float],
    edges: Iterable[Segment] = MEDIAPIPE_TREE_EDGES,
    root_index: int = 0,
) -> np.ndarray:
    """
    Full pipeline: directions from input + canonical lengths -> scaled points.

    Returns 21x3 points in the same local frame (root at origin).
    """
    dirs = extract_unit_directions(points_local, edges=edges, root_index=root_index)
    return reconstruct_from_directions(
        unit_dirs=dirs,
        link_lengths_m=link_lengths_m,
        edges=edges,
        root_index=root_index,
        root_position=(0.0, 0.0, 0.0),
        n_points=21,
    )
