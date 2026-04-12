"""Dong-style hand kinematics from 21 MediaPipe landmarks.

This module implements a deterministic, frame-wise extraction of angle
parameters for the long fingers (index, middle, ring, pinky).

Input:
    points: np.ndarray with shape (21, 3), MediaPipe landmark order.

Output:
    dict[finger_name] -> {
        "MCP_flex_beta": float degrees,
        "MCP_abd_gamma": float degrees,
        "PIP_beta": float degrees,
        "DIP_beta": float degrees,
    }
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# MediaPipe index order for long fingers.
FINGERS: Dict[str, Tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# ROM maxima per long-finger joint (degrees), aligned with DECISIONS.md Entry 9
# derived from Chen et al. (2013), using the upper bound where ranges are given.
ROM_MAX: Dict[str, Dict[str, float]] = {
    "index": {"MCP_flex_beta": 90.0, "MCP_abd_gamma": 60.0, "PIP_beta": 110.0, "DIP_beta": 90.0},
    "middle": {"MCP_flex_beta": 90.0, "MCP_abd_gamma": 45.0, "PIP_beta": 110.0, "DIP_beta": 90.0},
    "ring": {"MCP_flex_beta": 90.0, "MCP_abd_gamma": 45.0, "PIP_beta": 120.0, "DIP_beta": 90.0},
    "pinky": {"MCP_flex_beta": 90.0, "MCP_abd_gamma": 50.0, "PIP_beta": 135.0, "DIP_beta": 90.0},
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v.copy()
    return v / n


def _safe_acos(x: float) -> float:
    return float(np.degrees(np.arccos(np.clip(x, -1.0, 1.0))))


def _check_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected points with shape (21, 3), got {arr.shape}")
    return arr


def compute_palm_frame(points: np.ndarray) -> np.ndarray:
    """Builds homogeneous transform T_w0 from world to palm frame {0}.

    Uses the Dong construction:
      Y0 = norm(d13 - d5)
      Z0 = norm(norm(d9 - d0) x Y0)
      X0 = norm(Y0 x Z0)
      T_w0 = [R|d0], where R=[X0 Y0 Z0] and d0 is wrist.
    """
    p = _check_points(points)
    y0 = _normalize(p[13] - p[5])
    v = _normalize(p[9] - p[0])
    z0 = _normalize(np.cross(v, y0))
    x0 = _normalize(np.cross(y0, z0))

    t_w0 = np.eye(4, dtype=float)
    t_w0[:3, :3] = np.column_stack([x0, y0, z0])
    t_w0[:3, 3] = p[0]
    return t_w0


def to_local(t_w0: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    """Transforms one 3D point to palm-local coordinates."""
    h = np.append(np.asarray(point_world, dtype=float), 1.0)
    return (np.linalg.inv(t_w0) @ h)[:3]


def compute_mcp_angles(d_mcp_local: np.ndarray, d_pip_local: np.ndarray) -> Tuple[float, float]:
    """Computes MCP flexion (beta) and abduction (gamma) in degrees.

    beta  = arccos(Z_i[2])
    gamma = arccos(Y_i[1])
    """
    x_i = _normalize(np.asarray(d_pip_local, dtype=float) - np.asarray(d_mcp_local, dtype=float))
    z0 = np.array([0.0, 0.0, 1.0], dtype=float)

    y_i = np.cross(z0, x_i)
    if np.linalg.norm(y_i) < 1e-9:
        # Degenerate case: finger axis aligned with local z.
        return 90.0, 0.0
    y_i = _normalize(y_i)
    z_i = _normalize(np.cross(x_i, y_i))

    beta = _safe_acos(z_i[2])
    gamma = _safe_acos(y_i[1])
    return beta, gamma


def compute_bend_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle in degrees at p2 formed by p1->p2 and p2->p3."""
    v1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return _safe_acos(float(np.dot(v1, v2) / (n1 * n2)))


def compute_dong_angles(points: np.ndarray, round_digits: int | None = 2) -> Dict[str, Dict[str, float]]:
    """Computes Dong-style angles for long fingers from (21,3) points."""
    p = _check_points(points)
    t_w0 = compute_palm_frame(p)
    local = [to_local(t_w0, pt) for pt in p]

    out: Dict[str, Dict[str, float]] = {}
    for finger, (mcp_i, pip_i, dip_i, tip_i) in FINGERS.items():
        beta_mcp, gamma_mcp = compute_mcp_angles(local[mcp_i], local[pip_i])
        beta_pip = compute_bend_angle(p[mcp_i], p[pip_i], p[dip_i])
        beta_dip = compute_bend_angle(p[pip_i], p[dip_i], p[tip_i])
        values = {
            "MCP_flex_beta": beta_mcp,
            "MCP_abd_gamma": gamma_mcp,
            "PIP_beta": beta_pip,
            "DIP_beta": beta_dip,
        }
        if round_digits is not None:
            values = {k: round(v, round_digits) for k, v in values.items()}
        out[finger] = values

    return out


def summarize_open_palm_likeness(
    angles: Dict[str, Dict[str, float]],
    open_ratio_threshold: float = 0.20,
    include_dip_check: bool = True,
) -> Dict[str, float | bool | Dict[str, Dict[str, float]]]:
    """ROM-normalized open-palm diagnostic.

    A finger is considered open when:
      MCP_flex / ROM_MCP <= threshold
      PIP      / ROM_PIP <= threshold
      DIP      / ROM_DIP <= threshold   (optional)

    The hand is open if all long fingers satisfy this condition.
    """
    per_finger_ratios: Dict[str, Dict[str, float]] = {}
    open_flags: Dict[str, bool] = {}

    for finger in FINGERS:
        a = angles[finger]
        rom = ROM_MAX[finger]
        ratios = {
            "MCP_flex_ratio": abs(float(a["MCP_flex_beta"])) / rom["MCP_flex_beta"],
            "PIP_ratio": abs(float(a["PIP_beta"])) / rom["PIP_beta"],
            "DIP_ratio": abs(float(a["DIP_beta"])) / rom["DIP_beta"],
            "MCP_abd_ratio": abs(float(a["MCP_abd_gamma"])) / rom["MCP_abd_gamma"],
        }
        per_finger_ratios[finger] = {k: round(v, 4) for k, v in ratios.items()}

        is_open = (ratios["MCP_flex_ratio"] <= open_ratio_threshold) and (ratios["PIP_ratio"] <= open_ratio_threshold)
        if include_dip_check:
            is_open = is_open and (ratios["DIP_ratio"] <= open_ratio_threshold)
        open_flags[finger] = is_open

    open_score = float(np.mean([1.0 if open_flags[f] else 0.0 for f in FINGERS]))
    looks_open = all(open_flags.values())

    return {
        "open_ratio_threshold": round(open_ratio_threshold, 4),
        "include_dip_check": include_dip_check,
        "open_score": round(open_score, 4),
        "looks_open_palm": looks_open,
        "finger_open_flags": open_flags,
        "per_finger_ratios": per_finger_ratios,
    }
