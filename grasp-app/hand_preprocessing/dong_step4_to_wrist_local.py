"""
Paso 4 (Dong-lite): transformar world -> wrist local.

Basado en Dong & Payandeh (2025):
- Eq. (5)-(7): construccion de ejes X0, Y0, Z0
- Eq. (9): T0w = [R0w d0w; 0 1]
- Eq. (16): d_i^0 = (T0w)^(-1) d_i^w

Controles:
- SPACE: captura y muestra T0w, T0w^-1 y 21 puntos en marco local
- Q / ESC: salir
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Reduce ruido de logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)


def _open_capture(camera_index: int | str):
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if os.name == "posix" else [cv2.CAP_ANY]
    last_cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
        except TypeError:
            cap = cv2.VideoCapture(camera_index)
        if cap and cap.isOpened():
            return cap
        last_cap = cap
        if cap is not None:
            cap.release()
    return last_cap


def _camera_source(raw: str) -> int | str:
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Vector degenerado (norm={n:.3e})")
    return v / n


def _world_points(hand_world_landmarks) -> np.ndarray:
    return np.asarray([(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark], dtype=float)


def _extract_handedness(
    result,
    input_is_mirrored: bool = False,
) -> tuple[str | None, float | None]:
    """
    Read MediaPipe handedness output for the first detected hand.
    Returns (label, score), e.g. ("Left", 0.98). If unavailable, returns (None, None).
    MediaPipe handedness assumes mirrored selfie input. For non-mirrored camera/image input,
    Left/Right labels must be swapped.
    """
    handed = getattr(result, "multi_handedness", None)
    if not handed:
        return None, None
    try:
        cls = handed[0].classification[0]
    except (AttributeError, IndexError, TypeError):
        return None, None

    raw_label = getattr(cls, "label", None)
    label = str(raw_label).strip() if raw_label is not None else None
    if not label:
        label = None
    else:
        low = label.lower()
        if not input_is_mirrored:
            if low == "left":
                label = "Right"
            elif low == "right":
                label = "Left"
        elif low in ("left", "right"):
            label = low.capitalize()

    raw_score = getattr(cls, "score", None)
    score = float(raw_score) if raw_score is not None else None
    return label, score


def _extract_handedness_raw(result) -> tuple[str | None, float | None]:
    """
    Pure MediaPipe handedness: no swap, no mirror assumptions.
    """
    handed = getattr(result, "multi_handedness", None)
    if not handed:
        return None, None
    try:
        cls = handed[0].classification[0]
    except (AttributeError, IndexError, TypeError):
        return None, None

    raw_label = getattr(cls, "label", None)
    label = str(raw_label).strip() if raw_label is not None else None
    if label:
        low = label.lower()
        if low in ("left", "right"):
            label = low.capitalize()
    else:
        label = None

    raw_score = getattr(cls, "score", None)
    score = float(raw_score) if raw_score is not None else None
    return label, score


def _canonicalize_world_points_to_right(
    points_w: np.ndarray,
    hand_label: str | None,
) -> tuple[np.ndarray, str, bool]:
    """
    Canonical convention:
    - Right hand: keep as-is.
    - Left hand: reflect X in world frame so downstream kinematics uses right-hand convention.
    Returns (points_w_canonical, canonical_hand_label, reflected).
    """
    if points_w.shape != (21, 3):
        raise ValueError(f"points_w shape invalido: {points_w.shape} (esperado (21,3))")

    label = (hand_label or "").strip().lower()
    if label == "left":
        out = points_w.copy()
        out[:, 0] *= -1.0
        return out, "Right", True
    return points_w, "Right", False


def _compute_wrist_frame(points_w: np.ndarray) -> dict:
    # Dong Eq. (5)-(7):
    # Y0 = normalize(d13 - d5)
    # Z0 = normalize(normalize(d9 - d0) x Y0)
    # X0 = Y0 x Z0
    d0 = points_w[0]
    d5 = points_w[5]
    d9 = points_w[9]
    d13 = points_w[13]

    v_13_5 = d13 - d5
    v_9_0 = d9 - d0

    y0 = _normalize(v_13_5)
    v_9_0_hat = _normalize(v_9_0)
    z0 = _normalize(np.cross(v_9_0_hat, y0))
    x0 = _normalize(np.cross(y0, z0))

    r0w = np.column_stack((x0, y0, z0))

    # Dong Eq. (8):
    # beta  = asin(-X0_z)
    # gamma = atan2(X0_y, X0_x)
    # alpha = atan2(Y0_z, Z0_z)
    beta = float(np.arcsin(np.clip(-x0[2], -1.0, 1.0)))
    gamma = float(np.arctan2(x0[1], x0[0]))
    alpha = float(np.arctan2(y0[2], z0[2]))

    return {
        "d0": d0,
        "d5": d5,
        "d9": d9,
        "d13": d13,
        "v_13_5": v_13_5,
        "v_9_0": v_9_0,
        "v_9_0_hat": v_9_0_hat,
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "r0w": r0w,
        "alpha_rad": alpha,
        "beta_rad": beta,
        "gamma_rad": gamma,
        "alpha_deg": float(np.degrees(alpha)),
        "beta_deg": float(np.degrees(beta)),
        "gamma_deg": float(np.degrees(gamma)),
        "det_r0w": float(np.linalg.det(r0w)),
        "orth_err": float(np.linalg.norm(r0w.T @ r0w - np.eye(3))),
    }


def _build_t0w(r0w: np.ndarray, d0w: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = r0w
    t[:3, 3] = d0w
    return t


def _transform_world_to_local(points_w: np.ndarray, t0w_inv: np.ndarray) -> np.ndarray:
    ones = np.ones((points_w.shape[0], 1), dtype=float)
    pw_h = np.hstack((points_w, ones))  # [21,4]
    p0_h = (t0w_inv @ pw_h.T).T
    return p0_h[:, :3]


def _transform_local_to_world(points_0: np.ndarray, t0w: np.ndarray) -> np.ndarray:
    ones = np.ones((points_0.shape[0], 1), dtype=float)
    p0_h = np.hstack((points_0, ones))  # [21,4]
    pw_h = (t0w @ p0_h.T).T
    return pw_h[:, :3]


PALM_ROOT_IDS = (1, 5, 9, 13, 17)
FINGER_LINKS = (
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


# ============================================== #
# BLOQUE 1: Transformacion Global a Local (W->0) #
# ============================================== #
def _block_1_world_to_wrist_local(points_w: np.ndarray) -> dict:
    """
    Bloque 1: Transformacion Global a Local (W -> {0}).
    Dinamico por frame.
    """
    wf = _compute_wrist_frame(points_w)
    r0w = wf["r0w"]
    d0w = wf["d0"]
    t0w = _build_t0w(r0w, d0w)
    t0w_inv = np.linalg.inv(t0w)
    points_0 = _transform_world_to_local(points_w, t0w_inv)
    return {
        "wf": wf,
        "t0w": t0w,
        "t0w_inv": t0w_inv,
        "points_0": points_0,
    }


# ============================================================= #
# BLOQUE 2: Parametros Fijos Anatomicos (palma + long. dedos) #
# ============================================================= #
def _compute_palm_base_params(points_w: np.ndarray, points_0: np.ndarray, eps: float = 1e-8) -> dict:
    """
    Dong sec. 4.1 (Eq. 13, 16, 17):
    - Base lengths l0,i for i in {1,5,9,13,17}
    - Local root points d0i from Eq. (16), keeping measured 3D values (x,y,z)
    - Base angles theta_i from Eq. (17), using only (x,y) from d0i
    Notes:
    - Eq. (17) uses only XY components from d0i.
    - We keep measured Z from Eq. (16) for diagnostics in checks.
    """
    if points_w.shape != (21, 3):
        raise ValueError(f"points_w shape invalido: {points_w.shape} (esperado (21,3))")
    if points_0.shape != (21, 3):
        raise ValueError(f"points_0 shape invalido: {points_0.shape} (esperado (21,3))")

    d0w = points_w[0]
    lengths_m: dict[int, float] = {}
    thetas_rad: dict[int, float] = {}
    root_positions_from_eq16: dict[int, np.ndarray] = {}
    world_vectors: dict[int, np.ndarray] = {}
    norm_diffs: dict[int, float] = {}
    abs_local_z: dict[int, float] = {}

    for i in PALM_ROOT_IDS:
        d_i_w = points_w[i]
        d0i = points_0[i]

        # Eq. (13): l0,i = ||d_i^w - d_0^w||
        l0i = float(np.linalg.norm(d_i_w - d0w))

        # Eq. (17): theta_i = atan2(d0i_y, d0i_x) -> z is NOT used.
        xy_norm = float(np.hypot(d0i[0], d0i[1]))
        if xy_norm < eps:
            raise ValueError(f"Frame degenerado: kp{i} con proyeccion XY casi cero; theta{i} indefinido")
        theta_i = float(np.arctan2(d0i[1], d0i[0]))

        lengths_m[i] = l0i
        thetas_rad[i] = theta_i
        root_positions_from_eq16[i] = d0i
        world_vectors[i] = d_i_w - d0w
        norm_diffs[i] = abs(float(np.linalg.norm(d0i)) - l0i)
        abs_local_z[i] = abs(float(d0i[2]))

    thetas_deg = {i: float(np.degrees(thetas_rad[i])) for i in PALM_ROOT_IDS}
    return {
        "root_ids": PALM_ROOT_IDS,
        "lengths_m": lengths_m,
        "thetas_rad": thetas_rad,
        "thetas_deg": thetas_deg,
        "root_positions_from_eq16": root_positions_from_eq16,
        "world_vectors": world_vectors,
        "norm_diffs": norm_diffs,
        "abs_local_z": abs_local_z,
        "max_norm_diff": float(max(norm_diffs.values())),
        "max_abs_local_z": float(max(abs_local_z.values())),
    }


def _compute_finger_base_lengths(points_0: np.ndarray, eps: float = 1e-8) -> dict:
    """
    Dong finger base-length parameters (15 fixed link lengths):
    {l1,2..l3,4, l5,6..l7,8, ..., l17,18..l19,20}
    measured from observed wrist-local landmarks d_i^0.
    """
    if points_0.shape != (21, 3):
        raise ValueError(f"points_0 shape invalido: {points_0.shape} (esperado (21,3))")

    lengths_m: dict[tuple[int, int], float] = {}
    segments: dict[tuple[int, int], np.ndarray] = {}
    for a, b in FINGER_LINKS:
        seg = points_0[b] - points_0[a]
        l_ab = float(np.linalg.norm(seg))
        if l_ab < eps:
            raise ValueError(f"Frame degenerado: longitud l{a},{b} casi cero")
        segments[(a, b)] = seg
        lengths_m[(a, b)] = l_ab

    return {
        "links": FINGER_LINKS,
        "segments": segments,
        "lengths_m": lengths_m,
    }


def _freeze_palm_base_params(samples: list[dict]) -> dict:
    """
    Freeze palm-base params using calibration samples:
    - l0,i (Eq. 13)
    - theta_i (Eq. 17)
    """
    if not samples:
        raise ValueError("No hay muestras para congelar parametros de palma base")

    lengths_m: dict[int, float] = {}
    thetas_rad: dict[int, float] = {}
    eq15_positions: dict[int, np.ndarray] = {}

    for i in PALM_ROOT_IDS:
        l_vals = np.asarray([s["lengths_m"][i] for s in samples], dtype=float)
        t_vals = np.asarray([s["thetas_rad"][i] for s in samples], dtype=float)
        t_vals_unwrapped = np.unwrap(t_vals)

        l_med = float(np.median(l_vals))
        t_med = float(np.median(t_vals_unwrapped))
        t_med = float((t_med + np.pi) % (2.0 * np.pi) - np.pi)

        lengths_m[i] = l_med
        thetas_rad[i] = t_med
        eq15_positions[i] = np.array(
            [l_med * np.cos(t_med), l_med * np.sin(t_med), 0.0],
            dtype=float,
        )

    thetas_deg = {i: float(np.degrees(thetas_rad[i])) for i in PALM_ROOT_IDS}
    return {
        "root_ids": PALM_ROOT_IDS,
        "lengths_m": lengths_m,
        "thetas_rad": thetas_rad,
        "thetas_deg": thetas_deg,
        "eq15_positions": eq15_positions,
        "n_samples": len(samples),
        "is_frozen": True,
    }


def _freeze_finger_base_lengths(samples: list[dict]) -> dict:
    """
    Freeze the 15 finger link lengths using calibration samples (median per link).
    """
    if not samples:
        raise ValueError("No hay muestras para congelar longitudes de dedos")

    lengths_m: dict[tuple[int, int], float] = {}
    for a, b in FINGER_LINKS:
        vals = np.asarray([s["lengths_m"][(a, b)] for s in samples], dtype=float)
        lengths_m[(a, b)] = float(np.median(vals))

    return {
        "links": FINGER_LINKS,
        "lengths_m": lengths_m,
        "n_samples": len(samples),
        "is_frozen": True,
    }


def _block_2_palm_base_fixed(points_w: np.ndarray, points_0: np.ndarray, freeze_state: dict) -> dict:
    """
    Bloque 2: Parametros fijos anatomicos (calibration).
    - Siempre calcula parametros observados del frame actual.
    - Congela l0,i y theta_i de palma base tras N capturas validas.
    - Congela las 15 longitudes de dedos tras N capturas validas.
    """
    observed = _compute_palm_base_params(points_w=points_w, points_0=points_0)
    observed_finger = _compute_finger_base_lengths(points_0=points_0)

    target_frames = int(freeze_state.get("target_frames", 1))
    if target_frames < 1:
        target_frames = 1
    freeze_state["target_frames"] = target_frames

    samples: list[dict] = freeze_state.setdefault("samples", [])
    finger_samples: list[dict] = freeze_state.setdefault("finger_samples", [])
    frozen_palm = freeze_state.get("frozen_palm")
    frozen_finger = freeze_state.get("frozen_finger")
    had_frozen = frozen_palm is not None
    had_frozen_finger = frozen_finger is not None

    if frozen_palm is None:
        samples.append(observed)
        if len(samples) >= target_frames:
            frozen_palm = _freeze_palm_base_params(samples=samples)
            freeze_state["frozen_palm"] = frozen_palm

    if frozen_finger is None:
        finger_samples.append(observed_finger)
        if len(finger_samples) >= target_frames:
            frozen_finger = _freeze_finger_base_lengths(samples=finger_samples)
            freeze_state["frozen_finger"] = frozen_finger

    is_frozen = frozen_palm is not None
    is_finger_frozen = frozen_finger is not None
    just_froze = is_frozen and not had_frozen
    just_froze_finger = is_finger_frozen and not had_frozen_finger
    active_palm = frozen_palm if is_frozen else observed
    active_finger = frozen_finger if is_finger_frozen else observed_finger

    return {
        "observed_palm": observed,
        "observed_finger": observed_finger,
        "frozen_palm": frozen_palm,
        "frozen_finger": frozen_finger,
        "active_palm": active_palm,
        "active_finger": active_finger,
        "is_frozen": is_frozen,
        "is_finger_frozen": is_finger_frozen,
        "just_froze": just_froze,
        "just_froze_finger": just_froze_finger,
        "calib_count": len(samples),
        "finger_calib_count": len(finger_samples),
        "calib_target": target_frames,
    }


# ================================================================ #
# BLOQUE 3: Parametros Dinamicos Articulares (per-frame kinematics) #
# ================================================================ #
def _block_3_compute_mcp_first_layer(points_0: np.ndarray, palm_base: dict, eps: float = 1e-8) -> dict:
    """
    Dong sec. 4.2 first layer (Eq. 19-24), for MCP roots i in {1,5,9,13,17}.
    Strict Dong usage:
    - Eq. (20)-(22): orientation from observed wrist-local points (3D).
    - Eq. (23): translation d_i^0 from Eq. (15), i.e., planar with z=0.
    - Eq. (20): X_i^0 from segment i -> i+1 in wrist-local coordinates
    - Eq. (21): Y_i^0 from Z_0 x X_i^0
    - Eq. (22): Z_i^0 from X_i^0 x Y_i^0
    - Eq. (19): R_i^0 = [X_i^0 Y_i^0 Z_i^0]
    - Eq. (23): T_i^0 = [R_i^0 d_i^0; 0 1]
    - Eq. (24): beta_i = arccos((Z_i^0)_z), gamma_i = arccos((Y_i^0)_y)
    """
    if points_0.shape != (21, 3):
        raise ValueError(f"points_0 shape invalido: {points_0.shape} (esperado (21,3))")
    if not isinstance(palm_base, dict):
        raise ValueError("palm_base invalido: se esperaba dict con Eq.13/Eq.17")

    z0 = np.array([0.0, 0.0, 1.0], dtype=float)
    root_ids = PALM_ROOT_IDS
    child_ids = {i: i + 1 for i in root_ids}

    x_axes: dict[int, np.ndarray] = {}
    y_axes: dict[int, np.ndarray] = {}
    z_axes: dict[int, np.ndarray] = {}
    rot_mats: dict[int, np.ndarray] = {}
    tf_mats: dict[int, np.ndarray] = {}
    betas_rad: dict[int, float] = {}
    gammas_rad: dict[int, float] = {}
    betas_deg: dict[int, float] = {}
    gammas_deg: dict[int, float] = {}
    det_r: dict[int, float] = {}
    orth_err: dict[int, float] = {}
    segment_vecs: dict[int, np.ndarray] = {}
    d_i0_obs: dict[int, np.ndarray] = {}
    d_i0_planar: dict[int, np.ndarray] = {}
    trans_planar_delta_norm: dict[int, float] = {}

    for i in root_ids:
        j = child_ids[i]
        d_i0 = points_0[i]
        d_j0 = points_0[j]

        # Eq. (20): unit X axis points to the next joint.
        seg = d_j0 - d_i0
        x_i0 = _normalize(seg, eps=eps)

        # Eq. (21): unit Y axis from Z0 x X.
        y_num = np.cross(z0, x_i0)
        if float(np.linalg.norm(y_num)) < eps:
            raise ValueError(f"Frame degenerado MCP i={i}: Z0 x X{i} casi cero")
        y_i0 = _normalize(y_num, eps=eps)

        # Eq. (22): unit Z axis by right-hand rule.
        z_i0 = _normalize(np.cross(x_i0, y_i0), eps=eps)

        # Eq. (19): rotation matrix with columns [X Y Z].
        r_i0 = np.column_stack((x_i0, y_i0, z_i0))

        # Eq. (23): homogeneous transform wrt wrist frame {0} using Eq. (15) d_i^0.
        # d_i^0 = [l0,i cos(theta_i), l0,i sin(theta_i), 0]
        l0i = float(palm_base["lengths_m"][i])
        theta_i = float(palm_base["thetas_rad"][i])
        d_i0_t = np.array(
            [l0i * np.cos(theta_i), l0i * np.sin(theta_i), 0.0],
            dtype=float,
        )
        t_i0 = np.eye(4, dtype=float)
        t_i0[:3, :3] = r_i0
        t_i0[:3, 3] = d_i0_t

        # Eq. (24): MCP angle pair.
        beta_i = float(np.arccos(np.clip(z_i0[2], -1.0, 1.0)))
        gamma_i = float(np.arccos(np.clip(y_i0[1], -1.0, 1.0)))

        x_axes[i] = x_i0
        y_axes[i] = y_i0
        z_axes[i] = z_i0
        rot_mats[i] = r_i0
        tf_mats[i] = t_i0
        betas_rad[i] = beta_i
        gammas_rad[i] = gamma_i
        betas_deg[i] = float(np.degrees(beta_i))
        gammas_deg[i] = float(np.degrees(gamma_i))
        det_r[i] = float(np.linalg.det(r_i0))
        orth_err[i] = float(np.linalg.norm(r_i0.T @ r_i0 - np.eye(3)))
        segment_vecs[i] = seg
        d_i0_obs[i] = d_i0
        d_i0_planar[i] = d_i0_t
        trans_planar_delta_norm[i] = float(np.linalg.norm(d_i0_t - d_i0))

    max_orth_err = float(max(orth_err.values()))
    min_det = float(min(det_r.values()))
    max_det = float(max(det_r.values()))
    max_trans_planar_delta = float(max(trans_planar_delta_norm.values()))

    return {
        "root_ids": root_ids,
        "child_ids": child_ids,
        "z0": z0,
        "segment_vecs": segment_vecs,
        "x_axes": x_axes,
        "y_axes": y_axes,
        "z_axes": z_axes,
        "rot_mats": rot_mats,
        "tf_mats": tf_mats,
        "betas_rad": betas_rad,
        "gammas_rad": gammas_rad,
        "betas_deg": betas_deg,
        "gammas_deg": gammas_deg,
        "det_r": det_r,
        "orth_err": orth_err,
        "d_i0_obs": d_i0_obs,
        "d_i0_planar": d_i0_planar,
        "trans_planar_delta_norm": trans_planar_delta_norm,
        "max_orth_err": max_orth_err,
        "min_det": min_det,
        "max_det": max_det,
        "max_trans_planar_delta": max_trans_planar_delta,
    }


def _rotation_y(beta: float) -> np.ndarray:
    c = float(np.cos(beta))
    s = float(np.sin(beta))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _block_3_compute_second_third_layers(
    points_0: np.ndarray,
    mcp: dict,
    finger_base: dict | None = None,
    eps: float = 1e-8,
) -> dict:
    """
    Dong sec. 4.2 next layers (Eq. 29-36):
    - Second layer (PIP): beta_j, R_j^{j-1}, T_j^{j-1}
    - Third layer  (DIP): beta_{j+1}, R_{j+1}^{j}, T_{j+1}^{j}
    with j in {2,6,10,14,18}.
    """
    if points_0.shape != (21, 3):
        raise ValueError(f"points_0 shape invalido: {points_0.shape} (esperado (21,3))")
    if not isinstance(mcp, dict) or "tf_mats" not in mcp:
        raise ValueError("mcp invalido: se esperaba salida de first layer con T_i^0")

    second_ids = tuple(i + 1 for i in PALM_ROOT_IDS)  # {2,6,10,14,18}
    parent_ids = {j: j - 1 for j in second_ids}  # MCP parent
    third_ids = {j: j + 1 for j in second_ids}  # DIP
    fourth_ids = {j: j + 2 for j in second_ids}  # TIP

    d_j_jm1: dict[int, np.ndarray] = {}
    d_j1_jm1: dict[int, np.ndarray] = {}
    d_j2_jm1: dict[int, np.ndarray] = {}
    x_j_jm1: dict[int, np.ndarray] = {}
    x_jm1_jm1: dict[int, np.ndarray] = {}
    x_j1_jm1: dict[int, np.ndarray] = {}

    betas_j_rad: dict[int, float] = {}
    betas_j_deg: dict[int, float] = {}
    betas_j1_rad: dict[int, float] = {}
    betas_j1_deg: dict[int, float] = {}

    rot_j_jm1: dict[int, np.ndarray] = {}
    rot_j1_j: dict[int, np.ndarray] = {}
    tf_j_jm1: dict[int, np.ndarray] = {}
    tf_j1_j: dict[int, np.ndarray] = {}

    lengths_jm1_j_model: dict[int, float] = {}
    lengths_j_j1_model: dict[int, float] = {}
    d_j_jm1_model: dict[int, np.ndarray] = {}
    d_j1_j_model: dict[int, np.ndarray] = {}

    det_r_j: dict[int, float] = {}
    det_r_j1: dict[int, float] = {}
    orth_err_j: dict[int, float] = {}
    orth_err_j1: dict[int, float] = {}

    for j in second_ids:
        jm1 = parent_ids[j]
        j1 = third_ids[j]
        j2 = fourth_ids[j]

        t_jm1_0 = mcp["tf_mats"][jm1]
        t_jm1_0_inv = np.linalg.inv(t_jm1_0)

        def _to_parent_frame(p0: np.ndarray) -> np.ndarray:
            p_h = np.array([p0[0], p0[1], p0[2], 1.0], dtype=float)
            out = t_jm1_0_inv @ p_h
            return out[:3]

        # Eq. (30), Eq. (32): points in parent (j-1) frame.
        dj = _to_parent_frame(points_0[j])
        dj1 = _to_parent_frame(points_0[j1])
        dj2 = _to_parent_frame(points_0[j2])

        # Eq. (29): second-layer angle from relative orientation in parent frame.
        xj = dj1 - dj
        xjm1 = dj  # d_{j-1}^{j-1}=0 -> vector from parent origin to joint j
        n_xj = float(np.linalg.norm(xj))
        n_xjm1 = float(np.linalg.norm(xjm1))
        if n_xj < eps or n_xjm1 < eps:
            raise ValueError(f"Frame degenerado second layer j={j}: vector casi cero")
        cos_beta_j = float(np.dot(xj, xjm1) / (n_xj * n_xjm1))
        beta_j = float(np.arccos(np.clip(cos_beta_j, -1.0, 1.0)))

        # Eq. (31): third-layer angle from relative orientation in parent frame.
        xj1 = dj2 - dj1
        n_xj1 = float(np.linalg.norm(xj1))
        if n_xj1 < eps:
            raise ValueError(f"Frame degenerado third layer j={j}: vector casi cero")
        cos_beta_j1 = float(np.dot(xj1, xj) / (n_xj1 * n_xj))
        beta_j1 = float(np.arccos(np.clip(cos_beta_j1, -1.0, 1.0)))

        # Eq. (33), Eq. (34): rotations around Y.
        r_j_jm1 = _rotation_y(beta_j)
        r_j1_j = _rotation_y(beta_j1)

        # Eq. (35): translations by fixed link lengths.
        key_1 = (jm1, j)
        key_2 = (j, j1)
        if finger_base is not None and "lengths_m" in finger_base:
            l_jm1_j = float(finger_base["lengths_m"][key_1])
            l_j_j1 = float(finger_base["lengths_m"][key_2])
        else:
            l_jm1_j = n_xjm1
            l_j_j1 = n_xj

        d_j_model = np.array([l_jm1_j, 0.0, 0.0], dtype=float)
        d_j1_model = np.array([l_j_j1, 0.0, 0.0], dtype=float)

        # Eq. (36): homogeneous transforms.
        t_j_jm1 = np.eye(4, dtype=float)
        t_j_jm1[:3, :3] = r_j_jm1
        t_j_jm1[:3, 3] = d_j_model

        t_j1_j = np.eye(4, dtype=float)
        t_j1_j[:3, :3] = r_j1_j
        t_j1_j[:3, 3] = d_j1_model

        d_j_jm1[j] = dj
        d_j1_jm1[j] = dj1
        d_j2_jm1[j] = dj2
        x_j_jm1[j] = xj
        x_jm1_jm1[j] = xjm1
        x_j1_jm1[j] = xj1

        betas_j_rad[j] = beta_j
        betas_j_deg[j] = float(np.degrees(beta_j))
        betas_j1_rad[j] = beta_j1
        betas_j1_deg[j] = float(np.degrees(beta_j1))

        rot_j_jm1[j] = r_j_jm1
        rot_j1_j[j] = r_j1_j
        tf_j_jm1[j] = t_j_jm1
        tf_j1_j[j] = t_j1_j

        lengths_jm1_j_model[j] = l_jm1_j
        lengths_j_j1_model[j] = l_j_j1
        d_j_jm1_model[j] = d_j_model
        d_j1_j_model[j] = d_j1_model

        det_r_j[j] = float(np.linalg.det(r_j_jm1))
        det_r_j1[j] = float(np.linalg.det(r_j1_j))
        orth_err_j[j] = float(np.linalg.norm(r_j_jm1.T @ r_j_jm1 - np.eye(3)))
        orth_err_j1[j] = float(np.linalg.norm(r_j1_j.T @ r_j1_j - np.eye(3)))

    max_orth_err_j = float(max(orth_err_j.values()))
    max_orth_err_j1 = float(max(orth_err_j1.values()))

    return {
        "second_ids": second_ids,
        "parent_ids": parent_ids,
        "third_ids": third_ids,
        "fourth_ids": fourth_ids,
        "d_j_jm1": d_j_jm1,
        "d_j1_jm1": d_j1_jm1,
        "d_j2_jm1": d_j2_jm1,
        "x_j_jm1": x_j_jm1,
        "x_jm1_jm1": x_jm1_jm1,
        "x_j1_jm1": x_j1_jm1,
        "betas_j_rad": betas_j_rad,
        "betas_j_deg": betas_j_deg,
        "betas_j1_rad": betas_j1_rad,
        "betas_j1_deg": betas_j1_deg,
        "rot_j_jm1": rot_j_jm1,
        "rot_j1_j": rot_j1_j,
        "tf_j_jm1": tf_j_jm1,
        "tf_j1_j": tf_j1_j,
        "lengths_jm1_j_model": lengths_jm1_j_model,
        "lengths_j_j1_model": lengths_j_j1_model,
        "d_j_jm1_model": d_j_jm1_model,
        "d_j1_j_model": d_j1_j_model,
        "det_r_j": det_r_j,
        "det_r_j1": det_r_j1,
        "orth_err_j": orth_err_j,
        "orth_err_j1": orth_err_j1,
        "max_orth_err_j": max_orth_err_j,
        "max_orth_err_j1": max_orth_err_j1,
    }


def _block_3_compute_last_layer(
    second_third: dict,
    finger_base: dict | None = None,
    eps: float = 1e-8,
) -> dict:
    """
    Dong sec. 4.2 last layer (Eq. 45):
    - Fingertip frame has no DOF.
    - T_{j+2}^{j+1} = [I, d_{j+2}^{j+1}; 0, 1]
      with d_{j+2}^{j+1} = [l_{j+1,j+2}, 0, 0]^T.
    """
    if not isinstance(second_third, dict) or "second_ids" not in second_third:
        raise ValueError("second_third invalido: se esperaba salida de Eq.29-36")

    second_ids = second_third["second_ids"]  # {2,6,10,14,18}
    third_ids = second_third["third_ids"]  # j -> j+1 (DIP)
    fourth_ids = second_third["fourth_ids"]  # j -> j+2 (TIP)

    rot_j2_j1: dict[int, np.ndarray] = {}
    tf_j2_j1: dict[int, np.ndarray] = {}
    lengths_j1_j2_model: dict[int, float] = {}
    d_j2_j1_model: dict[int, np.ndarray] = {}
    det_r_j2: dict[int, float] = {}
    orth_err_j2: dict[int, float] = {}

    for j in second_ids:
        j1 = third_ids[j]
        j2 = fourth_ids[j]
        key = (j1, j2)

        if finger_base is not None and "lengths_m" in finger_base and key in finger_base["lengths_m"]:
            l_j1_j2 = float(finger_base["lengths_m"][key])
        else:
            # Fallback: observed DIP->TIP segment from the same parent-frame data.
            x_j1 = second_third["x_j1_jm1"][j]
            l_j1_j2 = float(np.linalg.norm(x_j1))

        if l_j1_j2 < eps:
            raise ValueError(f"Frame degenerado last layer j={j}: longitud l{j1},{j2} casi cero")

        # Eq. (45): no rotation at fingertip.
        r_j2_j1 = np.eye(3, dtype=float)
        d_j2 = np.array([l_j1_j2, 0.0, 0.0], dtype=float)
        t_j2_j1 = np.eye(4, dtype=float)
        t_j2_j1[:3, :3] = r_j2_j1
        t_j2_j1[:3, 3] = d_j2

        rot_j2_j1[j] = r_j2_j1
        tf_j2_j1[j] = t_j2_j1
        lengths_j1_j2_model[j] = l_j1_j2
        d_j2_j1_model[j] = d_j2
        det_r_j2[j] = float(np.linalg.det(r_j2_j1))
        orth_err_j2[j] = float(np.linalg.norm(r_j2_j1.T @ r_j2_j1 - np.eye(3)))

    max_orth_err_j2 = float(max(orth_err_j2.values()))

    return {
        "second_ids": second_ids,
        "third_ids": third_ids,
        "fourth_ids": fourth_ids,
        "rot_j2_j1": rot_j2_j1,
        "tf_j2_j1": tf_j2_j1,
        "lengths_j1_j2_model": lengths_j1_j2_model,
        "d_j2_j1_model": d_j2_j1_model,
        "det_r_j2": det_r_j2,
        "orth_err_j2": orth_err_j2,
        "max_orth_err_j2": max_orth_err_j2,
    }


def _rotation_matrix_to_quaternion_dong(r: np.ndarray) -> dict:
    """
    Dong sec. 5.4 (Eq. 58-60): convert 3x3 rotation matrix to quaternion (w, x, y, z).
    """
    if r.shape != (3, 3):
        raise ValueError(f"R shape invalida: {r.shape} (esperado (3,3))")

    m00, m01, m02 = float(r[0, 0]), float(r[0, 1]), float(r[0, 2])
    m10, m11, m12 = float(r[1, 0]), float(r[1, 1]), float(r[1, 2])
    m20, m21, m22 = float(r[2, 0]), float(r[2, 1]), float(r[2, 2])

    # Eq. (59)
    qw = 0.5 * np.sqrt(max(0.0, 1.0 + m00 + m11 + m22))
    qx_abs = 0.5 * np.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
    qy_abs = 0.5 * np.sqrt(max(0.0, 1.0 - m00 + m11 - m22))
    qz_abs = 0.5 * np.sqrt(max(0.0, 1.0 - m00 - m11 + m22))

    # Eq. (60): sign adjustments from off-diagonal terms.
    def _sign_nonzero(v: float) -> float:
        return -1.0 if v < 0.0 else 1.0

    sign_term_x = m21 - m12
    sign_term_y = m02 - m20
    sign_term_z = m10 - m01
    sign_in_x = qx_abs * sign_term_x
    sign_in_y = qy_abs * sign_term_y
    sign_in_z = qz_abs * sign_term_z
    sx = _sign_nonzero(sign_in_x)
    sy = _sign_nonzero(sign_in_y)
    sz = _sign_nonzero(sign_in_z)

    qx = qx_abs * sx
    qy = qy_abs * sy
    qz = qz_abs * sz

    q = np.array([qw, qx, qy, qz], dtype=float)
    n_before = float(np.linalg.norm(q))
    q_final = q.copy()
    if n_before > 1e-12:
        q_final /= n_before

    return {
        "r": r.copy(),
        "m": {
            "m00": m00, "m01": m01, "m02": m02,
            "m10": m10, "m11": m11, "m12": m12,
            "m20": m20, "m21": m21, "m22": m22,
        },
        "q_eq59_abs": np.array([qw, qx_abs, qy_abs, qz_abs], dtype=float),
        "sign_terms": {"x": sign_term_x, "y": sign_term_y, "z": sign_term_z},
        "sign_inputs": {"x": sign_in_x, "y": sign_in_y, "z": sign_in_z},
        "signs": {"x": sx, "y": sy, "z": sz},
        "q_eq60_signed": np.array([qw, qx, qy, qz], dtype=float),
        "norm_before": n_before,
        "q_final": q_final,
    }


def _block_3_compute_quaternions(mcp: dict, second_third: dict, last_layer: dict) -> dict:
    """
    Build quaternion outputs from rotation matrices (Eq. 57-64) for:
    - MCP roots i in {1,5,9,13,17}: R_i^0
    - PIP joints j in {2,6,10,14,18}: R_j^{j-1}
    - DIP joints j+1: R_{j+1}^j
    - TIP frames j+2: R_{j+2}^{j+1} (identity from Eq.45)
    """
    if not isinstance(mcp, dict) or "rot_mats" not in mcp:
        raise ValueError("mcp invalido para cuaterniones")
    if not isinstance(second_third, dict) or "rot_j_jm1" not in second_third or "rot_j1_j" not in second_third:
        raise ValueError("second_third invalido para cuaterniones")
    if not isinstance(last_layer, dict) or "rot_j2_j1" not in last_layer:
        raise ValueError("last_layer invalido para cuaterniones")

    q_by_joint: dict[int, np.ndarray] = {}
    q_details_by_joint: dict[int, dict] = {}
    parent_by_joint: dict[int, int] = {}

    # MCP: q_i from R_i^0
    for i in mcp["root_ids"]:
        details = _rotation_matrix_to_quaternion_dong(mcp["rot_mats"][i])
        q_details_by_joint[i] = details
        q_by_joint[i] = details["q_final"]
        parent_by_joint[i] = 0

    # PIP/DIP/TIP
    for j in second_third["second_ids"]:
        jm1 = second_third["parent_ids"][j]
        j1 = second_third["third_ids"][j]
        j2 = second_third["fourth_ids"][j]

        details_j = _rotation_matrix_to_quaternion_dong(second_third["rot_j_jm1"][j])
        q_details_by_joint[j] = details_j
        q_by_joint[j] = details_j["q_final"]
        parent_by_joint[j] = jm1

        details_j1 = _rotation_matrix_to_quaternion_dong(second_third["rot_j1_j"][j])
        q_details_by_joint[j1] = details_j1
        q_by_joint[j1] = details_j1["q_final"]
        parent_by_joint[j1] = j

        details_j2 = _rotation_matrix_to_quaternion_dong(last_layer["rot_j2_j1"][j])
        q_details_by_joint[j2] = details_j2
        q_by_joint[j2] = details_j2["q_final"]
        parent_by_joint[j2] = j1

    joint_order = tuple(sorted(q_by_joint.keys()))
    q_tuple_by_joint = {k: tuple(float(v) for v in q_by_joint[k]) for k in joint_order}
    return {
        "joint_order": joint_order,
        "parent_by_joint": parent_by_joint,
        "q_by_joint": q_by_joint,
        "q_details_by_joint": q_details_by_joint,
        "q_tuple_by_joint": q_tuple_by_joint,
    }


def _block_3_dynamic_joint_params(points_0: np.ndarray, palm_base: dict, finger_base: dict | None = None) -> dict:
    """
    Bloque 3: Parametros dinamicos articulares (per-frame kinematics).
    """
    mcp = _block_3_compute_mcp_first_layer(points_0=points_0, palm_base=palm_base)
    second_third = _block_3_compute_second_third_layers(points_0=points_0, mcp=mcp, finger_base=finger_base)
    last_layer = _block_3_compute_last_layer(second_third=second_third, finger_base=finger_base)
    quaternions = _block_3_compute_quaternions(mcp=mcp, second_third=second_third, last_layer=last_layer)
    return {"mcp": mcp, "second_third": second_third, "last_layer": last_layer, "quaternions": quaternions}


# ------------------------------ #
# Visualizacion y salida consola #
# ------------------------------ #
def _set_equal_axes_3d(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2])) / 2.0
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _set_shared_axes_3d(axes, points_list: list[np.ndarray]) -> None:
    all_pts = np.vstack(points_list)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2])) / 2.0
    radius = max(radius, 1e-3)
    for ax in axes:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)


def _plot_points_3d(
    ax,
    points: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    set_equal_axes: bool = True,
) -> None:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=28)
    for i, p in enumerate(points):
        ax.text(p[0], p[1], p[2], str(i), fontsize=7)
    for i, j in edges:
        seg = points[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if set_equal_axes:
        _set_equal_axes_3d(ax, points)


def _plot_world_vs_local(
    points_w: np.ndarray,
    points_0: np.ndarray,
    capture_id: int,
    edges: list[tuple[int, int]],
    block: bool = False,
    save_dir: str | None = None,
) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PLOT] matplotlib no disponible: {exc}", flush=True)
        return
    backend = str(matplotlib.get_backend())
    print(f"[PLOT] backend={backend} | block={block}", flush=True)

    fig = plt.figure(figsize=(12, 5))
    ncols = 2
    ax_w = fig.add_subplot(1, ncols, 1, projection="3d")
    ax_l = fig.add_subplot(1, ncols, 2, projection="3d")

    _plot_points_3d(ax_w, points_w, edges, "World (d_i^w)", set_equal_axes=False)
    _plot_points_3d(ax_l, points_0, edges, "Wrist Local (d_i^0)", set_equal_axes=False)
    axes = [ax_w, ax_l]
    points_for_limits = [points_w, points_0]
    _set_shared_axes_3d(axes, points_for_limits)

    fig.suptitle(f"Dong Step4 - Captura {capture_id}")
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(save_dir, f"step4_capture_{capture_id:03d}_{stamp}.png")
        fig.savefig(out_path, dpi=170)
        print(f"[PLOT] guardado en: {out_path}", flush=True)

    backend_lower = backend.lower()
    non_interactive_backends = {
        "agg",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "cairo",
    }
    if backend_lower in non_interactive_backends:
        print("[PLOT] backend no interactivo; se guarda PNG pero no se abre ventana.", flush=True)
        plt.close(fig)
        return

    if block:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.2)
        print("[PLOT] ventana solicitada (si no aparece, revisa DISPLAY/entorno grafico).", flush=True)


def _print_matrix(name: str, m: np.ndarray) -> None:
    print(f"{name} =", flush=True)
    for i in range(m.shape[0]):
        print("  [" + ", ".join(f"{m[i,j]:+.6f}" for j in range(m.shape[1])) + "]", flush=True)


def _print_vector(name: str, v: np.ndarray) -> None:
    print(f"{name} = [{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}]", flush=True)


def _print_points_world(points_w: np.ndarray) -> None:
    print("Entrada d_i^w (21 puntos world):", flush=True)
    for i, p in enumerate(points_w):
        print(f"[{i:02d}] x={p[0]:+.6f}  y={p[1]:+.6f}  z={p[2]:+.6f}", flush=True)


def _print_points_local(points_0: np.ndarray) -> None:
    print("d_i^0 (21 puntos en marco de muneca):", flush=True)
    for i, p in enumerate(points_0):
        print(f"[{i:02d}] x={p[0]:+.6f}  y={p[1]:+.6f}  z={p[2]:+.6f}", flush=True)


def _print_palm_base_params(out: dict) -> None:
    print("Paso A (Eq.13) - l0,i = ||d_i^w - d_0^w||", flush=True)
    print("  Entrada: d_i^w y d_0^w para i in {1,5,9,13,17}", flush=True)
    print("  Salida: l0,i", flush=True)
    for i in out["root_ids"]:
        v = out["world_vectors"][i]
        l_m = out["lengths_m"][i]
        print(
            f"  i={i}: (d_i^w-d_0^w)=[{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}] -> "
            f"l0,{i}={l_m:+.6f} m ({l_m * 100.0:+.2f} cm)",
            flush=True,
        )

    print("Paso B (Eq.16) - d_i^0 = (T0w)^-1 d_i^w", flush=True)
    print("  Entrada: T0w_inv y d_i^w", flush=True)
    print("  Salida: d_i^0 (roots i={1,5,9,13,17})", flush=True)
    for i in out["root_ids"]:
        p = out["root_positions_from_eq16"][i]
        print(f"  d0{i} = [{p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}]", flush=True)

    print("Paso C (Eq.17) - theta_i = atan2(d0i_y, d0i_x)", flush=True)
    print("  Entrada: (d0i_x, d0i_y)", flush=True)
    print("  Salida: theta_i", flush=True)
    for i in out["root_ids"]:
        p = out["root_positions_from_eq16"][i]
        theta_rad = out["thetas_rad"][i]
        theta_deg = out["thetas_deg"][i]
        print(
            f"  i={i}: atan2({p[1]:+.6f}, {p[0]:+.6f}) -> "
            f"theta{i}={theta_rad:+.6f} rad ({theta_deg:+.2f} deg)",
            flush=True,
        )


def _print_palm_base_frozen_params(out: dict) -> None:
    print("Parametros congelados (mediana de calibracion):", flush=True)
    print(f"  muestras usadas = {out['n_samples']}", flush=True)
    for i in out["root_ids"]:
        l_m = out["lengths_m"][i]
        t_rad = out["thetas_rad"][i]
        t_deg = out["thetas_deg"][i]
        p = out["eq15_positions"][i]
        print(
            f"  i={i}: l0,{i}={l_m:+.6f} m ({l_m * 100.0:+.2f} cm) | "
            f"theta{i}={t_rad:+.6f} rad ({t_deg:+.2f} deg) | "
            f"d0{i}(Eq.15)=[{p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}]",
            flush=True,
        )


def _print_finger_base_frozen_params(out: dict) -> None:
    print("Longitudes de dedos congeladas (15 links, mediana de calibracion):", flush=True)
    print(f"  muestras usadas = {out['n_samples']}", flush=True)
    for a, b in out["links"]:
        l_m = out["lengths_m"][(a, b)]
        print(f"  l{a},{b}={l_m:+.6f} m ({l_m * 100.0:+.2f} cm)", flush=True)


def _print_mcp_first_layer(out: dict) -> None:
    print("Resultados (Eq.19-24): beta MCP y gamma MCP por dedo raiz", flush=True)
    for i in out["root_ids"]:
        beta_r = out["betas_rad"][i]
        beta_d = out["betas_deg"][i]
        gamma_r = out["gammas_rad"][i]
        gamma_d = out["gammas_deg"][i]
        child = out["child_ids"][i]
        print(
            f"  i={i} (child={child}): "
            f"beta{i}={beta_r:+.6f} rad ({beta_d:+.2f} deg) | "
            f"gamma{i}={gamma_r:+.6f} rad ({gamma_d:+.2f} deg)",
            flush=True,
        )


def _print_second_third_layers(out: dict) -> None:
    print("Resultados (Eq.29-36): beta PIP, beta DIP y longitudes fijas del modelo", flush=True)
    for j in out["second_ids"]:
        parent = out["parent_ids"][j]
        child = out["third_ids"][j]
        beta_j_r = out["betas_j_rad"][j]
        beta_j_d = out["betas_j_deg"][j]
        beta_c_r = out["betas_j1_rad"][j]
        beta_c_d = out["betas_j1_deg"][j]
        l1_model = out["lengths_jm1_j_model"][j]  # l_{j-1,j} usada en modelo
        l2_model = out["lengths_j_j1_model"][j]   # l_{j,j+1} usada en modelo
        print(
            f"  j={j} (parent={parent}, child={child}): "
            f"beta{j}={beta_j_r:+.6f} rad ({beta_j_d:+.2f} deg) | "
            f"beta{child}={beta_c_r:+.6f} rad ({beta_c_d:+.2f} deg) | "
            f"l{parent},{j}={l1_model:+.6f} m | "
            f"l{j},{child}={l2_model:+.6f} m",
            flush=True,
        )


def _print_last_layer(out: dict) -> None:
    print("Resultados (Eq.45): transformacion DIP->TIP sin rotacion (solo traslacion fija)", flush=True)
    for j in out["second_ids"]:
        dip = out["third_ids"][j]
        tip = out["fourth_ids"][j]
        l_model = out["lengths_j1_j2_model"][j]
        print(
            f"  chain_j={j} (dip={dip}, tip={tip}): "
            f"R{tip}^{dip}=I | "
            f"d{tip}^{dip}=[{l_model:+.6f}, +0.000000, +0.000000] m | "
            f"l{dip},{tip}={l_model:+.6f} m",
            flush=True,
        )


def _print_quaternions(out: dict) -> None:
    print("Resultados (Eq.57-64): cuaternion final por joint (sin procedimiento)", flush=True)
    for joint in out["joint_order"]:
        parent = out["parent_by_joint"][joint]
        qf = out["q_by_joint"][joint]
        print(
            f"  Joint {joint} (R{joint}^{parent}): "
            f"q{joint}(w,x,y,z)=({qf[0]:+.6f}, {qf[1]:+.6f}, {qf[2]:+.6f}, {qf[3]:+.6f})",
            flush=True,
        )


def _print_single_joint_trace(
    joint: int,
    points_0: np.ndarray,
    mcp: dict,
    second_third: dict,
    last_layer: dict,
    quaternions: dict,
) -> None:
    if joint not in quaternions["q_by_joint"]:
        valid = ", ".join(str(j) for j in quaternions["joint_order"])
        print(f"Joint invalida para traza: {joint}. Disponibles: {valid}", flush=True)
        return

    parent = quaternions["parent_by_joint"][joint]
    details = quaternions["q_details_by_joint"][joint]
    print(f"Joint {joint} (R{joint}^{parent})", flush=True)
    print("  Entrada -> Transformacion -> Salida", flush=True)

    second_ids = set(second_third["second_ids"])
    dip_to_second = {second_third["third_ids"][j]: j for j in second_third["second_ids"]}
    tip_to_second = {second_third["fourth_ids"][j]: j for j in second_third["second_ids"]}

    if joint in set(mcp["root_ids"]):
        i = joint
        child = mcp["child_ids"][i]
        d_i = points_0[i]
        d_child = points_0[child]
        seg = mcp["segment_vecs"][i]
        seg_norm = float(np.linalg.norm(seg))
        print(f"  Tipo=MCP root i={i} (child={child}, parent=0)", flush=True)
        _print_vector(f"  Entrada d{i}^0", d_i)
        _print_vector(f"  Entrada d{child}^0", d_child)
        _print_vector(f"  Eq.20 seg=d{child}^0-d{i}^0", seg)
        print(f"  Eq.20 ||seg||={seg_norm:+.6f}", flush=True)
        _print_vector(f"  Eq.20 X{i}^0=normalize(seg)", mcp["x_axes"][i])
        _print_vector(f"  Eq.21 Y{i}^0=normalize(Z0xX{i}^0)", mcp["y_axes"][i])
        _print_vector(f"  Eq.22 Z{i}^0=normalize(X{i}^0xY{i}^0)", mcp["z_axes"][i])
        _print_matrix(f"  Eq.19 R{i}^0", mcp["rot_mats"][i])
        _print_vector(f"  Eq.23 d{i}^0_model", mcp["d_i0_planar"][i])
        _print_matrix(f"  Eq.23 T{i}^0", mcp["tf_mats"][i])
        print(
            f"  Eq.24 beta{i}={mcp['betas_rad'][i]:+.6f} rad ({mcp['betas_deg'][i]:+.2f} deg) | "
            f"gamma{i}={mcp['gammas_rad'][i]:+.6f} rad ({mcp['gammas_deg'][i]:+.2f} deg)",
            flush=True,
        )
    elif joint in second_ids:
        j = joint
        jm1 = second_third["parent_ids"][j]
        j1 = second_third["third_ids"][j]
        dj = second_third["d_j_jm1"][j]
        dj1 = second_third["d_j1_jm1"][j]
        xj = second_third["x_j_jm1"][j]
        xjm1 = second_third["x_jm1_jm1"][j]
        n_xj = float(np.linalg.norm(xj))
        n_xjm1 = float(np.linalg.norm(xjm1))
        cos_beta = float(np.dot(xj, xjm1) / (n_xj * n_xjm1))
        print(f"  Tipo=PIP j={j} (parent={jm1}, child={j1})", flush=True)
        _print_vector(f"  Eq.30 d{j}^{jm1}", dj)
        _print_vector(f"  Eq.32 d{j1}^{jm1}", dj1)
        _print_vector(f"  Eq.29 x_j=d{j1}^{jm1}-d{j}^{jm1}", xj)
        _print_vector(f"  Eq.29 x_jm1=d{j}^{jm1}", xjm1)
        print(
            f"  Eq.29 cos(beta{j})={cos_beta:+.6f} -> beta{j}={second_third['betas_j_rad'][j]:+.6f} rad "
            f"({second_third['betas_j_deg'][j]:+.2f} deg)",
            flush=True,
        )
        _print_matrix(f"  Eq.33 R{j}^{jm1}=Ry(beta{j})", second_third["rot_j_jm1"][j])
        _print_vector(f"  Eq.35 d{j}^{jm1}_model", second_third["d_j_jm1_model"][j])
        _print_matrix(f"  Eq.36 T{j}^{jm1}", second_third["tf_j_jm1"][j])
    elif joint in dip_to_second:
        j = dip_to_second[joint]
        jm1 = second_third["parent_ids"][j]
        j1 = second_third["third_ids"][j]
        j2 = second_third["fourth_ids"][j]
        xj1 = second_third["x_j1_jm1"][j]
        xj = second_third["x_j_jm1"][j]
        n_xj1 = float(np.linalg.norm(xj1))
        n_xj = float(np.linalg.norm(xj))
        cos_beta = float(np.dot(xj1, xj) / (n_xj1 * n_xj))
        print(f"  Tipo=DIP j+1={j1} (parent={j}, tip={j2}, parent_frame={jm1})", flush=True)
        _print_vector(f"  Eq.32 d{j1}^{jm1}", second_third["d_j1_jm1"][j])
        _print_vector(f"  Eq.32 d{j2}^{jm1}", second_third["d_j2_jm1"][j])
        _print_vector(f"  Eq.31 x_j1=d{j2}^{jm1}-d{j1}^{jm1}", xj1)
        _print_vector(f"  Eq.31 x_j=d{j1}^{jm1}-d{j}^{jm1}", xj)
        print(
            f"  Eq.31 cos(beta{j1})={cos_beta:+.6f} -> beta{j1}={second_third['betas_j1_rad'][j]:+.6f} rad "
            f"({second_third['betas_j1_deg'][j]:+.2f} deg)",
            flush=True,
        )
        _print_matrix(f"  Eq.34 R{j1}^{j}=Ry(beta{j1})", second_third["rot_j1_j"][j])
        _print_vector(f"  Eq.35 d{j1}^{j}_model", second_third["d_j1_j_model"][j])
        _print_matrix(f"  Eq.36 T{j1}^{j}", second_third["tf_j1_j"][j])
    elif joint in tip_to_second:
        j = tip_to_second[joint]
        j1 = second_third["third_ids"][j]
        j2 = second_third["fourth_ids"][j]
        print(f"  Tipo=TIP j+2={j2} (parent={j1})", flush=True)
        print("  Eq.45 sin DOF: R_tip^dip = I", flush=True)
        _print_matrix(f"  Eq.45 R{j2}^{j1}", last_layer["rot_j2_j1"][j])
        _print_vector(f"  Eq.45 d{j2}^{j1}_model", last_layer["d_j2_j1_model"][j])
        _print_matrix(f"  Eq.45 T{j2}^{j1}", last_layer["tf_j2_j1"][j])
    else:
        print("  Tipo=desconocido para traza detallada", flush=True)

    m = details["m"]
    q59 = details["q_eq59_abs"]
    q60 = details["q_eq60_signed"]
    qf = details["q_final"]
    st = details["sign_terms"]
    si = details["sign_inputs"]
    sg = details["signs"]
    n_before = float(details["norm_before"])
    print("  [Quaternion Eq.57-64]", flush=True)
    print(
        f"    Eq.58 m = "
        f"[[{m['m00']:+.6f}, {m['m01']:+.6f}, {m['m02']:+.6f}], "
        f"[{m['m10']:+.6f}, {m['m11']:+.6f}, {m['m12']:+.6f}], "
        f"[{m['m20']:+.6f}, {m['m21']:+.6f}, {m['m22']:+.6f}]]",
        flush=True,
    )
    print(
        f"    Eq.59 q_abs=(w,x,y,z)=({q59[0]:+.6f}, {q59[1]:+.6f}, {q59[2]:+.6f}, {q59[3]:+.6f})",
        flush=True,
    )
    print(
        f"    Eq.60 sign_terms: "
        f"(m21-m12)={st['x']:+.6f}, (m02-m20)={st['y']:+.6f}, (m10-m01)={st['z']:+.6f}",
        flush=True,
    )
    print(
        f"    Eq.60 sign_inputs: "
        f"x={si['x']:+.6f}, y={si['y']:+.6f}, z={si['z']:+.6f} -> "
        f"signs=({sg['x']:+.0f}, {sg['y']:+.0f}, {sg['z']:+.0f})",
        flush=True,
    )
    print(
        f"    Eq.60 q_signed=(w,x,y,z)=({q60[0]:+.6f}, {q60[1]:+.6f}, {q60[2]:+.6f}, {q60[3]:+.6f}) | "
        f"norm={n_before:+.6f}",
        flush=True,
    )
    print(
        f"    Final q{joint}=(w,x,y,z)=({qf[0]:+.6f}, {qf[1]:+.6f}, {qf[2]:+.6f}, {qf[3]:+.6f})",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 4 Dong-lite: world -> wrist local")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("webcam", "egocentric", "image"),
        default="webcam",
        help="Modo de ejecucion: webcam (stream mirrored), egocentric (stream sin mirror) o image (single frame).",
    )
    parser.add_argument("--camera", type=str, default="0", help="Indice de camara (0,1,2,...) o URL.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta de imagen para procesar un solo frame (sin camara).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Mostrar plot 3D world vs local en cada captura (SPACE).",
    )
    parser.add_argument(
        "--plot-block",
        action="store_true",
        help="Bloquear ejecucion hasta cerrar cada plot.",
    )
    parser.add_argument(
        "--plot-save-dir",
        type=str,
        default="/home/yareeez/AIST-hand/grasp-app/plots_step4",
        help="Directorio donde se guarda PNG de cada captura cuando --plot esta activo.",
    )
    parser.add_argument(
        "--palm-freeze-frames",
        type=int,
        default=1,
        # Default 1: upstream multiframe module already smooths landmarks (e.g., t=8).
        # Keeping this at 1 avoids double temporal averaging during calibration.
        # N=1  -> freeze uses the single observed sample directly.
        # N>1  -> freeze uses median across N valid calibration samples.
        help="Numero de capturas validas para congelar bloque 2 (palma base + 15 longitudes de dedos).",
    )
    parser.add_argument(
        "--input-is-mirrored",
        action="store_true",
        help=(
            "Indica que la entrada de camara/imagen YA viene espejada (selfie). "
            "Si se activa, no se hace swap Left/Right en handedness."
        ),
    )
    parser.add_argument(
        "--disable-canonical-right-hand",
        action="store_true",
        help=(
            "Desactiva la reflexion X para canonizar a mano derecha. "
            "Util para depurar visualizacion sin espejo."
        ),
    )
    parser.add_argument(
        "--trace-joint",
        type=int,
        default=None,
        help="Imprime trazabilidad completa de una sola joint (1..20): entrada, transformacion y salida.",
    )
    parser.add_argument(
        "--trace-joint-only",
        action="store_true",
        help="Con --trace-joint, omite el listado completo de cuaterniones y muestra solo la joint elegida.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "Modo dinamico: calibra por tiempo al inicio y luego calcula Step4 "
            "frame-by-frame en tiempo real."
        ),
    )
    parser.add_argument(
        "--calib-seconds",
        type=float,
        default=3.0,
        help="Duracion de calibracion inicial en segundos para --continuous.",
    )
    parser.add_argument(
        "--continuous-print-every",
        type=int,
        default=15,
        help="En --continuous, imprime un resumen en consola cada N frames procesados.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.trace_joint is not None and not (1 <= int(args.trace_joint) <= 20):
        print(f"--trace-joint fuera de rango: {args.trace_joint} (esperado 1..20)", flush=True)
        return 1
    if args.calib_seconds < 0.0:
        print(f"--calib-seconds invalido: {args.calib_seconds} (esperado >= 0)", flush=True)
        return 1
    if args.continuous_print_every < 1:
        print(
            f"--continuous-print-every invalido: {args.continuous_print_every} (esperado >= 1)",
            flush=True,
        )
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    edges = sorted(list(mp_hands.HAND_CONNECTIONS))
    palm_freeze_state = {
        "target_frames": max(1, int(args.palm_freeze_frames)),
        "samples": [],
        "finger_samples": [],
        "frozen_palm": None,
        "frozen_finger": None,
    }
    # In continuous mode, freeze is controlled by calibration time.
    # We keep collecting samples and force-freeze when calibration finishes.
    if args.continuous and args.calib_seconds > 0.0:
        palm_freeze_state["target_frames"] = 10**9

    def _force_freeze_from_samples() -> None:
        samples = palm_freeze_state.get("samples", [])
        finger_samples = palm_freeze_state.get("finger_samples", [])
        if palm_freeze_state.get("frozen_palm") is None and samples:
            palm_freeze_state["frozen_palm"] = _freeze_palm_base_params(samples=samples)
        if palm_freeze_state.get("frozen_finger") is None and finger_samples:
            palm_freeze_state["frozen_finger"] = _freeze_finger_base_lengths(samples=finger_samples)

    def _compute_pipeline_blocks(points_w: np.ndarray, include_block_3: bool = True) -> dict:
        block_1 = _block_1_world_to_wrist_local(points_w=points_w)
        points_0 = block_1["points_0"]
        block_2 = _block_2_palm_base_fixed(
            points_w=points_w,
            points_0=points_0,
            freeze_state=palm_freeze_state,
        )
        out = {
            "block_1": block_1,
            "points_0": points_0,
            "block_2": block_2,
        }
        if include_block_3:
            block_3 = _block_3_dynamic_joint_params(
                points_0=points_0,
                palm_base=block_2["active_palm"],
                finger_base=block_2["active_finger"],
            )
            out["block_3"] = block_3
        return out

    def run_pipeline(
        points_w: np.ndarray,
        has_2d: bool,
        capture_id: int,
        hand_label: str | None = None,
        hand_score: float | None = None,
        canonical_hand_label: str = "Right",
        reflected_to_canonical: bool = False,
        frame_mirrored: bool = False,
    ) -> None:
        computed = _compute_pipeline_blocks(points_w=points_w, include_block_3=True)
        block_1 = computed["block_1"]
        wf = block_1["wf"]
        t0w = block_1["t0w"]
        t0w_inv = block_1["t0w_inv"]
        points_0 = computed["points_0"]
        r0w = wf["r0w"]

        block_2 = computed["block_2"]
        palm_active = block_2["active_palm"]
        palm_observed = block_2["observed_palm"]
        palm_frozen = block_2["frozen_palm"]
        finger_active = block_2["active_finger"]
        finger_frozen = block_2["frozen_finger"]

        block_3 = computed["block_3"]
        mcp = block_3["mcp"]
        second_third = block_3["second_third"]
        last_layer = block_3["last_layer"]
        quaternions = block_3["quaternions"]

        handed_txt = hand_label if hand_label else "UNKNOWN"
        handed_score_txt = f"{hand_score:.3f}" if hand_score is not None else "NA"
        reflected_txt = "YES" if reflected_to_canonical else "NO"
        print(
            f"\n=== Step4 World->Local | Captura {capture_id} ===",
            flush=True,
        )
        print(
            f"[SOURCE] using=WORLD has_2d={has_2d} has_world=True n_points={len(points_w)} "
            f"hand_detected={handed_txt} score={handed_score_txt} "
            f"hand_canonical={canonical_hand_label} "
            f"landmark_reflected={reflected_txt} frame_mirrored={'YES' if frame_mirrored else 'NO'}",
            flush=True,
        )
        print("\n[BLOQUE 1] Transformacion Global a Local (W->0)", flush=True)
        print("\n[1] PASO INPUT", flush=True)
        print("Entrada: d_i^w (21 puntos world)", flush=True)
        print("Salida: d_i^w (mismo set, solo trazabilidad)", flush=True)
        _print_points_world(points_w)

        print("\n[2] PASO WRIST FRAME (Eq.5-8)", flush=True)
        print("Entrada: d0, d5, d9, d13 en world", flush=True)
        _print_vector("d0 (kp0)", wf["d0"])
        _print_vector("d5 (kp5)", wf["d5"])
        _print_vector("d9 (kp9)", wf["d9"])
        _print_vector("d13 (kp13)", wf["d13"])
        _print_vector("v_13_5 = d13-d5", wf["v_13_5"])
        _print_vector("v_9_0 = d9-d0", wf["v_9_0"])
        _print_vector("v_9_0_hat", wf["v_9_0_hat"])
        print("Salida: Y0, Z0, X0, R0w y Euler(alpha,beta,gamma)", flush=True)
        _print_vector("Y0", wf["y0"])
        _print_vector("Z0", wf["z0"])
        _print_vector("X0", wf["x0"])
        _print_matrix("R0w = [X0 Y0 Z0]", r0w)
        print(
            f"Euler (Dong Eq.8): alpha={wf['alpha_rad']:+.6f} rad ({wf['alpha_deg']:+.2f} deg) | "
            f"beta={wf['beta_rad']:+.6f} rad ({wf['beta_deg']:+.2f} deg) | "
            f"gamma={wf['gamma_rad']:+.6f} rad ({wf['gamma_deg']:+.2f} deg)",
            flush=True,
        )

        print("\n[3] PASO TRANSFORMACIONES (Eq.9)", flush=True)
        print("Entrada: R0w y d0w", flush=True)
        print("Salida: T0w y T0w_inv", flush=True)
        _print_matrix("T0w", t0w)
        _print_matrix("T0w_inv", t0w_inv)

        print("\n[4] PASO WORLD->LOCAL (Eq.16)", flush=True)
        print("Entrada: T0w_inv y d_i^w", flush=True)
        print("Salida: d_i^0 (21 puntos en marco de muneca)", flush=True)
        _print_points_local(points_0)

        print("\n[BLOQUE 2] Parametros Fijos Anatomicos (Base Calibration)", flush=True)
        if block_2["is_frozen"]:
            print(
                f"[PALM_BASE] mode=FROZEN calib={block_2['calib_count']}/{block_2['calib_target']}",
                flush=True,
            )
            if block_2["just_froze"]:
                print("[PALM_BASE] congelado en esta captura.", flush=True)
        else:
            print(
                f"[PALM_BASE] mode=CALIBRATING calib={block_2['calib_count']}/{block_2['calib_target']}",
                flush=True,
            )
        if block_2["is_finger_frozen"]:
            print(
                f"[FINGER_LENGTHS] mode=FROZEN calib={block_2['finger_calib_count']}/{block_2['calib_target']}",
                flush=True,
            )
            if block_2["just_froze_finger"]:
                print("[FINGER_LENGTHS] congelado en esta captura.", flush=True)
        else:
            print(
                f"[FINGER_LENGTHS] mode=CALIBRATING calib={block_2['finger_calib_count']}/{block_2['calib_target']}",
                flush=True,
            )

        print("\n[5] PASO PALM BASE DONG (Eq.13 -> Eq.16 -> Eq.17)", flush=True)
        _print_palm_base_params(palm_observed)
        if palm_frozen is not None:
            print("\n[5.FIX] PARAMETROS PALM BASE CONGELADOS", flush=True)
            _print_palm_base_frozen_params(palm_frozen)
        if finger_frozen is not None:
            print("\n[5.FIX.LINKS] LONGITUDES DE DEDOS CONGELADAS (15)", flush=True)
            _print_finger_base_frozen_params(finger_frozen)

        print("\n[BLOQUE 3] Parametros Dinamicos Articulares (Per-Frame Kinematics)", flush=True)
        print("\n[5.1] PASO MCP FIRST LAYER DONG (Eq.19 -> Eq.24)", flush=True)
        _print_mcp_first_layer(mcp)
        print("\n[5.2] PASO SECOND/THIRD LAYER DONG (Eq.29 -> Eq.36)", flush=True)
        _print_second_third_layers(second_third)
        print("\n[5.3] PASO LAST LAYER DONG (Eq.45)", flush=True)
        _print_last_layer(last_layer)
        print("\n[5.4] PASO QUATERNIONS DONG (Eq.57 -> Eq.64)", flush=True)
        if not args.trace_joint_only:
            _print_quaternions(quaternions)
        else:
            print("Resultados completos omitidos por --trace-joint-only", flush=True)
        if args.trace_joint is not None:
            print(f"\n[5.4.ONE] TRAZA SINGLE JOINT (joint={args.trace_joint})", flush=True)
            _print_single_joint_trace(
                joint=int(args.trace_joint),
                points_0=points_0,
                mcp=mcp,
                second_third=second_third,
                last_layer=last_layer,
                quaternions=quaternions,
            )
        if args.plot:
            _plot_world_vs_local(
                points_w=points_w,
                points_0=points_0,
                capture_id=capture_id,
                edges=edges,
                block=args.plot_block,
                save_dir=args.plot_save_dir,
            )

    if args.mode == "image":
        if not args.image:
            print("Modo image requiere --image /ruta/a/archivo", flush=True)
            return 1
        image = cv2.imread(args.image)
        if image is None:
            print(f"No se pudo leer la imagen: {args.image}", flush=True)
            return 1
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            result = hands.process(rgb)
            if not result.multi_hand_world_landmarks:
                print("No hay world_landmarks en la imagen.", flush=True)
                return 1
            has_2d = bool(result.multi_hand_landmarks)
            hand_label, hand_score = _extract_handedness(
                result,
                input_is_mirrored=args.input_is_mirrored,
            )
            points_w_raw = _world_points(result.multi_hand_world_landmarks[0])
            if args.disable_canonical_right_hand:
                points_w = points_w_raw
                canonical_hand_label = hand_label if hand_label else "RAW"
                reflected_to_canonical = False
            else:
                points_w, canonical_hand_label, reflected_to_canonical = _canonicalize_world_points_to_right(
                    points_w=points_w_raw,
                    hand_label=hand_label,
                )
            try:
                run_pipeline(
                    points_w=points_w,
                    has_2d=has_2d,
                    capture_id=1,
                    hand_label=hand_label,
                    hand_score=hand_score,
                    canonical_hand_label=canonical_hand_label,
                    reflected_to_canonical=reflected_to_canonical,
                    frame_mirrored=False,
                )
            except (ValueError, KeyError) as exc:
                print(f"Frame degenerado: {exc}", flush=True)
                return 1
        return 0
    if args.image:
        print("[WARN] mode=webcam ignora --image.", flush=True)

    source = _camera_source(args.camera)

    cap = _open_capture(source)
    if not cap or not cap.isOpened():
        print(f"No se pudo abrir la camara: {source!r}", flush=True)
        return 1

    capture_id = 0
    realtime_processed = 0
    continuous_calib_done = (not args.continuous) or (args.calib_seconds <= 0.0)
    continuous_calib_t0 = time.time()
    win = "Dong Step 4 - World to Wrist Local"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    if args.continuous:
        if continuous_calib_done:
            print(
                "Iniciado (modo continuo). Calibracion temporal omitida; calculo frame-by-frame activo. "
                "SPACE: snapshot detallado | Q/ESC: salir.",
                flush=True,
            )
        else:
            print(
                f"Iniciado (modo continuo). Calibracion inicial: {args.calib_seconds:.2f}s. "
                "Luego calculo frame-by-frame en tiempo real. "
                "SPACE: snapshot detallado | Q/ESC: salir.",
                flush=True,
            )
    else:
        print("Iniciado. SPACE: transformar a marco de muneca | Q/ESC: salir.", flush=True)
    frame_mirrored_webcam = args.mode == "webcam"
    if args.mode == "webcam":
        print("[MODE] webcam: frame_mirrored=YES | handedness MediaPipe raw | canonical_right=ON", flush=True)
    else:
        print("[MODE] egocentric: frame_mirrored=NO | handedness MediaPipe raw | canonical_right=ON", flush=True)
    if args.input_is_mirrored or args.disable_canonical_right_hand:
        print("[MODE] webcam puro ignora --input-is-mirrored y --disable-canonical-right-hand.", flush=True)
    if args.plot:
        print(
            f"[PLOT] activo: se intentara abrir ventana y guardar PNG en {args.plot_save_dir}",
            flush=True,
        )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if frame_mirrored_webcam:
                # Webcam mode: mirror input frame so MediaPipe handedness matches selfie convention.
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_2d = None
            hand_world = None
            hand_label = None
            hand_score = None
            hand_info = _extract_handedness_raw(result)
            hand_label, hand_score = hand_info
            if result.multi_hand_landmarks:
                hand_2d = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_2d, mp_hands.HAND_CONNECTIONS)
                h, w = frame.shape[:2]
                for idx, lm in enumerate(hand_2d.landmark):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.putText(
                        frame,
                        str(idx),
                        (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            if result.multi_hand_world_landmarks:
                hand_world = result.multi_hand_world_landmarks[0]

            hand_txt = hand_label if hand_label else "UNKNOWN"
            if hand_score is not None:
                hand_txt = f"{hand_txt}({hand_score:.2f})"
            points_w = None
            canonical_hand_label = "RAW"
            reflected_to_canonical = False
            if hand_world is not None:
                points_w_raw = _world_points(hand_world)
                points_w, canonical_hand_label, reflected_to_canonical = _canonicalize_world_points_to_right(
                    points_w=points_w_raw,
                    hand_label=hand_label,
                )

            if args.continuous and points_w is not None:
                try:
                    if not continuous_calib_done:
                        _compute_pipeline_blocks(points_w=points_w, include_block_3=False)
                        elapsed = time.time() - continuous_calib_t0
                        if elapsed >= args.calib_seconds:
                            _force_freeze_from_samples()
                            if (
                                palm_freeze_state.get("frozen_palm") is not None
                                and palm_freeze_state.get("frozen_finger") is not None
                            ):
                                continuous_calib_done = True
                                print(
                                    "[CONT] calibracion finalizada: "
                                    f"palm_samples={len(palm_freeze_state.get('samples', []))} | "
                                    f"finger_samples={len(palm_freeze_state.get('finger_samples', []))}",
                                    flush=True,
                                )
                    else:
                        computed = _compute_pipeline_blocks(points_w=points_w, include_block_3=True)
                        realtime_processed += 1
                        if realtime_processed % args.continuous_print_every == 0:
                            block_3 = computed["block_3"]
                            mcp = block_3["mcp"]
                            second_third = block_3["second_third"]
                            quats = block_3["quaternions"]
                            mcp_root_ids = set(mcp["root_ids"])
                            second_ids = set(second_third["second_ids"])
                            dip_to_second = {second_third["third_ids"][j]: j for j in second_third["second_ids"]}
                            tip_to_second = {second_third["fourth_ids"][j]: j for j in second_third["second_ids"]}
                            print(
                                "[CONT] "
                                f"frame={realtime_processed} hand={hand_txt} canonical={canonical_hand_label} "
                                "resultados por joint:",
                                flush=True,
                            )
                            for joint in quats["joint_order"]:
                                parent = quats["parent_by_joint"][joint]
                                q = quats["q_by_joint"][joint]
                                if joint in mcp_root_ids:
                                    beta = float(mcp["betas_deg"][joint])
                                    gamma = float(mcp["gammas_deg"][joint])
                                    line = (
                                        f"  Joint {joint} (R{joint}^{parent}): "
                                        f"beta{joint}={beta:+.2f}deg "
                                        f"gamma{joint}={gamma:+.2f}deg "
                                    )
                                elif joint in second_ids:
                                    beta = float(second_third["betas_j_deg"][joint])
                                    line = (
                                        f"  Joint {joint} (R{joint}^{parent}): "
                                        f"beta{joint}={beta:+.2f}deg "
                                    )
                                elif joint in dip_to_second:
                                    j = dip_to_second[joint]
                                    beta = float(second_third["betas_j1_deg"][j])
                                    line = (
                                        f"  Joint {joint} (R{joint}^{parent}): "
                                        f"beta{joint}={beta:+.2f}deg "
                                    )
                                elif joint in tip_to_second:
                                    line = f"  Joint {joint} (R{joint}^{parent}): "
                                else:
                                    line = f"  Joint {joint} (R{joint}^{parent}): "
                                print(
                                    line + f"q{joint}=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})",
                                    flush=True,
                                )
                except (ValueError, KeyError) as exc:
                    print(f"\nFrame degenerado (continuo): {exc}", flush=True)

            if args.continuous:
                if continuous_calib_done:
                    msg = (
                        f"REALTIME Step4 | HAND={hand_txt} | WORLD={'YES' if hand_world is not None else 'NO'} | "
                        f"frames={realtime_processed} | SPACE=snapshot | Q/ESC"
                    )
                else:
                    elapsed = time.time() - continuous_calib_t0
                    rem = max(0.0, args.calib_seconds - elapsed)
                    msg = (
                        f"CALIBRATING {rem:.1f}s | HAND={hand_txt} | WORLD={'YES' if hand_world is not None else 'NO'} | "
                        f"samples={len(palm_freeze_state.get('samples', []))} | Q/ESC"
                    )
            else:
                msg = (
                    f"SPACE: world->local | HAND={hand_txt} | "
                    f"WORLD={'YES' if hand_world is not None else 'NO'} | Q/ESC"
                )
            if hand_2d is None:
                msg = "No hand detected | " + msg
            cv2.putText(
                frame,
                msg,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if key == 32:  # SPACE
                if points_w is None:
                    print("\nNo hay world_landmarks en este frame.", flush=True)
                    continue

                try:
                    capture_id += 1
                    run_pipeline(
                        points_w=points_w,
                        has_2d=(hand_2d is not None),
                        capture_id=capture_id,
                        hand_label=hand_label,
                        hand_score=hand_score,
                        canonical_hand_label=canonical_hand_label,
                        reflected_to_canonical=reflected_to_canonical,
                        frame_mirrored=frame_mirrored_webcam,
                    )
                except (ValueError, KeyError) as exc:
                    print(f"\nFrame degenerado: {exc}", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
