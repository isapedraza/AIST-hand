"""
Dong Kinematics -- Kinematic feature extraction from 21-point hand landmarks.

Based on Dong & Payandeh (2025): Hand Kinematic Model Construction Based on
Tracking Landmarks.

This module provides pure kinematic computations without any camera, MediaPipe,
or visualization dependencies.  It can be used for:

  - Precalculating kinematic features for the training dataset
  - Real-time inference in the teleoperation pipeline
  - Any context that needs to go from 21 XYZ landmarks to joint angles/quaternions

Usage::

    from dong_kinematics import DongKinematics

    dk = DongKinematics(calibration_frames=1)
    result = dk.process(points_w)   # (21,3) ndarray -> dict of features
    dk.reset()                       # reinitialize calibration
"""

from __future__ import annotations

import numpy as np


# ================================================================ #
#                          Constants                                #
# ================================================================ #

PALM_ROOT_IDS: tuple[int, ...] = (1, 5, 9, 13, 17)

FINGER_LINKS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


# ================================================================ #
#                     Internal Helper Functions                     #
# ================================================================ #

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Vector degenerado (norm={n:.3e})")
    return v / n


# ================================================================ #
#           Block 1: World -> Wrist Local  (Dong Eq. 5-9, 16)      #
# ================================================================ #

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


# ================================================================ #
#    Block 2: Fixed Anatomical Parameters  (Dong Eq. 13, 15-17)    #
# ================================================================ #

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
#  Block 3: Dynamic Joint Parameters  (Dong Eq. 19-45, 57-64)     #
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
    - Third layer  (DIP): beta_{j+1}, R_{j+1}^j, T_{j+1}^j
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

    # CONVENCIÓN PARA ML (Double Cover): Si w < 0, invertimos todo el cuaternión
    # Esto garantiza que rotaciones físicas idénticas siempre den el mismo vector.
    if q_final[0] < 0.0:
        q_final = -q_final

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


# ================================================================ #
#                      Public Utilities                             #
# ================================================================ #

def canonicalize_to_right_hand(
    points_w: np.ndarray,
    hand_label: str | None,
) -> tuple[np.ndarray, str, bool]:
    """
    Canonical convention:
    - Right hand: keep as-is.
    - Left hand: reflect X in world frame so downstream kinematics uses right-hand convention.

    Args:
        points_w: (21, 3) world landmarks.
        hand_label: "Left", "Right", or None.

    Returns:
        (points_w_canonical, canonical_hand_label, reflected).
    """
    if points_w.shape != (21, 3):
        raise ValueError(f"points_w shape invalido: {points_w.shape} (esperado (21,3))")

    label = (hand_label or "").strip().lower()
    if label == "left":
        out = points_w.copy()
        out[:, 0] *= -1.0
        return out, "Right", True
    return points_w, "Right", False


# ================================================================ #
#                      DongKinematics Class                         #
# ================================================================ #

class DongKinematics:
    """
    Stateful wrapper for the Dong kinematic pipeline.

    Manages calibration state (freeze of fixed anatomical parameters) and
    provides a clean interface for computing kinematic features from 21 XYZ
    hand landmarks.

    Args:
        calibration_frames: Number of frames to collect before freezing
            anatomical parameters.  Use 1 for instant calibration (default).
            Use None for manual calibration mode (call force_freeze() explicitly).

    Usage::

        dk = DongKinematics(calibration_frames=1)
        result = dk.process(points_w)
        # result["quaternions"]  -> {joint_id: np.array([w,x,y,z])}
        # result["angles_deg"]   -> {joint_id: {"beta": float, ...}}
        # result["is_calibrated"] -> bool
    """

    def __init__(self, calibration_frames: int | None = 1):
        if calibration_frames is None:
            self._target_frames = 10**9  # manual mode: never auto-freeze
        else:
            self._target_frames = max(1, int(calibration_frames))
        self._freeze_state: dict = self._make_freeze_state()

    def _make_freeze_state(self) -> dict:
        return {
            "target_frames": self._target_frames,
            "samples": [],
            "finger_samples": [],
            "frozen_palm": None,
            "frozen_finger": None,
        }

    @property
    def is_calibrated(self) -> bool:
        """True when both palm and finger parameters are frozen."""
        return (
            self._freeze_state.get("frozen_palm") is not None
            and self._freeze_state.get("frozen_finger") is not None
        )

    @property
    def calibration_count(self) -> int:
        """Number of calibration samples collected so far."""
        return len(self._freeze_state.get("samples", []))

    def reset(self) -> None:
        """Clear calibration state and start fresh."""
        self._freeze_state = self._make_freeze_state()

    def calibrate(self, points_w: np.ndarray) -> None:
        """
        Collect a calibration sample without computing kinematic features.

        Use this during a calibration phase (e.g., timed window in live mode)
        when you only need to accumulate samples for parameter freezing.

        Args:
            points_w: (21, 3) world landmarks, canonicalized to right hand.
        """
        if points_w.shape != (21, 3):
            raise ValueError(f"points_w shape invalido: {points_w.shape} (esperado (21,3))")
        block_1 = _block_1_world_to_wrist_local(points_w)
        _block_2_palm_base_fixed(
            points_w=points_w,
            points_0=block_1["points_0"],
            freeze_state=self._freeze_state,
        )

    def force_freeze(self) -> None:
        """
        Force-freeze calibration parameters from whatever samples have been
        collected so far.  Call this after a timed calibration window.
        """
        samples = self._freeze_state.get("samples", [])
        finger_samples = self._freeze_state.get("finger_samples", [])
        if self._freeze_state.get("frozen_palm") is None and samples:
            self._freeze_state["frozen_palm"] = _freeze_palm_base_params(samples)
        if self._freeze_state.get("frozen_finger") is None and finger_samples:
            self._freeze_state["frozen_finger"] = _freeze_finger_base_lengths(finger_samples)

    def process(self, points_w: np.ndarray) -> dict:
        """
        Compute full kinematic features from 21 world landmarks.

        Also collects calibration samples if not yet frozen (auto-freezes
        after ``calibration_frames`` calls).

        Args:
            points_w: (21, 3) array of 3D landmark positions in world frame.
                      Must be canonicalized to right hand convention
                      (use ``canonicalize_to_right_hand()`` first).

        Returns:
            dict with keys:

            - ``quaternions``: dict[int, np.ndarray] -- joint_id -> [w,x,y,z]
            - ``joint_order``: tuple of joint ids in sorted order
            - ``parent_by_joint``: dict[int, int] -- joint_id -> parent_id
            - ``angles_deg``: dict[int, dict] -- joint_id -> angle dict
              (MCP: {"beta", "gamma"}, PIP/DIP: {"beta"})
            - ``angles_rad``: same in radians
            - ``wrist_euler_deg``: dict with "alpha", "beta", "gamma"
            - ``wrist_euler_rad``: same in radians
            - ``points_local``: (21, 3) ndarray in wrist frame
            - ``is_calibrated``: bool
            - ``raw``: dict with full pipeline output for debug

        Raises:
            ValueError: on degenerate frames.
        """
        if points_w.shape != (21, 3):
            raise ValueError(f"points_w shape invalido: {points_w.shape} (esperado (21,3))")

        # Block 1: world -> wrist local
        block_1 = _block_1_world_to_wrist_local(points_w)
        points_0 = block_1["points_0"]

        # Block 2: calibration / fixed anatomical params
        block_2 = _block_2_palm_base_fixed(
            points_w=points_w,
            points_0=points_0,
            freeze_state=self._freeze_state,
        )

        # Block 3: dynamic joint params (angles + quaternions)
        block_3 = _block_3_dynamic_joint_params(
            points_0=points_0,
            palm_base=block_2["active_palm"],
            finger_base=block_2["active_finger"],
        )

        # ---- Extract clean results ----
        quaternions = block_3["quaternions"]
        mcp = block_3["mcp"]
        second_third = block_3["second_third"]
        wf = block_1["wf"]

        # Build angles dicts
        angles_deg: dict[int, dict] = {}
        angles_rad: dict[int, dict] = {}

        # MCP roots: beta (flexion) + gamma (abduction)
        for i in mcp["root_ids"]:
            angles_deg[i] = {"beta": mcp["betas_deg"][i], "gamma": mcp["gammas_deg"][i]}
            angles_rad[i] = {"beta": mcp["betas_rad"][i], "gamma": mcp["gammas_rad"][i]}

        # PIP + DIP: beta (flexion) only
        for j in second_third["second_ids"]:
            j1 = second_third["third_ids"][j]
            # PIP
            angles_deg[j] = {"beta": second_third["betas_j_deg"][j]}
            angles_rad[j] = {"beta": second_third["betas_j_rad"][j]}
            # DIP
            angles_deg[j1] = {"beta": second_third["betas_j1_deg"][j]}
            angles_rad[j1] = {"beta": second_third["betas_j1_rad"][j]}

        return {
            "quaternions": quaternions["q_by_joint"],
            "joint_order": quaternions["joint_order"],
            "parent_by_joint": quaternions["parent_by_joint"],
            "angles_deg": angles_deg,
            "angles_rad": angles_rad,
            "wrist_euler_deg": {
                "alpha": wf["alpha_deg"],
                "beta": wf["beta_deg"],
                "gamma": wf["gamma_deg"],
            },
            "wrist_euler_rad": {
                "alpha": wf["alpha_rad"],
                "beta": wf["beta_rad"],
                "gamma": wf["gamma_rad"],
            },
            "points_local": points_0,
            "is_calibrated": self.is_calibrated,
            "raw": {
                "block_1": block_1,
                "block_2": block_2,
                "block_3": block_3,
            },
        }
