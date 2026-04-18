"""
MediaPipe Live -- Real-time hand kinematic visualization and debug tool.

Applies Dong & Payandeh (2025) kinematic extraction to live MediaPipe hand
landmarks from a webcam or image.  Uses dong_kinematics.py for all math.

Controls:
- SPACE: capture and print full kinematic snapshot
- Q / ESC: exit
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

from dong_kinematics import (
    DongKinematics,
    PALM_ROOT_IDS,
    canonicalize_to_right_hand,
)

# Reduce ruido de logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)


# ================================================================ #
#                    Camera / MediaPipe Utilities                    #
# ================================================================ #

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


# ================================================================ #
#                   Visualization (3D plots)                        #
# ================================================================ #

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


# ================================================================ #
#                    Console Print Utilities                        #
# ================================================================ #

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


# ================================================================ #
#                         Argument Parsing                          #
# ================================================================ #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe Live: Dong kinematics visualization")
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


# ================================================================ #
#              Snapshot Print  (SPACE key full dump)                 #
# ================================================================ #

def _print_full_snapshot(
    result: dict,
    points_w: np.ndarray,
    capture_id: int,
    has_2d: bool,
    hand_label: str | None,
    hand_score: float | None,
    canonical_hand_label: str,
    reflected_to_canonical: bool,
    frame_mirrored: bool,
    args: argparse.Namespace,
    edges: list[tuple[int, int]],
) -> None:
    """Print the full detailed snapshot (equivalent to SPACE key in the original)."""
    raw = result["raw"]
    block_1 = raw["block_1"]
    block_2 = raw["block_2"]
    block_3 = raw["block_3"]
    wf = block_1["wf"]
    t0w = block_1["t0w"]
    t0w_inv = block_1["t0w_inv"]
    points_0 = result["points_local"]
    r0w = wf["r0w"]

    palm_observed = block_2["observed_palm"]
    palm_frozen = block_2["frozen_palm"]
    finger_frozen = block_2["frozen_finger"]

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


# ================================================================ #
#                            Main Loop                              #
# ================================================================ #

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

    # ---- Create DongKinematics instance ----
    if args.continuous and args.calib_seconds > 0.0:
        # Manual calibration mode: collect samples during timed window, then force_freeze.
        dk = DongKinematics(calibration_frames=None)
    else:
        dk = DongKinematics(calibration_frames=max(1, int(args.palm_freeze_frames)))

    # ---- Image mode ----
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
                points_w, canonical_hand_label, reflected_to_canonical = canonicalize_to_right_hand(
                    points_w=points_w_raw,
                    hand_label=hand_label,
                )
            try:
                dk_result = dk.process(points_w)
                _print_full_snapshot(
                    result=dk_result,
                    points_w=points_w,
                    capture_id=1,
                    has_2d=has_2d,
                    hand_label=hand_label,
                    hand_score=hand_score,
                    canonical_hand_label=canonical_hand_label,
                    reflected_to_canonical=reflected_to_canonical,
                    frame_mirrored=False,
                    args=args,
                    edges=edges,
                )
            except (ValueError, KeyError) as exc:
                print(f"Frame degenerado: {exc}", flush=True)
                return 1
        return 0
    if args.image:
        print("[WARN] mode=webcam ignora --image.", flush=True)

    # ---- Camera mode ----
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
                points_w, canonical_hand_label, reflected_to_canonical = canonicalize_to_right_hand(
                    points_w=points_w_raw,
                    hand_label=hand_label,
                )

            if args.continuous and points_w is not None:
                try:
                    if not continuous_calib_done:
                        dk.calibrate(points_w)
                        elapsed = time.time() - continuous_calib_t0
                        if elapsed >= args.calib_seconds:
                            dk.force_freeze()
                            if dk.is_calibrated:
                                continuous_calib_done = True
                                print(
                                    "[CONT] calibracion finalizada: "
                                    f"palm_samples={dk.calibration_count}",
                                    flush=True,
                                )
                    else:
                        dk_result = dk.process(points_w)
                        realtime_processed += 1
                        if realtime_processed % args.continuous_print_every == 0:
                            raw_b3 = dk_result["raw"]["block_3"]
                            mcp = raw_b3["mcp"]
                            second_third = raw_b3["second_third"]
                            quats = raw_b3["quaternions"]
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
                        f"samples={dk.calibration_count} | Q/ESC"
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
                    dk_result = dk.process(points_w)
                    _print_full_snapshot(
                        result=dk_result,
                        points_w=points_w,
                        capture_id=capture_id,
                        has_2d=(hand_2d is not None),
                        hand_label=hand_label,
                        hand_score=hand_score,
                        canonical_hand_label=canonical_hand_label,
                        reflected_to_canonical=reflected_to_canonical,
                        frame_mirrored=frame_mirrored_webcam,
                        args=args,
                        edges=edges,
                    )
                except (ValueError, KeyError) as exc:
                    print(f"\nFrame degenerado: {exc}", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
