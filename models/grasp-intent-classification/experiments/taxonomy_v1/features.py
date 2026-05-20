"""
features.py -- Paso 2: derivacion del subespacio de observacion.
  - Vectores oseos (bone vectors)
  - Angulos articulares (joint flexion angles)
  - Sintesis por mediana (El Punto de Trial)
"""

import numpy as np
import pandas as pd
from config import (
    BONE_CONNECTIONS, ANGLE_TRIPLETS, ANGLE_NAMES,
    ABDUCTION_DEFS, THUMB_ABD_DEF, PALM_KP_A, PALM_KP_B, PALM_KP_O,
    ABDUCTION_NAMES, ALL_ANGLE_NAMES,
)


def compute_bone_vectors(xyz):
    """
    xyz: (N, 21, 3)
    Retorna: (N, 20, 3) -- 20 vectores oseos b_ij = j_child - j_parent.
    """
    bones = np.zeros((xyz.shape[0], len(BONE_CONNECTIONS), 3), dtype=np.float32)
    for i, (parent, child) in enumerate(BONE_CONNECTIONS):
        bones[:, i, :] = xyz[:, child, :] - xyz[:, parent, :]
    return bones


def _normalize(v):
    """Normaliza vectores (N, 3) por fila, evitando division por cero."""
    n = np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-8)
    return v / n


def compute_flexion_angles(xyz):
    """
    xyz: (N, 21, 3)
    Retorna: (N, 15) -- 15 angulos de flexion en radianes.
    Calculo: arccos del producto punto normalizado entre los vectores
    que convergen en cada articulacion (Chen et al. 2025, Eq. 10).
    """
    n = xyz.shape[0]
    angles = np.zeros((n, len(ANGLE_TRIPLETS)), dtype=np.float32)

    for i, (p, j, c) in enumerate(ANGLE_TRIPLETS):
        v_in  = xyz[:, j, :] - xyz[:, p, :]  # parent -> joint
        v_out = xyz[:, c, :] - xyz[:, j, :]   # joint -> child

        cos_a = np.sum(_normalize(v_in) * _normalize(v_out), axis=1)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles[:, i] = np.arccos(cos_a)

    return angles


def compute_abduction_angles(xyz):
    """
    xyz: (N, 21, 3)
    Retorna: (N, 5) -- 4 abduccion MCP (dedos largos) + 1 abduccion CMC (pulgar).

    Dedos largos: angulo entre el metacarpal (MCP-WRIST) y la falange
    proximal (PIP-MCP), ambos proyectados al plano de la palma.

    Pulgar: arcsin de la componente fuera del plano del metacarpo
    (palmar abduction en CMC).
    """
    n = xyz.shape[0]
    abd = np.zeros((n, 5), dtype=np.float32)

    # Normal del plano de la palma
    va = xyz[:, PALM_KP_A, :] - xyz[:, PALM_KP_O, :]  # kp5 - kp0
    vb = xyz[:, PALM_KP_B, :] - xyz[:, PALM_KP_O, :]  # kp17 - kp0
    palm_n = _normalize(np.cross(va, vb))  # (N, 3)

    # 4 dedos largos: proyeccion al plano de la palma
    for i, d in enumerate(ABDUCTION_DEFS):
        r = xyz[:, d["mcp"], :] - xyz[:, d["wrist"], :]  # metacarpal
        v = xyz[:, d["pip"], :] - xyz[:, d["mcp"], :]    # proximal

        # Proyectar al plano de la palma (eliminar componente normal)
        r_proj = r - np.sum(r * palm_n, axis=1, keepdims=True) * palm_n
        v_proj = v - np.sum(v * palm_n, axis=1, keepdims=True) * palm_n

        cos_a = np.sum(_normalize(r_proj) * _normalize(v_proj), axis=1)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        abd[:, i] = np.arccos(cos_a)

    # Pulgar: componente fuera del plano
    thumb_dir = xyz[:, THUMB_ABD_DEF["mcp"], :] - xyz[:, THUMB_ABD_DEF["cmc"], :]
    sin_a = np.abs(np.sum(_normalize(thumb_dir) * palm_n, axis=1))
    sin_a = np.clip(sin_a, -1.0, 1.0)
    abd[:, 4] = np.arcsin(sin_a)

    return abd


def compute_joint_angles(xyz):
    """
    xyz: (N, 21, 3)
    Retorna: (N, 20) -- 15 flexion + 5 abduccion en radianes.
    """
    flex = compute_flexion_angles(xyz)
    abd = compute_abduction_angles(xyz)
    return np.hstack([flex, abd])


def trial_median(df, angles):
    """
    Sintesis por Mediana -- El Punto de Trial.

    Agrupa frames por sequence_id y calcula la mediana de cada angulo
    sobre todos los frames del trial. Retorna un DataFrame con una fila
    por trial y las columnas: sequence_id, grasp_type, + ALL_ANGLE_NAMES.

    df:     DataFrame con columnas sequence_id y grasp_type.
    angles: (N, 20) angulos articulares correspondientes a df.
    """
    angle_df = pd.DataFrame(angles, columns=ALL_ANGLE_NAMES, index=df.index)
    angle_df["sequence_id"] = df["sequence_id"].values
    angle_df["grasp_type"]  = df["grasp_type"].values

    medians = angle_df.groupby("sequence_id").agg(
        {**{a: "median" for a in ALL_ANGLE_NAMES}, "grasp_type": "first"}
    ).reset_index()

    return medians


if __name__ == "__main__":
    # Quick sanity check
    from data import load_and_split, get_xyz

    train, _, _ = load_and_split()
    xyz = get_xyz(train).reshape(-1, 21, 3)

    bones = compute_bone_vectors(xyz[:5])
    print(f"Bone vectors shape: {bones.shape}  (esperado: 5, 20, 3)")

    angles = compute_joint_angles(xyz[:5])
    print(f"Joint angles shape: {angles.shape}  (esperado: 5, 20)")
    print(f"Flexion range: [{angles[:, :15].min():.3f}, {angles[:, :15].max():.3f}] rad")
    print(f"Abduction range: [{angles[:, 15:].min():.3f}, {angles[:, 15:].max():.3f}] rad")
