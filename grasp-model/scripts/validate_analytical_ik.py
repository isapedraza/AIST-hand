"""
Validación: IK analítico desde XYZ.
Compara pose derivada desde xyz_hog vs xyz_hamer para los mismos frames de S1.
Si r > 0.77 (mejor que Adam IK) → la representación es convention-independent.
"""

import pickle, io
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ─── Cargar MANO rest pose ────────────────────────────────────────────────────

class Dummy:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __setstate__(self, state): pass

class IgnoreChumpy(pickle.Unpickler):
    def find_class(self, module, name):
        if 'chumpy' in module or 'scipy.sparse' in module:
            return Dummy
        return super().find_class(module, name)

MANO_PATH = '/home/yareeez/hamer_ckpt/_DATA/data/mano_v1_2/models/MANO_RIGHT.pkl'
with open(MANO_PATH, 'rb') as f:
    raw = f.read()
mano = IgnoreChumpy(io.BytesIO(raw), encoding='latin1').load()

J_rest_raw = np.array(mano['J'], dtype=float)          # (16, 3)
parents_raw = np.array(mano['kintree_table'][0])        # (16,) uint32
v_template  = np.array(mano['v_template'], dtype=float) # (N, 3)

# Normalizar J_rest: WRIST en origen, escala wrist-INDEX_MCP = 0.1
# MANO joint 1 = INDEX_MCP (según _MANO_POSE_SLICE en tograph.py)
J_rest = J_rest_raw - J_rest_raw[0]
scale_rest = np.linalg.norm(J_rest[1])
J_rest = J_rest * 0.1 / scale_rest

# Normalizar v_template igual
v_template = (v_template - J_rest_raw[0]) * 0.1 / scale_rest

parents = parents_raw.astype(int)
parents[0] = -1  # raíz sin padre

# Verificar TIP vertex indices identificando cuál está más cerca de cada leaf joint
# TIP_VERTEX_IDX candidatos de HaMeR: [745, 317, 444, 556, 673]
# Leaf MANO joints: 3=INDEX_DIP, 6=MIDDLE_DIP, 9=PINKY_DIP, 12=RING_DIP, 15=THUMB_IP
TIP_CANDIDATES = [745, 317, 444, 556, 673]
leaf_joints = [3, 6, 9, 12, 15]

print("TIP vertices en rest pose vs leaf joints:")
for v_idx in TIP_CANDIDATES:
    vp = v_template[v_idx]
    dists = [np.linalg.norm(vp - J_rest[lj]) for lj in leaf_joints]
    closest = leaf_joints[np.argmin(dists)]
    print(f"  v[{v_idx}] = {np.round(vp,3)}  → closest leaf: MANO joint {closest}  (dist={min(dists):.4f})")

# Construcción del mapping TIP: leaf_joint → rest position
# OpenPose TIP order: [4=THUMB, 8=INDEX, 12=MIDDLE, 16=RING, 20=PINKY]
# Haremos el match correcto tras la impresión anterior.
# Para fines de validación, primero derivamos automáticamente:
# v_tip_rest[j] = vértice más cercano al leaf de cada dedo (en orden THUMB,INDEX,MIDDLE,RING,PINKY)
# leaf_by_finger = [THUMB_IP=15, INDEX_DIP=3, MIDDLE_DIP=6, RING_DIP=12, PINKY_DIP=9]
leaf_by_finger = [15, 3, 6, 12, 9]  # orden: THUMB, INDEX, MIDDLE, RING, PINKY

v_tip_rest = np.zeros((5, 3))
for j, lj in enumerate(leaf_by_finger):
    best_v = min(TIP_CANDIDATES, key=lambda v: np.linalg.norm(v_template[v] - J_rest[lj]))
    v_tip_rest[j] = v_template[best_v]

print("\nv_tip_rest asignados (orden: THUMB, INDEX, MIDDLE, RING, PINKY):")
for j, (lj, vtp) in enumerate(zip(leaf_by_finger, v_tip_rest)):
    print(f"  finger {j}: leaf_joint={lj}  tip_rest={np.round(vtp,3)}")

# ─── Mapping OpenPose → MANO joint ───────────────────────────────────────────
# Derivado de _MANO_POSE_SLICE en tograph.py
# OP_TO_MANO[i] = MANO joint index, -1 si es TIP
OP_TO_MANO = [0, 13, 14, 15, -1,  1,  2,  3, -1,  4,  5,  6, -1, 10, 11, 12, -1,  7,  8,  9, -1]

# ─── Rotación swing ──────────────────────────────────────────────────────────

def swing_rotation_matrix(v_from, v_to):
    """Rotación mínima (zero twist) que lleva v_from a v_to."""
    a = v_from / (np.linalg.norm(v_from) + 1e-8)
    b = v_to   / (np.linalg.norm(v_to)   + 1e-8)
    axis = np.cross(a, b)
    sin_t = np.linalg.norm(axis)
    cos_t = np.clip(np.dot(a, b), -1.0, 1.0)
    if sin_t < 1e-8:
        if cos_t > 0:
            return np.eye(3)
        # 180°: eje perpendicular
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        ax = np.cross(a, perp); ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    axis /= sin_t
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)

def rot_to_aa(R):
    """Convierte matriz de rotación 3x3 a axis-angle (3,)."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2 * np.sin(angle))
    return angle * axis

# ─── IK analítico ─────────────────────────────────────────────────────────────

def estimate_global_orient(j_world, J_rest):
    """
    Estima R_global[0] (orientación global del WRIST) desde las posiciones
    de los 5 hijos del WRIST (MCP/CMC joints) usando el algoritmo de Kabsch.
    R_global[0] satisface: j_world[c] ≈ R_global[0] @ J_rest[c]  para c en {1,4,7,10,13}
    """
    wrist_children = [1, 4, 7, 10, 13]
    rest_vecs  = np.array([J_rest[c]  for c in wrist_children])  # (5,3)
    world_vecs = np.array([j_world[c] for c in wrist_children])  # (5,3) — j_world[0]=0
    H = rest_vecs.T @ world_vecs  # (3,3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R


def analytical_ik(xyz_21):
    """
    xyz_21: (21, 3) en orden OpenPose, root-relative, wrist-INDEX_MCP=0.1
    Returns: (45,) axis-angle pose (convention-independent)
    """
    # Convertir a MANO order (16 joints)
    j_world = np.zeros((16, 3))
    for op_i, mano_i in enumerate(OP_TO_MANO):
        if mano_i >= 0:
            j_world[mano_i] = xyz_21[op_i]

    # TIP positions en world (orden: THUMB=4, INDEX=8, MIDDLE=12, RING=16, PINKY=20)
    tip_world = xyz_21[[4, 8, 12, 16, 20]]  # (5, 3)

    R_global = [np.eye(3)] * 16
    # R_global[0] = I: excluimos global orient (igual que R011)

    for i in range(1, 16):
        p = parents[i]
        if p == 0:
            continue  # global orient excluido
        b_rest_i  = J_rest[i]  - J_rest[p]
        b_world_i = j_world[i] - j_world[p]
        R_global[p] = swing_rotation_matrix(b_rest_i, b_world_i)

    # Para leaf joints (DIP×4 + IP_thumb): usar TIP como hijo virtual
    for j, lj in enumerate(leaf_by_finger):
        b_rest_tip  = v_tip_rest[j] - J_rest[lj]
        b_world_tip = tip_world[j]  - j_world[lj]
        R_global[lj] = swing_rotation_matrix(b_rest_tip, b_world_tip)

    # Rotaciones locales: pose[i] = R_global[parent[i]]^T @ R_global[i]
    pose_local = np.zeros(45)
    for i in range(1, 16):
        p = parents[i]
        R_local = R_global[p].T @ R_global[i]
        pose_local[(i-1)*3 : i*3] = rot_to_aa(R_local)

    return pose_local


def analytical_ik_global(xyz_21):
    """
    Variante: global swing -- sin acumulacion de cadena.
    Para cada hueso i, mide directamente swing(b_rest[i], b_world[i]).
    Sin multiplicacion de cadenas → sin propagacion de error.
    """
    j_world = np.zeros((16, 3))
    for op_i, mano_i in enumerate(OP_TO_MANO):
        if mano_i >= 0:
            j_world[mano_i] = xyz_21[op_i]
    tip_world = xyz_21[[4, 8, 12, 16, 20]]

    pose_global = np.zeros(45)
    for i in range(1, 16):
        p = parents[i]
        b_rest_i  = J_rest[i]  - J_rest[p]
        b_world_i = j_world[i] - j_world[p]
        R_swing = swing_rotation_matrix(b_rest_i, b_world_i)
        pose_global[(i-1)*3 : i*3] = rot_to_aa(R_swing)

    # Leaf joints: swing del hueso leaf→TIP
    for j, lj in enumerate(leaf_by_finger):
        p = lj
        b_rest_tip  = v_tip_rest[j] - J_rest[p]
        b_world_tip = tip_world[j]  - j_world[p]
        R_swing = swing_rotation_matrix(b_rest_tip, b_world_tip)
        # lj es joint 3,6,9,12,15 → pose index (lj-1)*3
        pose_global[(lj-1)*3 : lj*3] = rot_to_aa(R_swing)

    return pose_global

# ─── Cargar CSVs y comparar ───────────────────────────────────────────────────

HOG_CSV   = '/home/yareeez/AIST-hand/grasp-model/data/raw/hograspnet_mano.csv'
HAMER_CSV = '/home/yareeez/AIST-hand/grasp-model/data/raw/hograspnet_hamer_s1.csv'

# Nombres de joints en orden OpenPose (igual en ambos CSVs)
JOINT_NAMES = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
]
xyz_cols = [f'{j}_{ax}' for j in JOINT_NAMES for ax in ('x','y','z')]
key_cols  = ['sequence_id', 'cam', 'frame_id'] + xyz_cols

print("\nCargando CSVs...")
df_hog   = pd.read_csv(HOG_CSV,   usecols=key_cols)
df_hamer = pd.read_csv(HAMER_CSV, usecols=key_cols + ['grasp_type'])

# Merge por frame
df = df_hog.merge(df_hamer, on=['sequence_id', 'cam', 'frame_id'], suffixes=('_hog', '_hamer'))
print(f"Frames matcheados: {len(df):,}")

# Subsample para velocidad de prueba
N = min(500, len(df))
df_sample = df.sample(n=N, random_state=42)

print(f"Corriendo IK analítico sobre {N} frames...")

poses_hog   = []
poses_hamer = []

for _, row in df_sample.iterrows():
    xyz_hog = np.array([[row[f'{j}_x_hog'], row[f'{j}_y_hog'], row[f'{j}_z_hog']] for j in JOINT_NAMES])
    xyz_ham = np.array([[row[f'{j}_x_hamer'], row[f'{j}_y_hamer'], row[f'{j}_z_hamer']] for j in JOINT_NAMES])
    poses_hog.append(analytical_ik(xyz_hog))
    poses_hamer.append(analytical_ik(xyz_ham))

poses_hog_g   = []
poses_hamer_g = []
for _, row in df_sample.iterrows():
    xyz_hog = np.array([[row[f'{j}_x_hog'], row[f'{j}_y_hog'], row[f'{j}_z_hog']] for j in JOINT_NAMES])
    xyz_ham = np.array([[row[f'{j}_x_hamer'], row[f'{j}_y_hamer'], row[f'{j}_z_hamer']] for j in JOINT_NAMES])
    poses_hog_g.append(analytical_ik_global(xyz_hog))
    poses_hamer_g.append(analytical_ik_global(xyz_ham))
poses_hog_g   = np.array(poses_hog_g)
poses_hamer_g = np.array(poses_hamer_g)

poses_hog   = np.array(poses_hog)    # (N, 45)
poses_hamer = np.array(poses_hamer)  # (N, 45)

# Correlación y RMSE
flat_hog   = poses_hog.flatten()
flat_hamer = poses_hamer.flatten()
r, _    = pearsonr(flat_hog, flat_hamer)
rmse    = np.sqrt(np.mean((flat_hog - flat_hamer) ** 2))

print(f"\n{'='*50}")
print(f"RESULTADO IK ANALÍTICO")
print(f"{'='*50}")
print(f"  r    = {r:.4f}  (Adam IK baseline: 0.77)")
print(f"  RMSE = {rmse:.4f}")
print(f"  {'✓ MEJOR que Adam IK' if r > 0.77 else '✗ No mejor que Adam IK'}")

# Correlación por dimensión (45 params)
per_dim_r = [pearsonr(poses_hog[:, d], poses_hamer[:, d])[0] for d in range(45)]
print(f"\n  r por dim: min={min(per_dim_r):.3f}  median={np.median(per_dim_r):.3f}  max={max(per_dim_r):.3f}")
print(f"  dims con r > 0.8: {sum(r_d > 0.8 for r_d in per_dim_r)}/45")
print(f"  dims con r < 0.5: {sum(r_d < 0.5 for r_d in per_dim_r)}/45")

# Identificar qué joints corresponden a las dims con mayor/menor correlación
joint_names_mano = [
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP',
    'PINKY_MCP',  'PINKY_PIP',  'PINKY_DIP',
    'RING_MCP',   'RING_PIP',   'RING_DIP',
    'THUMB_CMC',  'THUMB_MCP',  'THUMB_IP',
]
print("\n  Correlación por joint (media de 3 axis-angle dims):")
for ji, jname in enumerate(joint_names_mano):
    dims = [ji*3, ji*3+1, ji*3+2]
    r_mean = np.mean([per_dim_r[d] for d in dims])
    print(f"    {jname:20s}  r_mean={r_mean:.3f}")

# Comparar con HOGraspNet pose original
pose_cols = [f'MANO_pose_{i:02d}' for i in range(45)]
df_hog_pose   = pd.read_csv(HOG_CSV,   usecols=['sequence_id','cam','frame_id'] + pose_cols)
df_hamer_pose = pd.read_csv(HAMER_CSV, usecols=['sequence_id','cam','frame_id'] + pose_cols)
df_orig = df_hog_pose.merge(df_hamer_pose, on=['sequence_id','cam','frame_id'], suffixes=('_hog','_hamer'))
df_orig_sample = df_orig[df_orig[['sequence_id','cam','frame_id']].apply(tuple,axis=1).isin(
    df_sample[['sequence_id','cam','frame_id']].apply(tuple,axis=1)
)]
if len(df_orig_sample) > 0:
    orig_hog   = df_orig_sample[[f'MANO_pose_{i:02d}_hog'   for i in range(45)]].values.flatten()
    orig_hamer = df_orig_sample[[f'MANO_pose_{i:02d}_hamer' for i in range(45)]].values.flatten()
    r_orig, _ = pearsonr(orig_hog, orig_hamer)
    print(f"\n  Referencia — r pose ORIGINAL (hog vs hamer): {r_orig:.4f}")

r_g, _ = pearsonr(poses_hog_g.flatten(), poses_hamer_g.flatten())
rmse_g = np.sqrt(np.mean((poses_hog_g.flatten() - poses_hamer_g.flatten())**2))
per_dim_r_g = [pearsonr(poses_hog_g[:, d], poses_hamer_g[:, d])[0] for d in range(45)]
print(f"\n{'='*50}")
print(f"GLOBAL SWING (sin acumulacion de cadena)")
print(f"{'='*50}")
print(f"  r    = {r_g:.4f}")
print(f"  RMSE = {rmse_g:.4f}")
print(f"  r por dim: min={min(per_dim_r_g):.3f}  median={np.median(per_dim_r_g):.3f}  max={max(per_dim_r_g):.3f}")
print(f"  dims con r > 0.8: {sum(r_d > 0.8 for r_d in per_dim_r_g)}/45")
print("  r por joint:")
for ji, jname in enumerate(joint_names_mano):
    dims = [ji*3, ji*3+1, ji*3+2]
    r_mean = np.mean([per_dim_r_g[d] for d in dims])
    print(f"    {jname:20s}  r_mean={r_mean:.3f}")
    print(f"  IK analítico {'MEJOR' if r > r_orig else 'peor'} que pose original")
