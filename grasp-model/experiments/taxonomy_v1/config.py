"""
config.py -- Synergy-Taxonomy Analysis v1
Todos los parametros del experimento en un solo lugar.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # grasp-model/
CSV_PATH     = PROJECT_ROOT / "data" / "raw" / "hograspnet.csv"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

# ── Split S1 (oficial HOGraspNet) ─────────────────────────────────────────────
TRAIN_SUBJECTS = list(range(11, 74))   # S11-S73
VAL_SUBJECTS   = list(range(1, 11))    # S01-S10
TEST_SUBJECTS  = list(range(74, 100))  # S74-S99

# ── Filtro de Existencia ──────────────────────────────────────────────────────
CONTACT_SUM_THRESHOLD = 0.0  # conservar frames con contact_sum > 0

# ── Joints (orden MediaPipe / HOGraspNet) ─────────────────────────────────────
JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Cadena cinematica: (parent, child) para cada bone.
# 20 bones que conectan los 21 keypoints.
BONE_CONNECTIONS = [
    (0, 1),    # WRIST -> THUMB_CMC
    (1, 2),    # THUMB_CMC -> THUMB_MCP
    (2, 3),    # THUMB_MCP -> THUMB_IP
    (3, 4),    # THUMB_IP -> THUMB_TIP
    (0, 5),    # WRIST -> INDEX_MCP
    (5, 6),    # INDEX_MCP -> INDEX_PIP
    (6, 7),    # INDEX_PIP -> INDEX_DIP
    (7, 8),    # INDEX_DIP -> INDEX_TIP
    (0, 9),    # WRIST -> MIDDLE_MCP
    (9, 10),   # MIDDLE_MCP -> MIDDLE_PIP
    (10, 11),  # MIDDLE_PIP -> MIDDLE_DIP
    (11, 12),  # MIDDLE_DIP -> MIDDLE_TIP
    (0, 13),   # WRIST -> RING_MCP
    (13, 14),  # RING_MCP -> RING_PIP
    (14, 15),  # RING_PIP -> RING_DIP
    (15, 16),  # RING_DIP -> RING_TIP
    (0, 17),   # WRIST -> PINKY_MCP
    (17, 18),  # PINKY_MCP -> PINKY_PIP
    (18, 19),  # PINKY_PIP -> PINKY_DIP
    (19, 20),  # PINKY_DIP -> PINKY_TIP
]

# Articulaciones donde se calcula angulo de flexion:
# (parent, joint, child) -- arccos del producto punto normalizado.
# WRIST (0) y TIPs (4,8,12,16,20) no tienen triplete valido.
ANGLE_TRIPLETS = [
    (0, 1, 2),    # THUMB_CMC
    (1, 2, 3),    # THUMB_MCP
    (2, 3, 4),    # THUMB_IP
    (0, 5, 6),    # INDEX_MCP
    (5, 6, 7),    # INDEX_PIP
    (6, 7, 8),    # INDEX_DIP
    (0, 9, 10),   # MIDDLE_MCP
    (9, 10, 11),  # MIDDLE_PIP
    (10, 11, 12), # MIDDLE_DIP
    (0, 13, 14),  # RING_MCP
    (13, 14, 15), # RING_PIP
    (14, 15, 16), # RING_DIP
    (0, 17, 18),  # PINKY_MCP
    (17, 18, 19), # PINKY_PIP
    (18, 19, 20), # PINKY_DIP
]

ANGLE_NAMES = [
    "THUMB_CMC_flex", "THUMB_MCP_flex", "THUMB_IP_flex",
    "INDEX_MCP_flex", "INDEX_PIP_flex", "INDEX_DIP_flex",
    "MIDDLE_MCP_flex", "MIDDLE_PIP_flex", "MIDDLE_DIP_flex",
    "RING_MCP_flex", "RING_PIP_flex", "RING_DIP_flex",
    "PINKY_MCP_flex", "PINKY_PIP_flex", "PINKY_DIP_flex",
]

# Abduccion: desviacion lateral en el plano de la palma.
# Para dedos largos: (WRIST, MCP_i, PIP_i) -- angulo entre metacarpal y proximal
# proyectados al plano de la palma.
# Para pulgar: componente fuera del plano del metacarpo (palmar abduction en CMC).
ABDUCTION_DEFS = [
    # (wrist, mcp, pip) para dedos largos
    {"name": "INDEX_MCP_abd",  "wrist": 0, "mcp": 5,  "pip": 6},
    {"name": "MIDDLE_MCP_abd", "wrist": 0, "mcp": 9,  "pip": 10},
    {"name": "RING_MCP_abd",   "wrist": 0, "mcp": 13, "pip": 14},
    {"name": "PINKY_MCP_abd",  "wrist": 0, "mcp": 17, "pip": 18},
]
THUMB_ABD_DEF = {"name": "THUMB_CMC_abd", "cmc": 1, "mcp": 2}

# Palm normal: cross(kp5 - kp0, kp17 - kp0)
PALM_KP_A = 5   # INDEX_MCP
PALM_KP_B = 17  # PINKY_MCP
PALM_KP_O = 0   # WRIST

ABDUCTION_NAMES = [d["name"] for d in ABDUCTION_DEFS] + [THUMB_ABD_DEF["name"]]

# 20 DOFs totales: 15 flexion + 5 abduccion
ALL_ANGLE_NAMES = ANGLE_NAMES + ABDUCTION_NAMES

# ── PCA (Paso 3) ─────────────────────────────────────────────────────────────
PCA_VARIANCE_TARGET = 0.85  # varianza acumulada objetivo

# ── Clustering (Paso 4) ──────────────────────────────────────────────────────
CLUSTER_K_RANGE = range(3, 13)  # rango de k a explorar

# ── MLP (Paso 5) ─────────────────────────────────────────────────────────────
MLP_HIDDEN_DIMS  = [256, 128]
MLP_DROPOUT      = 0.3
MLP_LR           = 1e-3
MLP_EPOCHS       = 50
MLP_BATCH_SIZE   = 256
MLP_PATIENCE     = 10
MLP_SEED         = 42

# ── Etiquetas HOGraspNet (28 clases) ─────────────────────────────────────────
CLASS_NAMES = {
    0:  "Large_Diameter",       1:  "Small_Diameter",
    2:  "Index_Finger_Extension", 3: "Extension_Type",
    4:  "Parallel_Extension",   5:  "Palmar",
    6:  "Medium_Wrap",          7:  "Adducted_Thumb",
    8:  "Light_Tool",           9:  "Distal",
    10: "Ring",                 11: "Power_Disk",
    12: "Power_Sphere",         13: "Sphere_4_Finger",
    14: "Sphere_3_Finger",      15: "Lateral",
    16: "Stick",                17: "Adduction_Grip",
    18: "Writing_Tripod",       19: "Lateral_Tripod",
    20: "Palmar_Pinch",         21: "Tip_Pinch",
    22: "Inferior_Pincer",      23: "Prismatic_3_Finger",
    24: "Precision_Disk",       25: "Precision_Sphere",
    26: "Quadpod",              27: "Tripod",
}
