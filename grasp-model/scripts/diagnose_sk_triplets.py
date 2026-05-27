"""
diagnose_sk_triplets.py -- Entry 86 supplement

Hipotesis: con pesos 1/sigma en S_k, poses humanas de puno se emparejan
con poses robot de MCP bajo (mano abierta). Con sigma, se emparejan con
MCP alto (puno). Si se confirma, explica la brecha de 50 deg de Entry 86.

Metodologia:
  - Humano: HaGRID clase=29 (puno) -- chain + quats desde CSV
  - Robot: muestra estratificada del NPZ (fist MCP>1.0, open MCP<0.3)
  - S_k = D_R + D_joints (D_ahg excluido -- mismo peso en ambos esquemas)
  - Para cada pose humana: argmin S_k = positivo seleccionado
  - Comparar MCP de positivos bajo 1/sigma vs sigma

Supuesto de frame: chain humana y robot ambas normalizadas por hand_length
(mismo pipeline que human_loader.py). Coherencia verificada implicitamente
por el hecho de que Run 20 entrena con este esquema.

Uso:
  python scripts/diagnose_sk_triplets.py
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from human_loader import _CHAIN_COLS, _QUAT_COLS

# ── Paths ──────────────────────────────────────────────────────────────
HAGRID_CSV = Path("/media/yareeez/94649A33649A1856/DATOS DE TESIS/"
                  "AIST-hand_grasp-model_data_cache/processed/hagrid_dong.csv")
ROBOT_NPZ  = Path("/media/yareeez/94649A33649A1856/DATOS DE TESIS/"
                  "AIST-hand_grasp-model_data_cache/processed/"
                  "valid_robot_poses_eigengrasp_dong.npz")

# ── S_k weights (same order: [mcp, pip, dip, tip=0]) ──────────────────
# 1/sigma (current, Run 20)
W_DR_INV = {
    "thumb":  [0.258, 0.544, 0.199, 0.0],
    "index":  [0.329, 0.325, 0.346, 0.0],
    "middle": [0.188, 0.362, 0.451, 0.0],
    "ring":   [0.238, 0.357, 0.405, 0.0],
    "pinky":  [0.197, 0.405, 0.398, 0.0],
}
W_DJ_INV = {  # [mcp_pos, pip_pos, dip_pos, tip_pos]
    "thumb":  [0.4499, 0.2534, 0.1484, 0.1484],
    "index":  [0.5282, 0.2435, 0.1381, 0.0902],
    "middle": [0.5630, 0.2259, 0.1267, 0.0844],
    "ring":   [0.5743, 0.2364, 0.1134, 0.0759],
    "pinky":  [0.5459, 0.2465, 0.1241, 0.0835],
}
# sigma (Run 24)
W_DR_SIG = {
    "thumb":  [0.3609, 0.1712, 0.4679, 0.0],
    "index":  [0.3375, 0.3416, 0.3209, 0.0],
    "middle": [0.5165, 0.2682, 0.2153, 0.0],
    "ring":   [0.4436, 0.2957, 0.2607, 0.0],
    "pinky":  [0.5047, 0.2455, 0.2498, 0.0],
}
W_DJ_SIG = {
    "thumb":  [0.1131, 0.2009, 0.3430, 0.3430],
    "index":  [0.0778, 0.1688, 0.2977, 0.4557],
    "middle": [0.0685, 0.1706, 0.3042, 0.4567],
    "ring":   [0.0623, 0.1513, 0.3153, 0.4711],
    "pinky":  [0.0707, 0.1565, 0.3108, 0.4620],
}

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# MCP flex joints in q24: FFJ3=3, MFJ3=7, RFJ3=11, LFJ3=16
MCP_FLEX_IDX = [3, 7, 11, 16]


def load_human_fist(n=200):
    """Load n fist poses from HaGRID class=29."""
    df = pd.read_csv(HAGRID_CSV)
    fist = df[df["grasp_type"] == 29].copy()

    # hand_length: wrist to middle tip (same as human_loader.py)
    wrist = fist[["WRIST_x", "WRIST_y", "WRIST_z"]].values
    mtip  = fist[["MIDDLE_FINGER_TIP_x", "MIDDLE_FINGER_TIP_y",
                   "MIDDLE_FINGER_TIP_z"]].values
    hl = np.linalg.norm(mtip - wrist, axis=1, keepdims=True)  # [N, 1]
    hl = np.maximum(hl, 1e-6)

    chain = fist[_CHAIN_COLS].values.astype(np.float32).reshape(-1, 5, 4, 3)
    chain = chain / hl[:, None, None, :]  # normalize by hand_length

    quats = fist[_QUAT_COLS].values.astype(np.float32).reshape(-1, 20, 4)

    idx = np.random.choice(len(fist), min(n, len(fist)), replace=False)
    return chain[idx], quats[idx]  # [N, 5, 4, 3], [N, 20, 4]


def load_robot_random(n=2000):
    """Load n robot poses sampled uniformly from full NPZ (continuous distribution)."""
    npz = np.load(ROBOT_NPZ)
    q    = npz["q"]       # [10M, 24] radians
    chain = npz["chain"]  # [10M, 5, 4, 3] normalized
    quats = npz["quats"]  # [10M, 15, 4]

    mcp_mean = q[:, MCP_FLEX_IDX].mean(axis=1)

    idx = np.random.choice(len(q), min(n, len(q)), replace=False)
    print(f"  Robot pool (random {n} from {len(q):,}): "
          f"MCP mean={mcp_mean[idx].mean():.3f} rad "
          f"({mcp_mean[idx].mean()*57.3:.1f} deg), "
          f"std={mcp_mean[idx].std():.3f}")

    return chain[idx], quats[idx], mcp_mean[idx]


def compute_sk(chain_h, quats_h, chain_r, quats_r, w_dr, w_dj):
    """
    S_k(h, r) = sum_subspace [D_R_sub(h,r) + D_joints_sub(h,r)]

    chain_h : [Nh, 5, 4, 3]
    quats_h : [Nh, 20, 4]   -- 4 per finger (MCP,PIP,DIP,TIP), TIP=identity
    chain_r : [Nr, 5, 4, 3]
    quats_r : [Nr, 15, 4]   -- 3 per finger (MCP,PIP,DIP), no TIP

    Returns S_k : [Nh, Nr]
    """
    Nh, Nr = len(chain_h), len(chain_r)
    S = np.zeros((Nh, Nr), dtype=np.float32)

    for fi, fname in enumerate(FINGERS):
        wr = np.array(w_dr[fname], dtype=np.float32)  # [4] (tip=0)
        wj = np.array(w_dj[fname], dtype=np.float32)  # [4]

        # D_R: 1 - dot(q_h, q_r)^2 per segment (MCP=0, PIP=1, DIP=2; TIP excluded)
        for seg in range(3):  # mcp, pip, dip
            if wr[seg] == 0.0:
                continue
            h_qi = fi * 4 + seg   # index in [Nh, 20, 4]
            r_qi = fi * 3 + seg   # index in [Nr, 15, 4]
            qh = quats_h[:, h_qi, :]  # [Nh, 4]
            qr = quats_r[:, r_qi, :]  # [Nr, 4]
            dot = (qh[:, None, :] * qr[None, :, :]).sum(axis=-1)  # [Nh, Nr]
            S += wr[seg] * (1.0 - dot ** 2)

        # D_joints: ||chain_h_j - chain_r_j|| per segment [mcp,pip,dip,tip]
        for seg in range(4):  # mcp, pip, dip, tip
            if wj[seg] == 0.0:
                continue
            ch = chain_h[:, fi, seg, :]  # [Nh, 3]
            cr = chain_r[:, fi, seg, :]  # [Nr, 3]
            diff = ch[:, None, :] - cr[None, :, :]  # [Nh, Nr, 3]
            dist = np.linalg.norm(diff, axis=-1)     # [Nh, Nr]
            S += wj[seg] * dist

    return S


def run_diagnostic():
    np.random.seed(42)
    print("\n=== S_k triplet selection diagnostic (continuous distribution) ===")
    print("Supuesto: human fist (HaGRID class=29) deberia emparejar con robot")
    print("de MCP alto. Esquema que selecciona positivos con MCP mas alto = mejor.\n")

    print("Loading human fist poses (HaGRID class=29)...")
    chain_h, quats_h = load_human_fist(n=200)
    print(f"  Human fist poses: {len(chain_h)}")

    print("\nLoading robot poses (random continuous distribution)...")
    chain_r, quats_r, mcp_r = load_robot_random(n=2000)
    print(f"  Pool MCP p10={np.percentile(mcp_r,10):.3f}  "
          f"p50={np.percentile(mcp_r,50):.3f}  "
          f"p90={np.percentile(mcp_r,90):.3f} rad")

    results = {}
    for scheme_name, w_dr, w_dj in [
        ("1/sigma (Run 20)", W_DR_INV, W_DJ_INV),
        ("sigma   (Run 24)", W_DR_SIG, W_DJ_SIG),
    ]:
        print(f"\n--- Scheme: {scheme_name} ---")
        S = compute_sk(chain_h, quats_h, chain_r, quats_r, w_dr, w_dj)

        pos_idx = S.argmin(axis=1)       # [Nh] index into robot pool
        pos_mcp = mcp_r[pos_idx]         # [Nh] MCP rad of selected positive

        mean_mcp = pos_mcp.mean()
        results[scheme_name] = mean_mcp
        print(f"  Mean MCP of selected positives : {mean_mcp:.4f} rad "
              f"({mean_mcp*57.3:.1f} deg)")
        print(f"  Std                            : {pos_mcp.std():.4f} rad")
        print(f"  p10 / p50 / p90                : "
              f"{np.percentile(pos_mcp,10)*57.3:.1f} / "
              f"{np.percentile(pos_mcp,50)*57.3:.1f} / "
              f"{np.percentile(pos_mcp,90)*57.3:.1f} deg")
        print(f"  S_k range: [{S.min():.4f}, {S.max():.4f}]")

    print("\n=== Interpretation ===")
    vals = list(results.values())
    diff = (vals[1] - vals[0]) * 57.3
    print(f"  MCP difference (sigma - 1/sigma): {diff:+.1f} deg")
    if abs(diff) < 2.0:
        print("  -> SIMILAR: S_k weight scheme no afecta calidad de positivos.")
        print("     Bottleneck es otro: margin, L_cont strength, o E_h.")
    elif diff > 0:
        print("  -> sigma selecciona positivos con MCP MAS ALTO.")
        print("     Sigma da mejor senal contrastiva para fist. Run 24 justificado.")
    else:
        print("  -> 1/sigma selecciona positivos con MCP MAS ALTO (inesperado).")


if __name__ == "__main__":
    run_diagnostic()
