#!/usr/bin/env python3
"""
Stage 1 training: shared latent space for cross-embodiment retargeting.
Yan et al. (2026) adaptation for dexterous hands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from cross_embodiment_sampler import CrossEmbodimentSampler
from quat_to_graph import QuatToGraph
from human_modules import HumanEncoder_E_h
from robot_modules import RobotEncoder_E_r, RobotDecoder_D_r
from shared_modules import SharedEncoder_E_X, SharedDecoder_D_X

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Change REPO_ROOT to your local path or Google Drive mount point in Colab.
REPO_ROOT  = Path("/home/yareeez/AIST-hand")
DEX_ROOT   = Path("/home/yareeez/dex-urdf")   # separate repo with URDFs

CSV_PATH    = REPO_ROOT / "grasp-model/data/processed/hograspnet_abl11.csv"
URDF_PATH   = DEX_ROOT  / "robots/hands/shadow_hand/shadow_hand_right.urdf"
HAND_CONFIG = REPO_ROOT / "grasp-model/data/hand_configs/shadow_hand_right.yaml"
CKPT_PATH   = REPO_ROOT / "grasp-model/checkpoints/stage1_latest.pt"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
B          = 1000    # batch size (target: 1e5, start small)
N_STEPS    = 10      # training steps (target: thousands)
LOG_EVERY  = 1       # print losses every N steps
CKPT_EVERY = 5       # save checkpoint every N steps

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
sampler = CrossEmbodimentSampler(
    csv_path         = CSV_PATH,
    urdf_path        = URDF_PATH,
    hand_config_path = HAND_CONFIG,
    split            = "train",
    device           = DEVICE,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
Z_DIM      = 16
SHARED_DIM = 1024
LR         = 1e-3
LAMBDA_C   = 10.0
LAMBDA_REC = 5.0
LAMBDA_LTC = 1.0
LAMBDA_TMP = 0.1

# probe batch to get J (number of robot joints) without hardcoding
_probe = sampler.get_batch_temporal(1)
J      = _probe["q_r"].shape[1]

quat_to_graph = QuatToGraph()
E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, heads=4, z_dim=Z_DIM).to(DEVICE)
E_r = RobotEncoder_E_r(n_joints=J, shared_dim=SHARED_DIM).to(DEVICE)
E_X = SharedEncoder_E_X(shared_dim=SHARED_DIM, z_dim=Z_DIM).to(DEVICE)
D_X = SharedDecoder_D_X(z_dim=Z_DIM, shared_dim=SHARED_DIM).to(DEVICE)
D_r = RobotDecoder_D_r(n_joints=J, shared_dim=SHARED_DIM).to(DEVICE)

optimizer = torch.optim.Adam(
    list(E_h.parameters()) + list(E_r.parameters()) +
    list(E_X.parameters()) + list(D_X.parameters()) +
    list(D_r.parameters()),
    lr=LR,
)

# ---------------------------------------------------------------------------
# Training loop skeleton
# ---------------------------------------------------------------------------
for step in range(N_STEPS):

    # === SAMPLER ===
    batch = sampler.get_batch_temporal(B)

    quats_h        = batch["quats_h"]        # [B, 20, 4]
    quats_h_t1     = batch["quats_h_t1"]     # [B, 20, 4]
    q_r            = batch["q_r"]            # [B, J]
    quats_h_sub    = batch["quats_h_sub"]    # [B, K, 4]
    quats_r_sub    = batch["quats_r_sub"]    # [B, K, 4]
    tips_h_sub     = batch["tips_h_sub"]     # [B, Fc, 3]
    tips_r_sub     = batch["tips_r_sub"]     # [B, Fc, 3]
    tips_h_t1      = batch["tips_h_t1"]      # [B, Fh, 3]
    common_labels  = batch["common_labels"]  # list[str]
    common_fingers = batch["common_fingers"] # list[str]

    # === HUMAN ===
    quats_both = torch.cat([quats_h, quats_h_t1], dim=0)          # [2B, 20, 4]
    graph      = quat_to_graph(quats_both)                          # PyG Batch de 2B grafos
    z_both     = E_h(graph.x, graph.edge_index, graph.batch)       # [2B, z_dim]
    z_t        = z_both[:B]                                         # [B, z_dim] frame t
    z_t1       = z_both[B:]                                         # [B, z_dim] frame t+1

    # === ROBOT ===
    h_r = E_r(q_r)          # [B, 1024]
    z_r = E_X(h_r)          # [B, 16]

    # === DECODE ===
    h_dec   = D_X(z_r)      # [B, 1024]     — para L_rec
    q_r_hat = D_r(h_dec)    # [B, J]        — para L_rec

    h_ltc      = D_X(z_t)              # [B, 1024]  — para L_ltc
    z_h_rt     = E_X(h_ltc)            # [B, 16]    — round-trip humano

    q_r_hat_t1 = D_r(D_X(z_t1))       # [B, J]     — robot retargeteado en t+1

    # === LOSSES ===
    # L_rec: robot reconstruction ‖q_r - q_r_hat‖
    L_rec = (q_r - q_r_hat).norm(dim=-1).mean()

    # L_ltc: human latent consistency ‖z_t - E_X(D_X(z_t))‖
    L_ltc = (z_t - z_h_rt).norm(dim=-1).mean()

    # L_contrastive: triplet loss sobre latentes humano+robot
    z_all    = torch.cat([z_t, z_r], dim=0)                              # [2B, 16]
    all_q    = torch.cat([quats_h_sub, quats_r_sub], dim=0)              # [2B, K, 4]
    all_tips = torch.cat([tips_h_sub.flatten(1), tips_r_sub.flatten(1)], dim=0)  # [2B, Fc*3]
    dot      = (all_q.unsqueeze(1) * all_q.unsqueeze(0)).sum(-1)         # [2B, 2B, K]
    D_R      = (1 - dot ** 2).sum(-1)                                    # [2B, 2B]
    D_ee     = (all_tips.unsqueeze(1) - all_tips.unsqueeze(0)).norm(dim=-1)  # [2B, 2B]
    S        = D_R + D_ee
    B2       = z_all.shape[0]
    eye      = torch.eye(B2, dtype=torch.bool, device=DEVICE)
    pos_idx  = S.masked_fill(eye, float('inf')).argmin(dim=1)
    neg_idx  = S.masked_fill(eye, float('-inf')).argmax(dim=1)
    d_pos    = (z_all - z_all[pos_idx]).norm(dim=-1)
    d_neg    = (z_all - z_all[neg_idx]).norm(dim=-1)
    L_cont   = torch.relu(d_pos - d_neg + 0.05).mean()

    # L_temporal: velocidad fingertip humano vs robot retargeteado
    fk_t   = sampler.robot_rnd.run_fk(q_r_hat)
    fk_t1  = sampler.robot_rnd.run_fk(q_r_hat_t1)
    _, _, meta_t  = sampler.robot_rnd.run_dong_stage2(fk_t,  HAND_CONFIG)
    _, _, meta_t1 = sampler.robot_rnd.run_dong_stage2(fk_t1, HAND_CONFIG)
    tips_r_t  = meta_t["tips"].to(DEVICE)    # [B, F, 3]
    tips_r_t1 = meta_t1["tips"].to(DEVICE)   # [B, F, 3]
    # seleccionar dedos comunes
    common_fingers = batch["common_fingers"]
    tip_labels     = meta_t["tip_labels"]
    common_idx_r   = [tip_labels.index(f) for f in common_fingers]
    tips_r_t_sub   = tips_r_t[:, common_idx_r, :]    # [B, Fc, 3]
    tips_r_t1_sub  = tips_r_t1[:, common_idx_r, :]   # [B, Fc, 3]
    # velocidades
    human_tip_labels = ["thumb", "index", "middle", "ring", "pinky"]
    common_idx_h     = [human_tip_labels.index(f) for f in common_fingers]
    tips_h_t1_sub    = tips_h_t1[:, common_idx_h, :]   # [B, Fc, 3]
    v_robot = tips_r_t1_sub - tips_r_t_sub              # [B, Fc, 3]
    v_human = tips_h_t1_sub - tips_h_sub                # [B, Fc, 3]
    L_temp  = (v_human - v_robot).norm(dim=-1).mean()

    L_total = LAMBDA_C * L_cont + LAMBDA_REC * L_rec + LAMBDA_LTC * L_ltc + LAMBDA_TMP * L_temp

    # === BACKWARD ===
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()

    if step % LOG_EVERY == 0:
        print(f"step {step:04d} | total={L_total.item():.4f} | cont={L_cont.item():.4f} rec={L_rec.item():.4f} ltc={L_ltc.item():.4f} temp={L_temp.item():.4f}")

    if step % CKPT_EVERY == 0:
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step":      step,
            "E_h":       E_h.state_dict(),
            "E_r":       E_r.state_dict(),
            "E_X":       E_X.state_dict(),
            "D_X":       D_X.state_dict(),
            "D_r":       D_r.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, CKPT_PATH)

print("done")
