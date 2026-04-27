#!/usr/bin/env python3
"""
Stage 1 training: shared latent space for cross-embodiment retargeting.
Yan et al. (2026) adaptation for dexterous hands.

Usage (local):
    python train_cross_emb.py

Usage (Colab):
    python train_cross_emb.py \
        --repo_root /content/AIST-hand \
        --dex_root  /content/drive/MyDrive/AIST-hand/dex-urdf \
        --csv_path  /content/drive/MyDrive/AIST-hand/hograspnet_abl11.csv \
        --ckpt_path /content/drive/MyDrive/AIST-hand/checkpoints/stage1_latest.pt \
        --b 20000 --n_steps 5000 --log_every 50 --ckpt_every 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--repo_root", default="/home/yareeez/AIST-hand")
    p.add_argument("--dex_root",  default="/home/yareeez/dex-urdf")
    p.add_argument("--csv_path",  default=None, help="Override CSV path")
    p.add_argument("--ckpt_path", default=None, help="Override checkpoint path")
    p.add_argument("--hand_config", default=None, help="Override hand config YAML")
    # Training
    p.add_argument("--b",          type=int,   default=1000)
    p.add_argument("--n_steps",    type=int,   default=10)
    p.add_argument("--log_every",  type=int,   default=1)
    p.add_argument("--ckpt_every", type=int,   default=5)
    p.add_argument("--lr_warmup",  type=int,   default=500)
    # Hyperparams
    p.add_argument("--z_dim",      type=int,   default=16)
    p.add_argument("--shared_dim", type=int,   default=1024)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda_c",   type=float, default=10.0)
    p.add_argument("--lambda_rec", type=float, default=5.0)
    p.add_argument("--lambda_ltc", type=float, default=1.0)
    p.add_argument("--lambda_tmp", type=float, default=0.1)
    p.add_argument("--n_triplets", type=int,   default=2048)
    p.add_argument("--margin",     type=float, default=0.3)
    return p.parse_args()


def main():
    args = p = _parse_args()

    REPO_ROOT = Path(args.repo_root)
    DEX_ROOT  = Path(args.dex_root)

    CSV_PATH    = Path(args.csv_path)    if args.csv_path    else REPO_ROOT / "grasp-model/data/processed/hograspnet_abl11.csv"
    CKPT_PATH   = Path(args.ckpt_path)  if args.ckpt_path   else REPO_ROOT / "grasp-model/checkpoints/stage1_latest.pt"
    HAND_CONFIG = Path(args.hand_config) if args.hand_config else REPO_ROOT / "grasp-model/data/hand_configs/shadow_hand_right.yaml"
    URDF_PATH   = DEX_ROOT / "robots/hands/shadow_hand/shadow_hand_right.urdf"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE} | B={args.b} | N_STEPS={args.n_steps} | margin={args.margin}")

    # Add scripts to path
    sys.path.insert(0, str(REPO_ROOT / "grasp-model/scripts"))
    sys.path.insert(0, str(REPO_ROOT / "grasp-model/src/cross_emb"))

    from cross_embodiment_sampler import CrossEmbodimentSampler
    from human_modules import HumanEncoder_E_h
    from robot_modules import RobotEncoder_E_r, RobotDecoder_D_r
    from shared_modules import SharedEncoder_E_X, SharedDecoder_D_X

    # ---------------------------------------------------------------------------
    # Sampler
    # ---------------------------------------------------------------------------
    sampler = CrossEmbodimentSampler(
        csv_path         = CSV_PATH,
        urdf_path        = URDF_PATH,
        hand_config_path = HAND_CONFIG,
        split            = "train",
        device           = DEVICE,
    )

    _probe = sampler.get_batch_temporal(1)
    J      = _probe["q_r"].shape[1]
    print(f"Robot joints J={J}")

    # ---------------------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------------------
    E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=args.z_dim).to(DEVICE)
    E_r = RobotEncoder_E_r(n_joints=J, shared_dim=args.shared_dim).to(DEVICE)
    E_X = SharedEncoder_E_X(shared_dim=args.shared_dim, z_dim=args.z_dim).to(DEVICE)
    D_X = SharedDecoder_D_X(z_dim=args.z_dim, shared_dim=args.shared_dim).to(DEVICE)
    D_r = RobotDecoder_D_r(n_joints=J, shared_dim=args.shared_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(E_h.parameters()) + list(E_r.parameters()) +
        list(E_X.parameters()) + list(D_X.parameters()) +
        list(D_r.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.n_steps - args.lr_warmup), eta_min=1e-5
    )

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for step in range(args.n_steps):

        batch          = sampler.get_batch_temporal(args.b)
        quats_h        = batch["quats_h"]
        quats_h_t1     = batch["quats_h_t1"]
        q_r            = batch["q_r"]
        quats_h_sub    = batch["quats_h_sub"]
        quats_r_sub    = batch["quats_r_sub"]
        tips_h_sub     = batch["tips_h_sub"]
        tips_r_sub     = batch["tips_r_sub"]
        tips_h_t1      = batch["tips_h_t1"]
        common_fingers = batch["common_fingers"]

        z_t  = E_h(quats_h)
        z_t1 = E_h(quats_h_t1)
        z_r  = E_X(E_r(q_r))

        q_r_hat    = D_r(D_X(z_r))
        z_h_rt     = E_X(D_X(z_t))
        q_r_hat_t1 = D_r(D_X(z_t1))

        L_rec = (q_r - q_r_hat).norm(dim=-1).mean()
        L_ltc = (z_t - z_h_rt).norm(dim=-1).mean()

        z_all    = torch.cat([z_t, z_r], dim=0)
        all_q    = torch.cat([quats_h_sub, quats_r_sub], dim=0)
        all_tips = torch.cat([tips_h_sub.flatten(1), tips_r_sub.flatten(1)], dim=0)
        B2  = z_all.shape[0]
        n   = min(args.n_triplets, B2)
        idx = torch.randperm(B2, device=DEVICE)[:n]
        zs  = z_all[idx];  qs = all_q[idx];  ts = all_tips[idx]
        dot     = (qs.unsqueeze(1) * qs.unsqueeze(0)).sum(-1)
        D_R     = (1 - dot ** 2).sum(-1)
        D_ee    = (ts.unsqueeze(1) - ts.unsqueeze(0)).norm(dim=-1)
        S       = D_R + D_ee
        eye     = torch.eye(n, dtype=torch.bool, device=DEVICE)
        pos_idx = S.masked_fill(eye, float('inf')).argmin(dim=1)
        neg_idx = S.masked_fill(eye, float('-inf')).argmax(dim=1)
        L_cont  = torch.relu((zs - zs[pos_idx]).norm(dim=-1) - (zs - zs[neg_idx]).norm(dim=-1) + args.margin).mean()

        fk_t,  fk_t1  = sampler.robot_rnd.run_fk(q_r_hat), sampler.robot_rnd.run_fk(q_r_hat_t1)
        _, _, meta_t  = sampler.robot_rnd.run_dong_stage2(fk_t,  HAND_CONFIG)
        _, _, meta_t1 = sampler.robot_rnd.run_dong_stage2(fk_t1, HAND_CONFIG)
        tip_labels    = meta_t["tip_labels"]
        common_idx_r  = [tip_labels.index(f) for f in common_fingers]
        human_labels  = ["thumb", "index", "middle", "ring", "pinky"]
        common_idx_h  = [human_labels.index(f) for f in common_fingers]
        tips_r_t_sub  = meta_t["tips"].to(DEVICE)[:, common_idx_r, :]
        tips_r_t1_sub = meta_t1["tips"].to(DEVICE)[:, common_idx_r, :]
        tips_h_t1_sub = tips_h_t1[:, common_idx_h, :]
        L_temp = ((tips_h_t1_sub - tips_h_sub) - (tips_r_t1_sub - tips_r_t_sub)).norm(dim=-1).mean()

        L_total = args.lambda_c * L_cont + args.lambda_rec * L_rec + args.lambda_ltc * L_ltc + args.lambda_tmp * L_temp

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        if step > args.lr_warmup:
            scheduler.step()

        if step % args.log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"step {step:05d} | total={L_total.item():.4f} | cont={L_cont.item():.4f} rec={L_rec.item():.4f} ltc={L_ltc.item():.4f} temp={L_temp.item():.4f} | lr={lr_now:.2e}")

        if step % args.ckpt_every == 0:
            torch.save({
                "step": step,
                "E_h":  E_h.state_dict(),
                "E_r":  E_r.state_dict(),
                "E_X":  E_X.state_dict(),
                "D_X":  D_X.state_dict(),
                "D_r":  D_r.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, CKPT_PATH)

    print("done")


if __name__ == "__main__":
    main()
