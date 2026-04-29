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
    p.add_argument("--hand_config",       default=None, help="Override hand config YAML")
    p.add_argument("--valid_poses_path",  default=None, help="Path to valid_robot_poses.npz (mode=VALID_NPZ). If omitted, uses random uniform sampling.")
    p.add_argument("--extra_human_csv",   default=None, help="Optional static human anchor CSV, e.g. HaGRID open/fist Dong features.")
    # Training
    p.add_argument("--b",          type=int,   default=1000)
    p.add_argument("--n_steps",    type=int,   default=10)
    p.add_argument("--log_every",  type=int,   default=1)
    p.add_argument("--ckpt_every", type=int,   default=5)
    p.add_argument("--lr_warmup",  type=int,   default=500)
    # Hyperparams
    p.add_argument("--z_dim",      type=int,   default=64)
    p.add_argument("--shared_dim", type=int,   default=1024)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda_c",   type=float, default=10.0)
    p.add_argument("--lambda_rec", type=float, default=5.0)
    p.add_argument("--lambda_ltc", type=float, default=1.0)
    p.add_argument("--lambda_tmp", type=float, default=0.1)
    p.add_argument(
        "--n_triplets",
        type=int,
        default=None,
        help="Legacy cap for sampled triplets per subspace. Omit or pass <=0 to use the full human+robot pool.",
    )
    p.add_argument("--margin",     type=float, default=0.05)
    p.add_argument("--extra_human_ratio", type=float, default=0.10)
    p.add_argument("--log_metric_stats", action="store_true", help="Log D_R/D_ee/S_k scale diagnostics by subspace.")
    p.add_argument(
        "--zero_wrj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Shadow WRJ2/WRJ1 targets and FK inputs to zero. Dong human input is wrist-local.",
    )
    return p.parse_args()


def main():
    args = p = _parse_args()

    REPO_ROOT = Path(args.repo_root)
    DEX_ROOT  = Path(args.dex_root)

    CSV_PATH    = Path(args.csv_path)    if args.csv_path    else REPO_ROOT / "grasp-model/data/processed/hograspnet_abl11.csv"
    CKPT_PATH   = Path(args.ckpt_path)  if args.ckpt_path   else REPO_ROOT / "grasp-model/checkpoints/stage1_latest.pt"
    HAND_CONFIG = Path(args.hand_config) if args.hand_config else REPO_ROOT / "grasp-model/data/hand_configs/shadow_hand_right.yaml"
    URDF_PATH   = DEX_ROOT / "robots/hands/shadow_hand/shadow_hand_right.urdf"
    EXTRA_HUMAN_CSV = Path(args.extra_human_csv) if args.extra_human_csv else None

    if EXTRA_HUMAN_CSV is not None and not EXTRA_HUMAN_CSV.exists():
        raise FileNotFoundError(f"extra_human_csv not found: {EXTRA_HUMAN_CSV}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    triplet_mode = "full_pool" if args.n_triplets is None or args.n_triplets <= 0 else str(args.n_triplets)
    print(
        f"Device: {DEVICE} | B={args.b} | N_STEPS={args.n_steps} "
        f"| triplets={triplet_mode} | margin={args.margin}"
    )
    print(f"zero_wrj={args.zero_wrj}")

    # Add scripts to path
    sys.path.insert(0, str(REPO_ROOT / "grasp-model/scripts"))
    sys.path.insert(0, str(REPO_ROOT / "grasp-model/src/cross_emb"))

    from cross_embodiment_sampler import CrossEmbodimentSampler
    from human_modules import HumanEncoder_E_h, SUBSPACE_LABEL_PREFIX, SUBSPACE_FINGERS
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
        valid_poses_path = args.valid_poses_path,
        extra_human_csv  = EXTRA_HUMAN_CSV,
        extra_human_ratio= args.extra_human_ratio,
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
    BEST_TOTAL_PATH = CKPT_PATH.parent / "stage1_best_total.pt"
    best_total = float("inf")

    for step in range(args.n_steps):

        batch          = sampler.get_batch_temporal(args.b)
        quats_h        = batch["quats_h"]
        quats_h_t1     = batch["quats_h_t1"]
        q_r            = batch["q_r"]
        if args.zero_wrj:
            q_r = q_r.clone()
            q_r[:, 0:2] = 0.0  # WRJ2, WRJ1 are not encoded in wrist-local Dong human signal.
        quats_h_sub    = batch["quats_h_sub"]
        quats_r_sub    = batch["quats_r_sub"]
        tips_h_sub     = batch["tips_h_sub"]
        tips_r_sub     = batch["tips_r_sub"]
        tips_h_t1      = batch["tips_h_t1"]
        common_fingers = batch["common_fingers"]
        common_labels  = batch["common_labels"]
        extra_human_count = batch.get("extra_human_count", 0)
        extra_human_by_class = batch.get("extra_human_by_class", {})

        z_t  = E_h(quats_h)       # [B, 3*z_dim]
        z_t1 = E_h(quats_h_t1)
        z_r  = E_X(E_r(q_r))     # [B, 3*z_dim]

        q_r_hat    = D_r(D_X(z_r))
        z_h_rt     = E_X(D_X(z_t))
        q_r_hat_t1 = D_r(D_X(z_t1))

        L_rec = (q_r - q_r_hat).norm(dim=-1).mean()
        L_ltc = (z_t - z_h_rt).norm(dim=-1).mean()

        # --- Per-subspace contrastive loss (Yan et al. 2026) ---
        z_t_subs = z_t.chunk(3, dim=-1)   # (z_thumb, z_prec, z_supp) each [B, z_dim]
        z_r_subs = z_r.chunk(3, dim=-1)
        L_cont = torch.tensor(0.0, device=DEVICE)
        metric_stats = {}
        for k, sub in enumerate(("thumb", "precision", "support")):
            prefixes   = SUBSPACE_LABEL_PREFIX[sub]
            sub_finger = SUBSPACE_FINGERS[sub]
            # Indices into common_labels / quats_*_sub for this subspace
            jidx = [i for i, l in enumerate(common_labels) if l.startswith(prefixes)]
            # Indices into common_fingers / tips_*_sub for this subspace
            tidx = [i for i, f in enumerate(common_fingers) if f in sub_finger]
            if not jidx:
                continue
            z_h_k = z_t_subs[k]                          # [B, z_dim]
            z_r_k = z_r_subs[k]
            q_h_k = quats_h_sub[:, jidx, :]              # [B, Jk, 4]
            q_r_k = quats_r_sub[:, jidx, :]
            t_h_k = tips_h_sub[:, tidx, :] if tidx else tips_h_sub[:, :0, :]
            t_r_k = tips_r_sub[:, tidx, :] if tidx else tips_r_sub[:, :0, :]

            z_all_k    = torch.cat([z_h_k, z_r_k], dim=0)
            q_all_k    = torch.cat([q_h_k, q_r_k], dim=0)
            t_all_k    = torch.cat([t_h_k, t_r_k], dim=0)
            B2  = z_all_k.shape[0]
            if B2 < 3:
                continue
            n   = B2 if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, B2)

            # Yan et al. describe randomly sampled triplets. Sample anchors and
            # two non-self candidates directly; compute S_k only for the pairs
            # used by the loss instead of allocating an all-pairs matrix.
            anchors = torch.randperm(B2, device=DEVICE)[:n]
            cand_a = torch.randint(0, B2 - 1, (n,), device=DEVICE)
            cand_a = cand_a + (cand_a >= anchors).long()
            cand_b = torch.randint(0, B2 - 1, (n,), device=DEVICE)
            cand_b = cand_b + (cand_b >= anchors).long()
            same = cand_b == cand_a
            if same.any():
                cand_b[same] = (cand_b[same] + 1) % B2
                cand_b[same] += (cand_b[same] == anchors[same]).long()
                cand_b[same] %= B2

            qa = q_all_k[anchors]
            ta = t_all_k[anchors]
            q_ca = q_all_k[cand_a]
            q_cb = q_all_k[cand_b]
            t_ca = t_all_k[cand_a]
            t_cb = t_all_k[cand_b]

            dot_a = (qa * q_ca).sum(-1)
            dot_b = (qa * q_cb).sum(-1)
            D_R_a = (1 - dot_a ** 2).mean(dim=-1)
            D_R_b = (1 - dot_b ** 2).mean(dim=-1)
            D_ee_a = (ta - t_ca).norm(dim=-1).mean(dim=-1)
            D_ee_b = (ta - t_cb).norm(dim=-1).mean(dim=-1)
            S_a = D_R_a + D_ee_a
            S_b = D_R_b + D_ee_b

            if args.log_metric_stats:
                D_R_pairs = torch.cat([D_R_a, D_R_b])
                D_ee_pairs = torch.cat([D_ee_a, D_ee_b])
                S_pairs = torch.cat([S_a, S_b])
                metric_stats[sub] = (
                    D_R_pairs.mean().item(),
                    D_ee_pairs.mean().item(),
                    S_pairs.mean().item(),
                )

            a_closer = S_a <= S_b
            pos_idx = torch.where(a_closer, cand_a, cand_b)
            neg_idx = torch.where(a_closer, cand_b, cand_a)
            L_cont  = L_cont + torch.relu(
                (z_all_k[anchors] - z_all_k[pos_idx]).norm(dim=-1)
                - (z_all_k[anchors] - z_all_k[neg_idx]).norm(dim=-1)
                + args.margin
            ).mean()

        q_r_hat_fk = q_r_hat
        q_r_hat_t1_fk = q_r_hat_t1
        if args.zero_wrj:
            q_r_hat_fk = q_r_hat.clone()
            q_r_hat_t1_fk = q_r_hat_t1.clone()
            q_r_hat_fk[:, 0:2] = 0.0
            q_r_hat_t1_fk[:, 0:2] = 0.0

        fk_t,  fk_t1  = sampler.robot_rnd.run_fk(q_r_hat_fk), sampler.robot_rnd.run_fk(q_r_hat_t1_fk)
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

        ckpt_payload = {
            "step": step,
            "E_h":  E_h.state_dict(),
            "E_r":  E_r.state_dict(),
            "E_X":  E_X.state_dict(),
            "D_X":  D_X.state_dict(),
            "D_r":  D_r.state_dict(),
            "zero_wrj": args.zero_wrj,
            "losses": {
                "total": L_total.item(),
                "cont": L_cont.item(),
                "rec": L_rec.item(),
                "ltc": L_ltc.item(),
                "temp": L_temp.item(),
            },
        }

        if L_total.item() < best_total:
            best_total = L_total.item()
            torch.save(ckpt_payload, BEST_TOTAL_PATH)

        if step % args.log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            best_flag = " *best_total" if L_total.item() == best_total else ""
            print(
                f"[step {step:05d}] loss total={L_total.item():.4f} "
                f"cont={L_cont.item():.4f} rec={L_rec.item():.4f} "
                f"ltc={L_ltc.item():.4f} temp={L_temp.item():.4f} "
                f"lr={lr_now:.2e}{best_flag}"
            )
            if extra_human_count:
                open_count = extra_human_by_class.get(28, 0)
                fist_count = extra_human_by_class.get(29, 0)
                print(f"  batch extra_human={extra_human_count} open={open_count} fist={fist_count}")
            if args.log_metric_stats and metric_stats:
                stats = " | ".join(
                    f"{sub}(D_R={dr:.4f}, D_ee={dee:.4f}, S={s:.4f})"
                    for sub, (dr, dee, s) in metric_stats.items()
                )
                print(f"  metric pairs {stats}")

        if step % args.ckpt_every == 0:
            torch.save({
                "step": step,
                "E_h":  E_h.state_dict(),
                "E_r":  E_r.state_dict(),
                "E_X":  E_X.state_dict(),
                "D_X":  D_X.state_dict(),
                "D_r":  D_r.state_dict(),
                "zero_wrj": args.zero_wrj,
                "losses": {
                    "total": L_total.item(),
                    "cont": L_cont.item(),
                    "rec": L_rec.item(),
                    "ltc": L_ltc.item(),
                    "temp": L_temp.item(),
                },
                "optimizer": optimizer.state_dict(),
            }, CKPT_PATH)

    print("done")


if __name__ == "__main__":
    main()
