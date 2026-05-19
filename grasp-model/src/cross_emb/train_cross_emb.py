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
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _set_seed(seed: int) -> None:
    """Fix all RNG sources used by the training pipeline.

    Covers: python random, numpy, torch CPU, torch CUDA (all devices).
    Sampler uses torch.randint/torch.rand -> respects torch global RNG.
    cudnn determinism intentionally not forced (MLP-only forward, no convs).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    p.add_argument("--z_dim",      type=int,   default=16)
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
    p.add_argument("--margin",      type=float, default=0.05)
    p.add_argument("--w_r", type=float, default=1.0, help="Weight for whole-hand D_R in S_hand.")
    p.add_argument("--w_joints", type=float, default=0.0, help="Deprecated Run20 S_k weight; unused in Run21.")
    p.add_argument("--w_ahg", type=float, default=0.0, help="Deprecated Run20 S_k weight; unused in Run21.")
    p.add_argument("--w_thumb_pos", type=float, default=10.0, help="Weight for Xin thumb-tip position term in S_hand.")
    p.add_argument("--w_tip_pos", type=float, default=1.0, help="Weight for Xin wrist-to-fingertip position term in S_hand.")
    p.add_argument("--w_tip_dir", type=float, default=10.0, help="Weight for Xin DIP-to-tip vector term in S_hand.")
    p.add_argument("--w_pinch", type=float, default=10.0, help="Weight for Xin thumb-to-fingertip pinch term in S_hand.")
    p.add_argument("--pinch_eps1_m", type=float, default=0.1, help="Xin pinch intent threshold in meters before morphology normalization.")
    p.add_argument("--pinch_eps2_m", type=float, default=0.01, help="Xin pinch contact threshold in meters before morphology normalization.")
    p.add_argument("--pinch_ref_hand_length_m", type=float, default=0.197, help="Reference hand length used to convert Xin meter thresholds to unit hand-length ratios.")
    p.add_argument("--pinch_sigmoid_w_m", type=float, default=10.0, help="Xin sigmoid slope in 1/m before morphology normalization.")
    p.add_argument("--extra_human_ratio", type=float, default=0.10)
    p.add_argument("--log_metric_stats", action="store_true", help="Log D_R/D_ee/S_k scale diagnostics by subspace.")
    p.add_argument(
        "--zero_wrj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Shadow WRJ2/WRJ1 targets and FK inputs to zero. Dong human input is wrist-local.",
    )
    p.add_argument("--resume_ckpt", default=None, help="Path to checkpoint to resume from. Loads model weights only; optimizer and scheduler reset fresh.")
    p.add_argument("--T_0", type=int, default=2000, help="CosineAnnealingWarmRestarts period (steps). LR resets to --lr every T_0 steps.")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fp16 autocast + GradScaler on CUDA. MLP forward/loss in fp16; FK and Dong stage2 stay fp32. Auto-disabled on CPU.",
    )
    p.add_argument(
        "--compile",
        dest="compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap E_h/E_r/E_X/D_X/D_r with torch.compile(mode='reduce-overhead'). Auto-disabled on CPU and on torch < 2.0.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Global seed. -1 (default) picks a random seed and logs it. Fixes torch/numpy/python RNG.",
    )
    p.add_argument(
        "--val_every",
        type=int,
        default=500,
        help="Evaluate on val split every N steps. 0 disables val. Best-by-val ckpt saved separately.",
    )
    p.add_argument(
        "--n_eval_batches",
        type=int,
        default=20,
        help="Number of batches to sample per eval pass.",
    )
    p.add_argument(
        "--b_eval",
        type=int,
        default=5000,
        help="Batch size for eval (val/test). Smaller than train --b for speed.",
    )
    return p.parse_args()


def main():
    args = p = _parse_args()

    # Fix RNG before anything that consumes it: sampler instantiation,
    # model weight init, batch sampling, autograd-side noise. Must run
    # before `from cross_embodiment_sampler import ...` triggers any
    # module-level randomness too (currently none, but cheap insurance).
    # seed=-1 -> pick random seed and log it for reproducibility.
    if args.seed < 0:
        args.seed = random.randint(0, 99_999)
        print(f"Seed: {args.seed} (auto-generated)")
    else:
        print(f"Seed: {args.seed}")
    _set_seed(args.seed)

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
    print(f"zero_wrj={args.zero_wrj} | scheduler=none (Yan-pure, constant lr={args.lr}) | resume={args.resume_ckpt or 'none'}")

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
        valid_poses_path = args.valid_poses_path,
        extra_human_csv  = EXTRA_HUMAN_CSV,
        extra_human_ratio= args.extra_human_ratio,
    )

    _probe = sampler.get_batch_temporal(1)
    J      = _probe["q_r"].shape[1]
    print(f"Robot joints J={J}")

    # ---------------------------------------------------------------------------
    # Val sampler (HOGraspNet S1 split: subjects 1-10, held out for selection).
    # Mid-training eval saves best-by-val ckpt without touching test split.
    # ---------------------------------------------------------------------------
    val_sampler = None
    if args.val_every and args.val_every > 0:
        # Val sampler: skip valid_poses NPZ to avoid loading the ~6 GB DONG_CACHE
        # tensor into GPU memory twice (train sampler already has it). RobotLoader
        # falls back to mode=RANDOM_UNIFORM for robot anchor poses. The metrics
        # (rs/nds/nvs) only depend on human poses + retargeted robot FK -- the
        # robot anchor distribution does not affect them. Only `rec` (reconstruction
        # of sampled q_r) is mildly affected.
        val_sampler = CrossEmbodimentSampler(
            csv_path         = CSV_PATH,
            urdf_path        = URDF_PATH,
            hand_config_path = HAND_CONFIG,
            split            = "val",
            device           = DEVICE,
            valid_poses_path = None,
            extra_human_csv  = None,
            extra_human_ratio= 0.0,
        )
        print(f"Val sampler: split=val, eval every {args.val_every} steps over {args.n_eval_batches} batches of B={args.b_eval}.")

    # ---------------------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------------------
    E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=args.z_dim).to(DEVICE)
    E_r = RobotEncoder_E_r(n_joints=J, shared_dim=args.shared_dim).to(DEVICE)
    E_X = SharedEncoder_E_X(shared_dim=args.shared_dim, z_dim=args.z_dim).to(DEVICE)
    D_X = SharedDecoder_D_X(z_dim=args.z_dim, shared_dim=args.shared_dim).to(DEVICE)
    D_r = RobotDecoder_D_r(n_joints=J, shared_dim=args.shared_dim).to(DEVICE)

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=DEVICE)
        E_h.load_state_dict(ckpt["E_h"])
        E_r.load_state_dict(ckpt["E_r"])
        E_X.load_state_dict(ckpt["E_X"])
        D_X.load_state_dict(ckpt["D_X"])
        D_r.load_state_dict(ckpt["D_r"])
        print(f"Resumed weights from {args.resume_ckpt} (step {ckpt.get('step', '?')})")
        print("Optimizer and scheduler reset fresh.")

    # torch.compile MUST run AFTER state_dict load: compiling traces the forward
    # graph; loading weights afterwards into the wrapped module is supported but
    # avoiding the recompile is cleaner. Compile only on CUDA -- CPU compile
    # works but yields little gain and slows first step heavily.
    use_compile = bool(args.compile) and DEVICE == "cuda" and hasattr(torch, "compile")
    if use_compile:
        try:
            E_h = torch.compile(E_h, mode="reduce-overhead")
            E_r = torch.compile(E_r, mode="reduce-overhead")
            E_X = torch.compile(E_X, mode="reduce-overhead")
            D_X = torch.compile(D_X, mode="reduce-overhead")
            D_r = torch.compile(D_r, mode="reduce-overhead")
            print("torch.compile: enabled (mode=reduce-overhead) on E_h/E_r/E_X/D_X/D_r")
        except Exception as e:
            print(f"torch.compile failed ({e}); falling back to eager.")
            use_compile = False
    else:
        print("torch.compile: disabled")

    optimizer = torch.optim.Adam(
        list(E_h.parameters()) + list(E_r.parameters()) +
        list(E_X.parameters()) + list(D_X.parameters()) +
        list(D_r.parameters()),
        lr=args.lr,
    )
    # Yan et al. (2026) replication: Adam with constant LR, no scheduler.
    # Tanh saturation at latent output provides implicit late-stage damping.
    # `--T_0` and `--lr_warmup` flags retained for backward compat but unused.

    use_amp = bool(args.amp) and DEVICE == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP: {'enabled (fp16)' if use_amp else 'disabled'}")

    # ---------------------------------------------------------------------------
    # D_R per-joint weights: w_j = (1/sigma_j) / sum(1/sigma)
    # sigma_j = std(1 - dot_j^2) over HOGraspNet train pairs, human only.
    # Order per subspace: [mcp, pip, dip, tip]. tip=0 (always identity in Dong).
    # Precomputed offline from hograspnet_abl11.csv (50k random pairs, 10k frames).
    # ---------------------------------------------------------------------------
    _sk_w = {
        "thumb":  [0.258, 0.544, 0.199, 0.0],
        "index":  [0.329, 0.325, 0.346, 0.0],
        "middle": [0.188, 0.362, 0.451, 0.0],
        "ring":   [0.238, 0.357, 0.405, 0.0],
        "pinky":  [0.197, 0.405, 0.398, 0.0],
    }
    sk_weights_dr = {sub: torch.tensor(w, device=DEVICE) for sub, w in _sk_w.items()}
    pinch_eps1 = args.pinch_eps1_m / args.pinch_ref_hand_length_m
    pinch_eps2 = args.pinch_eps2_m / args.pinch_ref_hand_length_m
    pinch_sigmoid_w = args.pinch_sigmoid_w_m * args.pinch_ref_hand_length_m
    if pinch_eps2 >= pinch_eps1:
        raise ValueError(f"pinch_eps2 must be lower than pinch_eps1 after normalization: {pinch_eps2} >= {pinch_eps1}")
    print(
        f"Xin morphology ratios: eps1={pinch_eps1:.4f}, eps2={pinch_eps2:.4f}, "
        f"sigmoid_w={pinch_sigmoid_w:.4f}"
    )

    _seg_order = ["mcp", "pip", "dip", "tip"]

    def _dr_weights_for_labels(labels: list[str]) -> torch.Tensor:
        vals = []
        for label in labels:
            finger, seg = label.split("_", 1)
            vals.append(sk_weights_dr[finger][_seg_order.index(seg)])
        w = torch.stack(vals).to(DEVICE)
        return w / w.sum().clamp(min=1e-8)

    def _sigmoid_xin(d: torch.Tensor, w: float) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(w * (d - pinch_eps1)))

    def _pinch_rescale(d: torch.Tensor) -> torch.Tensor:
        mid = pinch_eps1 / (pinch_eps1 - pinch_eps2) * (d - pinch_eps2)
        return torch.where(
            d < pinch_eps2,
            torch.zeros_like(d),
            torch.where(d > pinch_eps1, d, mid),
        )

    def _s_hand_components(
        q1: torch.Tensor,
        q2: torch.Tensor,
        chain1: torch.Tensor,
        chain2: torch.Tensor,
        common_labels: list[str],
        common_fingers: list[str],
    ) -> dict[str, torch.Tensor]:
        # q*: [N, K, 4], chain*: [N, F, 4, 3] in wrist-local hand-length units.
        dot = (q1 * q2).sum(-1)
        w_dr = _dr_weights_for_labels(common_labels)
        D_R = (w_dr * (1 - dot ** 2)).sum(dim=-1)

        n = chain1.shape[0]
        zero = torch.zeros(n, device=chain1.device, dtype=chain1.dtype)
        tips1 = chain1[:, :, 3, :]
        tips2 = chain2[:, :, 3, :]
        dirs1 = chain1[:, :, 3, :] - chain1[:, :, 2, :]
        dirs2 = chain2[:, :, 3, :] - chain2[:, :, 2, :]
        D_tip_dir = ((dirs1 - dirs2) ** 2).sum(dim=-1).sum(dim=-1)

        if "thumb" not in common_fingers:
            D_thumb_pos = zero
            D_pinch = zero
            D_tip_pos = ((tips1 - tips2) ** 2).sum(dim=-1).sum(dim=-1)
        else:
            thumb_idx = common_fingers.index("thumb")
            pinch_idx = [i for i, f in enumerate(common_fingers) if f != "thumb"]
            D_thumb_pos = ((tips1[:, thumb_idx] - tips2[:, thumb_idx]) ** 2).sum(dim=-1)
            if not pinch_idx:
                D_pinch = zero
                D_tip_pos = ((tips1 - tips2) ** 2).sum(dim=-1).sum(dim=-1)
            else:
                thumb1 = tips1[:, thumb_idx]
                thumb2 = tips2[:, thumb_idx]
                gamma1 = tips1[:, pinch_idx, :] - thumb1[:, None, :]
                gamma2 = tips2[:, pinch_idx, :] - thumb2[:, None, :]
                d = gamma1.norm(dim=-1)
                gamma_hat = gamma1 / d[..., None].clamp(min=1e-8)
                s = _sigmoid_xin(d, pinch_sigmoid_w)
                stilde_nonthumb = _sigmoid_xin(d, -pinch_sigmoid_w)
                target = _pinch_rescale(d)[..., None] * gamma_hat
                D_pinch = (s * ((gamma2 - target) ** 2).sum(dim=-1)).sum(dim=-1)

                stilde = torch.ones(n, len(common_fingers), device=chain1.device, dtype=chain1.dtype)
                stilde[:, pinch_idx] = stilde_nonthumb
                stilde[:, thumb_idx] = _sigmoid_xin(d.min(dim=-1).values, -pinch_sigmoid_w)
                D_tip_pos = (stilde * ((tips1 - tips2) ** 2).sum(dim=-1)).sum(dim=-1)

        S = (
            args.w_r * D_R
            + args.w_thumb_pos * D_thumb_pos
            + args.w_tip_pos * D_tip_pos
            + args.w_tip_dir * D_tip_dir
            + args.w_pinch * D_pinch
        )
        return {
            "D_R": D_R,
            "D_thumb_pos": D_thumb_pos,
            "D_tip_pos": D_tip_pos,
            "D_tip_dir": D_tip_dir,
            "D_pinch": D_pinch,
            "S": S,
        }

    def _sd(m: torch.nn.Module) -> dict:
        # Strip the OptimizedModule wrapper from torch.compile so saved keys
        # match a bare (uncompiled) module. Loading is unaffected -- callers
        # always load into bare modules before compile.
        return getattr(m, "_orig_mod", m).state_dict()

    # ---------------------------------------------------------------------------
    # Eval helper -- runs retargeting on val/test split and returns RS/NDS/NVS.
    #
    # Metrics (adapted from Yan et al. 2026 Sec. 4.3.1 for hand domain):
    #   RS  = sum_j (1 - <q_h_j, q_r_j>^2)        (D_R, per-joint quaternion sim)
    #   NDS = sum over (Fc, 4) of ||p_h - p_r||   (D_joints over chain, not just EE)
    #   NVS = ||v_h_tips - v_r_tips||_2           (per-fingertip velocity diff)
    #
    # Differences vs Yan:
    #   - RS uses uniform sum (no Jarque-Bou weights here -- weights are for the
    #     contrastive triplet selection during training, not for eval comparison).
    #   - NDS uses 4 chain positions per finger (mcp,pip,dip,tip) instead of 1 EE.
    #   - NVS averages over 5 fingertips instead of 1 hand EE.
    #
    # No gradient. Compatible with torch.compile-wrapped modules via _orig_mod.
    # ---------------------------------------------------------------------------
    def _compute_eval_metrics(eval_sampler, n_batches: int, b_eval: int) -> dict | None:
        if eval_sampler is None:
            return None
        E_h_ = getattr(E_h, "_orig_mod", E_h)
        E_r_ = getattr(E_r, "_orig_mod", E_r)
        E_X_ = getattr(E_X, "_orig_mod", E_X)
        D_X_ = getattr(D_X, "_orig_mod", D_X)
        D_r_ = getattr(D_r, "_orig_mod", D_r)
        rs_sum = nds_sum = nvs_sum = 0.0
        s_hand_sum = pinch_err_sum = pinch_dist_err_sum = tip_dir_err_sum = 0.0
        rec_sum = 0.0
        n_done = 0
        human_labels_global = ["thumb", "index", "middle", "ring", "pinky"]
        for m in (E_h_, E_r_, E_X_, D_X_, D_r_):
            m.eval()
        with torch.no_grad():
            for _ in range(n_batches):
                batch = eval_sampler.get_batch_temporal(b_eval)
                quats_h        = batch["quats_h"]
                quats_h_t1     = batch["quats_h_t1"]
                quats_h_sub    = batch["quats_h_sub"]
                chain_h_sub    = batch["chain_h_sub"]
                tips_h_sub     = batch["tips_h_sub"]
                tips_h_t1      = batch["tips_h_t1"]
                q_r_data       = batch["q_r"]
                common_fingers = batch["common_fingers"]
                common_labels  = batch["common_labels"]

                # Retarget: human Dong -> latent -> robot joint angles.
                z_t        = E_h_(quats_h)
                z_t1       = E_h_(quats_h_t1)
                q_r_hat    = D_r_(D_X_(z_t)).float()
                q_r_hat_t1 = D_r_(D_X_(z_t1)).float()
                if args.zero_wrj:
                    q_r_hat    = q_r_hat.clone();    q_r_hat[:, 0:2]    = 0.0
                    q_r_hat_t1 = q_r_hat_t1.clone(); q_r_hat_t1[:, 0:2] = 0.0

                # FK + full Dong stage2 for both timesteps (need quats+chain+tips).
                B = q_r_hat.shape[0]
                q_combined = torch.cat([q_r_hat, q_r_hat_t1], dim=0)
                fk_combined = eval_sampler.robot_rnd.run_fk(q_combined)
                quats_r_all, joint_labels_r, meta_r = eval_sampler.robot_rnd.run_dong_stage2(
                    fk_combined, HAND_CONFIG
                )

                # Subset robot Dong outputs to common joints/fingers.
                joint_idx_r = [joint_labels_r.index(l) for l in common_labels]
                quats_r_sub_t = quats_r_all[:B, joint_idx_r]                  # [B, Jk, 4]
                # Align hemispheres (w>=0 not guaranteed match per pair).
                sign = torch.sign((quats_h_sub * quats_r_sub_t).sum(-1, keepdim=True))
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                quats_r_sub_t = quats_r_sub_t * sign

                # RS: D_R uniform sum (Yan eq 1).
                dot = (quats_h_sub * quats_r_sub_t).sum(-1)                    # [B, Jk]
                rs  = (1 - dot ** 2).sum(-1).mean().item()

                # NDS: chain L2 over 4 positions per finger (hand domain adaptation).
                chain_r_dict = meta_r["chain_positions"]
                chain_r_t = torch.stack(
                    [chain_r_dict[f][:B] for f in common_fingers], dim=1
                )                                                              # [B, Fc, 4, 3]
                nds = (chain_h_sub - chain_r_t).norm(dim=-1).sum(dim=(-2, -1)).mean().item()

                # NVS: per-fingertip velocity diff.
                tips_r_all = meta_r["tips"]                                    # [2B, F, 3]
                tip_labels_r = meta_r["tip_labels"]
                tip_idx_r  = [tip_labels_r.index(f) for f in common_fingers]
                tip_idx_h  = [human_labels_global.index(f) for f in common_fingers]
                tips_r_t   = tips_r_all[:B, tip_idx_r]
                tips_r_t1  = tips_r_all[B:, tip_idx_r]
                tips_h_t1_sub = tips_h_t1[:, tip_idx_h]
                v_h = tips_h_t1_sub - tips_h_sub
                v_r = tips_r_t1   - tips_r_t
                nvs = (v_h - v_r).norm(dim=-1).mean().item()

                # L_rec on data pair: how well does the decoder reconstruct sampled q_r?
                z_r_       = E_X_(E_r_(q_r_data))
                q_r_hat_rd = D_r_(D_X_(z_r_))
                rec        = (q_r_data - q_r_hat_rd).norm(dim=-1).mean().item()

                comps = _s_hand_components(
                    quats_h_sub,
                    quats_r_sub_t,
                    chain_h_sub,
                    chain_r_t,
                    common_labels,
                    common_fingers,
                )
                s_hand = comps["S"].mean().item()
                tip_dir_err = comps["D_tip_dir"].mean().item()
                pinch_err = comps["D_pinch"].mean().item()
                if "thumb" in common_fingers and len(common_fingers) > 1:
                    thumb_i = common_fingers.index("thumb")
                    pinch_i = [i for i, f in enumerate(common_fingers) if f != "thumb"]
                    gh = chain_h_sub[:, pinch_i, 3, :] - chain_h_sub[:, thumb_i, 3, :][:, None, :]
                    gr = chain_r_t[:, pinch_i, 3, :] - chain_r_t[:, thumb_i, 3, :][:, None, :]
                    pinch_dist_err = (gh.norm(dim=-1) - gr.norm(dim=-1)).abs().mean().item()
                else:
                    pinch_dist_err = 0.0

                rs_sum  += rs
                nds_sum += nds
                nvs_sum += nvs
                rec_sum += rec
                s_hand_sum += s_hand
                pinch_err_sum += pinch_err
                pinch_dist_err_sum += pinch_dist_err
                tip_dir_err_sum += tip_dir_err
                n_done  += 1
        for m in (E_h_, E_r_, E_X_, D_X_, D_r_):
            m.train()
        if n_done == 0:
            return None
        return {
            "rs":  rs_sum  / n_done,
            "nds": nds_sum / n_done,
            "nvs": nvs_sum / n_done,
            "rec": rec_sum / n_done,
            "s_hand": s_hand_sum / n_done,
            "pinch_err": pinch_err_sum / n_done,
            "pinch_dist_err": pinch_dist_err_sum / n_done,
            "tip_dir_err": tip_dir_err_sum / n_done,
            "n_batches": n_done,
            "b_eval": b_eval,
        }

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_TOTAL_PATH = CKPT_PATH.parent / "stage1_best_total.pt"
    BEST_VAL_PATH   = CKPT_PATH.parent / "stage1_best_val.pt"
    best_total      = float("inf")
    best_val_score  = float("inf")
    last_val_metrics: dict | None = None

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
        tips_h_t1      = batch["tips_h_t1"]
        chain_h_sub    = batch["chain_h_sub"]   # [B, Fc, 4, 3]
        chain_r_sub    = batch["chain_r_sub"]   # [B, Fc, 4, 3]
        common_fingers = batch["common_fingers"]
        common_labels  = batch["common_labels"]
        extra_human_count = batch.get("extra_human_count", 0)
        extra_human_by_class = batch.get("extra_human_by_class", {})

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            z_t  = E_h(quats_h)       # [B, z_dim]
            z_t1 = E_h(quats_h_t1)
            z_r  = E_X(E_r(q_r))     # [B, z_dim]

            q_r_hat    = D_r(D_X(z_r))
            z_h_rt     = E_X(D_X(z_t))
            q_r_hat_t1 = D_r(D_X(z_t1))

            L_rec = (q_r - q_r_hat).norm(dim=-1).mean()
            L_ltc = (z_t - z_h_rt).norm(dim=-1).mean()

        # --- Whole-hand contrastive loss with functional S_hand ---
        z_all = torch.cat([z_t, z_r], dim=0)
        q_all = torch.cat([quats_h_sub, quats_r_sub], dim=0)
        chain_all = torch.cat([chain_h_sub, chain_r_sub], dim=0)
        B2 = z_all.shape[0]
        n = B2 if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, B2)
        metric_stats = {}
        if B2 < 3:
            L_cont = torch.tensor(0.0, device=DEVICE)
        else:
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

            with torch.no_grad():
                comps_a = _s_hand_components(
                    q_all[anchors],
                    q_all[cand_a],
                    chain_all[anchors],
                    chain_all[cand_a],
                    common_labels,
                    common_fingers,
                )
                comps_b = _s_hand_components(
                    q_all[anchors],
                    q_all[cand_b],
                    chain_all[anchors],
                    chain_all[cand_b],
                    common_labels,
                    common_fingers,
                )
                S_a = comps_a["S"]
                S_b = comps_b["S"]
                a_closer = S_a <= S_b
                pos_idx = torch.where(a_closer, cand_a, cand_b)
                neg_idx = torch.where(a_closer, cand_b, cand_a)

                if args.log_metric_stats:
                    merged = {k: torch.cat([comps_a[k], comps_b[k]]) for k in comps_a.keys()}
                    metric_stats["hand"] = tuple(
                        merged[k].mean().item()
                        for k in ("D_R", "D_thumb_pos", "D_tip_pos", "D_tip_dir", "D_pinch", "S")
                    ) + (
                        merged["S"].std().item(),
                        merged["S"].min().item(),
                        merged["S"].max().item(),
                    )

            L_cont = torch.relu(
                (z_all[anchors] - z_all[pos_idx]).norm(dim=-1)
                - (z_all[anchors] - z_all[neg_idx]).norm(dim=-1)
                + args.margin
            ).mean()

        # FK + run_dong_tips_only run in fp32 (pytorch-kinematics not validated
        # for fp16). `q_r_hat`/`q_r_hat_t1` may be fp16 under AMP autocast --
        # `.float()` upcasts and preserves the autograd path back to MLPs.
        q_r_hat_fk    = q_r_hat.float()
        q_r_hat_t1_fk = q_r_hat_t1.float()
        if args.zero_wrj:
            q_r_hat_fk    = q_r_hat_fk.clone()
            q_r_hat_t1_fk = q_r_hat_t1_fk.clone()
            q_r_hat_fk[:, 0:2] = 0.0
            q_r_hat_t1_fk[:, 0:2] = 0.0

        # Fuse the t and t+1 FK calls into one batched forward, then split.
        # Halves pytorch-kinematics launch / Python overhead; identical math.
        # L_temp only needs fingertip positions, so use the lightweight
        # `run_dong_tips_only` path (skip block3 + mat_to_quat).
        B_fk = q_r_hat_fk.shape[0]
        q_combined    = torch.cat([q_r_hat_fk, q_r_hat_t1_fk], dim=0)    # [2B, J]
        fk_combined   = sampler.robot_rnd.run_fk(q_combined)
        tips_all, tip_labels = sampler.robot_rnd.run_dong_tips_only(fk_combined, HAND_CONFIG)
        common_idx_r  = [tip_labels.index(f) for f in common_fingers]
        human_labels  = ["thumb", "index", "middle", "ring", "pinky"]
        common_idx_h  = [human_labels.index(f) for f in common_fingers]
        tips_r_all    = tips_all.to(DEVICE)[:, common_idx_r, :]          # [2B, Fc, 3]
        tips_r_t_sub  = tips_r_all[:B_fk]
        tips_r_t1_sub = tips_r_all[B_fk:]
        tips_h_t1_sub = tips_h_t1[:, common_idx_h, :]
        L_temp = ((tips_h_t1_sub - tips_h_sub) - (tips_r_t1_sub - tips_r_t_sub)).norm(dim=-1).mean()

        L_total = args.lambda_c * L_cont + args.lambda_rec * L_rec + args.lambda_ltc * L_ltc + args.lambda_tmp * L_temp

        optimizer.zero_grad()
        # GradScaler scales loss before backward to keep fp16 grads in range,
        # then unscales before clip+step. With AMP off, scaler is a no-op.
        scaler.scale(L_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(E_h.parameters()) + list(E_r.parameters()) +
            list(E_X.parameters()) + list(D_X.parameters()) +
            list(D_r.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        # Constant LR: no scheduler.step().

        ckpt_payload = {
            "step": step,
            "seed": args.seed,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "E_h":  _sd(E_h),
            "E_r":  _sd(E_r),
            "E_X":  _sd(E_X),
            "D_X":  _sd(D_X),
            "D_r":  _sd(D_r),
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

        # Val eval: every `args.val_every` steps (skip step 0).
        if val_sampler is not None and step > 0 and step % args.val_every == 0:
            last_val_metrics = _compute_eval_metrics(
                val_sampler, args.n_eval_batches, args.b_eval
            )
            if last_val_metrics is not None:
                # Run21 functional score: S_hand plus small smoothness/reconstruction terms.
                score = last_val_metrics["s_hand"] + 0.1 * last_val_metrics["nvs"] + 0.1 * last_val_metrics["rec"]
                print(
                    f"[step {step:05d}] VAL rs={last_val_metrics['rs']:.4f} "
                    f"nds={last_val_metrics['nds']:.4f} nvs={last_val_metrics['nvs']:.4f} "
                    f"rec={last_val_metrics['rec']:.4f} s_hand={last_val_metrics['s_hand']:.4f} "
                    f"pinch={last_val_metrics['pinch_err']:.4f} tip_dir={last_val_metrics['tip_dir_err']:.4f} "
                    f"score={score:.4f}"
                    f"{' *best_val' if score < best_val_score else ''}"
                )
                if score < best_val_score:
                    best_val_score = score
                    val_payload = dict(ckpt_payload)
                    val_payload["val_metrics"] = last_val_metrics
                    val_payload["val_score"]   = score
                    torch.save(val_payload, BEST_VAL_PATH)

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
                dr, dt, dp, dd, dpin, s, std, mn, mx = metric_stats["hand"]
                weighted = (
                    f"wD_R={args.w_r * dr:.4f}, "
                    f"wthumb={args.w_thumb_pos * dt:.4f}, "
                    f"wtip={args.w_tip_pos * dp:.4f}, "
                    f"wdir={args.w_tip_dir * dd:.4f}, "
                    f"wpinch={args.w_pinch * dpin:.4f}"
                )
                print(
                    f"  S_hand pairs D_R={dr:.4f}, D_thumb_pos={dt:.4f}, "
                    f"D_tip_pos={dp:.4f}, D_tip_dir={dd:.4f}, D_pinch={dpin:.4f}, "
                    f"S_mean={s:.4f}, S_std={std:.4f}, S_min={mn:.4f}, S_max={mx:.4f} | {weighted}"
                )

        if step % args.ckpt_every == 0:
            torch.save({
                "step": step,
                "E_h":  _sd(E_h),
                "E_r":  _sd(E_r),
                "E_X":  _sd(E_X),
                "D_X":  _sd(D_X),
                "D_r":  _sd(D_r),
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

    # ---------------------------------------------------------------------------
    # Final test eval (HOGraspNet S1 split, subjects 74-99, held out for reporting).
    # Computed once after training. Saved alongside the best-val checkpoint.
    # ---------------------------------------------------------------------------
    print("\n=== Final test eval (subjects 74-99) ===")
    test_sampler = CrossEmbodimentSampler(
        csv_path         = CSV_PATH,
        urdf_path        = URDF_PATH,
        hand_config_path = HAND_CONFIG,
        split            = "test",
        device           = DEVICE,
        valid_poses_path = None,           # avoid GPU OOM from double DONG_CACHE
        extra_human_csv  = None,
        extra_human_ratio= 0.0,
    )
    test_metrics = _compute_eval_metrics(test_sampler, args.n_eval_batches, args.b_eval)
    if test_metrics is not None:
        score = test_metrics["s_hand"] + 0.1 * test_metrics["nvs"] + 0.1 * test_metrics["rec"]
        print(
            f"TEST rs={test_metrics['rs']:.4f} nds={test_metrics['nds']:.4f} "
            f"nvs={test_metrics['nvs']:.4f} rec={test_metrics['rec']:.4f} "
            f"s_hand={test_metrics['s_hand']:.4f} pinch={test_metrics['pinch_err']:.4f} "
            f"tip_dir={test_metrics['tip_dir_err']:.4f} score={score:.4f}"
        )

        # Annotate best-val ckpt with test metrics for final reporting.
        if BEST_VAL_PATH.exists():
            ck = torch.load(BEST_VAL_PATH, map_location="cpu", weights_only=False)
            ck["test_metrics"] = test_metrics
            ck["test_score"]   = score
            torch.save(ck, BEST_VAL_PATH)
            print(f"Test metrics annotated on {BEST_VAL_PATH}.")
        # Also annotate best-total ckpt.
        if BEST_TOTAL_PATH.exists():
            ck = torch.load(BEST_TOTAL_PATH, map_location="cpu", weights_only=False)
            ck["test_metrics"] = test_metrics
            ck["test_score"]   = score
            torch.save(ck, BEST_TOTAL_PATH)
            print(f"Test metrics annotated on {BEST_TOTAL_PATH}.")

    print("done")


if __name__ == "__main__":
    main()
