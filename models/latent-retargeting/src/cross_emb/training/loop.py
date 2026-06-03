"""Stage 1 training loop for cross-embodiment retargeting.

Usage (local):
    python -m cross_emb.training.loop

Usage (Colab via shim):
    python models/latent-retargeting/scripts/train_cross_emb.py \\
        --repo_root /content/AIST-hand \\
        --dex_root  /content/drive/MyDrive/AIST-hand/dex-urdf \\
        --csv_path  /content/drive/MyDrive/AIST-hand/hograspnet_abl11.csv \\
        --ckpt_path /content/drive/MyDrive/AIST-hand/checkpoints/stage1_latest.pt \\
        --b 20000 --n_steps 5000 --log_every 50 --ckpt_every 500
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from cross_emb.loaders import CrossEmbodimentSampler
from cross_emb.nn.human_modules import HumanEncoder_E_h, HumanEncoder_E_h_single, HumanEncoder_E_h_hybrid
from cross_emb.nn.robot_modules import RobotEncoder_E_r, RobotDecoder_D_r, RobotDecoder_D_r_residual
from cross_emb.nn.shared_modules import SharedEncoder_E_X, SharedDecoder_D_X
from .config import _parse_args
from .losses import d_r_yan, l_joint, load_l_joint_config, xin_sk_full, xin_sk_per_finger


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


def _compute_eval_metrics(
    eval_sampler,
    n_batches: int,
    b_eval: int,
    E_h,
    E_r,
    E_X,
    D_X,
    D_r,
    hand_config: Path,
    zero_wrj: bool,
    residual_decoder: bool = False,
) -> dict | None:
    """Run retargeting on a split and return RS/NDS/NVS/rec metrics.

    Metrics (adapted from Yan et al. 2026 Sec. 4.3.1 for hand domain):
      RS  = sum_j (1 - <q_h_j, q_r_j>^2)        (D_R, per-joint quaternion sim)
      NDS = sum over (Fc, 4) of ||p_h - p_r||   (D_joints over chain, not just EE)
      NVS = ||v_h_tips - v_r_tips||_2           (per-fingertip velocity diff)
    """
    if eval_sampler is None:
        return None
    E_h_ = getattr(E_h, "_orig_mod", E_h)
    E_r_ = getattr(E_r, "_orig_mod", E_r)
    E_X_ = getattr(E_X, "_orig_mod", E_X)
    D_X_ = getattr(D_X, "_orig_mod", D_X)
    D_r_ = getattr(D_r, "_orig_mod", D_r)
    rs_sum = nds_sum = nvs_sum = 0.0
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
            q_r_hat    = (D_r_(z_t)    if residual_decoder else D_r_(D_X_(z_t))).float()
            q_r_hat_t1 = (D_r_(z_t1)  if residual_decoder else D_r_(D_X_(z_t1))).float()
            if zero_wrj:
                q_r_hat    = q_r_hat.clone();    q_r_hat[:, 0:2]    = 0.0
                q_r_hat_t1 = q_r_hat_t1.clone(); q_r_hat_t1[:, 0:2] = 0.0

            # FK + full Dong stage2 for both timesteps (need quats+chain+tips).
            B = q_r_hat.shape[0]
            q_combined = torch.cat([q_r_hat, q_r_hat_t1], dim=0)
            fk_combined = eval_sampler.robot_rnd.run_fk(q_combined)
            quats_r_all, joint_labels_r, meta_r = eval_sampler.robot_rnd.run_dong_stage2(
                fk_combined, hand_config
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
            q_r_hat_rd = D_r_(z_r_) if residual_decoder else D_r_(D_X_(z_r_))
            rec        = (q_r_data - q_r_hat_rd).norm(dim=-1).mean().item()

            rs_sum  += rs
            nds_sum += nds
            nvs_sum += nvs
            rec_sum += rec
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
        "n_batches": n_done,
        "b_eval": b_eval,
    }


def main() -> None:
    args = p = _parse_args()

    # seed=-1 -> pick random seed and log it for reproducibility.
    if args.seed < 0:
        args.seed = random.randint(0, 99_999)
        print(f"Seed: {args.seed} (auto-generated)")
    else:
        print(f"Seed: {args.seed}")
    _set_seed(args.seed)

    REPO_ROOT = Path(args.repo_root)
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    DEX_ROOT  = Path(args.dex_root)

    CSV_PATH    = Path(args.csv_path)    if args.csv_path    else REPO_ROOT / "human/datasets/hograspnet/processed/hograspnet_abl11.csv"
    CKPT_PATH   = Path(args.ckpt_path)  if args.ckpt_path   else PACKAGE_ROOT / "checkpoints/stage1_latest.pt"
    HAND_CONFIG = Path(args.hand_config) if args.hand_config else REPO_ROOT / "robot/hands/shadow_hand/shadow_hand_right.yaml"
    ROBOT_YAML  = Path(args.robot_yaml_path) if args.robot_yaml_path else REPO_ROOT / "robot/hands/shadow_hand/robot.yaml"
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

    # L_joint configuration. Read xin_position_weight per joint from robot.yaml
    # semantic_roles (Xin et al. 2025: 0.5 for FFJ4/MFJ4/RFJ4/LFJ4 abduction,
    # 0.1 for THJ5 thumb rotation, 0 elsewhere). joint_names comes from the
    # pytorch_kinematics chain and matches the order of q_r columns.
    robot_joint_names = sampler.robot_rnd.chain_joint_names
    if args.lambda_joint > 0:
        l_joint_w_pos = load_l_joint_config(ROBOT_YAML)
        active = {k: v for k, v in l_joint_w_pos.items() if k in robot_joint_names and v != 0.0}
        print(f"L_joint enabled (lambda_joint={args.lambda_joint}): active weights = {active}")
    else:
        l_joint_w_pos = {}
        print("L_joint disabled (lambda_joint=0).")

    # ---------------------------------------------------------------------------
    # Val sampler (HOGraspNet S1 split: subjects 1-10, held out for selection).
    # ---------------------------------------------------------------------------
    val_sampler = None
    if args.val_every and args.val_every > 0:
        # Val sampler: skip valid_poses NPZ to avoid loading the ~6 GB DONG_CACHE
        # tensor into GPU memory twice. RobotLoader falls back to mode=RANDOM_UNIFORM.
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
    # Run 25: --single_latent switches E_h to one global projection head and
    # routes the shared E_X/D_X through a single z_dim_total latent rather
    # than 5 per-finger subspaces. Per-finger contrastive loop is replaced by
    # a single triplet on xin_sk_full.
    if args.single_latent and args.hybrid:
        raise ValueError("--single_latent and --hybrid are mutually exclusive.")
    if args.residual_decoder and not args.hybrid:
        raise ValueError("--residual_decoder requires --hybrid.")
    if args.single_latent:
        z_dim_total = args.z_dim_total
        E_h = HumanEncoder_E_h_single(in_dim=4, hidden_dim=32, z_dim_total=z_dim_total).to(DEVICE)
        print(f"{'='*60}")
        print(f"  LATENT MODE : SINGLE (z_dim_total={z_dim_total})")
        print(f"{'='*60}")
    elif args.hybrid:
        z_dim_total = args.z_dim_global + 5 * args.z_dim
        E_h = HumanEncoder_E_h_hybrid(in_dim=4, hidden_dim=32,
                                      z_dim=args.z_dim, z_dim_global=args.z_dim_global).to(DEVICE)
        print(f"{'='*60}")
        print(f"  LATENT MODE : HYBRID (z_global={args.z_dim_global} + 5x{args.z_dim} = {z_dim_total})")
        print(f"{'='*60}")
    else:
        z_dim_total = 5 * args.z_dim
        E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=args.z_dim).to(DEVICE)
        print(f"{'='*60}")
        print(f"  LATENT MODE : PER-FINGER (5 subspaces, z_dim={args.z_dim}, total={z_dim_total})")
        print(f"{'='*60}")
    E_r = RobotEncoder_E_r(n_joints=J, shared_dim=args.shared_dim).to(DEVICE)
    E_X = SharedEncoder_E_X(shared_dim=args.shared_dim, z_dim_total=z_dim_total).to(DEVICE)
    D_X = SharedDecoder_D_X(shared_dim=args.shared_dim, z_dim_total=z_dim_total).to(DEVICE)
    if args.hybrid and args.residual_decoder:
        D_r = RobotDecoder_D_r_residual(
            z_dim_global=args.z_dim_global, z_dim_finger=args.z_dim, n_joints=J
        ).to(DEVICE)
        print(f"  DECODER     : RESIDUAL (D_base[{args.z_dim_global}→{J}] + 5×D_k per finger)")
    else:
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

    # torch.compile MUST run AFTER state_dict load.
    # Compile only on CUDA -- CPU compile works but yields little gain.
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

    def _sd(m: torch.nn.Module) -> dict:
        # Strip the OptimizedModule wrapper from torch.compile so saved keys
        # match a bare (uncompiled) module.
        return getattr(m, "_orig_mod", m).state_dict()

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
        tips_r_sub     = batch["tips_r_sub"]
        tips_h_t1      = batch["tips_h_t1"]
        chain_h_sub    = batch["chain_h_sub"]   # [B, Fc, 4, 3]
        chain_r_sub    = batch["chain_r_sub"]   # [B, Fc, 4, 3]
        common_fingers = batch["common_fingers"]
        common_labels  = batch["common_labels"]
        extra_human_count = batch.get("extra_human_count", 0)
        extra_human_by_class = batch.get("extra_human_by_class", {})

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            z_t  = E_h(quats_h)       # [B, 5*z_dim]
            z_t1 = E_h(quats_h_t1)
            z_r  = E_X(E_r(q_r))     # [B, 5*z_dim]

            q_r_hat    = D_r(z_r)    if args.residual_decoder else D_r(D_X(z_r))
            z_h_rt     = E_X(D_X(z_t))
            q_r_hat_t1 = D_r(z_t1)  if args.residual_decoder else D_r(D_X(z_t1))

            L_rec = (q_r - q_r_hat).norm(dim=-1).mean()
            L_ltc = (z_t - z_h_rt).norm(dim=-1).mean()

        # --- Contrastive loss with Xin Cartesian S_k ---
        #
        # Replaces the quaternion-based S_k (D_R + D_joints + D_ahg) with
        # symmetric Cartesian metrics from Xin et al. 2025:
        #   - fingertip_pos:  wrist -> tip vector match
        #   - pinch:          thumb -> primary finger vector match (index/mid/ring)
        #   - fingertip_rot:  last-segment unit vector match
        # Operates over the FULL hand (5 fingers in common_fingers ordering)
        # because pinch needs the thumb position even when scoring index/mid/ring.
        #
        # --single_latent: one triplet on xin_sk_full over the whole hand.
        # else:            five triplets (one per finger subspace) on xin_sk_per_finger.
        tips_all  = torch.cat([tips_h_sub,  tips_r_sub],  dim=0)   # [2B, Fc, 3]
        chain_all = torch.cat([chain_h_sub, chain_r_sub], dim=0)   # [2B, Fc, 4, 3]
        quats_all = torch.cat([quats_h_sub, quats_r_sub], dim=0)   # [2B, Jc, 4]
        z_all_full = torch.cat([z_t, z_r], dim=0)                  # [2B, z_dim_total]
        metric_stats = {}

        if args.hybrid:
            # Idea I: 6 blocks = 1 coarse global + 5 fine per-finger.
            # z = [z_global | z_thumb | z_index | z_middle | z_ring | z_pinky].
            # SIX independent triplets, each with its OWN oracle (NOT a shared
            # global selection -- that is Idea A / Run 34, discarded):
            #   - global block: whole-hand oracle (xin_sk_full + lam_dr*d_r_yan)
            #   - each fine block k: per-finger oracle (xin_sk_per_finger)
            # Independent sampling per block preserves compositionality: the
            # decoder learns to combine a coordinated base (global) with local
            # per-finger modulation.
            G = args.z_dim_global
            L_cont = torch.tensor(0.0, device=DEVICE)
            B2 = z_all_full.shape[0]

            # --- Global block triplet (whole-hand oracle) ---
            if B2 >= 3:
                n = B2 if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, B2)
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
                    tips_a   = tips_all[anchors]
                    tips_ca  = tips_all[cand_a]
                    tips_cb  = tips_all[cand_b]
                    chain_a  = chain_all[anchors]
                    chain_ca = chain_all[cand_a]
                    chain_cb = chain_all[cand_b]

                    S_a = xin_sk_full(tips_a, tips_ca, chain_a, chain_ca,
                                      lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid)
                    S_b = xin_sk_full(tips_a, tips_cb, chain_a, chain_cb,
                                      lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid)

                    if args.lam_dr > 0:
                        quats_a  = quats_all[anchors]
                        quats_ca = quats_all[cand_a]
                        quats_cb = quats_all[cand_b]
                        D_R_a = d_r_yan(quats_a, quats_ca)
                        D_R_b = d_r_yan(quats_a, quats_cb)
                        S_a = S_a + args.lam_dr * D_R_a
                        S_b = S_b + args.lam_dr * D_R_b

                    if args.log_metric_stats:
                        S_pairs = torch.cat([S_a, S_b])
                        metric_stats["global"] = (
                            S_pairs.mean().item(), S_pairs.std().item(),
                            S_pairs.min().item(), S_pairs.max().item(),
                        )
                        if args.lam_dr > 0:
                            DR_pairs = torch.cat([D_R_a, D_R_b])
                            metric_stats["D_R"] = (
                                DR_pairs.mean().item(), DR_pairs.std().item(),
                                DR_pairs.min().item(), DR_pairs.max().item(),
                            )

                    a_closer = S_a <= S_b
                    pos_idx = torch.where(a_closer, cand_a, cand_b)
                    neg_idx = torch.where(a_closer, cand_b, cand_a)

                z_glob = z_all_full[:, :G]
                L_cont = L_cont + torch.relu(
                    (z_glob[anchors] - z_glob[pos_idx]).norm(dim=-1)
                    - (z_glob[anchors] - z_glob[neg_idx]).norm(dim=-1)
                    + args.margin
                ).mean()

            # --- Fine per-finger triplets (per-finger oracle, independent sampling) ---
            for k, sub in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                if sub not in common_fingers:
                    continue
                finger_idx = common_fingers.index(sub)
                z_sub = z_all_full[:, G + k * args.z_dim: G + (k + 1) * args.z_dim]
                if B2 < 3:
                    continue
                n = B2 if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, B2)
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
                    tips_a   = tips_all[anchors]
                    tips_ca  = tips_all[cand_a]
                    tips_cb  = tips_all[cand_b]
                    chain_a  = chain_all[anchors]
                    chain_ca = chain_all[cand_a]
                    chain_cb = chain_all[cand_b]

                    S_a = xin_sk_per_finger(
                        tips_a, tips_ca, chain_a, chain_ca, finger_idx=finger_idx,
                        lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid,
                    )
                    S_b = xin_sk_per_finger(
                        tips_a, tips_cb, chain_a, chain_cb, finger_idx=finger_idx,
                        lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid,
                    )

                    if args.log_metric_stats:
                        S_pairs = torch.cat([S_a, S_b])
                        metric_stats[sub] = (
                            S_pairs.mean().item(), S_pairs.std().item(),
                            S_pairs.min().item(), S_pairs.max().item(),
                        )

                    a_closer = S_a <= S_b
                    pos_idx = torch.where(a_closer, cand_a, cand_b)
                    neg_idx = torch.where(a_closer, cand_b, cand_a)

                L_cont = L_cont + torch.relu(
                    (z_sub[anchors] - z_sub[pos_idx]).norm(dim=-1)
                    - (z_sub[anchors] - z_sub[neg_idx]).norm(dim=-1)
                    + args.margin
                ).mean()
        elif args.single_latent:
            L_cont = torch.tensor(0.0, device=DEVICE)
            B2 = z_all_full.shape[0]
            if B2 >= 3:
                n = B2 if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, B2)
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
                    tips_a   = tips_all[anchors]
                    tips_ca  = tips_all[cand_a]
                    tips_cb  = tips_all[cand_b]
                    chain_a  = chain_all[anchors]
                    chain_ca = chain_all[cand_a]
                    chain_cb = chain_all[cand_b]

                    S_a = xin_sk_full(tips_a, tips_ca, chain_a, chain_ca,
                                      lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid)
                    S_b = xin_sk_full(tips_a, tips_cb, chain_a, chain_cb,
                                      lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid)

                    # Yan-style D_R term: uniform sum over common-joint quaternions.
                    # Run 27B ablation. lam_dr=0 (default) skips this branch and
                    # preserves Run 25/26 behavior exactly.
                    if args.lam_dr > 0:
                        quats_a  = quats_all[anchors]
                        quats_ca = quats_all[cand_a]
                        quats_cb = quats_all[cand_b]
                        D_R_a = d_r_yan(quats_a, quats_ca)
                        D_R_b = d_r_yan(quats_a, quats_cb)
                        S_a = S_a + args.lam_dr * D_R_a
                        S_b = S_b + args.lam_dr * D_R_b

                    if args.log_metric_stats:
                        S_pairs = torch.cat([S_a, S_b])
                        metric_stats["hand"] = (
                            S_pairs.mean().item(),
                            S_pairs.std().item(),
                            S_pairs.min().item(),
                            S_pairs.max().item(),
                        )
                        if args.lam_dr > 0:
                            DR_pairs = torch.cat([D_R_a, D_R_b])
                            metric_stats["D_R"] = (
                                DR_pairs.mean().item(),
                                DR_pairs.std().item(),
                                DR_pairs.min().item(),
                                DR_pairs.max().item(),
                            )

                    a_closer = S_a <= S_b
                    pos_idx = torch.where(a_closer, cand_a, cand_b)
                    neg_idx = torch.where(a_closer, cand_b, cand_a)

                L_cont = torch.relu(
                    (z_all_full[anchors] - z_all_full[pos_idx]).norm(dim=-1)
                    - (z_all_full[anchors] - z_all_full[neg_idx]).norm(dim=-1)
                    + args.margin
                ).mean()
        else:
            z_t_subs = z_t.chunk(5, dim=-1)   # (z_thumb, z_index, z_middle, z_ring, z_pinky)
            z_r_subs = z_r.chunk(5, dim=-1)
            L_cont = torch.tensor(0.0, device=DEVICE)
            for k, sub in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                if sub not in common_fingers:
                    continue
                finger_idx = common_fingers.index(sub)

                z_h_k = z_t_subs[k]                          # [B, z_dim]
                z_r_k = z_r_subs[k]
                z_all_k = torch.cat([z_h_k, z_r_k], dim=0)   # [2B, z_dim]
                B2 = z_all_k.shape[0]
                if B2 < 3:
                    continue
                n = B2 if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, B2)

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

                # Triplet selection metrics feed the hard S_a <= S_b mask only.
                # Gradient flows exclusively through z_all_k below.
                with torch.no_grad():
                    tips_a   = tips_all[anchors]                  # [n, Fc, 3]
                    tips_ca  = tips_all[cand_a]
                    tips_cb  = tips_all[cand_b]
                    chain_a  = chain_all[anchors]                 # [n, Fc, 4, 3]
                    chain_ca = chain_all[cand_a]
                    chain_cb = chain_all[cand_b]

                    S_a = xin_sk_per_finger(
                        tips_a, tips_ca, chain_a, chain_ca,
                        finger_idx=finger_idx,
                        lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid,
                    )
                    S_b = xin_sk_per_finger(
                        tips_a, tips_cb, chain_a, chain_cb,
                        finger_idx=finger_idx,
                        lam_fp=args.lam_fp, lam_pinch=args.lam_pinch, lam_fr=args.lam_fr, lam_mid=args.lam_mid,
                    )

                    if args.log_metric_stats:
                        S_pairs = torch.cat([S_a, S_b])
                        metric_stats[sub] = (
                            S_pairs.mean().item(),
                            S_pairs.std().item(),
                            S_pairs.min().item(),
                            S_pairs.max().item(),
                        )

                    a_closer = S_a <= S_b
                    pos_idx = torch.where(a_closer, cand_a, cand_b)
                    neg_idx = torch.where(a_closer, cand_b, cand_a)

                L_cont = L_cont + torch.relu(
                    (z_all_k[anchors] - z_all_k[pos_idx]).norm(dim=-1)
                    - (z_all_k[anchors] - z_all_k[neg_idx]).norm(dim=-1)
                    + args.margin
                ).mean()

        # FK + run_dong_tips_only run in fp32 (pytorch-kinematics not validated
        # for fp16). `.float()` upcasts and preserves the autograd path back to MLPs.
        q_r_hat_fk    = q_r_hat.float()
        q_r_hat_t1_fk = q_r_hat_t1.float()
        if args.zero_wrj:
            q_r_hat_fk    = q_r_hat_fk.clone()
            q_r_hat_t1_fk = q_r_hat_t1_fk.clone()
            q_r_hat_fk[:, 0:2] = 0.0
            q_r_hat_t1_fk[:, 0:2] = 0.0

        # Fuse t and t+1 FK calls into one batched forward, then split.
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

        if args.lambda_joint > 0 and l_joint_w_pos:
            L_jnt = l_joint(q_r_hat.float(), robot_joint_names, l_joint_w_pos)
        else:
            L_jnt = torch.zeros((), device=DEVICE)

        L_total = (args.lambda_c   * L_cont
                 + args.lambda_rec * L_rec
                 + args.lambda_ltc * L_ltc
                 + args.lambda_tmp * L_temp
                 + args.lambda_joint * L_jnt)

        optimizer.zero_grad()
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
                "jnt": L_jnt.item(),
            },
        }

        if L_total.item() < best_total:
            best_total = L_total.item()
            torch.save(ckpt_payload, BEST_TOTAL_PATH)

        if val_sampler is not None and step > 0 and step % args.val_every == 0:
            last_val_metrics = _compute_eval_metrics(
                val_sampler, args.n_eval_batches, args.b_eval,
                E_h, E_r, E_X, D_X, D_r,
                HAND_CONFIG, args.zero_wrj,
                residual_decoder=args.residual_decoder,
            )
            if last_val_metrics is not None:
                score = (
                    last_val_metrics["rs"]
                    + last_val_metrics["nds"]
                    + args.lambda_tmp * last_val_metrics["nvs"]
                )
                print(
                    f"[step {step:05d}] VAL rs={last_val_metrics['rs']:.4f} "
                    f"nds={last_val_metrics['nds']:.4f} nvs={last_val_metrics['nvs']:.4f} "
                    f"rec={last_val_metrics['rec']:.4f} score={score:.4f}"
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
                f"jnt={L_jnt.item():.4f} lr={lr_now:.2e}{best_flag}"
            )
            if extra_human_count:
                open_count = extra_human_by_class.get(28, 0)
                fist_count = extra_human_by_class.get(29, 0)
                print(f"  batch extra_human={extra_human_count} open={open_count} fist={fist_count}")
            if args.log_metric_stats and metric_stats:
                stats = " | ".join(
                    f"{sub}(S_mean={s:.4f}, S_std={std:.4f}, S_min={mn:.4f}, S_max={mx:.4f})"
                    for sub, (s, std, mn, mx) in metric_stats.items()
                )
                print(f"  Xin S_k pairs {stats}")

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
    test_metrics = _compute_eval_metrics(
        test_sampler, args.n_eval_batches, args.b_eval,
        E_h, E_r, E_X, D_X, D_r,
        HAND_CONFIG, args.zero_wrj,
        residual_decoder=args.residual_decoder,
    )
    if test_metrics is not None:
        score = test_metrics["rs"] + test_metrics["nds"] + args.lambda_tmp * test_metrics["nvs"]
        print(
            f"TEST rs={test_metrics['rs']:.4f} nds={test_metrics['nds']:.4f} "
            f"nvs={test_metrics['nvs']:.4f} rec={test_metrics['rec']:.4f} score={score:.4f}"
        )

        if BEST_VAL_PATH.exists():
            ck = torch.load(BEST_VAL_PATH, map_location="cpu", weights_only=False)
            ck["test_metrics"] = test_metrics
            ck["test_score"]   = score
            torch.save(ck, BEST_VAL_PATH)
            print(f"Test metrics annotated on {BEST_VAL_PATH}.")
        if BEST_TOTAL_PATH.exists():
            ck = torch.load(BEST_TOTAL_PATH, map_location="cpu", weights_only=False)
            ck["test_metrics"] = test_metrics
            ck["test_score"]   = score
            torch.save(ck, BEST_TOTAL_PATH)
            print(f"Test metrics annotated on {BEST_TOTAL_PATH}.")

    print("done")


if __name__ == "__main__":
    main()
