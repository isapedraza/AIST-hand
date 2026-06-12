from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--repo_root", default="/home/yareeez/AIST-hand")
    p.add_argument("--dex_root",  default="/home/yareeez/dex-urdf")
    p.add_argument("--csv_path",  default=None, help="Override CSV path")
    p.add_argument("--ckpt_path", default=None, help="Override checkpoint path")
    p.add_argument("--hand_config",       default=None, help="Override hand config YAML")
    p.add_argument("--valid_poses_path",  default=None, help="Path to valid_robot_poses.npz (mode=VALID_NPZ). If omitted, uses random uniform sampling.")
    p.add_argument("--urdf_path",         default=None, help="Override robot URDF path. Default: Shadow URDF under --dex_root.")
    p.add_argument("--robots", nargs='+', default=None,
                   help="Robot names to train (e.g. --robots shadow allegro leap). "
                        "Each name maps to robot/hand-configs/{name}.yaml in the repo.")
    p.add_argument("--eigengrasp_path",   default=None, help="Path to eigengrasp NPZ (PCA basis). Enables EIGENGRASP_ONLINE sampling: sample from PCA space + MuJoCo collision filter live. Requires --mjcf_path. Ignored if --valid_poses_path set.")
    p.add_argument("--mjcf_path",         default=None, help="Path to MJCF XML for MuJoCo collision filtering in EIGENGRASP_ONLINE mode.")
    p.add_argument("--n_knobs",           type=int, default=9, help="Number of eigengrasp components to sample (default 9 = ~91%% variance).")
    p.add_argument("--freeze_shared",     action="store_true", help="Freeze E_h, E_X, D_X — only train E_r/D_r. Use when adding a new robot to an existing checkpoint.")
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
    p.add_argument("--margin",      type=float, default=0.05)
    p.add_argument("--w_r",     type=float, default=1.0, help="Weight for D_R in S_k (ahg mode).")
    p.add_argument("--w_joints", type=float, default=1.0, help="Weight for D_joints in S_k (ahg mode).")
    p.add_argument("--w_ahg",   type=float, default=1.0, help="Weight for D_ahg in S_k (ahg mode). S_k = w_r*D_R + w_joints*D_joints + w_ahg*D_ahg.")
    # Contrastive loss mode
    p.add_argument("--contrastive_mode", choices=["triplet", "infonce"], default="triplet",
                   help="Contrastive loss type. 'triplet'=Yan margin loss (default). 'infonce'=NT-Xent adaptive (SiMHand Eq.2+3).")
    p.add_argument("--infonce_tau", type=float, default=0.5,
                   help="Temperature τ for InfoNCE/NT-Xent loss.")
    p.add_argument("--infonce_n", type=int, default=512,
                   help="Pool size N subsampled from [2B] per subspace for InfoNCE. 512 → 511 negatives per anchor.")
    # Xin S_k toggle (--sk_metric xin replaces D_joints+D_ahg with Xin Cartesian terms)
    p.add_argument("--sk_metric", choices=["ahg", "xin"], default="ahg",
                   help="Similarity metric for triplet selection. 'ahg'=current (D_R+D_joints+D_ahg). 'xin'=Xin Cartesian terms + optional D_R via --lam_dr.")
    p.add_argument("--lam_dr",          type=float, default=0.0,  help="D_R weight added to Xin S_k (xin mode only). 0=disabled.")
    p.add_argument("--lam_tip",         type=float, default=0.0,  help="Xin wrist->tip position weight (non-thumb fingers).")
    p.add_argument("--lam_thumb_tip",   type=float, default=0.0,  help="Xin wrist->tip position weight (thumb).")
    p.add_argument("--lam_finger",      type=float, default=0.0,  help="Xin dense MCP/PIP/DIP/TIP chain weight (non-thumb).")
    p.add_argument("--lam_thumb_finger",type=float, default=0.0,  help="Xin dense chain weight (thumb).")
    p.add_argument("--lam_pinch",       type=float, default=0.0,  help="Xin thumb->finger pinch vector weight (index/middle/ring).")
    p.add_argument("--lam_tip_rot",     type=float, default=0.0,  help="Xin DIP->TIP unit vector weight.")
    p.add_argument("--xin_switching",   action="store_true",
                   help="Enable run21-paper-sk sigmoid switching for pinch (Xin Eq. 197/216). Suppresses tip_pos when pinch active.")
    p.add_argument("--pinch_eps1_m",           type=float, default=0.1,   help="Xin pinch intent threshold (meters).")
    p.add_argument("--pinch_eps2_m",           type=float, default=0.01,  help="Xin pinch contact threshold (meters).")
    p.add_argument("--pinch_sigmoid_w_m",      type=float, default=10.0,  help="Xin sigmoid slope (1/m).")
    p.add_argument("--pinch_ref_hand_length_m",type=float, default=0.197, help="Reference hand length (m) for normalizing pinch thresholds.")
    p.add_argument("--extra_human_ratio", type=float, default=0.10)
    p.add_argument("--log_metric_stats", action="store_true", help="Log D_R/D_ee/S_k scale diagnostics by subspace.")
    p.add_argument(
        "--zero_wrj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Shadow WRJ2/WRJ1 targets and FK inputs to zero. Dong human input is wrist-local.",
    )
    p.add_argument("--primitive_sample", action="store_true",
                   help="Sample robot poses from DexGrasp-Zero primitive space (M_h). No valid_poses NPZ needed. "
                        "Joints coupled by finger/type; primitives built from URDF FK at startup.")
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
        "--human_rot_repr",
        choices=["quat", "r6"],
        default="quat",
        help="Rotation representation for human poses. 'r6' uses 6D (Zhou et al. 2019) for better SO(3) continuity.",
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
    p.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Skip the final test-split eval pass. Recommended for multi-robot runs where it can crash.",
    )
    return p.parse_args()
