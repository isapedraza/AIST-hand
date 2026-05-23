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
    p.add_argument("--w_r",      type=float, default=1.0, help="[DEPRECATED in Run 25] Legacy D_R weight in S_k. Ignored by Xin Cartesian S_k.")
    p.add_argument("--w_joints", type=float, default=1.0, help="[DEPRECATED in Run 25] Legacy D_joints weight. Ignored by Xin Cartesian S_k.")
    p.add_argument("--w_ahg",    type=float, default=1.0, help="[DEPRECATED in Run 25] Legacy D_ahg weight. Ignored by Xin Cartesian S_k.")
    p.add_argument("--lam_tip_pos",   type=float, default=1.0,  help="Weight for L_fingertip_pos (wrist->tip) in Xin S_k, non-thumb fingers (Xin Eq. 2).")
    p.add_argument("--lam_thumb_pos", type=float, default=10.0, help="Weight for L_thumb_pos (wrist->thumb tip) in Xin S_k (Xin Eq. 1). Separate from lam_tip_pos per paper.")
    p.add_argument("--lam_pinch",     type=float, default=10.0, help="Weight for L_pinch (thumb->primary finger vector) in Xin S_k (Xin Eq. 3).")
    p.add_argument("--lam_tip_rot",   type=float, default=10.0, help="Weight for L_fingertip_rot (DIP->tip unit vector) in Xin S_k (Xin Eq. 4).")
    p.add_argument("--lam_pip_pos",   type=float, default=1.0,  help="Weight for PIP position term in Xin S_k (DexMV-style; not in Xin paper).")
    p.add_argument("--lam_dr",    type=float, default=0.0,  help="Weight for Yan-style D_R quaternion term added to S_k. 0 disables (default). Active only in --single_latent path; uniform sum over common-joint quaternions.")
    p.add_argument("--lambda_joint",    type=float, default=0.0,
                   help="Weight for L_joint (joint position regularization). 0 disables.")
    p.add_argument("--robot_yaml_path", default=None,
                   help="Override path to robot.yaml (semantic_roles -> L_joint weights). Defaults to repo_root/robot/hands/shadow_hand/robot.yaml")
    p.add_argument("--single_latent", action="store_true",
                   help="Use HumanEncoder_E_h_single (one projection head, z_dim_total) and xin_sk_full instead of 5 per-finger subspaces.")
    p.add_argument("--z_dim_total", type=int, default=320,
                   help="Total latent dimension when --single_latent is set. Default 320 = 5*64 for capacity parity with the 5-subspace baseline.")
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
