"""Stage 1 training loop for cross-embodiment retargeting.

Usage (single robot, legacy):
    python -m cross_emb.training.loop

Usage (multi-robot):
    python -m cross_emb.training.loop --robots shadow allegro

Usage (add new robot to existing checkpoint):
    python -m cross_emb.training.loop \\
        --robots allegro \\
        --resume_ckpt checkpoints/stage1_latest.pt \\
        --freeze_shared

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
from cross_emb.nn.human_modules import HumanEncoder_E_h, SUBSPACE_LABEL_PREFIX, SUBSPACE_FINGERS
from cross_emb.nn.robot_modules import RobotEncoder_E_r, RobotDecoder_D_r
from cross_emb.nn.shared_modules import SharedEncoder_E_X, SharedDecoder_D_X
from .config import _parse_args
from cross_emb.rotations import d_r_pose, rot6d_to_matrix
from .losses import (
    _sk_w, _sk_wj, _ahg, xin_sk_per_finger,
    compute_W_linear, nt_xent_adaptive,
    compute_pairwise_S_ahg, compute_pairwise_S_xin,
)


def _set_seed(seed: int) -> None:
    """Fix all RNG sources used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(p: str, repo_root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else repo_root / path


def _load_robot_configs(args, repo_root: Path) -> list[dict]:
    """Return list of per-robot config dicts.

    Each dict: name, urdf, hand_config, valid_poses (Path|None), zero_wrj (bool).
    Falls back to legacy single-robot CLI args if --robots not provided.
    """
    if args.robots:
        import yaml
        cfgs = []
        for name in args.robots:
            yaml_path = repo_root / f"robot/hand-configs/{name}.yaml"
            with open(yaml_path) as f:
                rcfg = yaml.safe_load(f)
            cfgs.append({
                "name":        name,
                "urdf":        _resolve_path(rcfg["urdf"],        repo_root),
                "hand_config": yaml_path,
                "valid_poses": _resolve_path(rcfg["valid_poses"], repo_root) if rcfg.get("valid_poses") else None,
                "zero_wrj":    bool(rcfg.get("zero_wrj", False)),
                "eigengrasp":  _resolve_path(rcfg["eigengrasp"],  repo_root) if rcfg.get("eigengrasp") else None,
                "mjcf":        _resolve_path(rcfg["mjcf"],        repo_root) if rcfg.get("mjcf") else None,
                "n_knobs":     int(rcfg.get("n_knobs", 9)),
            })
        return cfgs

    # Legacy single-robot path
    DEX_ROOT = Path(args.dex_root)
    urdf        = Path(args.urdf_path)        if args.urdf_path         else DEX_ROOT / "robots/hands/shadow_hand/shadow_hand_right.urdf"
    hand_config = Path(args.hand_config)      if args.hand_config        else repo_root / "robot/hand-configs/shadow.yaml"
    valid_poses = Path(args.valid_poses_path) if args.valid_poses_path   else None
    eigengrasp  = Path(args.eigengrasp_path)  if args.eigengrasp_path    else None
    mjcf        = Path(args.mjcf_path)        if args.mjcf_path          else None
    return [{"name": "robot", "urdf": urdf, "hand_config": hand_config,
             "valid_poses": valid_poses, "zero_wrj": bool(args.zero_wrj),
             "eigengrasp": eigengrasp, "mjcf": mjcf, "n_knobs": args.n_knobs}]


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
    rot_repr: str = "quat",
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
            pose_h         = batch["pose_h"]
            pose_h_t1      = batch["pose_h_t1"]
            pose_h_sub     = batch["pose_h_sub"]
            chain_h_sub    = batch["chain_h_sub"]
            tips_h_sub     = batch["tips_h_sub"]
            tips_h_t1      = batch["tips_h_t1"]
            q_r_data       = batch["q_r"]
            common_fingers = batch["common_fingers"]
            common_labels  = batch["common_labels"]

            # Retarget: human Dong -> latent -> robot joint angles.
            z_t        = E_h_(pose_h)
            z_t1       = E_h_(pose_h_t1)
            q_r_hat    = D_r_(D_X_(z_t)).float()
            q_r_hat_t1 = D_r_(D_X_(z_t1)).float()
            if zero_wrj:
                q_r_hat    = q_r_hat.clone();    q_r_hat[:, 0:2]    = 0.0
                q_r_hat_t1 = q_r_hat_t1.clone(); q_r_hat_t1[:, 0:2] = 0.0

            # FK + full Dong stage2 for both timesteps.
            B = q_r_hat.shape[0]
            q_combined = torch.cat([q_r_hat, q_r_hat_t1], dim=0)
            fk_combined = eval_sampler.robot_rnd.run_fk(q_combined)
            quats_r_all, joint_labels_r, meta_r = eval_sampler.robot_rnd.run_dong_stage2(
                fk_combined, hand_config
            )

            joint_idx_r = [joint_labels_r.index(l) for l in common_labels]
            if rot_repr == "r6":
                pose_r_sub_t = meta_r["rot6"][:B, joint_idx_r]
            else:
                quats_r_sub_t = quats_r_all[:B, joint_idx_r]
                sign = torch.sign((pose_h_sub * quats_r_sub_t).sum(-1, keepdim=True))
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                pose_r_sub_t = quats_r_sub_t * sign

            rs = d_r_pose(pose_h_sub, pose_r_sub_t, rot_repr).mean().item()

            chain_r_dict = meta_r["chain_positions"]
            chain_r_t = torch.stack(
                [chain_r_dict[f][:B] for f in common_fingers], dim=1
            )
            nds = (chain_h_sub - chain_r_t).norm(dim=-1).sum(dim=(-2, -1)).mean().item()

            tips_r_all = meta_r["tips"]
            tip_labels_r = meta_r["tip_labels"]
            tip_idx_r  = [tip_labels_r.index(f) for f in common_fingers]
            tip_idx_h  = [human_labels_global.index(f) for f in common_fingers]
            tips_r_t   = tips_r_all[:B, tip_idx_r]
            tips_r_t1  = tips_r_all[B:, tip_idx_r]
            tips_h_t1_sub = tips_h_t1[:, tip_idx_h]
            v_h = tips_h_t1_sub - tips_h_sub
            v_r = tips_r_t1   - tips_r_t
            nvs = (v_h - v_r).norm(dim=-1).mean().item()

            z_r_       = E_X_(E_r_(q_r_data))
            q_r_hat_rd = D_r_(D_X_(z_r_))
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

_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
_SEG_ORDER    = ["mcp", "pip", "dip", "tip"]


def _cross_robot_contrastive(all_rd, args, sk_weights_dr, device) -> torch.Tensor:
    """Cross-embodiment triplet loss over a single pooled latent space.

    Yan et al. (2026): one contrastive pool per subspace containing human +
    ALL robots, with triplets mined across any embodiment. Unlike the legacy
    per-robot path, this aligns robot<->robot directly (not only transitively
    through the human).

    Cross-embodiment safety is ADAPTIVE per subspace (morphologies differ per
    finger, not only per hand):
      - D_R compares only the joints shared by every robot participating in the
        subspace (intersection of common_labels -> e.g. just `mcp` once a
        2-link finger like Barrett is in the pool).
      - The Xin tip/pinch/tip_rot terms use mini-tips [thumb_tip, finger_tip],
        so a hand contributes a subspace only if it actually has that finger;
        the per-hand finger count (3/5) never enters a fixed [N,5,3] layout.
      - The dense chain terms (D_joints, D_ahg, lam_finger) are dropped: they
        require identical chain lengths, which Barrett (2 pts) breaks.
    """
    if args.contrastive_mode != "triplet":
        raise NotImplementedError(
            "Cross-robot contrastive supports --contrastive_mode triplet only."
        )
    _pinch_eps1  = args.pinch_eps1_m / args.pinch_ref_hand_length_m
    _pinch_eps2  = args.pinch_eps2_m / args.pinch_ref_hand_length_m
    _pinch_sig_w = args.pinch_sigmoid_w_m * args.pinch_ref_hand_length_m

    L_cont = torch.tensor(0.0, device=device)

    for fidx, sub in enumerate(_FINGER_ORDER):
        prefix = sub + "_"
        parts = [rd for rd in all_rd if sub in rd["common_fingers"]]
        if not parts:
            continue

        # --- Adaptive shared joints for D_R (intersection across robots) ---
        shared = None
        for rd in parts:
            labs = {l for l in rd["common_labels"] if l.startswith(prefix)}
            shared = labs if shared is None else (shared & labs)
        shared_sorted = [f"{sub}_{s}" for s in _SEG_ORDER if f"{sub}_{s}" in shared]
        use_dr = args.lam_dr > 0 and len(shared_sorted) > 0
        if use_dr:
            seg_idx = torch.tensor(
                [_SEG_ORDER.index(l.split("_")[1]) for l in shared_sorted], device=device
            )
            w_dr = sk_weights_dr[sub][seg_idx]
            w_dr = w_dr / w_dr.sum().clamp(min=1e-8)

        # --- Build the pooled tensors (human + each robot) ---
        z_list, q_list, tips_list, chain_list = [], [], [], []
        for rd in parts:
            cf, cl = rd["common_fingers"], rd["common_labels"]
            t_i, f_i = cf.index("thumb"), cf.index(sub)
            zc_h = rd["z_t"].chunk(5, dim=-1)[fidx]
            zc_r = rd["z_r"].chunk(5, dim=-1)[fidx]

            def _mini(tips_sub, chain_sub):
                B = tips_sub.shape[0]
                tp = tips_sub.new_zeros(B, 5, 3)
                tp[:, 0]    = tips_sub[:, t_i]   # thumb tip
                tp[:, fidx] = tips_sub[:, f_i]   # current finger tip
                cp = chain_sub.new_zeros(B, 5, 4, 3)
                cp[:, fidx, 3] = chain_sub[:, f_i, -1]   # tip
                cp[:, fidx, 2] = chain_sub[:, f_i, -2]   # pre-tip (last segment)
                if fidx != 0:
                    cp[:, 0, 3] = chain_sub[:, t_i, -1]
                    cp[:, 0, 2] = chain_sub[:, t_i, -2]
                return tp, cp

            tp_h, cp_h = _mini(rd["tips_h_sub"], rd["chain_h_sub"])
            tp_r, cp_r = _mini(rd["tips_r_sub"], rd["chain_r_sub"])
            z_list     += [zc_h, zc_r]
            tips_list  += [tp_h, tp_r]
            chain_list += [cp_h, cp_r]
            if use_dr:
                ih = [cl.index(l) for l in shared_sorted]
                q_list += [rd["pose_h_sub"][:, ih, :], rd["pose_r_sub"][:, ih, :]]

        z_pool     = torch.cat(z_list, dim=0)
        tips_pool  = torch.cat(tips_list, dim=0)
        chain_pool = torch.cat(chain_list, dim=0)
        q_pool     = torch.cat(q_list, dim=0) if use_dr else None
        N = z_pool.shape[0]
        if N < 3:
            continue

        eff_tip = args.lam_thumb_tip if sub == "thumb" else args.lam_tip

        # --- Triplet mining over the pooled batch ---
        n       = N if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, N)
        anchors = torch.randperm(N, device=device)[:n]
        cand_a  = torch.randint(0, N - 1, (n,), device=device); cand_a += (cand_a >= anchors).long()
        cand_b  = torch.randint(0, N - 1, (n,), device=device); cand_b += (cand_b >= anchors).long()
        same = cand_b == cand_a
        if same.any():
            cand_b[same] = (cand_b[same] + 1) % N
            cand_b[same] += (cand_b[same] == anchors[same]).long()
            cand_b[same] %= N

        with torch.no_grad():
            def _S(cand):
                s = xin_sk_per_finger(
                    tips_pool[anchors], tips_pool[cand],
                    chain_pool[anchors], chain_pool[cand],
                    finger_idx=fidx,
                    lam_tip=eff_tip, lam_finger=0.0,   # lam_finger=0: cross-robot safe
                    lam_pinch=args.lam_pinch, lam_tip_rot=args.lam_tip_rot,
                    enable_switching=args.xin_switching,
                    pinch_eps1=_pinch_eps1, pinch_eps2=_pinch_eps2, pinch_sigmoid_w=_pinch_sig_w,
                )
                if use_dr:
                    qa, qc = q_pool[anchors], q_pool[cand]
                    if args.human_rot_repr == "r6":
                        Ra, Rc = rot6d_to_matrix(qa), rot6d_to_matrix(qc)
                        tr  = torch.matmul(Ra.transpose(-1, -2), Rc).diagonal(dim1=-2, dim2=-1).sum(-1)
                        per = (1.0 - ((tr - 1) * 0.5).clamp(-1, 1)) * 0.5
                    else:
                        per = 1.0 - (qa * qc).sum(-1) ** 2
                    s = s + args.lam_dr * (w_dr * per).sum(-1)
                return s

            S_a, S_b = _S(cand_a), _S(cand_b)
            a_closer = S_a <= S_b
            pos_idx  = torch.where(a_closer, cand_a, cand_b)
            neg_idx  = torch.where(a_closer, cand_b, cand_a)

        L_cont = L_cont + torch.relu(
            (z_pool[anchors] - z_pool[pos_idx]).norm(dim=-1)
            - (z_pool[anchors] - z_pool[neg_idx]).norm(dim=-1)
            + args.margin
        ).mean()

    return L_cont


def main() -> None:
    args = p = _parse_args()

    if args.seed < 0:
        args.seed = random.randint(0, 99_999)
        print(f"Seed: {args.seed} (auto-generated)")
    else:
        print(f"Seed: {args.seed}")
    _set_seed(args.seed)

    REPO_ROOT    = Path(args.repo_root)
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]

    if args.csv_path:
        CSV_PATH = Path(args.csv_path)
    elif args.human_rot_repr == "r6":
        CSV_PATH = REPO_ROOT / "human/datasets/hograspnet/processed/hograspnet_abl14_r6.csv"
    else:
        CSV_PATH = REPO_ROOT / "human/datasets/hograspnet/processed/hograspnet_abl11.csv"
    CKPT_PATH       = Path(args.ckpt_path)      if args.ckpt_path      else PACKAGE_ROOT / "checkpoints/stage1_latest.pt"
    EXTRA_HUMAN_CSV = Path(args.extra_human_csv) if args.extra_human_csv else None

    if EXTRA_HUMAN_CSV is not None and not EXTRA_HUMAN_CSV.exists():
        raise FileNotFoundError(f"extra_human_csv not found: {EXTRA_HUMAN_CSV}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    triplet_mode = "full_pool" if args.n_triplets is None or args.n_triplets <= 0 else str(args.n_triplets)
    print(
        f"Device: {DEVICE} | B={args.b} | N_STEPS={args.n_steps} "
        f"| triplets={triplet_mode} | margin={args.margin}"
    )
    print(f"scheduler=none (Yan-pure, constant lr={args.lr}) | resume={args.resume_ckpt or 'none'}")

    # ---------------------------------------------------------------------------
    # Robot configs
    # ---------------------------------------------------------------------------
    robot_cfgs = _load_robot_configs(args, REPO_ROOT)
    print(f"Robots ({len(robot_cfgs)}): {[cfg['name'] for cfg in robot_cfgs]}")

    # ---------------------------------------------------------------------------
    # Samplers — one per robot. Human data is identical across robots, so the
    # first sampler builds the HumanLoader and the rest share it (one CSV read,
    # one RAM copy) instead of re-loading the human CSV per robot.
    # ---------------------------------------------------------------------------
    shared_human = shared_extra = None
    for cfg in robot_cfgs:
        cfg["sampler"] = CrossEmbodimentSampler(
            csv_path         = CSV_PATH,
            urdf_path        = cfg["urdf"],
            hand_config_path = cfg["hand_config"],
            split            = "train",
            device           = DEVICE,
            valid_poses_path = cfg["valid_poses"],
            extra_human_csv  = EXTRA_HUMAN_CSV,
            extra_human_ratio= args.extra_human_ratio,
            human_rot_repr   = args.human_rot_repr,
            primitive_sample = args.primitive_sample,
            eigengrasp_path  = cfg.get("eigengrasp"),
            mjcf_path        = cfg.get("mjcf"),
            n_knobs          = cfg.get("n_knobs", 9),
            human_loader     = shared_human,
            extra_human_loader = shared_extra,
        )
        if shared_human is None:
            shared_human = cfg["sampler"].human_loader
            shared_extra = cfg["sampler"].extra_human_loader
        _probe    = cfg["sampler"].get_batch_temporal(1)
        cfg["J"]  = _probe["q_r"].shape[1]
        print(f"  {cfg['name']}: J={cfg['J']}  zero_wrj={cfg['zero_wrj']}")

    # Val samplers — share one val HumanLoader across robots (same as train).
    shared_human_val = None
    for cfg in robot_cfgs:
        if args.val_every and args.val_every > 0:
            cfg["val_sampler"] = CrossEmbodimentSampler(
                csv_path         = CSV_PATH,
                urdf_path        = cfg["urdf"],
                hand_config_path = cfg["hand_config"],
                split            = "val",
                device           = DEVICE,
                valid_poses_path = None,   # avoid double DONG_CACHE GPU load
                extra_human_csv  = None,
                extra_human_ratio= 0.0,
                human_rot_repr   = args.human_rot_repr,
                primitive_sample = args.primitive_sample,
                human_loader     = shared_human_val,
            )
            if shared_human_val is None:
                shared_human_val = cfg["val_sampler"].human_loader
        else:
            cfg["val_sampler"] = None

    if any(cfg["val_sampler"] for cfg in robot_cfgs):
        print(f"Val samplers: eval every {args.val_every} steps over {args.n_eval_batches} batches of B={args.b_eval}.")

    # ---------------------------------------------------------------------------
    # Models — shared + per-robot
    # ---------------------------------------------------------------------------
    human_in_dim = 6 if args.human_rot_repr == "r6" else 4
    E_h = HumanEncoder_E_h(in_dim=human_in_dim, hidden_dim=32, z_dim=args.z_dim).to(DEVICE)
    E_X = SharedEncoder_E_X(shared_dim=args.shared_dim, z_dim=args.z_dim).to(DEVICE)
    D_X = SharedDecoder_D_X(z_dim=args.z_dim, shared_dim=args.shared_dim).to(DEVICE)

    for cfg in robot_cfgs:
        cfg["E_r"] = RobotEncoder_E_r(n_joints=cfg["J"], shared_dim=args.shared_dim).to(DEVICE)
        cfg["D_r"] = RobotDecoder_D_r(n_joints=cfg["J"], shared_dim=args.shared_dim).to(DEVICE)

    # ---------------------------------------------------------------------------
    # Resume checkpoint
    # ---------------------------------------------------------------------------
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=DEVICE)
        E_h.load_state_dict(ckpt["E_h"])
        E_X.load_state_dict(ckpt["E_X"])
        D_X.load_state_dict(ckpt["D_X"])
        if "robots" in ckpt:
            for cfg in robot_cfgs:
                if cfg["name"] in ckpt["robots"]:
                    cfg["E_r"].load_state_dict(ckpt["robots"][cfg["name"]]["E_r"])
                    cfg["D_r"].load_state_dict(ckpt["robots"][cfg["name"]]["D_r"])
                    print(f"  Resumed {cfg['name']} E_r/D_r from checkpoint.")
                else:
                    print(f"  {cfg['name']} not in checkpoint — E_r/D_r initialized fresh.")
        elif "E_r" in ckpt:
            # Legacy single-robot checkpoint
            if len(robot_cfgs) == 1:
                robot_cfgs[0]["E_r"].load_state_dict(ckpt["E_r"])
                robot_cfgs[0]["D_r"].load_state_dict(ckpt["D_r"])
                print(f"  Resumed (legacy) E_r/D_r for {robot_cfgs[0]['name']}.")
            else:
                print("  Legacy checkpoint has single E_r/D_r but multiple robots configured — E_r/D_r initialized fresh for all.")
        print(f"Resumed shared weights from {args.resume_ckpt} (step {ckpt.get('step', '?')}). Optimizer reset fresh.")

    # ---------------------------------------------------------------------------
    # Freeze shared (for adding new robots to existing latent space)
    # ---------------------------------------------------------------------------
    if args.freeze_shared:
        for m in (E_h, E_X, D_X):
            for param in m.parameters():
                param.requires_grad_(False)
        print("Shared networks (E_h, E_X, D_X) FROZEN — training only E_r/D_r.")

    # ---------------------------------------------------------------------------
    # torch.compile
    # ---------------------------------------------------------------------------
    use_compile = bool(args.compile) and DEVICE == "cuda" and hasattr(torch, "compile")
    if use_compile:
        try:
            E_h = torch.compile(E_h, mode="reduce-overhead")
            E_X = torch.compile(E_X, mode="reduce-overhead")
            D_X = torch.compile(D_X, mode="reduce-overhead")
            for cfg in robot_cfgs:
                cfg["E_r"] = torch.compile(cfg["E_r"], mode="reduce-overhead")
                cfg["D_r"] = torch.compile(cfg["D_r"], mode="reduce-overhead")
            print(f"torch.compile: enabled (mode=reduce-overhead) on all modules")
        except Exception as e:
            print(f"torch.compile failed ({e}); falling back to eager.")
            use_compile = False
    else:
        print("torch.compile: disabled")

    # ---------------------------------------------------------------------------
    # Optimizer — shared params (unless frozen) + all robot E_r/D_r
    # ---------------------------------------------------------------------------
    trainable_params = (
        [] if args.freeze_shared else
        list(E_h.parameters()) + list(E_X.parameters()) + list(D_X.parameters())
    )
    for cfg in robot_cfgs:
        trainable_params += list(cfg["E_r"].parameters()) + list(cfg["D_r"].parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    # Yan et al. (2026): Adam with constant LR, no scheduler.

    use_amp = bool(args.amp) and DEVICE == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP: {'enabled (fp16)' if use_amp else 'disabled'}")
    print(f"freeze_shared={args.freeze_shared}")

    # Build device tensors from offline-computed per-joint weight constants.
    sk_weights_dr     = {sub: torch.tensor(w, device=DEVICE) for sub, w in _sk_w.items()}
    sk_weights_joints = {sub: torch.tensor(w, device=DEVICE) for sub, w in _sk_wj.items()}

    def _sd(m: torch.nn.Module) -> dict:
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

    multi = len(robot_cfgs) > 1
    if multi and args.sk_metric != "xin":
        print(f"[multi-robot] sk_metric={args.sk_metric} ignored; cross-robot uses Xin tip/pinch + adaptive D_R.")

    def _mem(tag: str) -> None:
        if args.mem_debug and DEVICE == "cuda":
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            m = torch.cuda.max_memory_allocated() / 1e9
            print(f"[mem] {tag:24s} alloc={a:6.2f}G reserved={r:6.2f}G peak={m:6.2f}G", flush=True)

    if args.mem_debug and DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _mem("before step0")

    for step in range(args.n_steps):

        # Accumulate losses over all robots.
        L_total = torch.tensor(0.0, device=DEVICE)
        accum_cont = accum_rec = accum_ltc = accum_temp = 0.0
        last_batch: dict | None = None
        metric_stats: dict = {}
        all_rd: list[dict] = []   # multi-robot: stash latents for cross-robot contrastive

        for cfg in robot_cfgs:
            E_r         = cfg["E_r"]
            D_r         = cfg["D_r"]
            sampler_r   = cfg["sampler"]
            hand_config = cfg["hand_config"]
            zero_wrj    = cfg["zero_wrj"]

            batch          = sampler_r.get_batch_temporal(args.b)
            last_batch     = batch
            pose_h         = batch["pose_h"]
            pose_h_t1      = batch["pose_h_t1"]
            q_r            = batch["q_r"]
            if zero_wrj:
                q_r = q_r.clone()
                q_r[:, 0:2] = 0.0
            pose_h_sub     = batch["pose_h_sub"]
            pose_r_sub     = batch["pose_r_sub"]
            tips_h_sub     = batch["tips_h_sub"]
            tips_r_sub     = batch["tips_r_sub"]
            tips_h_t1      = batch["tips_h_t1"]
            chain_h_sub    = batch["chain_h_sub"]
            chain_r_sub    = batch["chain_r_sub"]
            common_fingers = batch["common_fingers"]
            common_labels  = batch["common_labels"]

            try:
              with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                z_t  = E_h(pose_h)
                z_t1 = E_h(pose_h_t1)
                z_r  = E_X(E_r(q_r))

                q_r_hat        = D_r(D_X(z_r))    # reconstruction (robot→latent→robot) for L_rec
                z_h_rt         = E_X(D_X(z_t))
                q_r_from_h_t   = D_r(D_X(z_t))    # retargeted from human t  (for L_temp)
                q_r_from_h_t1  = D_r(D_X(z_t1))   # retargeted from human t+1 (for L_temp)

                L_rec = (q_r - q_r_hat).norm(dim=-1).mean()
                L_ltc = (z_t - z_h_rt).norm(dim=-1).mean()
            except torch.cuda.OutOfMemoryError:
                if DEVICE == "cuda":
                    print(f"[mem] OOM at step {step} during forward of robot '{cfg['name']}'", flush=True)
                    print(torch.cuda.memory_summary(), flush=True)
                raise

            # --- Per-subspace contrastive loss (Yan et al. 2026) ---
            # Single robot: legacy per-robot pool (human + this robot), below.
            # Multi robot: this loop is skipped; the contrastive is computed
            #   cross-robot AFTER the robot loop (see _cross_robot_contrastive),
            #   pooling human + ALL robots so triplets cross embodiments directly.
            z_t_subs = z_t.chunk(5, dim=-1)
            z_r_subs = z_r.chunk(5, dim=-1)
            L_cont = torch.tensor(0.0, device=DEVICE)

            _use_xin = args.sk_metric == "xin"
            if _use_xin and not multi:
                _tips_r_sub    = chain_r_sub[:, :, 3, :]
                _tips_all_xin  = torch.cat([tips_h_sub,  _tips_r_sub],  dim=0)
                _chain_all_xin = torch.cat([chain_h_sub, chain_r_sub],  dim=0)
                _finger_order  = ["thumb", "index", "middle", "ring", "pinky"]
                _pinch_eps1    = args.pinch_eps1_m / args.pinch_ref_hand_length_m
                _pinch_eps2    = args.pinch_eps2_m / args.pinch_ref_hand_length_m
                _pinch_sig_w   = args.pinch_sigmoid_w_m * args.pinch_ref_hand_length_m

            _subspace_iter = () if multi else enumerate(("thumb", "index", "middle", "ring", "pinky"))
            for k, sub in _subspace_iter:
                prefixes   = SUBSPACE_LABEL_PREFIX[sub]
                sub_finger = SUBSPACE_FINGERS[sub]
                jidx = [i for i, l in enumerate(common_labels) if l.startswith(prefixes)]
                tidx = [i for i, f in enumerate(common_fingers) if f in sub_finger]
                if not jidx:
                    continue
                z_h_k = z_t_subs[k]
                z_r_k = z_r_subs[k]
                q_h_k     = pose_h_sub[:, jidx, :]
                q_r_k     = pose_r_sub[:, jidx, :]
                chain_h_k = chain_h_sub[:, tidx, :, :] if tidx else chain_h_sub[:, :0, :, :]
                chain_r_k = chain_r_sub[:, tidx, :, :] if tidx else chain_r_sub[:, :0, :, :]

                z_all_k     = torch.cat([z_h_k, z_r_k], dim=0)
                q_all_k     = torch.cat([q_h_k, q_r_k], dim=0)
                chain_all_k = torch.cat([chain_h_k, chain_r_k], dim=0)
                B2 = z_all_k.shape[0]
                if B2 < 3:
                    continue

                if args.contrastive_mode == "infonce":
                    N_inf   = min(args.infonce_n, B2)
                    inf_idx = torch.randperm(B2, device=DEVICE)[:N_inf]
                    z_inf   = z_all_k[inf_idx]

                    with torch.no_grad():
                        _seg_order = ["mcp", "pip", "dip", "tip"]
                        _jlabs     = [common_labels[j].split("_")[1] for j in jidx]
                        _dr_idx    = torch.tensor([_seg_order.index(s) for s in _jlabs], device=DEVICE)
                        _w_dr      = sk_weights_dr[sub][_dr_idx]
                        _w_dr      = _w_dr / _w_dr.sum().clamp(min=1e-8)

                        if _use_xin:
                            _fidx       = _finger_order.index(sub)
                            _eff_tip    = args.lam_thumb_tip    if sub == "thumb" else args.lam_tip
                            _eff_finger = args.lam_thumb_finger if sub == "thumb" else args.lam_finger
                            tips_inf  = _tips_all_xin[inf_idx]
                            chain_inf = _chain_all_xin[inf_idx]
                            S_pw = compute_pairwise_S_xin(
                                tips_inf, chain_inf, finger_idx=_fidx,
                                lam_tip=_eff_tip, lam_finger=_eff_finger,
                                lam_pinch=args.lam_pinch, lam_tip_rot=args.lam_tip_rot,
                                enable_switching=args.xin_switching,
                                pinch_eps1=_pinch_eps1, pinch_eps2=_pinch_eps2,
                                pinch_sigmoid_w=_pinch_sig_w,
                                lam_dr=args.lam_dr,
                                q_pool=q_all_k[inf_idx], w_dr=_w_dr, w_r=1.0,
                                rot_repr=args.human_rot_repr,
                            )
                        else:
                            S_pw = compute_pairwise_S_ahg(
                                q_all_k[inf_idx], chain_all_k[inf_idx],
                                w_dr=_w_dr, w_joints=sk_weights_joints[sub],
                                w_r=args.w_r, w_joints_scale=args.w_joints, w_ahg=args.w_ahg,
                                rot_repr=args.human_rot_repr,
                            )
                        W_pw = compute_W_linear(S_pw)

                    L_cont = L_cont + nt_xent_adaptive(z_inf, W_pw, tau=args.infonce_tau)
                    continue

                # Triplet path
                n       = B2 if args.n_triplets is None or args.n_triplets <= 0 else min(args.n_triplets, B2)
                anchors = torch.randperm(B2, device=DEVICE)[:n]
                cand_a  = torch.randint(0, B2 - 1, (n,), device=DEVICE)
                cand_a  = cand_a + (cand_a >= anchors).long()
                cand_b  = torch.randint(0, B2 - 1, (n,), device=DEVICE)
                cand_b  = cand_b + (cand_b >= anchors).long()
                same = cand_b == cand_a
                if same.any():
                    cand_b[same] = (cand_b[same] + 1) % B2
                    cand_b[same] += (cand_b[same] == anchors[same]).long()
                    cand_b[same] %= B2

                with torch.no_grad():
                    qa       = q_all_k[anchors]
                    chain_a  = chain_all_k[anchors]
                    q_ca     = q_all_k[cand_a]
                    q_cb     = q_all_k[cand_b]
                    chain_ca = chain_all_k[cand_a]
                    chain_cb = chain_all_k[cand_b]

                    if args.human_rot_repr == "r6":
                        R_a  = rot6d_to_matrix(qa)
                        R_ca = rot6d_to_matrix(q_ca)
                        R_cb = rot6d_to_matrix(q_cb)
                        t_a = torch.matmul(R_a.transpose(-1, -2), R_ca).diagonal(dim1=-2, dim2=-1).sum(-1)
                        t_b = torch.matmul(R_a.transpose(-1, -2), R_cb).diagonal(dim1=-2, dim2=-1).sum(-1)
                        per_jdr_a = (1.0 - ((t_a - 1) * 0.5).clamp(-1, 1)) * 0.5
                        per_jdr_b = (1.0 - ((t_b - 1) * 0.5).clamp(-1, 1)) * 0.5
                    else:
                        dot_a = (qa * q_ca).sum(-1)
                        dot_b = (qa * q_cb).sum(-1)
                        per_jdr_a = 1.0 - dot_a ** 2
                        per_jdr_b = 1.0 - dot_b ** 2

                    _seg_order = ["mcp", "pip", "dip", "tip"]
                    _jlabs     = [common_labels[j].split("_")[1] for j in jidx]
                    _dr_idx    = torch.tensor([_seg_order.index(s) for s in _jlabs], device=DEVICE)
                    _w_dr      = sk_weights_dr[sub][_dr_idx]
                    _w_dr      = _w_dr / _w_dr.sum().clamp(min=1e-8)
                    D_R_a      = (_w_dr * per_jdr_a).sum(dim=-1)
                    D_R_b      = (_w_dr * per_jdr_b).sum(dim=-1)

                    if _use_xin:
                        _fidx = _finger_order.index(sub)
                        _eff_tip    = args.lam_thumb_tip    if sub == "thumb" else args.lam_tip
                        _eff_finger = args.lam_thumb_finger if sub == "thumb" else args.lam_finger
                        S_a = xin_sk_per_finger(
                            _tips_all_xin[anchors], _tips_all_xin[cand_a],
                            _chain_all_xin[anchors], _chain_all_xin[cand_a],
                            finger_idx=_fidx,
                            lam_tip=_eff_tip, lam_finger=_eff_finger,
                            lam_pinch=args.lam_pinch, lam_tip_rot=args.lam_tip_rot,
                            enable_switching=args.xin_switching,
                            pinch_eps1=_pinch_eps1, pinch_eps2=_pinch_eps2, pinch_sigmoid_w=_pinch_sig_w,
                        )
                        S_b = xin_sk_per_finger(
                            _tips_all_xin[anchors], _tips_all_xin[cand_b],
                            _chain_all_xin[anchors], _chain_all_xin[cand_b],
                            finger_idx=_fidx,
                            lam_tip=_eff_tip, lam_finger=_eff_finger,
                            lam_pinch=args.lam_pinch, lam_tip_rot=args.lam_tip_rot,
                            enable_switching=args.xin_switching,
                            pinch_eps1=_pinch_eps1, pinch_eps2=_pinch_eps2, pinch_sigmoid_w=_pinch_sig_w,
                        )
                        if args.lam_dr > 0:
                            S_a = S_a + args.lam_dr * D_R_a
                            S_b = S_b + args.lam_dr * D_R_b
                    else:
                        _w_joints  = sk_weights_joints[sub]
                        D_joints_a = (_w_joints * (chain_a  - chain_ca).norm(dim=-1)).sum(dim=(-2, -1))
                        D_joints_b = (_w_joints * (chain_a  - chain_cb).norm(dim=-1)).sum(dim=(-2, -1))
                        D_ahg_a = _ahg(chain_a, chain_ca)
                        D_ahg_b = _ahg(chain_a, chain_cb)
                        S_a = args.w_r * D_R_a + args.w_joints * D_joints_a + args.w_ahg * D_ahg_a
                        S_b = args.w_r * D_R_b + args.w_joints * D_joints_b + args.w_ahg * D_ahg_b

                    if args.log_metric_stats:
                        S_pairs = torch.cat([S_a, S_b])
                        metric_stats[f"{cfg['name']}/{sub}"] = (
                            D_R_a.mean().item(),
                            0.0 if _use_xin else D_joints_a.mean().item(),
                            0.0 if _use_xin else D_ahg_a.mean().item(),
                            S_pairs.mean().item(),
                            S_pairs.std().item(),
                            S_pairs.min().item(),
                            S_pairs.max().item(),
                        )

                    a_closer = S_a <= S_b
                    pos_idx  = torch.where(a_closer, cand_a, cand_b)
                    neg_idx  = torch.where(a_closer, cand_b, cand_a)

                L_cont = L_cont + torch.relu(
                    (z_all_k[anchors] - z_all_k[pos_idx]).norm(dim=-1)
                    - (z_all_k[anchors] - z_all_k[neg_idx]).norm(dim=-1)
                    + args.margin
                ).mean()

            # --- Temporal loss (per-robot FK + Dong tips) ---
            q_r_hat_fk    = q_r_from_h_t.float()
            q_r_hat_t1_fk = q_r_from_h_t1.float()
            if zero_wrj:
                q_r_hat_fk    = q_r_hat_fk.clone();    q_r_hat_fk[:, 0:2]    = 0.0
                q_r_hat_t1_fk = q_r_hat_t1_fk.clone(); q_r_hat_t1_fk[:, 0:2] = 0.0

            B_fk       = q_r_hat_fk.shape[0]
            q_combined = torch.cat([q_r_hat_fk, q_r_hat_t1_fk], dim=0)
            fk_combined = sampler_r.robot_rnd.run_fk(q_combined)
            tips_all, tip_labels = sampler_r.robot_rnd.run_dong_tips_only(fk_combined, hand_config)
            common_idx_r  = [tip_labels.index(f) for f in common_fingers]
            human_labels  = ["thumb", "index", "middle", "ring", "pinky"]
            common_idx_h  = [human_labels.index(f) for f in common_fingers]
            tips_r_all    = tips_all.to(DEVICE)[:, common_idx_r, :]
            tips_r_t_sub  = tips_r_all[:B_fk]
            tips_r_t1_sub = tips_r_all[B_fk:]
            tips_h_t1_sub = tips_h_t1[:, common_idx_h, :]
            L_temp = ((tips_h_t1_sub - tips_h_sub) - (tips_r_t1_sub - tips_r_t_sub)).norm(dim=-1).mean()

            L_robot = (args.lambda_c   * L_cont
                     + args.lambda_rec * L_rec
                     + args.lambda_ltc * L_ltc
                     + args.lambda_tmp * L_temp)
            L_total      = L_total + L_robot
            accum_cont  += L_cont.item()
            accum_rec   += L_rec.item()
            accum_ltc   += L_ltc.item()
            accum_temp  += L_temp.item()

            if multi:
                all_rd.append({
                    "z_t": z_t, "z_r": z_r,
                    "pose_h_sub": pose_h_sub, "pose_r_sub": pose_r_sub,
                    "tips_h_sub": tips_h_sub, "tips_r_sub": tips_r_sub,
                    "chain_h_sub": chain_h_sub, "chain_r_sub": chain_r_sub,
                    "common_labels": common_labels, "common_fingers": common_fingers,
                })

            if step == 0:
                _mem(f"pass1 done {cfg['name']}")

        # --- Cross-robot contrastive (one pooled latent space, Yan-style) ---
        if multi:
            L_cont_pooled = _cross_robot_contrastive(all_rd, args, sk_weights_dr, DEVICE)
            L_total = L_total + args.lambda_c * L_cont_pooled
            # Scale so the downstream `accum_cont / n_r` reporting yields the true value.
            accum_cont = L_cont_pooled.item() * len(robot_cfgs)
            if step == 0:
                _mem("pass2 contrastive")

        # --- Backward ---
        try:
            optimizer.zero_grad()
            scaler.scale(L_total).backward()
            scaler.unscale_(optimizer)
            all_grad_params = (
                [] if args.freeze_shared else
                list(E_h.parameters()) + list(E_X.parameters()) + list(D_X.parameters())
            )
            for cfg in robot_cfgs:
                all_grad_params += list(cfg["E_r"].parameters()) + list(cfg["D_r"].parameters())
            torch.nn.utils.clip_grad_norm_(all_grad_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        except torch.cuda.OutOfMemoryError:
            if DEVICE == "cuda":
                print(f"[mem] OOM at step {step} during backward", flush=True)
                print(torch.cuda.memory_summary(), flush=True)
            raise
        if step == 0:
            _mem("after backward")

        # --- Checkpoint payload (uses averaged losses across robots) ---
        n_r = len(robot_cfgs)
        ckpt_payload = {
            "step":   step,
            "seed":   args.seed,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "E_h":    _sd(E_h),
            "E_X":    _sd(E_X),
            "D_X":    _sd(D_X),
            "robots": {
                cfg["name"]: {"E_r": _sd(cfg["E_r"]), "D_r": _sd(cfg["D_r"])}
                for cfg in robot_cfgs
            },
            "losses": {
                "total": L_total.item(),
                "cont":  accum_cont  / n_r,
                "rec":   accum_rec   / n_r,
                "ltc":   accum_ltc   / n_r,
                "temp":  accum_temp  / n_r,
            },
        }
        # Legacy compat: expose top-level E_r/D_r when single robot
        if n_r == 1:
            ckpt_payload["E_r"]      = _sd(robot_cfgs[0]["E_r"])
            ckpt_payload["D_r"]      = _sd(robot_cfgs[0]["D_r"])
            ckpt_payload["zero_wrj"] = robot_cfgs[0]["zero_wrj"]

        if L_total.item() < best_total:
            best_total = L_total.item()
            torch.save(ckpt_payload, BEST_TOTAL_PATH)

        # --- Val eval (per robot, average score) ---
        if last_val_metrics is None:
            last_val_metrics = {}
        if args.val_every and step > 0 and step % args.val_every == 0:
            val_scores = []
            for cfg in robot_cfgs:
                if cfg["val_sampler"] is None:
                    continue
                m = _compute_eval_metrics(
                    cfg["val_sampler"], args.n_eval_batches, args.b_eval,
                    E_h, cfg["E_r"], E_X, D_X, cfg["D_r"],
                    cfg["hand_config"], cfg["zero_wrj"],
                    rot_repr=args.human_rot_repr,
                )
                if m is not None:
                    last_val_metrics[cfg["name"]] = m
                    score = m["rs"] + m["nds"] + args.lambda_tmp * m["nvs"]
                    val_scores.append(score)
                    print(
                        f"[step {step:05d}] VAL {cfg['name']} rs={m['rs']:.4f} "
                        f"nds={m['nds']:.4f} nvs={m['nvs']:.4f} rec={m['rec']:.4f} score={score:.4f}"
                    )
            if val_scores:
                avg_score = sum(val_scores) / len(val_scores)
                best_flag = ""
                if avg_score < best_val_score:
                    best_val_score = avg_score
                    best_flag = " *best_val"
                    val_payload = dict(ckpt_payload)
                    val_payload["val_metrics"] = last_val_metrics
                    val_payload["val_score"]   = avg_score
                    torch.save(val_payload, BEST_VAL_PATH)
                print(f"[step {step:05d}] VAL avg_score={avg_score:.4f}{best_flag}")

        if step % args.log_every == 0:
            lr_now    = optimizer.param_groups[0]["lr"]
            best_flag = " *best_total" if L_total.item() == best_total else ""
            robot_tag = f" [{','.join(cfg['name'] for cfg in robot_cfgs)}]" if n_r > 1 else ""
            print(
                f"[step {step:05d}]{robot_tag} loss total={L_total.item():.4f} "
                f"cont={accum_cont/n_r:.4f} rec={accum_rec/n_r:.4f} "
                f"ltc={accum_ltc/n_r:.4f} temp={accum_temp/n_r:.4f} "
                f"lr={lr_now:.2e}{best_flag}"
            )
            if last_batch is not None:
                extra_human_count = last_batch.get("extra_human_count", 0)
                if extra_human_count:
                    extra_human_by_class = last_batch.get("extra_human_by_class", {})
                    open_count = extra_human_by_class.get(28, 0)
                    fist_count = extra_human_by_class.get(29, 0)
                    print(f"  batch extra_human={extra_human_count} open={open_count} fist={fist_count}")
            if args.log_metric_stats and metric_stats:
                stats = " | ".join(
                    f"{key}(D_R={dr:.4f}, D_joints={dj:.4f}, D_ahg={da:.4f}, S_mean={s:.4f}, S_std={std:.4f}, S_min={mn:.4f}, S_max={mx:.4f})"
                    for key, (dr, dj, da, s, std, mn, mx) in metric_stats.items()
                )
                print(f"  metric pairs {stats}")

        if step % args.ckpt_every == 0:
            torch.save({
                **ckpt_payload,
                "optimizer": optimizer.state_dict(),
            }, CKPT_PATH)

    # ---------------------------------------------------------------------------
    # Final test eval
    # ---------------------------------------------------------------------------
    if args.skip_final_eval:
        print("\n=== Final test eval SKIPPED (--skip_final_eval) ===")
        eval_cfgs = []
    else:
        print("\n=== Final test eval (subjects 74-99) ===")
        eval_cfgs = robot_cfgs
    for cfg in eval_cfgs:
        test_sampler = CrossEmbodimentSampler(
            csv_path         = CSV_PATH,
            urdf_path        = cfg["urdf"],
            hand_config_path = cfg["hand_config"],
            split            = "test",
            device           = DEVICE,
            valid_poses_path = None,
            extra_human_csv  = None,
            extra_human_ratio= 0.0,
            human_rot_repr   = args.human_rot_repr,
            primitive_sample = args.primitive_sample,
        )
        test_metrics = _compute_eval_metrics(
            test_sampler, args.n_eval_batches, args.b_eval,
            E_h, cfg["E_r"], E_X, D_X, cfg["D_r"],
            cfg["hand_config"], cfg["zero_wrj"],
            rot_repr=args.human_rot_repr,
        )
        if test_metrics:
            print(
                f"TEST {cfg['name']}: rs={test_metrics['rs']:.4f}  nds={test_metrics['nds']:.4f}  "
                f"nvs={test_metrics['nvs']:.4f}  rec={test_metrics['rec']:.4f}  "
                f"(n_batches={test_metrics['n_batches']} b={test_metrics['b_eval']})"
            )

    torch.save({**ckpt_payload, "optimizer": optimizer.state_dict()}, CKPT_PATH)
    print(f"Saved final checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()
