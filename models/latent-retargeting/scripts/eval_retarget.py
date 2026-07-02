"""
Offline eval: RS / NDS / NVS on HOGraspNet test split (subjects 74-99).

Usage (local):
    python eval_retarget.py \
        --ckpt /home/yareeez/Downloads/stage1_allegro_bodex_objbal_udhm.pt \
        --csv  /path/to/hograspnet_abl14_r6.csv

Usage (Colab cell):
    !python {REPO_ROOT}/models/latent-retargeting/scripts/eval_retarget.py \
        --ckpt {CKPT_PATH} --csv {CSV_PATH}

Args:
    --ckpt       Path to checkpoint (.pt).
    --csv        HOGraspNet r6 CSV used during training.
    --repo_root  Repo root (default: auto-detected).
    --robots     Subset of robots to eval, e.g. --robots shadow allegro.
                 Default: all robots stored in the checkpoint.
    --n_batches  Eval batches (default 20).
    --b_eval     Batch size for eval (default 5000).
    --device     torch device (default cpu; use cuda if available).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = _SCRIPT_DIR.parents[2]   # scripts/ -> latent-retargeting/ -> models/ -> AIST-hand/

# Make sure the src package is importable without pip install -e
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RS/NDS/NVS eval for retargeting checkpoints")
    parser.add_argument("--ckpt",      required=True, help="Checkpoint .pt file")
    parser.add_argument("--csv",       required=True, help="HOGraspNet r6 CSV (hograspnet_abl14_r6.csv)")
    parser.add_argument("--repo_root", default=str(REPO_ROOT), help="AIST-hand repo root")
    parser.add_argument("--robots",    nargs="*", default=None,
                        help="Robots to eval (default: all in checkpoint)")
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--b_eval",    type=int, default=5000)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    device    = torch.device(args.device)
    csv_path  = Path(args.csv)
    ckpt_path = Path(args.ckpt)

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    print(f"Checkpoint : {ckpt_path}")
    print(f"CSV        : {csv_path}")
    print(f"Device     : {device}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})

    z_dim       = int(cfg.get("z_dim",          16))
    shared_dim  = int(cfg.get("shared_dim",    1024))
    rot_repr    = str(cfg.get("human_rot_repr", "r6"))
    h_encoder   = str(cfg.get("human_encoder", "spatial"))
    tmp_window  = int(cfg.get("temporal_window",  5))
    prim_sample = bool(cfg.get("primitive_sample", False))

    robot_names_in_ckpt = list(ckpt.get("robots", {}).keys())
    if not robot_names_in_ckpt:
        robot_names_in_ckpt = ["robot"]   # legacy single-robot ckpt
    robots_to_eval = args.robots or robot_names_in_ckpt

    print(f"Robots in ckpt : {robot_names_in_ckpt}")
    print(f"Evaluating     : {robots_to_eval}")
    print(f"z_dim={z_dim}  shared_dim={shared_dim}  rot_repr={rot_repr}")
    print()

    from cross_emb.nn.human_modules  import HumanEncoder_E_h
    from cross_emb.nn.shared_modules import SharedEncoder_E_X, SharedDecoder_D_X
    from cross_emb.nn.robot_modules  import RobotEncoder_E_r, RobotDecoder_D_r
    from cross_emb.loaders           import CrossEmbodimentSampler
    from cross_emb.training.loop     import _compute_eval_metrics

    human_in_dim = 6 if rot_repr == "r6" else 4
    E_h = HumanEncoder_E_h(in_dim=human_in_dim, hidden_dim=32, z_dim=z_dim).to(device)
    E_X = SharedEncoder_E_X(shared_dim=shared_dim, z_dim=z_dim).to(device)
    D_X = SharedDecoder_D_X(z_dim=z_dim, shared_dim=shared_dim).to(device)
    E_h.load_state_dict(ckpt["E_h"])
    E_X.load_state_dict(ckpt["E_X"])
    D_X.load_state_dict(ckpt["D_X"])
    E_h.eval(); E_X.eval(); D_X.eval()

    results = {}
    for robot_name in robots_to_eval:
        hand_config_path = repo_root / f"robot/hand-configs/{robot_name}.yaml"
        if not hand_config_path.exists():
            print(f"[{robot_name}] hand-config not found: {hand_config_path} — skip")
            continue
        with open(hand_config_path) as f:
            rcfg = yaml.safe_load(f)
        urdf_path = repo_root / rcfg["urdf"]
        zero_wrj  = bool(rcfg.get("zero_wrj", False))

        print(f"Building test sampler for {robot_name} ...")
        test_sampler = CrossEmbodimentSampler(
            csv_path          = csv_path,
            urdf_path         = urdf_path,
            hand_config_path  = hand_config_path,
            split             = "test",
            device            = device,
            valid_poses_path  = None,
            extra_human_csv   = None,
            extra_human_ratio = 0.0,
            human_rot_repr    = rot_repr,
            temporal_window   = tmp_window,
            primitive_sample  = prim_sample,
        )
        J = len(test_sampler.robot_rnd.chain_joint_names)

        E_r = RobotEncoder_E_r(n_joints=J, shared_dim=shared_dim).to(device)
        D_r = RobotDecoder_D_r(n_joints=J, shared_dim=shared_dim).to(device)

        robots_ckpt = ckpt.get("robots", {})
        if robot_name in robots_ckpt:
            E_r.load_state_dict(robots_ckpt[robot_name]["E_r"])
            D_r.load_state_dict(robots_ckpt[robot_name]["D_r"])
        elif "E_r" in ckpt:
            E_r.load_state_dict(ckpt["E_r"])
            D_r.load_state_dict(ckpt["D_r"])
        else:
            print(f"  [{robot_name}] no weights in checkpoint — skip")
            continue

        print(f"Running eval  ({args.n_batches} batches × {args.b_eval}) ...")
        m = _compute_eval_metrics(
            test_sampler, args.n_batches, args.b_eval,
            E_h, E_r, E_X, D_X, D_r,
            hand_config_path, zero_wrj,
            rot_repr=rot_repr,
            human_encoder=h_encoder,
        )
        if m is None:
            print(f"  [{robot_name}] eval returned None")
            continue

        results[robot_name] = m
        print(f"\n=== {robot_name} (J={J}) ===")
        print(f"  RS  = {m['rs']:.4f}   (lower = better; 0 = perfect angular match)")
        print(f"  NDS = {m['nds']:.4f}   (lower = better; 0 = perfect tip/chain position match)")
        print(f"  NVS = {m['nvs']:.4f}   (lower = better; 0 = perfect velocity match)")
        print(f"  rec = {m['rec']:.4f}   (robot AE reconstruction error)")
        print(f"  batches={m['n_batches']}  b_eval={m['b_eval']}")
        print()

    if len(results) > 1:
        print("=== Summary ===")
        print(f"{'robot':<12} {'RS':>8} {'NDS':>8} {'NVS':>8} {'rec':>8}")
        for name, m in results.items():
            print(f"{name:<12} {m['rs']:>8.4f} {m['nds']:>8.4f} {m['nvs']:>8.4f} {m['rec']:>8.4f}")


if __name__ == "__main__":
    main()
