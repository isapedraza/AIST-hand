#!/usr/bin/env python3
"""Evaluate UDHM cross-embodiment coherence.

Tests that open/close poses map to sensible UDHM values, and that
the same semantic state (open/close) is similar across embodiments.

Checks:
1. Per-robot: close → flex slots > open flex slots
2. Cross-embodiment: human close ≈ robot close (UDHM L1), human open ≈ robot open
3. Sanity: all values in [-1, 1], no NaN/inf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "latent-retargeting" / "src"))

from cross_emb.loaders.robot_loader import RobotLoader, _load_hand_config, _dong_run_stage2
from cross_emb.loaders.robot_primitives import build_primitives, robot_to_udhm
from cross_emb.loaders.human_loader import HumanLoader
from cross_emb.loaders.human_to_udhm import human_to_udhm
from cross_emb.loaders.udhm_stage3 import UDHM22_SLOTS

FLEX_SLOTS = [i for i, s in enumerate(UDHM22_SLOTS) if "flex" in s or "ip" in s]
ABD_SLOTS  = [i for i, s in enumerate(UDHM22_SLOTS) if "abd" in s or "spread" in s]

N_SAMPLES = 200


def load_robot_udhm(robot_name: str, urdf: str, config: str,
                    open_npz: str, close_npz: str, qkey: str) -> tuple:
    """Returns (udhm_open [N,22], udhm_close [N,22], tabla)."""
    urdf_path   = REPO_ROOT / urdf
    config_path = REPO_ROOT / config

    loader = RobotLoader(str(urdf_path))
    tabla  = build_primitives(loader, config_path)

    def load_q(path, key) -> torch.Tensor:
        d = np.load(REPO_ROOT / path)
        q = torch.from_numpy(d[key].astype(np.float32))
        # Shadow NPZ has 24-col (includes 2 wrist joints at front); loader chain = 24
        idx = torch.randperm(len(q))[:N_SAMPLES]
        return q[idx]

    q_open  = load_q(open_npz,  qkey)
    q_close = load_q(close_npz, qkey)

    # Shadow qpos24 includes wrist (cols 0-1) which are in chain; others are 22-col only.
    # Pad 22-col to 24-col if needed.
    J = len(loader.chain_joint_names)
    if q_open.shape[1] < J:
        pad = torch.zeros(q_open.shape[0], J - q_open.shape[1])
        q_open  = torch.cat([pad, q_open],  dim=1)
        q_close = torch.cat([pad, q_close], dim=1)

    with torch.no_grad():
        udhm_open  = robot_to_udhm(q_open,  tabla)
        udhm_close = robot_to_udhm(q_close, tabla)

    print(f"\n{'─'*50}")
    print(f"Robot: {robot_name}  (J={J}, primitives={list(tabla.keys())})")
    _check_sanity(udhm_open,  f"{robot_name} open")
    _check_sanity(udhm_close, f"{robot_name} close")
    _check_flex_direction(udhm_open, udhm_close, robot_name)
    return udhm_open, udhm_close, tabla


def load_human_udhm(csv_path: str) -> tuple:
    """Returns (udhm_open [N,22], udhm_close [N,22]) from hagrid CSV."""
    from cross_emb.loaders.human_loader import DONG_LABELS, _pose_cols_and_dim

    pose_cols, pose_dim = _pose_cols_and_dim("quat")
    df = pd.read_csv(REPO_ROOT / csv_path)

    pose_np = df[pose_cols].values.astype(np.float32)  # [N, 20*pose_dim]
    n_joints = len(DONG_LABELS)
    pose_t = torch.from_numpy(pose_np.reshape(len(df), n_joints, pose_dim))

    closed_idx = np.where(df["anchor_label"] == "closed_fist")[0][:N_SAMPLES]
    open_idx   = np.where(df["anchor_label"] == "open_hand")[0][:N_SAMPLES]

    pose_close = pose_t[closed_idx]
    pose_open  = pose_t[open_idx]

    with torch.no_grad():
        udhm_close = human_to_udhm(pose_close, DONG_LABELS)
        udhm_open  = human_to_udhm(pose_open,  DONG_LABELS)

    print(f"\n{'─'*50}")
    print(f"Human (hagrid): closed_fist n={len(closed_idx)}, open_hand n={len(open_idx)}")
    _check_sanity(udhm_open,  "human open")
    _check_sanity(udhm_close, "human close")
    _check_flex_direction(udhm_open, udhm_close, "human")
    return udhm_open, udhm_close


def _check_sanity(udhm: torch.Tensor, label: str):
    if torch.isnan(udhm).any() or torch.isinf(udhm).any():
        print(f"  [FAIL] {label}: NaN/Inf found")
        return
    vmin, vmax = udhm.min().item(), udhm.max().item()
    out_range = ((udhm.abs() > 1.0).sum().item())
    status = "OK" if out_range == 0 else f"WARN {out_range} vals outside [-1,1]"
    print(f"  sanity {label}: range=[{vmin:.3f}, {vmax:.3f}]  {status}")


def _check_flex_direction(udhm_open: torch.Tensor, udhm_close: torch.Tensor, name: str):
    flex_open  = udhm_open[:,  FLEX_SLOTS].mean().item()
    flex_close = udhm_close[:, FLEX_SLOTS].mean().item()
    ok = flex_close > flex_open
    sym = "✓" if ok else "✗"
    print(f"  flex direction {name}: open={flex_open:.4f}  close={flex_close:.4f}  {sym}")


def _cross_compare(name_a: str, udhm_a_close, udhm_a_open,
                   name_b: str, udhm_b_close, udhm_b_open):
    """Per-slot mean values and L1 diff for close and open states."""
    n_close = min(len(udhm_a_close), len(udhm_b_close))
    n_open  = min(len(udhm_a_open),  len(udhm_b_open))

    # Per-slot means
    a_close_mu = udhm_a_close[:n_close].mean(dim=0)   # [22]
    b_close_mu = udhm_b_close[:n_close].mean(dim=0)
    a_open_mu  = udhm_a_open[:n_open].mean(dim=0)
    b_open_mu  = udhm_b_open[:n_open].mean(dim=0)

    close_diff = (a_close_mu - b_close_mu).abs()
    open_diff  = (a_open_mu  - b_open_mu).abs()

    print(f"\n  ── {name_a} ↔ {name_b} per-slot ──")
    print(f"  {'slot':<22} {'A_close':>8} {'B_close':>8} {'|diff|':>8} │ {'A_open':>8} {'B_open':>8} {'|diff|':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} ┼ {'─'*8} {'─'*8} {'─'*8}")
    for i, slot in enumerate(UDHM22_SLOTS):
        ac = a_close_mu[i].item()
        bc = b_close_mu[i].item()
        ao = a_open_mu[i].item()
        bo = b_open_mu[i].item()
        dc = close_diff[i].item()
        do_ = open_diff[i].item()
        # skip always-zero slots
        if abs(ac) < 1e-4 and abs(bc) < 1e-4 and abs(ao) < 1e-4 and abs(bo) < 1e-4:
            continue
        flag = " !" if dc > 0.2 else ""
        print(f"  {slot:<22} {ac:>8.3f} {bc:>8.3f} {dc:>8.3f}{flag} │ {ao:>8.3f} {bo:>8.3f} {do_:>8.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("UDHM Cross-Embodiment Evaluation")
    print("=" * 60)

    robots = [
        ("shadow", "robot/hands/shadow_hand/shadow_hand_right.urdf",
         "robot/hand-configs/shadow.yaml",
         "robot/hands/shadow_hand/datasets/processed/synthetic_open_hand_shadow_qpos.npz",
         "robot/hands/shadow_hand/datasets/processed/synthetic_close_hand_shadow_qpos.npz",
         "qpos24"),
        ("leap", "robot/hands/leap_hand/leap_hand_right.urdf",
         "robot/hand-configs/leap.yaml",
         "robot/hands/leap_hand/datasets/processed/synthetic_open_leap.npz",
         "robot/hands/leap_hand/datasets/processed/synthetic_close_leap.npz",
         "qpos16"),
        ("allegro", "robot/hands/allegro_hand/allegro_hand_right.urdf",
         "robot/hand-configs/allegro.yaml",
         "robot/hands/allegro_hand/datasets/processed/synthetic_open_allegro.npz",
         None,  # no close — use valid poses
         "qpos16"),
        ("barrett", "robot/hands/barrett_hand/bhand_model.urdf",
         "robot/hand-configs/barrett.yaml",
         "robot/hands/barrett_hand/datasets/processed/synthetic_open_barrett.npz",
         "robot/hands/barrett_hand/datasets/processed/synthetic_close_barrett.npz",
         "qpos8"),
        ("inspire", "robot/hands/inspire_hand/inspire_hand_right.urdf",
         "robot/hand-configs/inspire.yaml",
         "robot/hands/inspire_hand/datasets/processed/synthetic_open_inspire.npz",
         "robot/hands/inspire_hand/datasets/processed/synthetic_close_inspire.npz",
         "qpos12"),
    ]

    results = {}
    for robot_name, urdf, config, open_npz, close_npz, qkey in robots:
        if close_npz is None:
            # Allegro: use valid poses as "close" proxy
            close_npz_path = "robot/hands/allegro_hand/datasets/processed/valid_robot_poses_allegro.npz"
            d = np.load(REPO_ROOT / close_npz_path)
            close_qkey = "q" if "q" in d.files else [k for k in d.files if "qpos" in k][0]
            close_npz = close_npz_path
            qkey_close = close_qkey
        else:
            qkey_close = qkey

        # Temporarily patch load_robot_udhm for allegro's different close key
        urdf_path   = REPO_ROOT / urdf
        config_path = REPO_ROOT / config
        if not urdf_path.exists():
            print(f"\nSkipping {robot_name}: URDF not found")
            continue

        loader = RobotLoader(str(urdf_path))
        tabla  = build_primitives(loader, config_path)

        def load_q(path, key, J):
            d = np.load(REPO_ROOT / path)
            q = torch.from_numpy(d[key].astype(np.float32))
            idx = torch.randperm(len(q))[:N_SAMPLES]
            q = q[idx]
            if q.shape[1] < J:
                pad = torch.zeros(q.shape[0], J - q.shape[1])
                q = torch.cat([pad, q], dim=1)
            return q

        J = len(loader.chain_joint_names)
        q_open  = load_q(open_npz,  qkey,       J)
        q_close = load_q(close_npz, qkey_close, J)

        with torch.no_grad():
            udhm_open  = robot_to_udhm(q_open,  tabla)
            udhm_close = robot_to_udhm(q_close, tabla)

        print(f"\n{'─'*50}")
        print(f"Robot: {robot_name}  (J={J}, primitives={list(tabla.keys())})")
        _check_sanity(udhm_open,  f"{robot_name} open")
        _check_sanity(udhm_close, f"{robot_name} close")
        _check_flex_direction(udhm_open, udhm_close, robot_name)
        results[robot_name] = (udhm_open, udhm_close)

    # Human
    human_open, human_close = load_human_udhm("human/datasets/hagrid_dong.csv")
    results["human"] = (human_open, human_close)

    # Cross-embodiment comparisons
    print(f"\n{'='*60}")
    print("Cross-embodiment (human ↔ robot):")
    for robot_name, (r_open, r_close) in results.items():
        if robot_name == "human":
            continue
        _cross_compare("human", human_close, human_open,
                       robot_name, r_close, r_open)

    print(f"\n{'='*60}\n")
