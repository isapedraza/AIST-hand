#!/usr/bin/env python3
"""Verify UDHM stage-3 on LEAP (Tests #1-3, same protocol as shadow in ESTADO-2026-06-16).

Test #1: open vs close (synthetic) -> flexion delta, sane sign/magnitude.
Test #2: isolate abduction -- excite ONLY the index abduction joint (if_rot,
         chain_joint_names index 1), check ONLY index_mcp_abd moves, zero leak
         into flexion slots.
Test #3: cross-embodiment match -- human (hagrid_dong_r6.csv, open_hand/closed_fist
         labels) vs LEAP synthetic open/close, distance matrix sanity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models/latent-retargeting/src"))

from cross_emb.loaders.robot_loader import RobotLoader  # noqa: E402
from cross_emb.loaders.dong_math import dong_run_stage2  # noqa: E402
from cross_emb.loaders.udhm_stage3 import udhm_run_stage3, UDHM22_SLOTS  # noqa: E402
from cross_emb.loaders.human_loader import DONG_LABELS, _R6_COLS  # noqa: E402

LEAP_URDF = REPO_ROOT / "robot/hands/leap_hand/leap_hand_right.urdf"
LEAP_CFG  = REPO_ROOT / "robot/hand-configs/leap.yaml"
PROC = REPO_ROOT / "robot/hands/leap_hand/datasets/processed"
HAGRID_CSV = REPO_ROOT / "human/datasets/hagrid_dong_r6.csv"

SLOT_IDX = {n: i for i, n in enumerate(UDHM22_SLOTS)}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def leap_udhm_from_qpos(loader: RobotLoader, config: dict, qpos16: torch.Tensor) -> torch.Tensor:
    fk_out = loader.run_fk(qpos16)
    quats, labels, _meta = dong_run_stage2(fk_out, config)
    return udhm_run_stage3(quats, labels)


def main() -> None:
    config = load_config(LEAP_CFG)
    loader = RobotLoader(urdf_path=LEAP_URDF, device="cpu")
    print(f"chain_joint_names ({len(loader.chain_joint_names)}): {loader.chain_joint_names}")

    # ------------------------------------------------------------------
    print("\n=== TEST #1: open vs close (synthetic LEAP) ===")
    d_open  = np.load(PROC / "synthetic_open_leap.npz")
    d_close = np.load(PROC / "synthetic_close_leap.npz")
    q_open  = torch.from_numpy(d_open["qpos16"]).float()
    q_close = torch.from_numpy(d_close["qpos16"]).float()

    u_open  = leap_udhm_from_qpos(loader, config, q_open)
    u_close = leap_udhm_from_qpos(loader, config, q_close)

    flex_slots = [n for n in UDHM22_SLOTS if "flex" in n]
    flex_idx = [SLOT_IDX[n] for n in flex_slots]
    mean_open_flex  = u_open[:,  flex_idx].mean().item()
    mean_close_flex = u_close[:, flex_idx].mean().item()
    print(f"mean flex (open)  = {mean_open_flex:+.4f}")
    print(f"mean flex (close) = {mean_close_flex:+.4f}")
    print(f"delta = {mean_close_flex - mean_open_flex:+.4f}  (expect close > open, clearly)")

    # ------------------------------------------------------------------
    print("\n=== TEST #2: isolate abduction (index finger, joint 'if_rot') ===")
    abd_joint_idx = loader.chain_joint_names.index("0")  # if_rot = URDF joint "0" per finger group; verify below
    print(f"chain_joint_names[1] = {loader.chain_joint_names[1]!r} (expect if_rot/abduction per build_leap_eigengrasps order)")

    J = len(loader.chain_joint_names)
    q_base = torch.zeros(1, J)
    deltas = [-0.3, -0.15, 0.0, 0.15, 0.3]
    q_batch = q_base.repeat(len(deltas), 1)
    for i, d in enumerate(deltas):
        q_batch[i, 1] = d  # index 1 = if_rot per JOINTS16 canonical order

    u_batch = leap_udhm_from_qpos(loader, config, q_batch)
    abd_val  = u_batch[:, SLOT_IDX["index_mcp_abd"]]
    flex_val = u_batch[:, SLOT_IDX["index_mcp_flex"]]
    pip_val  = u_batch[:, SLOT_IDX["index_pip_flex"]]
    dip_val  = u_batch[:, SLOT_IDX["index_dip_flex"]]
    print(f"{'delta_q':>8}  {'mcp_abd':>9}  {'mcp_flex':>9}  {'pip_flex':>9}  {'dip_flex':>9}")
    for i, d in enumerate(deltas):
        print(f"{d:8.3f}  {abd_val[i].item():9.5f}  {flex_val[i].item():9.5f}  "
              f"{pip_val[i].item():9.5f}  {dip_val[i].item():9.5f}")
    leak = (flex_val.abs().max() + pip_val.abs().max() + dip_val.abs().max()).item()
    print(f"leak into flexion slots (should be ~0) = {leak:.6f}")
    print(f"abd range (should track delta_q monotonically) = "
          f"{abd_val.min().item():.4f} .. {abd_val.max().item():.4f}")

    # ------------------------------------------------------------------
    print("\n=== TEST #3: cross-embodiment match (human hagrid vs LEAP synthetic) ===")
    df = pd.read_csv(HAGRID_CSV)
    open_mask  = df["anchor_label"] == "open_hand"
    close_mask = df["anchor_label"] == "closed_fist"
    print(f"human rows: open_hand={open_mask.sum()}  closed_fist={close_mask.sum()}")

    def human_udhm(mask: pd.Series, n: int = 500) -> torch.Tensor:
        sub = df[mask].sample(n=min(n, mask.sum()), random_state=0)
        r6 = sub[_R6_COLS].values.astype(np.float32).reshape(-1, 20, 6)
        r6 = torch.from_numpy(r6)
        return udhm_run_stage3(r6, DONG_LABELS)

    h_open  = human_udhm(open_mask)
    h_close = human_udhm(close_mask)

    h_open_mean  = h_open.mean(0)
    h_close_mean = h_close.mean(0)
    r_open_mean  = u_open.mean(0)
    r_close_mean = u_close.mean(0)

    def dist(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).norm().item()

    print(f"d(human_open,  leap_open)  = {dist(h_open_mean, r_open_mean):.4f}")
    print(f"d(human_open,  leap_close) = {dist(h_open_mean, r_close_mean):.4f}  (expect larger)")
    print(f"d(human_close, leap_close) = {dist(h_close_mean, r_close_mean):.4f}")
    print(f"d(human_close, leap_open)  = {dist(h_close_mean, r_open_mean):.4f}  (expect larger)")


if __name__ == "__main__":
    main()
