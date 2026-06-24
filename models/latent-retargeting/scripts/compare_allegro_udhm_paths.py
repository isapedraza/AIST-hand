#!/usr/bin/env python3
"""UDHM open vs close comparison across 3 paths.

For the contrastive to work, open→close must produce the same directional
change in UDHM across all embodiments (flex up, abd down or neutral).

Paths tested:
  1. Shadow   stage3       -- reference (known good)
  2. Allegro  stage3       -- Fix C hypothesis
  3. Allegro  qpos-direct  -- current bug

Run from AIST-hand/:
    python models/latent-retargeting/scripts/compare_allegro_udhm_paths.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO / "models" / "latent-retargeting" / "src"))

from cross_emb.loaders.udhm_stage3 import udhm_run_stage3, UDHM22_SLOTS
from cross_emb.loaders.robot_loader import RobotLoader
from cross_emb.loaders.robot_primitives import build_primitives, robot_to_udhm

N = 500

SHADOW_CLOSE = REPO / "robot/hands/shadow_hand/datasets/processed/synthetic_close_hand_shadow_qpos.npz"
SHADOW_OPEN  = REPO / "robot/hands/shadow_hand/datasets/processed/synthetic_open_hand_shadow_qpos.npz"
SHADOW_URDF  = REPO / "robot/hands/shadow_hand/shadow_hand_right.urdf"
SHADOW_CFG   = REPO / "robot/hand-configs/shadow.yaml"

ALLEGRO_CLOSE = REPO / "robot/hands/allegro_hand/datasets/processed/synthetic_close_allegro.npz"
ALLEGRO_OPEN  = REPO / "robot/hands/allegro_hand/datasets/processed/synthetic_open_allegro.npz"
ALLEGRO_URDF  = REPO / "robot/hands/allegro_hand/allegro_hand_right.urdf"
ALLEGRO_CFG   = REPO / "robot/hand-configs/allegro.yaml"


def qpos_to_udhm_stage3(loader: RobotLoader, cfg: Path, q: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fk = loader.run_fk(q)
        quats, labels, _ = loader.run_dong_stage2(fk, cfg)
        return udhm_run_stage3(quats, labels)


def qpos_to_udhm_qposdirect(q: torch.Tensor, table: dict) -> torch.Tensor:
    with torch.no_grad():
        return robot_to_udhm(q, table)


def print_open_close(name: str, udhm_open: torch.Tensor, udhm_close: torch.Tensor) -> None:
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"  {'slot':<28} {'open mean':>10} {'close mean':>10} {'delta':>10}  dir")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10}  ─────")
    for i, slot in enumerate(UDHM22_SLOTS):
        o = udhm_open[:, i].mean().item()
        c = udhm_close[:, i].mean().item()
        delta = c - o
        if abs(o) < 1e-4 and abs(c) < 1e-4:
            direction = "DEAD"
        elif abs(delta) < 1e-3:
            direction = "flat"
        elif delta > 0:
            direction = "close↑"
        else:
            direction = "close↓"
        print(f"  {slot:<28} {o:>+10.4f} {c:>+10.4f} {delta:>+10.4f}  {direction}")


def main() -> None:
    for p in [SHADOW_CLOSE, SHADOW_OPEN, SHADOW_URDF, SHADOW_CFG,
              ALLEGRO_CLOSE, ALLEGRO_OPEN, ALLEGRO_URDF, ALLEGRO_CFG]:
        if not p.exists():
            sys.exit(f"NOT FOUND: {p}")

    # --- Shadow ---
    print("Loading Shadow loader...")
    shadow_loader = RobotLoader(SHADOW_URDF)

    d_sc = np.load(SHADOW_CLOSE, allow_pickle=True)
    d_so = np.load(SHADOW_OPEN,  allow_pickle=True)
    q_sc = torch.from_numpy(d_sc["qpos24"][:N]).float()
    q_so = torch.from_numpy(d_so["qpos24"][:N]).float()

    udhm_sc = qpos_to_udhm_stage3(shadow_loader, SHADOW_CFG, q_sc)
    udhm_so = qpos_to_udhm_stage3(shadow_loader, SHADOW_CFG, q_so)

    # --- Allegro ---
    print("Loading Allegro loader + primitives table...")
    allegro_loader = RobotLoader(ALLEGRO_URDF)
    table = build_primitives(allegro_loader, ALLEGRO_CFG)
    print(f"  roles: { {f: list(r.keys()) for f, r in table.items()} }")

    d_ac = np.load(ALLEGRO_CLOSE, allow_pickle=True)
    d_ao = np.load(ALLEGRO_OPEN,  allow_pickle=True)
    q_ac = torch.from_numpy(d_ac["qpos16"]).float()          # only 5 close poses
    q_ao = torch.from_numpy(d_ao["qpos16"][:N]).float()

    print(f"  Allegro close poses: {len(q_ac)}  open poses: {len(q_ao)}")

    udhm_ac_s3 = qpos_to_udhm_stage3(allegro_loader, ALLEGRO_CFG, q_ac)
    udhm_ao_s3 = qpos_to_udhm_stage3(allegro_loader, ALLEGRO_CFG, q_ao)

    udhm_ac_qp = qpos_to_udhm_qposdirect(q_ac, table)
    udhm_ao_qp = qpos_to_udhm_qposdirect(q_ao, table)

    # --- Print ---
    print_open_close("Shadow   stage3  [REFERENCE]", udhm_so, udhm_sc)
    print_open_close("Allegro  stage3  [Fix C]",     udhm_ao_s3, udhm_ac_s3)
    print_open_close("Allegro  qpos-direct  [BUG]",  udhm_ao_qp, udhm_ac_qp)


if __name__ == "__main__":
    main()

