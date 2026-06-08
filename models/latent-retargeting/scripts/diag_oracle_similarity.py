#!/usr/bin/env python3
"""Does the contrastive ORACLE rate human-fist closer to robot-fist than to
robot-open?  If yes -> oracle fine, gap is margin/cone.  If no -> oracle is the
root cause (it pairs human-closed with robot-open -> contrastive pushes the
closed hand toward the open pose).

Lower S = more similar.
"""
import sys
from pathlib import Path
import numpy as np
import torch

SRC = Path("/home/yareeez/AIST-hand/models/latent-retargeting/src")
sys.path.insert(0, str(SRC))
from cross_emb.training.losses import xin_sk_full, d_r_yan
from cross_emb.loaders.human_loader import StaticHumanAnchorLoader

DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
N = 1000
torch.manual_seed(0); np.random.seed(0)

# ---- ROBOT: filter cache into closed / open ----
d = np.load(f"{DATA}/valid_robot_poses_eigengrasp_dong.npz", mmap_mode="r")
mcp = np.asarray(d["q"][:, [3, 7, 11, 16]])           # FFJ3,MFJ3,RFJ3,LFJ3
closed = np.where((mcp > 1.2).all(1))[0]
openi  = np.where((mcp < 0.2).all(1))[0]
print(f"robot poses: closed={len(closed)}  open={len(openi)}")
rc = np.random.choice(closed, N, replace=len(closed) < N)
ro = np.random.choice(openi,  N, replace=len(openi) < N)

# robot tips/chain (5 fingers) and quats (15 joints)
def grab(idx):
    return (torch.from_numpy(np.asarray(d["tips"][idx])).float(),
            torch.from_numpy(np.asarray(d["chain"][idx])).float(),
            torch.from_numpy(np.asarray(d["quats"][idx])).float())
tips_rc, chain_rc, q_rc = grab(rc)
tips_ro, chain_ro, q_ro = grab(ro)

# ---- HUMAN: closed_fist from hagrid (grasp_type 29) ----
hl = StaticHumanAnchorLoader(f"{DATA}/hagrid_dong.csv")
ci = hl._class_indices[29]                # closed_fist
oi = hl._class_indices[28]                # open_hand
def hgrab(idx_t):
    idx = idx_t[torch.randint(0, len(idx_t), (N,))]
    return hl._tips[idx], hl._chain[idx], hl._quats[idx]
tips_hc, chain_hc, q_hc = hgrab(ci)
tips_ho, chain_ho, q_ho = hgrab(oi)

# human quats[20] -> common 15 joints (drop tips) to match robot quats[15]
H2R = [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18]
q_hc15, q_ho15 = q_hc[:, H2R, :], q_ho[:, H2R, :]

def report(name, ta, ca, qa, tb, cb, qb):
    s = xin_sk_full(ta, tb, ca, cb).mean().item()
    dr = d_r_yan(qa, qb).mean().item()
    print(f"  {name:34} Xin_S_k = {s:8.3f}   D_R = {dr:7.3f}")

print("=" * 64)
print("ORACLE on HUMAN-CLOSED fist vs robot targets (lower = more similar)")
report("human-CLOSED vs robot-CLOSED", tips_hc, chain_hc, q_hc15, tips_rc, chain_rc, q_rc)
report("human-CLOSED vs robot-OPEN",   tips_hc, chain_hc, q_hc15, tips_ro, chain_ro, q_ro)
print("-" * 64)
print("Control: HUMAN-OPEN vs robot targets")
report("human-OPEN   vs robot-CLOSED", tips_ho, chain_ho, q_ho15, tips_rc, chain_rc, q_rc)
report("human-OPEN   vs robot-OPEN",   tips_ho, chain_ho, q_ho15, tips_ro, chain_ro, q_ro)
print("-" * 64)
print("Sanity (within robot): should say closed~closed similar, closed~open far")
report("robot-CLOSED vs robot-CLOSED", tips_rc, chain_rc, q_rc, tips_rc[torch.randperm(N)], chain_rc[torch.randperm(N)], q_rc[torch.randperm(N)])
report("robot-CLOSED vs robot-OPEN",   tips_rc, chain_rc, q_rc, tips_ro, chain_ro, q_ro)
print("=" * 64)
print("VERDICT: if 'human-CLOSED vs robot-CLOSED' < 'human-CLOSED vs robot-OPEN'")
print("  -> oracle CORRECT (gap is margin/cone). Else -> ORACLE is root cause.")
