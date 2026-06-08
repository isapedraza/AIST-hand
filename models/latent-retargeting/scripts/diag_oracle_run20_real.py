#!/usr/bin/env python3
"""FAITHFUL replica of Run 20's contrastive oracle (commit 847459f28).

S_k = 1.0*D_R + 1.2*D_joints + 0.07*D_ahg, per finger subspace, where
  D_R      = sum_j w_dr_j  * (1 - <q_a,q_b>^2)   over common joints mcp,pip,dip
  D_joints = sum   w_jt    * ||chain_a - chain_b||  over mcp,pip,dip,tip
  D_ahg    = angle term to base+tip critical joints
with the exact offline 1/sigma weights Run 20 used.

Question (settled here, against the REAL oracle): does Run 20's S_k rate a
human fist closer to a robot fist than to a robot open hand?
"""
import sys
from pathlib import Path
import numpy as np
import torch

SRC = Path("/home/yareeez/AIST-hand/models/latent-retargeting/src")
sys.path.insert(0, str(SRC))
from cross_emb.loaders.human_loader import StaticHumanAnchorLoader

DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
N = 1000
W_R, W_JOINTS, W_AHG = 1.0, 1.2, 0.07
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# Exact Run-20 offline weights (order [mcp,pip,dip,tip]).
SK_W_DR = {
    "thumb":[0.258,0.544,0.199,0.0], "index":[0.329,0.325,0.346,0.0],
    "middle":[0.188,0.362,0.451,0.0], "ring":[0.238,0.357,0.405,0.0],
    "pinky":[0.197,0.405,0.398,0.0]}
SK_W_JT = {
    "thumb":[0.4499,0.2534,0.1484,0.1484], "index":[0.5282,0.2435,0.1381,0.0902],
    "middle":[0.5630,0.2259,0.1267,0.0844], "ring":[0.5743,0.2364,0.1134,0.0759],
    "pinky":[0.5459,0.2465,0.1241,0.0835]}

torch.manual_seed(0); np.random.seed(0)

# ---- ROBOT closed/open from cache ----
d = np.load(f"{DATA}/valid_robot_poses_eigengrasp_dong.npz", mmap_mode="r")
mcpq = np.asarray(d["q"][:, [3,7,11,16]])
closed = np.where((mcpq > 1.2).all(1))[0]
openi  = np.where((mcpq < 0.2).all(1))[0]
rc = np.random.choice(closed, N, replace=len(closed)<N)
ro = np.random.choice(openi,  N, replace=len(openi)<N)
# robot quats [15] -> reshape per finger 3 joints (mcp,pip,dip); chain [5,4,3]
qr_c = torch.from_numpy(np.asarray(d["quats"][rc])).float()   # [N,15,4]
qr_o = torch.from_numpy(np.asarray(d["quats"][ro])).float()
chr_c = torch.from_numpy(np.asarray(d["chain"][rc])).float()  # [N,5,4,3]
chr_o = torch.from_numpy(np.asarray(d["chain"][ro])).float()

# ---- HUMAN closed/open ----
hl = StaticHumanAnchorLoader(f"{DATA}/hagrid_dong.csv")
def hsamp(cls):
    idx = hl._class_indices[cls]; idx = idx[torch.randint(0,len(idx),(N,))]
    return hl._quats[idx], hl._chain[idx]   # quats[N,20,4], chain[N,5,4,3]
qh_c, chh_c = hsamp(29)   # closed_fist
qh_o, chh_o = hsamp(28)   # open_hand
# human quats[20] per finger order mcp,pip,dip,tip -> take mcp,pip,dip = offsets 0,1,2
def h_finger_q(q, f): return q[:, 4*f:4*f+3, :]      # [N,3,4]
def r_finger_q(q, f): return q[:, 3*f:3*f+3, :]      # [N,3,4]

def d_r_term(qa, qb, sub):
    w = torch.tensor(SK_W_DR[sub][:3]); w = w / w.sum()
    dot = (qa * qb).sum(-1)                 # [N,3]
    return (w * (1 - dot**2)).sum(-1)       # [N]

def d_joints_term(ca, cb, sub):
    w = torch.tensor(SK_W_JT[sub])          # [4]
    return (w * (ca - cb).norm(dim=-1)).sum(-1)   # ca,cb [N,4,3] -> [N]

def d_ahg_term(ca, cb):
    # ca,cb: [N,4,3] single finger. critical = base(0)+tip(3) -> 2 pts.
    j = ca; crit = ca[:, [0,3], :]
    uj = j / j.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    uc = crit / crit.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    a1 = torch.acos(torch.bmm(uj, uc.transpose(1,2)).clamp(-1+1e-6,1-1e-6))
    j2 = cb; crit2 = cb[:, [0,3], :]
    uj2 = j2 / j2.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    uc2 = crit2 / crit2.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    a2 = torch.acos(torch.bmm(uj2, uc2.transpose(1,2)).clamp(-1+1e-6,1-1e-6))
    return (a1 - a2).abs().sum(dim=(-2,-1))

def S_k(qh, chh, qr, chr_):
    DR = DJ = DA = 0.0
    for f, sub in enumerate(FINGERS):
        DR = DR + d_r_term(h_finger_q(qh,f), r_finger_q(qr,f), sub)
        DJ = DJ + d_joints_term(chh[:,f], chr_[:,f], sub)
        DA = DA + d_ahg_term(chh[:,f], chr_[:,f])
    S = W_R*DR + W_JOINTS*DJ + W_AHG*DA
    return S.mean().item(), (W_R*DR).mean().item(), (W_JOINTS*DJ).mean().item(), (W_AHG*DA).mean().item()

print("="*78)
print("RUN 20 REAL ORACLE S_k = 1.0*D_R + 1.2*D_joints + 0.07*D_ahg  (lower=more similar)")
print(f"{'pair':30} | {'S_k':>9} | {'1.0*D_R':>9} | {'1.2*Djt':>9} | {'0.07*Dahg':>10}")
for name,(qh,chh),(qr,chr_) in [
    ("human-CLOSED vs robot-CLOSED",(qh_c,chh_c),(qr_c,chr_c)),
    ("human-CLOSED vs robot-OPEN",  (qh_c,chh_c),(qr_o,chr_o)),
    ("human-OPEN   vs robot-CLOSED",(qh_o,chh_o),(qr_c,chr_c)),
    ("human-OPEN   vs robot-OPEN",  (qh_o,chh_o),(qr_o,chr_o)),
]:
    s,dr,dj,da = S_k(qh,chh,qr,chr_)
    print(f"{name:30} | {s:9.3f} | {dr:9.3f} | {dj:9.3f} | {da:10.3f}")
print("-"*78)
print("VERDICT: if 'h-CLOSED vs r-CLOSED' S_k < 'h-CLOSED vs r-OPEN' S_k -> oracle correct.")
print("Also watch which COMPONENT (D_R vs D_joints) drives / inverts it.")
