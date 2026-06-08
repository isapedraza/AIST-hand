#!/usr/bin/env python3
"""Cheap (no-train) test of the user's two ideas:
  (1) let D_R drive (we isolate D_R here, w_joints=0),
  (2) drop D_R's internal per-joint 1/sigma weights (uniform instead).

Metric = closure margin = D_R(human-closed, robot-OPEN) - D_R(human-closed, robot-CLOSED).
  > 0  -> oracle correctly rates a human fist closer to a robot fist.
  larger -> stronger closure signal.
Compare WEIGHTED (Run 20 1/sigma) vs UNIFORM. Also per-joint (mcp/pip/dip).
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
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
SK_W_DR = {  # [mcp,pip,dip] (tip dropped: robot has no tip joint)
    "thumb":[0.258,0.544,0.199], "index":[0.329,0.325,0.346],
    "middle":[0.188,0.362,0.451], "ring":[0.238,0.357,0.405], "pinky":[0.197,0.405,0.398]}

torch.manual_seed(0); np.random.seed(0)
d = np.load(f"{DATA}/valid_robot_poses_eigengrasp_dong.npz", mmap_mode="r")
mcpq = np.asarray(d["q"][:, [3,7,11,16]])
rc = np.random.choice(np.where((mcpq>1.2).all(1))[0], N)
ro = np.random.choice(np.where((mcpq<0.2).all(1))[0], N)
qr_c = torch.from_numpy(np.asarray(d["quats"][rc])).float()   # [N,15,4]
qr_o = torch.from_numpy(np.asarray(d["quats"][ro])).float()
hl = StaticHumanAnchorLoader(f"{DATA}/hagrid_dong.csv")
def hs(c):
    i = hl._class_indices[c]; i = i[torch.randint(0,len(i),(N,))]; return hl._quats[i]
qh_c, qh_o = hs(29), hs(28)
def hq(q,f): return q[:, 4*f:4*f+3, :]     # human mcp,pip,dip
def rq(q,f): return q[:, 3*f:3*f+3, :]     # robot mcp,pip,dip

def D_R(qh, qr, weights):  # weights: dict sub->[3] or "uniform"
    tot = 0.0
    for f, sub in enumerate(FINGERS):
        dot = (hq(qh,f) * rq(qr,f)).sum(-1)        # [N,3]
        d2 = 1 - dot**2
        if weights == "uniform":
            w = torch.ones(3)/3
        else:
            w = torch.tensor(weights[sub]); w = w/w.sum()
        tot = tot + (w * d2).sum(-1)
    return tot.mean().item()

def D_R_joint(qh, qr, jslot):  # single joint slot 0=mcp,1=pip,2=dip, summed over fingers
    tot = 0.0
    for f in range(5):
        dot = (hq(qh,f)[:,jslot,:] * rq(qr,f)[:,jslot,:]).sum(-1)
        tot = tot + (1 - dot**2)
    return tot.mean().item()

print("="*70)
print("D_R closure signal (no training). Margin = D_R(h-closed,r-OPEN) - D_R(h-closed,r-CLOSED)")
print("  >0 correct; larger = stronger closure signal.\n")
for tag, w in [("WEIGHTED (Run20 1/sigma)", SK_W_DR), ("UNIFORM (no internal weights)", "uniform")]:
    cc = D_R(qh_c, qr_c, w); co = D_R(qh_c, qr_o, w)
    oc = D_R(qh_o, qr_c, w); oo = D_R(qh_o, qr_o, w)
    print(f"{tag}")
    print(f"   h-closed: vs r-closed={cc:.4f}  vs r-open={co:.4f}   CLOSURE MARGIN={co-cc:+.4f}")
    print(f"   h-open:   vs r-closed={oc:.4f}  vs r-open={oo:.4f}   OPEN MARGIN  ={oc-oo:+.4f}")
    print()

print("-"*70)
print("Per-joint (uniform, single joint, summed over 5 fingers): where is the closure signal?")
for jslot, nm in [(0,"MCP"),(1,"PIP"),(2,"DIP")]:
    cc = D_R_joint(qh_c, qr_c, jslot); co = D_R_joint(qh_c, qr_o, jslot)
    print(f"   {nm}: h-closed vs r-closed={cc:.4f}  vs r-open={co:.4f}   margin={co-cc:+.4f}")
