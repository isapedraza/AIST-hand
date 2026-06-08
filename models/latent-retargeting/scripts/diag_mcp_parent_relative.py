#!/usr/bin/env python3
"""Settle it: is the human-vs-robot MCP mismatch caused by the FRAME?

Current Dong MCP = segment orientation in WRIST frame (absolute) -> human 40, robot 135 (mismatch).
Test: recompute MCP as a PARENT-RELATIVE flexion angle = angle between
  metacarpal vector (wrist->MCP) and proximal-phalanx vector (MCP->PIP),
i.e. the same KIND of bend angle the PIP uses. Angles are scale-invariant.

If parent-relative MCP converges (human ~ robot) like PIP did -> frame was the cause (claim TRUE).
If it stays 40 vs 135 -> claim FALSE, drop it.
"""
import numpy as np, torch

DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
import sys; sys.path.insert(0, "/home/yareeez/AIST-hand/models/latent-retargeting/src")
from cross_emb.loaders.human_loader import StaticHumanAnchorLoader
N = 1000
np.random.seed(0); torch.manual_seed(0)

# robot chain from cache (closed/open by q-MCP). chain [.,5,4,3] = mcp,pip,dip,tip, wrist=origin.
d = np.load(f"{DATA}/valid_robot_poses_eigengrasp_dong.npz", mmap_mode="r")
mcpq = np.asarray(d["q"][:, [3,7,11,16]])
rc = np.random.choice(np.where((mcpq>1.2).all(1))[0], N)
ro = np.random.choice(np.where((mcpq<0.2).all(1))[0], N)
chr_c = torch.from_numpy(np.asarray(d["chain"][rc])).float()
chr_o = torch.from_numpy(np.asarray(d["chain"][ro])).float()

hl = StaticHumanAnchorLoader(f"{DATA}/hagrid_dong.csv")
def hch(c):
    i = hl._class_indices[c]; i = i[torch.randint(0,len(i),(N,))]; return hl._chain[i]
chh_c, chh_o = hch(29), hch(28)

def angle_between(u, v):  # [...,3] -> deg
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.rad2deg(torch.acos((u*v).sum(-1).clamp(-1+1e-6, 1-1e-6)))

def parent_rel(chain, f):
    # chain[:,f] = [mcp,pip,dip,tip] positions, wrist=origin
    mcp = chain[:, f, 0]; pip = chain[:, f, 1]; dip = chain[:, f, 2]
    mcp_flex = angle_between(mcp - 0.0, pip - mcp)     # metacarpal(wrist->mcp) vs phalanx(mcp->pip)
    pip_flex = angle_between(pip - mcp, dip - pip)     # phalanx vs middle (the comparable one)
    return mcp_flex.mean().item(), pip_flex.mean().item()

print("="*72)
print("PARENT-RELATIVE flexion angle (deg). MCP = bend wrist->mcp vs mcp->pip.")
print(f"{'finger':8} | {'MCP robot-clo':>13} {'MCP human-clo':>13} | {'PIP robot-clo':>13} {'PIP human-clo':>13}")
for f, nm in [(1,"INDEX"),(2,"MIDDLE")]:
    rmcp, rpip = parent_rel(chr_c, f)
    hmcp, hpip = parent_rel(chh_c, f)
    print(f"{nm:8} | {rmcp:13.1f} {hmcp:13.1f} | {rpip:13.1f} {hpip:13.1f}")
print("-"*72)
print("For reference, the CURRENT (wrist-frame) Dong MCP gave robot~135 vs human~40.")
print("If parent-relative MCP robot ~ human now -> the frame WAS the cause.")
