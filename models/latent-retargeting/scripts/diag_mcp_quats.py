#!/usr/bin/env python3
"""Why is the MCP D_R signal inverted? Print raw mean MCP quaternions and the
rotation angle they encode, for human/robot x closed/open. Distinguishes:
  (1) human MCP carries abduction / measures something else than robot FFJ3
  (2) different quaternion frame/axis convention
  (3) human 'fist' barely flexes the MCP (small angle)
  (4) hemisphere (w sign) handled differently
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
np.random.seed(0); torch.manual_seed(0)

d = np.load(f"{DATA}/valid_robot_poses_eigengrasp_dong.npz", mmap_mode="r")
mcpq = np.asarray(d["q"][:, [3,7,11,16]])
rc = np.random.choice(np.where((mcpq>1.2).all(1))[0], N)
ro = np.random.choice(np.where((mcpq<0.2).all(1))[0], N)
qr_c = torch.from_numpy(np.asarray(d["quats"][rc])).float()
qr_o = torch.from_numpy(np.asarray(d["quats"][ro])).float()
hl = StaticHumanAnchorLoader(f"{DATA}/hagrid_dong.csv")
def hs(c):
    i = hl._class_indices[c]; i = i[torch.randint(0,len(i),(N,))]; return hl._quats[i]
qh_c, qh_o = hs(29), hs(28)

def angle_deg(q):  # q [N,4] wxyz -> mean rotation angle
    w = q[:,0].abs().clamp(0,1)
    return torch.rad2deg(2*torch.acos(w)).mean().item()

def fmt(q):  # mean quaternion
    m = q.mean(0)
    return f"[w={m[0]:+.3f} x={m[1]:+.3f} y={m[2]:+.3f} z={m[3]:+.3f}]"

# index finger MCP: human quats idx 4, robot quats idx 3
for fname, hmcp, rmcp in [("INDEX", 4, 3), ("MIDDLE", 8, 6)]:
    print("="*72)
    print(f"{fname} MCP quaternion (mean over {N})  + rotation angle")
    hc = qh_c[:, hmcp, :]; ho = qh_o[:, hmcp, :]
    rcq = qr_c[:, rmcp, :]; roq = qr_o[:, rmcp, :]
    print(f"  human  CLOSED {fmt(hc)}  angle={angle_deg(hc):5.1f} deg")
    print(f"  human  OPEN   {fmt(ho)}  angle={angle_deg(ho):5.1f} deg")
    print(f"  robot  CLOSED {fmt(rcq)}  angle={angle_deg(rcq):5.1f} deg")
    print(f"  robot  OPEN   {fmt(roq)}  angle={angle_deg(roq):5.1f} deg")
    # dot of mean-aligned per-sample (cross pairs)
    def md(a,b): return (a*b).sum(-1).mean().item()
    print(f"  <human-closed, robot-closed> = {md(hc,rcq):+.3f}   <human-closed, robot-open> = {md(hc,roq):+.3f}")
