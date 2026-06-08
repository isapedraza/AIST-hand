#!/usr/bin/env python3
"""Diagnostic: where does closure die in Run 20 -- encoder or decoder?

Robot-side autoencode test. Feed LABELED closed/open robot poses through the
robot path (E_r -> E_X -> D_X -> D_r) and check whether the MCP flexion joints
survive the round-trip. If closed poses reconstruct to closed MCP, the decoder
CAN produce closure -> problem is human-side alignment. If closed poses
reconstruct to open MCP, the robot autoencoder collapses closure -> decoder/
bottleneck is the culprit.
"""
import sys
from pathlib import Path
import numpy as np
import torch

SRC = Path("/home/yareeez/AIST-hand/models/latent-retargeting/src")
sys.path.insert(0, str(SRC))
from cross_emb.nn.shared_modules import SharedEncoder_E_X, SharedDecoder_D_X
from cross_emb.nn.robot_modules import RobotEncoder_E_r, RobotDecoder_D_r

CKPT = "/home/yareeez/Downloads/stage1_best_run20.pt"
DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
N = 3000

# MCP flexion joints in qpos24 (J3 = base flexion). idx confirmed vs joint_names.
MCP = {"FFJ3": 3, "MFJ3": 7, "RFJ3": 11, "LFJ3": 16}
# PIP (J2) for context
PIP = {"FFJ2": 4, "MFJ2": 8, "RFJ2": 12, "LFJ2": 17}

torch.manual_seed(0)
c = torch.load(CKPT, map_location="cpu", weights_only=False)

E_r = RobotEncoder_E_r(n_joints=24, shared_dim=1024)
E_X = SharedEncoder_E_X(shared_dim=1024, z_dim=16)
D_X = SharedDecoder_D_X(z_dim=16, shared_dim=1024)
D_r = RobotDecoder_D_r(n_joints=24, shared_dim=1024)
E_r.load_state_dict(c["E_r"]); E_X.load_state_dict(c["E_X"])
D_X.load_state_dict(c["D_X"]); D_r.load_state_dict(c["D_r"])
for m in (E_r, E_X, D_X, D_r): m.eval()

dc = np.load(f"{DATA}/synthetic_close_hand_shadow_qpos.npz")
do = np.load(f"{DATA}/synthetic_open_hand_shadow_qpos.npz")

def take(d):
    q = torch.from_numpy(d["qpos24"][:N].astype(np.float32)).clone()
    q[:, 0:2] = 0.0   # zero_wrj=True
    return q

qc, qo = take(dc), take(do)

@torch.no_grad()
def autoenc(q):
    z = E_X(E_r(q))
    qh = D_r(D_X(z))
    return z, qh

zc, qch = autoenc(qc)
zo, qoh = autoenc(qo)

print("=" * 68)
print("RECONSTRUCTION ERROR (robot autoencode quality)")
print(f"  closed: ||q - q_hat|| mean = {(qc - qch).norm(dim=-1).mean():.3f}")
print(f"  open  : ||q - q_hat|| mean = {(qo - qoh).norm(dim=-1).mean():.3f}")

print("=" * 68)
print("MCP FLEXION (J3) -- rad.  closed input should stay HIGH (~1.5), open LOW")
print(f"{'joint':6} | {'IN closed':>9} {'OUT closed':>10} | {'IN open':>8} {'OUT open':>9} | {'OUT gap':>8}")
for nm, i in MCP.items():
    inc, outc = qc[:, i].mean().item(), qch[:, i].mean().item()
    ino, outo = qo[:, i].mean().item(), qoh[:, i].mean().item()
    print(f"{nm:6} | {inc:9.3f} {outc:10.3f} | {ino:8.3f} {outo:9.3f} | {outc-outo:8.3f}")

print("-" * 68)
print("PIP (J2) for context")
for nm, i in PIP.items():
    inc, outc = qc[:, i].mean().item(), qch[:, i].mean().item()
    ino, outo = qo[:, i].mean().item(), qoh[:, i].mean().item()
    print(f"{nm:6} | {inc:9.3f} {outc:10.3f} | {ino:8.3f} {outo:9.3f} | {outc-outo:8.3f}")

print("=" * 68)
print("LATENT SEPARATION (does z distinguish closed from open?)")
dz = (zc.mean(0) - zo.mean(0)).norm().item()
within = 0.5 * (zc.std(0).norm().item() + zo.std(0).norm().item())
print(f"  ||mean z_closed - mean z_open|| = {dz:.3f}")
print(f"  avg within-class z std (L2)     = {within:.3f}")
print(f"  separation ratio (between/within) = {dz / max(within,1e-6):.2f}")
