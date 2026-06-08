#!/usr/bin/env python3
"""Part 2: feed LABELED human closed/open hands through Run 20 full path
(E_h -> D_X -> D_r) and check (a) does E_h separate closed from open,
(b) does it land in the robot-closed region, (c) does the robot MCP close.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SRC = Path("/home/yareeez/AIST-hand/models/latent-retargeting/src")
sys.path.insert(0, str(SRC))
from cross_emb.nn.human_modules import HumanEncoder_E_h
from cross_emb.nn.shared_modules import SharedEncoder_E_X, SharedDecoder_D_X
from cross_emb.nn.robot_modules import RobotEncoder_E_r, RobotDecoder_D_r

CKPT = "/home/yareeez/Downloads/stage1_best_run20.pt"
DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
MCP = {"FFJ3": 3, "MFJ3": 7, "RFJ3": 11, "LFJ3": 16}

c = torch.load(CKPT, map_location="cpu", weights_only=False)
E_h = HumanEncoder_E_h(in_dim=4, hidden_dim=32, z_dim=16)
E_r = RobotEncoder_E_r(n_joints=24, shared_dim=1024)
E_X = SharedEncoder_E_X(shared_dim=1024, z_dim=16)
D_X = SharedDecoder_D_X(z_dim=16, shared_dim=1024)
D_r = RobotDecoder_D_r(n_joints=24, shared_dim=1024)
E_h.load_state_dict(c["E_h"]); E_r.load_state_dict(c["E_r"]); E_X.load_state_dict(c["E_X"])
D_X.load_state_dict(c["D_X"]); D_r.load_state_dict(c["D_r"])
for m in (E_h, E_r, E_X, D_X, D_r): m.eval()

# --- human hands ---
df = pd.read_csv(f"{DATA}/hagrid_dong.csv")
qcols = [f"q{j}_{ch}" for j in range(1, 21) for ch in ("w", "x", "y", "z")]
def quats(label):
    sub = df[df["anchor_label"] == label]
    arr = sub[qcols].values.astype(np.float32).reshape(-1, 20, 4)
    return torch.from_numpy(arr)
qh_closed, qh_open = quats("closed_fist"), quats("open_hand")
print(f"human closed_fist: {len(qh_closed)}   open_hand: {len(qh_open)}")

@torch.no_grad()
def human_path(qh):
    z = E_h(qh)
    qr = D_r(D_X(z))
    return z, qr
zh_c, qr_c = human_path(qh_closed)
zh_o, qr_o = human_path(qh_open)

# robot reference latents (same as part 1)
dc = np.load(f"{DATA}/synthetic_close_hand_shadow_qpos.npz")
do = np.load(f"{DATA}/synthetic_open_hand_shadow_qpos.npz")
def robot_z(d, n=3000):
    q = torch.from_numpy(d["qpos24"][:n].astype(np.float32)).clone(); q[:, 0:2] = 0.0
    with torch.no_grad():
        return E_X(E_r(q))
zr_c, zr_o = robot_z(dc), robot_z(do)

print("=" * 70)
print("(a) DOES E_h SEPARATE human closed from open?")
dz = (zh_c.mean(0) - zh_o.mean(0)).norm().item()
within = 0.5 * (zh_c.std(0).norm().item() + zh_o.std(0).norm().item())
print(f"   ||z_h_closed - z_h_open|| = {dz:.3f}   within-std = {within:.3f}   ratio = {dz/max(within,1e-6):.2f}")

print("=" * 70)
print("(b) WHERE does human-closed land vs robot regions?")
mzhc, mzho = zh_c.mean(0), zh_o.mean(0)
mzrc, mzro = zr_c.mean(0), zr_o.mean(0)
print(f"   human-CLOSED -> robot-closed dist = {(mzhc-mzrc).norm():.3f} | robot-open dist = {(mzhc-mzro).norm():.3f}")
print(f"   human-OPEN   -> robot-closed dist = {(mzho-mzrc).norm():.3f} | robot-open dist = {(mzho-mzro).norm():.3f}")
print(f"   (robot-closed vs robot-open are {(mzrc-mzro).norm():.3f} apart)")

print("=" * 70)
print("(c) ROBOT MCP from human input (the symptom).  closed should be HIGH")
print(f"{'joint':6} | {'human-closed out':>16} | {'human-open out':>14}")
for nm, i in MCP.items():
    print(f"{nm:6} | {qr_c[:,i].mean():16.3f} | {qr_o[:,i].mean():14.3f}")
print(f"\n   For reference robot-autoencode gave closed~1.5, open~0.0")

print("=" * 70)
print("(d) IS IT JUST A MANIFOLD OFFSET?  shift human-closed z by")
print("    (robot_closed_mean - human_closed_mean), then decode.")
with torch.no_grad():
    offset = (mzrc - mzhc)              # global translation human->robot closed region
    zh_c_shift = zh_c + offset
    qr_c_shift = D_r(D_X(zh_c_shift))
    # also: decode nearest-robot snap (upper bound = pure robot-closed)
for nm, i in MCP.items():
    print(f"   {nm:6} shifted-human-closed out = {qr_c_shift[:,i].mean():.3f}  (raw was, see above; robot-AE ~1.5)")
