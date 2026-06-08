#!/usr/bin/env python3
"""Apples-to-apples: ROBOT-closed vs HUMAN-closed MCP, BOTH through the same
Dong pipeline (same quaternion convention). Robot poses come from the dedicated
synthetic closed/open NPZ (no cherry-pick filter). Human from hagrid anchors.
"""
import sys
from pathlib import Path
import numpy as np
import torch

SRC = Path("/home/yareeez/AIST-hand/models/latent-retargeting/src")
sys.path.insert(0, str(SRC))
from cross_emb.loaders.human_loader import StaticHumanAnchorLoader
from cross_emb.loaders.robot_loader import RobotLoader

DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
URDF = "/home/yareeez/AIST-hand/robot/hands/shadow_hand/shadow_hand_right.urdf"
HAND = "/home/yareeez/AIST-hand/robot/hands/shadow_hand/shadow_hand_right.yaml"
N = 500
torch.manual_seed(0); np.random.seed(0)

rl = RobotLoader(URDF, device="cpu")
chain_names = rl.chain_joint_names
print("chain_joint_names:", chain_names)

def robot_dong(npz_name):
    d = np.load(f"{DATA}/{npz_name}")
    names24 = ["WRJ2", "WRJ1"] + [str(s) for s in d["joint_names"]]   # qpos24 order
    q24 = d["qpos24"][:N].astype(np.float32)
    # reorder q24 -> chain order
    # chain names may be like 'rh_FFJ3'; match by suffix.
    def col_for(cn):
        for i, nm in enumerate(names24):
            if cn == nm or cn.endswith(nm) or nm.endswith(cn):
                return i
        return None
    cols = [col_for(cn) for cn in chain_names]
    if any(c is None for c in cols):
        print("WARN unmatched chain joints:", [cn for cn, c in zip(chain_names, cols) if c is None])
    q = torch.tensor(q24[:, [c if c is not None else 0 for c in cols]])
    fk = rl.run_fk(q)
    quats, labels, _ = rl.run_dong_stage2(fk, HAND)
    return quats, labels

qc_r, labels = robot_dong("synthetic_close_hand_shadow_qpos.npz")
qo_r, _      = robot_dong("synthetic_open_hand_shadow_qpos.npz")
print("robot dong labels:", labels)

hl = StaticHumanAnchorLoader(f"{DATA}/hagrid_dong.csv")
def hsamp(c):
    i = hl._class_indices[c]; i = i[torch.randint(0, len(i), (N,))]; return hl._quats[i]
qc_h, qo_h = hsamp(29), hsamp(28)
HUMAN_LABELS = hl.labels  # DONG_LABELS, 20

def ang(q):  # [N,4]->mean deg
    return torch.rad2deg(2*torch.acos(q[:,0].abs().clamp(0,1))).mean().item()

def get(quats, labels, name):
    return quats[:, labels.index(name), :]

print("="*70)
print("SAME Dong pipeline both sides. MCP + PIP flexion angle (deg).")
for hjoint, rjoint, tag in [("index_mcp","index_mcp","INDEX MCP"),
                            ("middle_mcp","middle_mcp","MIDDLE MCP"),
                            ("index_pip","index_pip","INDEX PIP"),
                            ("middle_pip","middle_pip","MIDDLE PIP")]:
    rc = ang(get(qc_r, labels, rjoint)); ro = ang(get(qo_r, labels, rjoint))
    hc = ang(get(qc_h, HUMAN_LABELS, hjoint)); ho = ang(get(qo_h, HUMAN_LABELS, hjoint))
    print(f"{tag:11} | robot closed={rc:5.1f}  open={ro:5.1f}  | human closed={hc:5.1f}  open={ho:5.1f}")
