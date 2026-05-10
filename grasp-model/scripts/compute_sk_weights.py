#!/usr/bin/env python3
"""
Compute per-segment weights for D_R and D_joints in S_k.

Outputs weights ready to hardcode in train_cross_emb.py.

D_R weights  : w_j = (1/sigma_j) / sum(1/sigma)   [low-variance joints get high weight]
D_joints weights: w_j = sigma_j / sum(sigma)       [high-variance segments get high weight]

Usage:
    cd /home/yareeez/AIST-hand/grasp-model
    /home/yareeez/AIST-hand/.venv/bin/python scripts/compute_sk_weights.py
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src/cross_emb")

from human_loader import HumanLoader

CSV_PATH  = "data/processed/hograspnet_abl11.csv"
N_PAIRS   = 50_000
SEED      = 42

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
SEGMENTS = ["mcp", "pip", "dip", "tip"]

# Finger index -> columns in _chain [N, 5, 4, 3]
FINGER_IDX = {f: i for i, f in enumerate(FINGERS)}

print("Loading HumanLoader (train split)...")
loader = HumanLoader(CSV_PATH, split="train", device="cpu")
Q  = loader._quats   # [N, 20, 4]  quaternions
C  = loader._chain   # [N, 5, 4, 3] chain positions (normalized by hand_length)
N  = Q.shape[0]
print(f"Frames: {N}")

rng = np.random.default_rng(SEED)
ia  = rng.integers(0, N, N_PAIRS)
ib  = rng.integers(0, N, N_PAIRS)

# Joint index within the 20-joint array per finger
# HumanLoader order: thumb=[0,1,2,3], index=[4,5,6,7], middle=[8,9,10,11], ring=[12,13,14,15], pinky=[16,17,18,19]
JOINT_IDX = {
    "thumb":  [0, 1, 2, 3],
    "index":  [4, 5, 6, 7],
    "middle": [8, 9, 10, 11],
    "ring":   [12, 13, 14, 15],
    "pinky":  [16, 17, 18, 19],
}

print("\n" + "="*60)
print("D_R weights  (1/sigma normalized, tip excluded)")
print("  sigma_j = std(1 - dot(q_a_j, q_b_j)^2)")
print("="*60)

dr_weights = {}
for finger in FINGERS:
    jidx = JOINT_IDX[finger]
    sigmas = []
    for seg_i, seg in enumerate(SEGMENTS):
        ji = jidx[seg_i]
        qa = Q[ia, ji, :]  # [P, 4]
        qb = Q[ib, ji, :]
        if isinstance(qa, torch.Tensor):
            dot = (qa * qb).sum(-1)
            d   = 1 - dot**2
            sigma = d.std().item()
        else:
            dot = (qa * qb).sum(-1)
            d   = 1 - dot**2
            sigma = d.std()
        sigmas.append(sigma)

    # Exclude tip (index 3) from D_R -- always identity quaternion in Dong
    sigmas_notip = sigmas[:3]
    inv   = [1/s for s in sigmas_notip]
    total = sum(inv)
    w     = [v/total for v in inv] + [0.0]  # tip=0

    dr_weights[finger] = w
    print(f"\n{finger}:")
    for seg, s, wi in zip(SEGMENTS, sigmas, w):
        print(f"  {seg}: sigma={s:.4f}  w={wi:.4f}")

print("\n\nHardcode block for _sk_w (D_R, 1/sigma):")
print("_sk_w = {")
for finger in FINGERS:
    w = dr_weights[finger]
    print(f'    "{finger}": [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}, {w[3]:.4f}],')
print("}")

print("\n" + "="*60)
print("D_joints weights  (sigma normalized)")
print("  sigma_j = std(||chain_j_a - chain_j_b||)")
print("="*60)

dj_weights = {}
for finger in FINGERS:
    fi = FINGER_IDX[finger]
    sigmas = []
    for seg_i, seg in enumerate(SEGMENTS):
        ca = C[ia, fi, seg_i, :]  # [P, 3]
        cb = C[ib, fi, seg_i, :]
        if isinstance(ca, torch.Tensor):
            delta = (ca - cb).norm(dim=-1)
            sigma = delta.std().item()
        else:
            delta = np.linalg.norm(ca - cb, axis=-1)
            sigma = delta.std()
        sigmas.append(sigma)

    total = sum(sigmas)
    w     = [s/total for s in sigmas]

    dj_weights[finger] = w
    print(f"\n{finger}:")
    for seg, s, wi in zip(SEGMENTS, sigmas, w):
        print(f"  {seg}: sigma={s:.4f}  w={wi:.4f}")

print("\n\nHardcode block for _sk_wj (D_joints, sigma):")
print("_sk_wj = {")
for finger in FINGERS:
    w = dj_weights[finger]
    print(f'    "{finger}": [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}, {w[3]:.4f}],')
print("}")
