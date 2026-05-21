"""Loss-related constants and metric helpers for the cross-embodiment training loop."""
from __future__ import annotations

import torch

# D_R per-joint weights: w_j = (1/sigma_j) / sum(1/sigma)
# sigma_j = std(1 - dot_j^2) over HOGraspNet train pairs, human only.
# Order per subspace: [mcp, pip, dip, tip]. tip=0 (always identity in Dong).
# Precomputed offline from hograspnet_abl11.csv (50k random pairs, 10k frames).
_sk_w: dict[str, list[float]] = {
    "thumb":  [0.258, 0.544, 0.199, 0.0],
    "index":  [0.329, 0.325, 0.346, 0.0],
    "middle": [0.188, 0.362, 0.451, 0.0],
    "ring":   [0.238, 0.357, 0.405, 0.0],
    "pinky":  [0.197, 0.405, 0.398, 0.0],
}

# D_joints per-segment weights: w_j = (1/sigma_j) / sum(1/sigma)
# sigma_j = std(||chain_j_a - chain_j_b||) over HOGraspNet train pairs, human only.
# Order per subspace: [mcp, pip, dip, tip]. MCP highest weight (least variation).
# Precomputed offline from hograspnet_abl11.csv (50k random pairs, 10k frames).
_sk_wj: dict[str, list[float]] = {
    "thumb":  [0.4499, 0.2534, 0.1484, 0.1484],
    "index":  [0.5282, 0.2435, 0.1381, 0.0902],
    "middle": [0.5630, 0.2259, 0.1267, 0.0844],
    "ring":   [0.5743, 0.2364, 0.1134, 0.0759],
    "pinky":  [0.5459, 0.2465, 0.1241, 0.0835],
}


def _ahg(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """AHG-style angle histogram distance between two chain batches.

    Args:
        c1, c2: [n, Fk, 4, 3] wrist-local joint chain positions.

    Returns:
        [n] scalar distance per sample.
    """
    n_s = c1.shape[0]
    Fk  = c1.shape[1]
    joints   = c1.view(n_s, Fk * 4, 3)
    critical = torch.cat([c1[:, :, 0, :], c1[:, :, 3, :]], dim=1)  # [n, 2*Fk, 3]
    u_j = joints   / joints.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_c = critical / critical.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos  = torch.bmm(u_j, u_c.transpose(1, 2)).clamp(-1 + 1e-6, 1 - 1e-6)
    ang1 = torch.acos(cos)                                           # [n, Fk*4, 2*Fk]
    joints2   = c2.view(n_s, Fk * 4, 3)
    critical2 = torch.cat([c2[:, :, 0, :], c2[:, :, 3, :]], dim=1)
    u_j2 = joints2   / joints2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_c2 = critical2 / critical2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos2 = torch.bmm(u_j2, u_c2.transpose(1, 2)).clamp(-1 + 1e-6, 1 - 1e-6)
    ang2 = torch.acos(cos2)
    return (ang1 - ang2).abs().sum(dim=(-2, -1))                     # [n]
