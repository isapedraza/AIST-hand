"""Loss-related constants and metric helpers for the cross-embodiment training loop."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from cross_emb.rotations import rot6d_to_matrix

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

# UDHM per-SLOT weights for the lam_udhm term (decomposed abd/flex). Keyed by slot
# name (order-independent; loop.py looks up by name and re-normalizes per finger).
# w = (1/sigma) / sum(1/sigma). sigma = std over HOGraspNet (abl11, 60k frames).
# CRITICAL: abduction sigma is computed on LOW-FLEXION frames only -- the Dong gamma
# extraction degrades under heavy flexion (verified: middle/ring abd std jumps ~5x
# from low to high flexion), so the naive full-dataset sigma is flexion-induced noise
# that would wrongly down-weight abduction. flex/pip/dip use full-dataset sigma (real).
# Result: abduction gets fair high weight on all fingers (it is genuinely low-variance).
# middle/ring mcp_flex boosted to index level: 1/sigma gives them 0.135/0.150 vs
# index 0.182 because they have higher dataset variance, but all MCP flex joints
# deserve equal supervisory pressure across fingers.
# thumb_mcp_abd and pinky_twist = 0 (Dong does not model them; zero-padded anyway).
_udhm_w: dict[str, float] = {
    "thumb_cmc_flex": 0.297, "thumb_cmc_spread": 0.220, "thumb_mcp_flex": 0.299,
    "thumb_mcp_abd": 0.0,    "thumb_ip_flex": 0.185,
    "index_mcp_abd": 0.483,  "index_mcp_flex": 0.182,  "index_pip_flex": 0.165,  "index_dip_flex": 0.170,
    "middle_mcp_abd": 0.566, "middle_mcp_flex": 0.200,  "middle_pip_flex": 0.143, "middle_dip_flex": 0.156,
    "ring_mcp_abd": 0.595,   "ring_mcp_flex": 0.200,   "ring_pip_flex": 0.124,   "ring_dip_flex": 0.131,
    "pinky_mcp_abd": 0.363,  "pinky_mcp_flex": 0.253,  "pinky_pip_flex": 0.187,  "pinky_dip_flex": 0.197,
    "pinky_twist": 0.0,
}


def dense_finger_pose(
    chain_a: torch.Tensor,
    chain_b: torch.Tensor,
    finger_idx: int,
) -> torch.Tensor:
    """Dense wrist-local finger pose distance over MCP/PIP/DIP/TIP.

    Args:
        chain_a, chain_b: [N, 5, 4, 3] MCP/PIP/DIP/TIP per finger, normalized.
        finger_idx:       0=thumb, 1=index, 2=middle, 3=ring, 4=pinky.

    Returns:
        [N] mean squared Cartesian distance over the 4 chain points.
    """
    diff = chain_a[:, finger_idx, :, :] - chain_b[:, finger_idx, :, :]
    return (diff ** 2).sum(dim=-1).mean(dim=-1)


def _last_segment_unit(chain: torch.Tensor, finger_idx: int, eps: float = 1e-6) -> torch.Tensor:
    """Unit-norm direction of the last finger segment (DIP -> TIP).

    Args:
        chain:      [N, 5, 4, 3] joint chain positions (wrist-local, normalized).
        finger_idx: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky.

    Returns:
        [N, 3] unit direction vector.
    """
    r = chain[:, finger_idx, 3, :] - chain[:, finger_idx, 2, :]
    return r / (r.norm(dim=-1, keepdim=True) + eps)


def _sigmoid_xin(d: torch.Tensor, w: float, eps1: float) -> torch.Tensor:
    """Xin-style sigmoid switching weight (Xin 2025 Eq. 197, ported from run21-paper-sk).

    Positive w -> gate ~1 when d << eps1 (pinch active).
    Negative w -> complement gate (tip_pos suppression when pinch active).
    """
    return 1.0 / (1.0 + torch.exp(w * (d - eps1)))


def _pinch_rescale(d: torch.Tensor, eps1: float, eps2: float) -> torch.Tensor:
    """Piecewise distance rescale l(d) (Xin 2025 Eq. 216, ported from run21-paper-sk).

    d < eps2:           collapses to 0 (in-contact)
    eps2 <= d <= eps1:  linear interp 0 -> eps1
    d > eps1:           identity
    """
    mid = eps1 / (eps1 - eps2) * (d - eps2)
    return torch.where(
        d < eps2,
        torch.zeros_like(d),
        torch.where(d > eps1, d, mid),
    )


def xin_sk_per_finger(
    tips_a: torch.Tensor,
    tips_b: torch.Tensor,
    chain_a: torch.Tensor,
    chain_b: torch.Tensor,
    finger_idx: int,
    lam_tip: float = 0.0,
    lam_finger: float = 0.0,
    lam_pinch: float = 0.0,
    lam_tip_rot: float = 0.0,
    enable_switching: bool = False,
    pinch_eps1: float = 0.508,
    pinch_eps2: float = 0.0508,
    pinch_sigmoid_w: float = 1.97,
) -> torch.Tensor:
    """Per-finger Cartesian similarity (Xin et al. 2025), normalized space.

    Two independent position signals (use either or both):
      lam_tip:    wrist->TIP only (Xin Eq. 1/2).
      lam_finger: dense MCP/PIP/DIP/TIP chain (Run 31+).

    lam_pinch:   thumb->primary finger vector (Xin Eq. 3, fingers 1-3 only).
    lam_tip_rot: DIP->TIP unit vector (Xin Eq. 4).

    When enable_switching=True (run21-paper-sk formulation):
      - Pinch uses sigmoid gate s(d_a) with piecewise target rescaling.
      - tip_pos for fingers 1-3 is suppressed by complement gate stilde(d_a).
      - Thumb (0) and pinky (4) are never gated.

    Args:
        tips_a, tips_b:   [N, 5, 3]    wrist-relative, normalized by hand_length.
        chain_a, chain_b: [N, 5, 4, 3] MCP/PIP/DIP/TIP per finger, normalized.
        finger_idx:                    0=thumb, 1=index, 2=middle, 3=ring, 4=pinky.
        pinch_eps1, pinch_eps2:        thresholds in normalized (hand-length) space.
        pinch_sigmoid_w:               sigmoid sharpness in normalized space.
    """
    out = tips_a.new_zeros(tips_a.shape[0])

    # tip_pos: wrist -> TIP
    if lam_tip > 0:
        v_a = tips_a[:, finger_idx, :]
        v_b = tips_b[:, finger_idx, :]
        tip_pos_raw = ((v_a - v_b) ** 2).sum(dim=-1)
        if enable_switching and finger_idx in (1, 2, 3):
            g_a    = tips_a[:, finger_idx, :] - tips_a[:, 0, :]
            d_a    = g_a.norm(dim=-1)
            stilde = _sigmoid_xin(d_a, -pinch_sigmoid_w, pinch_eps1)
            out    = out + lam_tip * stilde * tip_pos_raw
        else:
            out = out + lam_tip * tip_pos_raw

    # dense finger pose: MCP/PIP/DIP/TIP chain
    if lam_finger > 0:
        out = out + lam_finger * dense_finger_pose(chain_a, chain_b, finger_idx)

    # pinch: thumb -> primary finger vector (index/middle/ring only)
    if lam_pinch > 0 and finger_idx in (1, 2, 3):
        g_a = tips_a[:, finger_idx, :] - tips_a[:, 0, :]
        g_b = tips_b[:, finger_idx, :] - tips_b[:, 0, :]
        if enable_switching:
            d_a         = g_a.norm(dim=-1)
            gamma_a_hat = g_a / d_a.clamp(min=1e-8).unsqueeze(-1)
            s           = _sigmoid_xin(d_a,  pinch_sigmoid_w, pinch_eps1)
            target      = _pinch_rescale(d_a, pinch_eps1, pinch_eps2).unsqueeze(-1) * gamma_a_hat
            pinch       = s * ((g_b - target) ** 2).sum(dim=-1)
        else:
            pinch = ((g_a - g_b) ** 2).sum(dim=-1)
        out = out + lam_pinch * pinch

    # tip_rot: DIP -> TIP unit vector
    if lam_tip_rot > 0:
        r_a_hat = _last_segment_unit(chain_a, finger_idx)
        r_b_hat = _last_segment_unit(chain_b, finger_idx)
        tip_rot = ((r_a_hat - r_b_hat) ** 2).sum(dim=-1)
        out = out + lam_tip_rot * tip_rot

    return out


def xin_sk_full(
    tips_a: torch.Tensor,
    tips_b: torch.Tensor,
    chain_a: torch.Tensor,
    chain_b: torch.Tensor,
    lam_tip: float = 0.0,
    lam_thumb_tip: float = 0.0,
    lam_finger: float = 0.0,
    lam_thumb_finger: float = 0.0,
    lam_pinch: float = 0.0,
    lam_tip_rot: float = 0.0,
    enable_switching: bool = False,
    pinch_eps1: float = 0.508,
    pinch_eps2: float = 0.0508,
    pinch_sigmoid_w: float = 1.97,
) -> torch.Tensor:
    """Full-hand S_k sum over 5 fingers.

    Thumb uses lam_thumb_tip / lam_thumb_finger; others use lam_tip / lam_finger.
    """
    s = tips_a.new_zeros(tips_a.shape[0])
    for f in range(5):
        eff_tip    = lam_thumb_tip    if f == 0 else lam_tip
        eff_finger = lam_thumb_finger if f == 0 else lam_finger
        s = s + xin_sk_per_finger(
            tips_a, tips_b, chain_a, chain_b, f,
            lam_tip=eff_tip, lam_finger=eff_finger,
            lam_pinch=lam_pinch, lam_tip_rot=lam_tip_rot,
            enable_switching=enable_switching,
            pinch_eps1=pinch_eps1, pinch_eps2=pinch_eps2, pinch_sigmoid_w=pinch_sigmoid_w,
        )
    return s


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


# ---------------------------------------------------------------------------
# InfoNCE / NT-Xent adaptive (SiMHand Eq. 2+3, --contrastive_mode infonce)
# ---------------------------------------------------------------------------

def compute_W_linear(S_matrix: torch.Tensor) -> torch.Tensor:
    """[N,N] pairwise S_k distance → [N,N] weight matrix. Per-row (SiMHand Eq.2).

    W[i,j] = (S_max_i - S_k[i,j]) / (S_max_i - S_min_i). Diagonal = 0.
    """
    N = S_matrix.shape[0]
    mask_self = torch.eye(N, dtype=torch.bool, device=S_matrix.device)
    S_inf = S_matrix.masked_fill(mask_self, float('inf'))
    s_min = S_inf.min(dim=1, keepdim=True).values
    S_ninf = S_matrix.masked_fill(mask_self, float('-inf'))
    s_max = S_ninf.max(dim=1, keepdim=True).values
    W = (s_max - S_matrix) / (s_max - s_min).clamp(min=1e-8)
    return W.masked_fill(mask_self, 0.0)


def nt_xent_adaptive(z_pool: torch.Tensor, W: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """NT-Xent with per-pair adaptive weights (SiMHand Eq. 2+3).

    Positive = argmax W per row (dynamic nearest neighbor by S_k).
    z_pool: [N, D]. W: [N, N] weight matrix, diagonal 0.
    """
    N = z_pool.shape[0]
    device = z_pool.device
    z = F.normalize(z_pool, dim=-1)
    sim = torch.mm(z, z.T)                                          # [N, N]
    mask_self = torch.eye(N, dtype=torch.bool, device=device)
    pos_idx = W.masked_fill(mask_self, -1.0).argmax(dim=1)          # [N]
    arange  = torch.arange(N, device=device)
    w_pos   = W[arange, pos_idx]
    pos_sim = sim[arange, pos_idx]
    num     = torch.exp(w_pos * pos_sim / tau)
    weighted = (W * sim / tau).masked_fill(mask_self, float('-inf'))
    denom   = torch.exp(weighted).sum(dim=1)
    return -torch.log(num / (denom + 1e-8)).mean()


def compute_pairwise_S_ahg(
    q_pool: torch.Tensor,
    chain_pool: torch.Tensor,
    w_dr: torch.Tensor,
    w_joints: torch.Tensor,
    w_r: float,
    w_joints_scale: float,
    w_ahg: float,
    rot_repr: str = "quat",
) -> torch.Tensor:
    """Pairwise S_k (AHG mode): D_R + D_joints + D_ahg for all N×N pairs.

    q_pool:     [N, Jk, F]   rotations (F=4 quat or F=6 r6)
    chain_pool: [N, Fk, 4, 3] wrist-local joint positions
    w_dr:       [Jk]  per-joint D_R weights
    w_joints:   [4]   per-segment D_joints weights
    Returns:    [N, N]
    """
    N  = q_pool.shape[0]
    Fk = chain_pool.shape[1]

    # D_R pairwise
    if rot_repr == "r6":
        R    = rot6d_to_matrix(q_pool)                              # [N, Jk, 3, 3]
        Ra   = R.unsqueeze(1)                                       # [N, 1, Jk, 3, 3]
        Rb   = R.unsqueeze(0)                                       # [1, N, Jk, 3, 3]
        tr   = torch.matmul(Ra.transpose(-1, -2), Rb).diagonal(dim1=-2, dim2=-1).sum(-1)  # [N, N, Jk]
        per_jdr = (1.0 - ((tr - 1) * 0.5).clamp(-1, 1)) * 0.5
    else:
        qa      = q_pool.unsqueeze(1)                               # [N, 1, Jk, 4]
        qb      = q_pool.unsqueeze(0)
        per_jdr = 1.0 - (qa * qb).sum(-1) ** 2                     # [N, N, Jk]
    D_R = (w_dr * per_jdr).sum(dim=-1)                             # [N, N]

    # D_joints pairwise
    ca = chain_pool.unsqueeze(1)                                    # [N, 1, Fk, 4, 3]
    cb = chain_pool.unsqueeze(0)                                    # [1, N, Fk, 4, 3]
    D_joints = (w_joints * (ca - cb).norm(dim=-1)).sum(dim=(-2, -1))  # [N, N]

    # D_ahg pairwise: compute per-sample angle histograms, diff pairwise
    joints   = chain_pool.reshape(N, Fk * 4, 3)
    critical = torch.cat([chain_pool[:, :, 0, :], chain_pool[:, :, 3, :]], dim=1)  # [N, 2*Fk, 3]
    u_j = joints   / joints.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_c = critical / critical.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ang = torch.acos(torch.bmm(u_j, u_c.transpose(1, 2)).clamp(-1 + 1e-6, 1 - 1e-6))  # [N, Fk*4, 2*Fk]
    D_ahg = (ang.unsqueeze(1) - ang.unsqueeze(0)).abs().sum(dim=(-2, -1))              # [N, N]

    return w_r * D_R + w_joints_scale * D_joints + w_ahg * D_ahg


def compute_pairwise_S_xin(
    tips_pool: torch.Tensor,
    chain_pool: torch.Tensor,
    finger_idx: int,
    lam_tip: float,
    lam_finger: float,
    lam_pinch: float,
    lam_tip_rot: float,
    enable_switching: bool,
    pinch_eps1: float,
    pinch_eps2: float,
    pinch_sigmoid_w: float,
    lam_dr: float = 0.0,
    q_pool: torch.Tensor | None = None,
    w_dr: torch.Tensor | None = None,
    w_r: float = 1.0,
    rot_repr: str = "quat",
) -> torch.Tensor:
    """Pairwise S_k (Xin mode) for all N×N pairs, vectorized.

    Mirrors xin_sk_per_finger with broadcasting over the N×N grid.
    tips_pool:  [N, 5, 3]      all 5 fingers (wrist-relative, normalized)
    chain_pool: [N, 5, 4, 3]   all 5 fingers MCP/PIP/DIP/TIP
    Returns:    [N, N]
    """
    N   = tips_pool.shape[0]
    out = tips_pool.new_zeros(N, N)

    # tip_pos
    if lam_tip > 0:
        v  = tips_pool[:, finger_idx, :]                            # [N, 3]
        tip_pos_raw = ((v.unsqueeze(1) - v.unsqueeze(0)) ** 2).sum(dim=-1)  # [N, N]
        if enable_switching and finger_idx in (1, 2, 3):
            d      = (tips_pool[:, finger_idx, :] - tips_pool[:, 0, :]).norm(dim=-1)  # [N]
            stilde = _sigmoid_xin(d, -pinch_sigmoid_w, pinch_eps1)  # [N]
            out    = out + lam_tip * stilde.unsqueeze(1) * tip_pos_raw
        else:
            out = out + lam_tip * tip_pos_raw

    # dense finger pose
    if lam_finger > 0:
        ca = chain_pool[:, finger_idx, :, :].unsqueeze(1)          # [N, 1, 4, 3]
        cb = chain_pool[:, finger_idx, :, :].unsqueeze(0)          # [1, N, 4, 3]
        out = out + lam_finger * ((ca - cb) ** 2).sum(dim=-1).mean(dim=-1)  # [N, N]

    # pinch
    if lam_pinch > 0 and finger_idx in (1, 2, 3):
        g  = tips_pool[:, finger_idx, :] - tips_pool[:, 0, :]      # [N, 3]
        ga = g.unsqueeze(1)
        gb = g.unsqueeze(0)
        if enable_switching:
            d        = g.norm(dim=-1)
            gamma    = g / d.clamp(min=1e-8).unsqueeze(-1)
            s        = _sigmoid_xin(d, pinch_sigmoid_w, pinch_eps1)
            target   = _pinch_rescale(d, pinch_eps1, pinch_eps2).unsqueeze(-1) * gamma  # [N, 3]
            pinch    = s.unsqueeze(1) * ((gb - target.unsqueeze(1)) ** 2).sum(dim=-1)   # [N, N]
        else:
            pinch = ((ga - gb) ** 2).sum(dim=-1)
        out = out + lam_pinch * pinch

    # tip_rot
    if lam_tip_rot > 0:
        r     = chain_pool[:, finger_idx, 3, :] - chain_pool[:, finger_idx, 2, :]
        r_hat = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ra, rb = r_hat.unsqueeze(1), r_hat.unsqueeze(0)
        out   = out + lam_tip_rot * ((ra - rb) ** 2).sum(dim=-1)

    # optional D_R
    if lam_dr > 0 and q_pool is not None and w_dr is not None:
        if rot_repr == "r6":
            R    = rot6d_to_matrix(q_pool)
            Ra   = R.unsqueeze(1)
            Rb   = R.unsqueeze(0)
            tr   = torch.matmul(Ra.transpose(-1, -2), Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
            per_jdr = (1.0 - ((tr - 1) * 0.5).clamp(-1, 1)) * 0.5
        else:
            qa      = q_pool.unsqueeze(1)
            qb      = q_pool.unsqueeze(0)
            per_jdr = 1.0 - (qa * qb).sum(-1) ** 2
        out = out + lam_dr * w_r * (w_dr * per_jdr).sum(dim=-1)

    return out
