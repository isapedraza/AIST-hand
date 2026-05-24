"""Loss-related constants and metric helpers for the cross-embodiment training loop."""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

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


# ---------------------------------------------------------------------------
# Xin-style Cartesian similarity metric (Run 25+)
#
# Replaces the joint-quaternion-based S_k (D_R + D_joints + D_ahg). Operates
# in the wrist-relative normalized space (both human and robot tips/chain are
# divided by their own hand_length). Symmetric, no switching weights, no eps
# dependency.
#
# Adapted from Xin et al. 2025 (eqs. 178, 192, 224). Run31 densifies
# Xin's fingertip/thumb position term from wrist->TIP only to a full
# per-finger chain pose over MCP/PIP/DIP/TIP:
#   - finger_pose:     mean wrist-local chain point match (overall hand shape)
#   - pinch:           thumb -> primary finger vector match (precise pinching)
#   - fingertip_rot:   DIP/IP -> TIP unit vector match (orientation)
#
# Switching weights and metric-unit eps thresholds are intentionally omitted:
# Xin uses them for per-frame NLopt optimization, not similarity ranking.
# For thumb the "DIP/IP -> TIP" reduces to IP -> TIP thanks to the loader
# convention (human_loader._CHAIN_COLS: thumb slot 2 holds THUMB_IP, not a
# duplicated TIP). Robot side already aligns (thdistal -> thtip).
# Vectors are normalized to unit length before comparison to isolate the
# orientation signal (Xin gets this implicitly via human_hand_scale; our
# pipeline normalizes per-body, so explicit normalization is the equivalent).
# ---------------------------------------------------------------------------


def _last_segment_unit(chain: torch.Tensor, finger_idx: int, eps: float = 1e-6) -> torch.Tensor:
    """Unit-norm direction of the last finger segment (DIP/IP -> TIP).

    Args:
        chain:      [N, 5, 4, 3] joint chain positions (wrist-local, normalized).
        finger_idx: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky.

    Returns:
        [N, 3] unit direction vector.
    """
    r = chain[:, finger_idx, 3, :] - chain[:, finger_idx, 2, :]
    return r / (r.norm(dim=-1, keepdim=True) + eps)


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
) -> torch.Tensor:
    """Per-finger Cartesian similarity. Symmetric, operates in normalized space.

    Two independent position signals (use either or both):
      lam_tip:    wrist->TIP only (1 point) -- pure Xin Eq.1/2.
      lam_finger: MCP/PIP/DIP/TIP chain (4 points) -- Run31 dense extension.
    Plus:
      lam_pinch:   L_pinch -- thumb->primary finger vector (Xin Eq.3, fingers 1-3 only).
      lam_tip_rot: L_fingertip_rot -- DIP/IP->TIP unit vector (Xin Eq.4).

    Each term gated: lam == 0 -> term not computed.

    Args:
        tips_a, tips_b:   [N, 5, 3]    wrist-relative, normalized by hand_length.
        chain_a, chain_b: [N, 5, 4, 3] MCP/PIP/DIP/TIP per finger, normalized.
        finger_idx:                    finger index in [0..4].
    """
    out = tips_a.new_zeros(tips_a.shape[0])

    if lam_tip > 0:
        v_a = tips_a[:, finger_idx, :]
        v_b = tips_b[:, finger_idx, :]
        tip = ((v_a - v_b) ** 2).sum(dim=-1)
        out = out + lam_tip * tip

    if lam_finger > 0:
        out = out + lam_finger * dense_finger_pose(chain_a, chain_b, finger_idx)

    if lam_pinch > 0 and finger_idx in (1, 2, 3):
        g_a = tips_a[:, finger_idx, :] - tips_a[:, 0, :]
        g_b = tips_b[:, finger_idx, :] - tips_b[:, 0, :]
        pinch = ((g_a - g_b) ** 2).sum(dim=-1)
        out = out + lam_pinch * pinch

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
) -> torch.Tensor:
    """Full-hand symmetric S_k (sum over 5 fingers).

    Thumb uses lam_thumb_tip / lam_thumb_finger; others use lam_tip / lam_finger.
    Matches Xin weight structure (thumb tip = 10x other fingertips).
    """
    s = tips_a.new_zeros(tips_a.shape[0])
    for f in range(5):
        eff_tip    = lam_thumb_tip    if f == 0 else lam_tip
        eff_finger = lam_thumb_finger if f == 0 else lam_finger
        s = s + xin_sk_per_finger(
            tips_a, tips_b, chain_a, chain_b, f,
            lam_tip=eff_tip, lam_finger=eff_finger,
            lam_pinch=lam_pinch, lam_tip_rot=lam_tip_rot,
        )
    return s


def global_pinch_similarity(
    tips_a: torch.Tensor,
    tips_b: torch.Tensor,
) -> torch.Tensor:
    """Full-hand thumb-to-primary-finger pinch-vector distance.

    Compares thumb->index, thumb->middle, and thumb->ring relative vectors.
    This is intentionally inter-finger only: it does not duplicate absolute
    thumb tip or dense finger-chain position terms.
    """
    out = tips_a.new_zeros(tips_a.shape[0])
    for finger_idx in (1, 2, 3):
        g_a = tips_a[:, finger_idx, :] - tips_a[:, 0, :]
        g_b = tips_b[:, finger_idx, :] - tips_b[:, 0, :]
        out = out + ((g_a - g_b) ** 2).sum(dim=-1)
    return out


# ---------------------------------------------------------------------------
# D_R -- Yan-style rotation similarity (Yan et al. 2026, eq. 1).
#
# Operates on quaternions over the joints common to both embodiments.
# Uniform sum, no per-joint weights:
#   D_R(x_A, x_B) = sum_j (1 - <q_A^j, q_B^j>^2)
#
# 1 - dot^2 is the squared geodesic distance on the unit quaternion sphere
# (equivalent to sin^2(theta/2) where theta is the SO(3) geodesic angle).
# Joints absent in either embodiment are filtered upstream by `common_labels`,
# so this function just consumes the already-aligned [B, J, 4] tensors.
# Tip joints in Dong are always identity; their contribution is ~0 and does
# not need explicit exclusion.
# ---------------------------------------------------------------------------


def d_r_yan(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """Yan rotation similarity, uniform sum over joints.

    Args:
        q_a, q_b: [N, J, 4] unit quaternions, joint-aligned.

    Returns:
        [N] non-negative scalar (lower = more similar).
    """
    dot = (q_a * q_b).sum(dim=-1)            # [N, J]
    return (1.0 - dot ** 2).sum(dim=-1)      # [N]


# ---------------------------------------------------------------------------
# L_joint -- joint position regularization (Xin Sec. III-B)
#
# Pulls specific joints (abduction joints + thumb rotation) toward 0 to
# prevent unnatural / mechanically unfavorable configurations. Weights and
# joint roles are read from the per-robot YAML (`semantic_roles`), keeping
# this loss generic across hand morphologies.
# ---------------------------------------------------------------------------


def load_l_joint_config(robot_yaml_path: str | Path) -> dict[str, float]:
    """Read `xin_position_weight` per joint from robot.yaml semantic_roles.

    Looks under components.hand.semantic_roles.* for any dict-shaped role
    with `xin_position_weight` and a list of `joints`. Returns a flat
    {joint_name: weight} mapping.
    """
    with open(robot_yaml_path) as f:
        cfg = yaml.safe_load(f)
    roles = cfg.get("components", {}).get("hand", {}).get("semantic_roles", {}) or {}
    w_pos: dict[str, float] = {}
    for role_name, role_cfg in roles.items():
        if not isinstance(role_cfg, dict):
            continue  # skip placeholders like "special_constraints: []"
        w = float(role_cfg.get("xin_position_weight", 0.0))
        if w == 0.0:
            continue
        for jn in role_cfg.get("joints", []) or []:
            w_pos[jn] = w
    return w_pos


def l_joint(
    q_r: torch.Tensor,
    joint_names: list[str],
    w_pos: dict[str, float],
) -> torch.Tensor:
    """Joint position regularization: sum_j w_j * (q_j - 0)^2, mean over batch.

    Args:
        q_r:         [B, J] robot joint positions (radians).
        joint_names: list of length J with the joint name per column of q_r.
        w_pos:       {joint_name: weight}, typically from load_l_joint_config.

    Returns:
        Scalar loss tensor.
    """
    loss = q_r.new_zeros(())
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    for jn, w in w_pos.items():
        if w == 0.0:
            continue
        j = name_to_idx.get(jn)
        if j is None:
            continue
        loss = loss + w * (q_r[:, j] ** 2).mean()
    return loss
