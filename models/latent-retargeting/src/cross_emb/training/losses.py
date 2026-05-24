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
# Adapted from Xin et al. 2025 (eqs. 178, 192, 224). Three terms:
#   - fingertip_pos:   wrist -> tip vector match (overall hand shape)
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


def _sigmoid_xin(d: torch.Tensor, w: float, eps1: float) -> torch.Tensor:
    """Xin-style sigmoid switching weight (Xin 2025 Eq. 197, ported from Run 21 paper-sk).

    sigmoid(x, c, w) = 1 / (1 + exp(w * (x - c)))

    Positive w  -> gate ~1 when d << eps1, ~0 when d >> eps1 (pinch active gate).
    Negative w  -> complement gate (tip_pos suppression when pinch is active).
    """
    return 1.0 / (1.0 + torch.exp(w * (d - eps1)))


def _pinch_rescale(d: torch.Tensor, eps1: float, eps2: float) -> torch.Tensor:
    """Piecewise distance rescale l(d) (Xin 2025 Eq. 216, ported from Run 21 paper-sk).

    Three zones:
      d < eps2:           target collapses to 0 (in-contact)
      eps2 <= d <= eps1:  linear interp from 0 to eps1
      d > eps1:           identity (target = d)
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
    lam_tip_pos: float = 1.0,
    lam_pinch: float = 10.0,
    lam_tip_rot: float = 10.0,
    lam_pip_pos: float = 1.0,
    enable_switching: bool = False,
    pinch_eps1: float = 0.508,
    pinch_eps2: float = 0.0508,
    pinch_sigmoid_w: float = 1.97,
) -> torch.Tensor:
    """Per-finger Cartesian similarity. Symmetric, operates in normalized space.

    Terms follow Xin et al. 2025 nomenclature:
      tip_pos: L_fingertip_pos / L_thumb_pos -- wrist->tip vector (Xin Eqs. 1-2)
      pinch:   L_pinch -- thumb->primary finger vector (Xin Eq. 3, fingers 1-3 only)
      tip_rot: L_fingertip_rot -- DIP/IP->TIP unit vector (Xin Eq. 4)
      pip_pos: PIP position (DexMV-style; not in Xin paper)

    When enable_switching=True (Run 30+):
      - Pinch uses Xin sigmoid switching with target rescaling (ported from Run 21
        paper-sk / Xin code). gamma_hat and d come from anchor (tips_a); target is
        l(d_a) * gamma_hat_a; D_pinch = s(d_a) * ||gamma_b - target||^2.
      - tip_pos for fingers in (1,2,3) is gated by stilde = sigmoid(d_a, eps1, -w)
        (complement of pinch gate). Thumb (finger_idx=0) and pinky (finger_idx=4)
        are not gated -- Xin's tilde_s is only defined for primary fingers, and
        thumb's tip_pos lives exclusively in lam_thumb_pos (no double-counting).

    Args:
        tips_a, tips_b:   [N, 5, 3]    wrist-relative, normalized by hand_length.
        chain_a, chain_b: [N, 5, 4, 3] MCP/PIP/DIP/TIP per finger, normalized.
        finger_idx:                    finger index in [0..4].
        enable_switching:              if True, apply Xin sigmoid switching.
        pinch_eps1, pinch_eps2:        thresholds already in normalized space
                                       (eps_m / pinch_ref_hand_length_m).
        pinch_sigmoid_w:               sigmoid sharpness already in normalized
                                       space (w_m * pinch_ref_hand_length_m).

    Returns:
        [N] non-negative scalar similarity (lower = more similar).
    """
    # L_fingertip_pos / L_thumb_pos: wrist -> tip vector
    v_a = tips_a[:, finger_idx, :]
    v_b = tips_b[:, finger_idx, :]
    tip_pos_raw = ((v_a - v_b) ** 2).sum(dim=-1)

    # L_pinch + stilde gating (only for primary fingers index/middle/ring).
    if finger_idx in (1, 2, 3):
        g_a = tips_a[:, finger_idx, :] - tips_a[:, 0, :]
        g_b = tips_b[:, finger_idx, :] - tips_b[:, 0, :]
        if enable_switching:
            d_a         = g_a.norm(dim=-1)
            gamma_a_hat = g_a / d_a.clamp(min=1e-8).unsqueeze(-1)
            s           = _sigmoid_xin(d_a,  pinch_sigmoid_w, pinch_eps1)
            stilde      = _sigmoid_xin(d_a, -pinch_sigmoid_w, pinch_eps1)
            target      = _pinch_rescale(d_a, pinch_eps1, pinch_eps2).unsqueeze(-1) * gamma_a_hat
            pinch       = s * ((g_b - target) ** 2).sum(dim=-1)
            tip_pos     = stilde * tip_pos_raw
        else:
            pinch   = ((g_a - g_b) ** 2).sum(dim=-1)
            tip_pos = tip_pos_raw
    else:
        # Thumb (0) and pinky (4): no pinch term, no stilde gating.
        pinch   = torch.zeros_like(tip_pos_raw)
        tip_pos = tip_pos_raw

    # L_fingertip_rot: DIP/IP -> TIP unit vector (Xin Eq. 4)
    r_a_hat = _last_segment_unit(chain_a, finger_idx)
    r_b_hat = _last_segment_unit(chain_b, finger_idx)
    tip_rot = ((r_a_hat - r_b_hat) ** 2).sum(dim=-1)

    # PIP position: wrist -> PIP vector (DexMV-style; not in Xin paper)
    pip_a = chain_a[:, finger_idx, 1, :]
    pip_b = chain_b[:, finger_idx, 1, :]
    pip_pos = ((pip_a - pip_b) ** 2).sum(dim=-1)

    return lam_tip_pos * tip_pos + lam_pinch * pinch + lam_tip_rot * tip_rot + lam_pip_pos * pip_pos


def xin_sk_full(
    tips_a: torch.Tensor,
    tips_b: torch.Tensor,
    chain_a: torch.Tensor,
    chain_b: torch.Tensor,
    lam_tip_pos: float = 1.0,
    lam_thumb_pos: float = 10.0,
    lam_pinch: float = 10.0,
    lam_tip_rot: float = 10.0,
    lam_pip_pos: float = 1.0,
    enable_switching: bool = False,
    pinch_eps1: float = 0.508,
    pinch_eps2: float = 0.0508,
    pinch_sigmoid_w: float = 1.97,
) -> torch.Tensor:
    """Full-hand symmetric S_k (sum over all 5 fingers).

    Thumb tip uses lam_thumb_pos (L_thumb_pos, Xin Eq. 1).
    Other fingers use lam_tip_pos (L_fingertip_pos, Xin Eq. 2).
    Matches Xin et al. 2025 weight structure: thumb tip = 10x other fingertips.

    enable_switching / pinch_eps* / pinch_sigmoid_w propagate to xin_sk_per_finger.
    """
    s = tips_a.new_zeros(tips_a.shape[0])
    for f in range(5):
        eff_tip_pos = lam_thumb_pos if f == 0 else lam_tip_pos
        s = s + xin_sk_per_finger(
            tips_a, tips_b, chain_a, chain_b, f,
            lam_tip_pos=eff_tip_pos, lam_pinch=lam_pinch, lam_tip_rot=lam_tip_rot, lam_pip_pos=lam_pip_pos,
            enable_switching=enable_switching,
            pinch_eps1=pinch_eps1, pinch_eps2=pinch_eps2, pinch_sigmoid_w=pinch_sigmoid_w,
        )
    return s


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
