"""
Xin et al. retargeting metrics, adapted as differentiable training losses.

Originally from "Analyzing Key Objectives in Human-to-Robot Retargeting for
Dexterous Manipulation" (Yu et al.). The original code uses NLopt SLSQP and
treats sigmoid/stilde gates as constants; here we apply them as torch losses
on the human path (`q_r_hat = D_r(D_X(E_h(quats_h)))`), so gradient can flow
back into the decoder.

All metrics expect tensors in the same normalized space used by
`chain_h_sub` / `chain_r_sub` in the Stage 1 batch: wrist-local, divided by
`hand_length`. Five fingers in order [thumb, index, middle, ring, pinky],
chain points in order [MCP, PIP, DIP, TIP].

Shapes:
    tips_*  : [B, 5, 3]
    chain_* : [B, 5, 4, 3]
Outputs of per-finger fns: [B]
Output of xin_total       : [B]   (sum across fingers, weighted by lams)
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml


# ---------------------------------------------------------------------------
# Switching helpers (ported from Run 30 losses.py)
# ---------------------------------------------------------------------------

def _sigmoid_xin(d: torch.Tensor, w: float, eps1: float) -> torch.Tensor:
    """Xin pinch sigmoid: 1 / (1 + exp(w * (d - eps1))).
    Positive w  -> gate ~1 when d << eps1 (pinch active).
    Negative w  -> complement (suppress tip_pos when pinching).
    """
    return 1.0 / (1.0 + torch.exp(w * (d - eps1)))


def _pinch_rescale(d: torch.Tensor, eps1: float, eps2: float) -> torch.Tensor:
    """Piecewise rescale used inside D_pinch target construction.
        d < eps2        -> 0           (treat as in-contact)
        eps2 <= d <= eps1 -> k*(d - eps2)  (linear ramp, k = eps1/(eps1-eps2))
        d > eps1        -> d           (identity, no rescaling)
    Not C1 at the seams; in practice gradient stays well-defined except on a
    measure-zero set, and the contact region (d < eps2) being flat is the
    desired stable behavior.
    """
    k   = eps1 / (eps1 - eps2)
    mid = k * (d - eps2)
    return torch.where(
        d < eps2,
        torch.zeros_like(d),
        torch.where(d > eps1, d, mid),
    )


# ---------------------------------------------------------------------------
# Per-finger Xin components
# ---------------------------------------------------------------------------

def xin_tip_pos(
    tips_a: torch.Tensor,        # [B, 5, 3] human (anchor)
    tips_b: torch.Tensor,        # [B, 5, 3] robot from q_r_hat
    finger_idx: int,
    *,
    enable_switching: bool = False,
    eps1: float = 0.508,
    sigmoid_w: float = 1.97,
    pinch_anchor_thumb: torch.Tensor | None = None,  # [B, 3] thumb tip on anchor
) -> torch.Tensor:
    """
    Xin D_tip_pos (and D_thumb_pos when finger_idx == 0).
    Squared Euclidean between wrist-relative fingertip positions.

    When switching is enabled and finger_idx in {1, 2, 3} (index/middle/ring),
    multiply by stilde = sigmoid(-w*(d - eps1)) where d is the anchor's
    thumb-to-finger distance: suppress tip_pos contribution when pinching.
    Thumb (0) and pinky (4) are never gated.
    """
    v_a = tips_a[:, finger_idx, :]
    v_b = tips_b[:, finger_idx, :]
    raw = ((v_a - v_b) ** 2).sum(dim=-1)             # [B]

    if enable_switching and finger_idx in (1, 2, 3) and pinch_anchor_thumb is not None:
        g_a = tips_a[:, finger_idx, :] - pinch_anchor_thumb
        d_a = g_a.norm(dim=-1)
        stilde = _sigmoid_xin(d_a, -sigmoid_w, eps1).detach()
        raw = stilde * raw
    return raw


def xin_pinch(
    tips_a: torch.Tensor,
    tips_b: torch.Tensor,
    finger_idx: int,
    *,
    enable_switching: bool = False,
    eps1: float = 0.508,
    eps2: float = 0.0508,
    sigmoid_w: float = 1.97,
) -> torch.Tensor:
    """
    Xin D_pinch. Compares thumb-to-finger vectors. Returns zero for thumb (0)
    and pinky (4) -- only index/middle/ring participate.

    Without switching: simple ||g_a - g_b||^2 where g = tip_finger - tip_thumb.
    With switching: target on anchor is rescaled via stilde; gate by sigmoid
    weighted to be ~1 when anchor distance is small (pinch active).
    """
    if finger_idx in (0, 4):
        return tips_a.new_zeros(tips_a.shape[0])

    g_a = tips_a[:, finger_idx, :] - tips_a[:, 0, :]
    g_b = tips_b[:, finger_idx, :] - tips_b[:, 0, :]

    if not enable_switching:
        return ((g_a - g_b) ** 2).sum(dim=-1)

    d_a         = g_a.norm(dim=-1)
    gamma_a_hat = g_a / d_a.clamp(min=1e-8).unsqueeze(-1)
    s           = _sigmoid_xin(d_a,  sigmoid_w, eps1).detach()
    rescaled    = _pinch_rescale(d_a, eps1, eps2).detach()
    target      = rescaled.unsqueeze(-1) * gamma_a_hat.detach()
    return s * ((g_b - target) ** 2).sum(dim=-1)


def xin_tip_rot(
    chain_a: torch.Tensor,       # [B, 5, 4, 3]
    chain_b: torch.Tensor,
    finger_idx: int,
) -> torch.Tensor:
    """
    Xin D_tip_rot: raw DIP->TIP vector L2-squared diff (NOT unit-normalized).
    Faithful to the paper. Input chain is already in hand-length-normalized
    space, so the vector magnitude is ~distal_length / hand_length
    (~0.10-0.15 for anthropomorphic hands). Cross-embodiment comparability
    is preserved by the upstream hand_length normalization, not by
    unit-normalizing the direction here.
    """
    r_a = chain_a[:, finger_idx, 3, :] - chain_a[:, finger_idx, 2, :]
    r_b = chain_b[:, finger_idx, 3, :] - chain_b[:, finger_idx, 2, :]
    return ((r_a - r_b) ** 2).sum(dim=-1)


def xin_pip_pos(
    chain_a: torch.Tensor,
    chain_b: torch.Tensor,
    finger_idx: int,
) -> torch.Tensor:
    """DexMV-style PIP position match. Squared Euclidean on chain index 1."""
    pip_a = chain_a[:, finger_idx, 1, :]
    pip_b = chain_b[:, finger_idx, 1, :]
    return ((pip_a - pip_b) ** 2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def xin_total(
    tips_h: torch.Tensor,        # [B, 5, 3] human (anchor)
    tips_r: torch.Tensor,        # [B, 5, 3] robot from q_r_hat (with gradient)
    chain_h: torch.Tensor,       # [B, 5, 4, 3]
    chain_r: torch.Tensor,
    *,
    lam_thumb_pos: float = 10.0,
    lam_tip_pos:   float = 1.0,
    lam_pinch:     float = 10.0,
    lam_tip_rot:   float = 10.0,
    lam_pip_pos:   float = 1.0,
    enable_switching: bool = True,
    eps1:      float = 0.508,
    eps2:      float = 0.0508,
    sigmoid_w: float = 1.97,
    return_breakdown: bool = False,
):
    """
    Aggregate Xin retargeting loss across 5 fingers.

    Per-finger weights:
      finger 0 (thumb)         -> tip_pos uses lam_thumb_pos, no pinch
      fingers 1..3 (i/m/r)     -> tip_pos uses lam_tip_pos, full pinch
      finger 4 (pinky)         -> tip_pos uses lam_tip_pos, no pinch
    `tip_rot` and `pip_pos` apply to all 5 fingers.

    Returns
    -------
    L : Tensor[B]
        Sum across fingers. Caller usually `.mean()`s before adding to L_total.
    breakdown : dict (only if return_breakdown=True)
        Per-component running sums, each Tensor[B], for dry-run logging.
        Keys: "tip_pos", "pinch", "tip_rot", "pip_pos".
    """
    B   = tips_h.shape[0]
    out = tips_h.new_zeros(B)
    bd  = {
        "tip_pos": tips_h.new_zeros(B),
        "pinch":   tips_h.new_zeros(B),
        "tip_rot": tips_h.new_zeros(B),
        "pip_pos": tips_h.new_zeros(B),
    }

    thumb_anchor = tips_h[:, 0, :]  # for tip_pos stilde gate
    for f in range(5):
        w_tp = lam_thumb_pos if f == 0 else lam_tip_pos
        tp = xin_tip_pos(
            tips_h, tips_r, f,
            enable_switching=enable_switching,
            eps1=eps1, sigmoid_w=sigmoid_w,
            pinch_anchor_thumb=thumb_anchor,
        )
        pn = xin_pinch(
            tips_h, tips_r, f,
            enable_switching=enable_switching,
            eps1=eps1, eps2=eps2, sigmoid_w=sigmoid_w,
        )
        tr = xin_tip_rot(chain_h, chain_r, f)
        pp = xin_pip_pos(chain_h, chain_r, f)

        out      = out      + w_tp * tp + lam_pinch * pn + lam_tip_rot * tr + lam_pip_pos * pp
        bd["tip_pos"] = bd["tip_pos"] + w_tp * tp
        bd["pinch"]   = bd["pinch"]   + lam_pinch * pn
        bd["tip_rot"] = bd["tip_rot"] + lam_tip_rot * tr
        bd["pip_pos"] = bd["pip_pos"] + lam_pip_pos * pp

    if return_breakdown:
        return out, bd
    return out


# ---------------------------------------------------------------------------
# L_joint -- joint position regularization (Xin Sec. III-B, ported from Run 25)
# ---------------------------------------------------------------------------

def load_l_joint_config(hand_config_path: str | Path) -> dict[str, float]:
    """Read per-joint Xin position weights from the hand YAML.

    Expects a top-level `semantic_roles` block with role -> {joints, xin_position_weight}.
    Returns a flat {joint_name: weight} map. Missing block returns {}.
    """
    with open(hand_config_path) as f:
        cfg = yaml.safe_load(f)
    roles = cfg.get("semantic_roles", {}) or {}
    w_pos: dict[str, float] = {}
    for role_name, role_cfg in roles.items():
        if not isinstance(role_cfg, dict):
            continue
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
    """L_joint = sum_j w_j * (q_r[:, j] ** 2).mean(). Scalar tensor.

    Pulls listed joints toward 0 (rest). Joints not in w_pos are untouched.
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
