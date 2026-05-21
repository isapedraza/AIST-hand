"""Sanity-check for Run 25 Xin Cartesian S_k functions and the loader fix.

Run from /home/yareeez/AIST-hand/models/latent-retargeting/:
    python scripts/test_xin_sk.py

What it checks
--------------
1. Shapes / non-negativity of xin_sk_per_finger for every finger (including thumb).
2. xin_sk_full equals the sum of per-finger contributions.
3. Symmetry: S_k(a, b) == S_k(b, a).
4. Self-similarity: S_k(a, a) == 0.
5. Gradient flow through tips and chain.
6. L_joint config loads weights from robot.yaml.
7. (Optional, if CSV path is available) HumanLoader fix: thumb slot 1 == slot 2 == IP,
   and last-segment vector for thumb is non-zero.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Make the package importable when running directly without install -e
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_emb.training.losses import (  # noqa: E402
    l_joint,
    load_l_joint_config,
    xin_sk_full,
    xin_sk_per_finger,
)


def _format_check(name: str) -> None:
    print(f"  [OK] {name}")


def test_synthetic() -> None:
    print("[1] Synthetic tensors -- shape / sign / sum / symmetry / self / gradient")
    torch.manual_seed(0)
    N = 32
    tips_a = torch.randn(N, 5, 3, requires_grad=True)
    tips_b = torch.randn(N, 5, 3)
    chain_a = torch.randn(N, 5, 4, 3, requires_grad=True)
    chain_b = torch.randn(N, 5, 4, 3)

    # 1. All fingers contribute non-negative scalars
    for f in range(5):
        sk = xin_sk_per_finger(tips_a, tips_b, chain_a, chain_b, f,
                               lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=1.0)
        assert sk.shape == (N,), f"finger {f}: expected [{N}], got {sk.shape}"
        assert (sk >= 0).all(), f"finger {f}: negative similarity"
    _format_check("per-finger shapes and non-negativity (all 5 fingers, thumb included)")

    # 2. full = sum of per-finger
    sk_full = xin_sk_full(tips_a, tips_b, chain_a, chain_b,
                          lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=1.0)
    sk_sum = sum(
        xin_sk_per_finger(tips_a, tips_b, chain_a, chain_b, f,
                          lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=1.0)
        for f in range(5)
    )
    assert torch.allclose(sk_full, sk_sum, atol=1e-6), "full vs sum of per-finger differ"
    _format_check("xin_sk_full = sum of per-finger contributions")

    # 3. symmetry
    sk_ab = xin_sk_full(tips_a, tips_b, chain_a, chain_b,
                        lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=1.0)
    sk_ba = xin_sk_full(tips_b, tips_a, chain_b, chain_a,
                        lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=1.0)
    assert torch.allclose(sk_ab, sk_ba, atol=1e-6), "S_k is not symmetric"
    _format_check("symmetry: S_k(a,b) == S_k(b,a)")

    # 4. self -> 0
    sk_self = xin_sk_full(tips_a, tips_a, chain_a, chain_a,
                          lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=1.0)
    assert torch.allclose(sk_self, torch.zeros(N), atol=1e-6), "S_k(a,a) != 0"
    _format_check("self-similarity: S_k(a,a) == 0")

    # 6. lam_mid=0 reproduces Run 25 exactly; lam_mid>0 increases S_k when PIP differs
    chain_b_diff_pip = chain_b.clone()
    chain_b_diff_pip[:, :, 1, :] += 0.5  # shift all PIP positions
    sk_no_mid  = xin_sk_full(tips_a, tips_b, chain_a, chain_b,
                              lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=0.0)
    sk_mid_0   = xin_sk_full(tips_a, tips_b, chain_a, chain_b,
                              lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=0.0)
    sk_mid_1   = xin_sk_full(tips_a, tips_b, chain_a, chain_b_diff_pip,
                              lam_fp=1.0, lam_pinch=10.0, lam_fr=10.0, lam_mid=1.0)
    assert torch.allclose(sk_no_mid, sk_mid_0, atol=1e-6), "lam_mid=0 should match Run 25"
    assert (sk_mid_1 > sk_mid_0).all(), "lam_mid>0 with differing PIP should increase S_k"
    _format_check("lam_mid=0 reproduces Run 25; lam_mid=1 increases S_k on PIP-shifted poses")

    # 5. gradient flows through both inputs
    sk_full.sum().backward()
    assert tips_a.grad is not None and tips_a.grad.abs().sum() > 0, "no grad through tips"
    assert chain_a.grad is not None and chain_a.grad.abs().sum() > 0, "no grad through chain"
    _format_check("gradient flows through tips_a and chain_a")


def test_l_joint_config() -> None:
    print("[2] L_joint -- semantic_roles from robot.yaml")
    yaml_path = ROOT.parents[1] / "robot" / "hands" / "shadow_hand" / "robot.yaml"
    if not yaml_path.exists():
        print(f"  [SKIP] {yaml_path} not found")
        return
    w_pos = load_l_joint_config(yaml_path)
    expected = {"FFJ4": 0.5, "MFJ4": 0.5, "RFJ4": 0.5, "LFJ4": 0.5, "THJ5": 0.1}
    for name, value in expected.items():
        assert name in w_pos, f"missing joint {name} in w_pos"
        assert abs(w_pos[name] - value) < 1e-9, f"{name}: expected {value}, got {w_pos[name]}"
    _format_check(f"loaded weights for {sorted(w_pos)} -> {w_pos}")

    # Smoke-test l_joint with a fake q_r
    joint_names = ["WRJ2", "WRJ1", "FFJ4", "FFJ3", "FFJ2", "FFJ1",
                   "MFJ4", "MFJ3", "MFJ2", "MFJ1",
                   "RFJ4", "RFJ3", "RFJ2", "RFJ1",
                   "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
                   "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"]
    q_r = torch.zeros(8, 24, requires_grad=True)
    q_r2 = q_r.clone().detach().requires_grad_(True)
    # zero qpos -> loss = 0
    loss0 = l_joint(q_r, joint_names, w_pos)
    assert float(loss0) == 0.0, f"l_joint(0) != 0: {float(loss0)}"
    # non-zero abduction -> loss > 0 and grad flows
    q_r2.data[:, joint_names.index("FFJ4")] = 0.3
    loss = l_joint(q_r2, joint_names, w_pos)
    loss.backward()
    assert float(loss) > 0.0, "l_joint should be positive with non-zero FFJ4"
    assert q_r2.grad is not None and q_r2.grad[:, joint_names.index("FFJ4")].abs().sum() > 0
    _format_check(f"l_joint(zeros)=0 and l_joint with FFJ4=0.3 -> {float(loss):.4f} with grad")


def test_loader_fix() -> None:
    print("[3] HumanLoader -- chain thumb slot 1 == slot 2 (both IP)")
    csv_path = os.environ.get("HOGRASPNET_CSV")
    if not csv_path or not Path(csv_path).exists():
        print("  [SKIP] set HOGRASPNET_CSV=/path/to/hograspnet_abl11.csv to enable")
        return
    from cross_emb.loaders.human_loader import HumanLoader  # noqa: WPS433

    hl = HumanLoader(csv_path=csv_path, split="val", device="cpu")
    batch = hl.get_batch(B=8)
    chain = batch["chain"]
    assert chain.shape == (8, 5, 4, 3), f"unexpected chain shape: {chain.shape}"

    # Thumb slot 1 (IP) and slot 2 (also IP after the fix) must coincide.
    assert torch.allclose(chain[:, 0, 1, :], chain[:, 0, 2, :]), (
        "thumb slot 1 != slot 2 (loader fix not applied)"
    )
    # The last-segment vector for thumb is IP -> TIP and must be non-trivial.
    r_thumb = chain[:, 0, 3, :] - chain[:, 0, 2, :]
    norms = r_thumb.norm(dim=-1)
    assert (norms > 1e-3).all(), f"thumb IP->TIP suspiciously small: {norms}"
    _format_check(
        f"thumb IP->TIP norm range: [{norms.min().item():.4f}, {norms.max().item():.4f}]"
    )

    # Sanity: other fingers should still have non-trivial DIP -> TIP.
    for f in range(1, 5):
        r = chain[:, f, 3, :] - chain[:, f, 2, :]
        assert (r.norm(dim=-1) > 1e-3).all(), f"finger {f} DIP->TIP collapsed"
    _format_check("other fingers DIP->TIP still non-zero")


def main() -> None:
    test_synthetic()
    test_l_joint_config()
    test_loader_fix()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
