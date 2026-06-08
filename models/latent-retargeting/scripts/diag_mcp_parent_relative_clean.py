#!/usr/bin/env python3
"""CLEAN re-run of the MCP parent-relative frame test (SIGUIENTE.md).

Why this exists
---------------
The previous test (diag_mcp_parent_relative.py) selected the robot "fist" by
filtering eigengrasp poses with q[:,[3,7,11,16]] > 1.2. That did NOT guarantee a
real fist: its PIP control came out at ~24 deg instead of ~90 deg, so the result
was inconclusive (dirty input, not a model finding).

This version uses the GOOD robot fist/open poses: the synthetic Shadow Hand qpos
NPZ (real fist by construction), runs MuJoCo FK to get the chain, and computes the
SAME parent-relative bend angles. A PIP control is printed FIRST: if robot PIP
closed ~ 90 deg, the fist is real and the test is valid.

Single question
---------------
Does the PARENT-RELATIVE MCP converge human ~ robot (like the PIP did)?
  - converges  -> the wrist-frame MCP mismatch was a FRAME artifact. Fix = MCP frame.
  - stays apart -> claim FALSE. Stop touching the model (SIGUIENTE.md exit).
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/yareeez/AIST-hand/models/latent-retargeting/src")
from cross_emb.loaders.robot_loader import RobotLoader, _dong_run_stage2, _load_hand_config
from cross_emb.loaders.human_loader import StaticHumanAnchorLoader

DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
URDF = "/home/yareeez/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"
HAND_CONFIG = "/home/yareeez/AIST-hand/robot/hands/shadow_hand/shadow_hand_right.yaml"

N = 1000
np.random.seed(0)
torch.manual_seed(0)


def fk_chain(qpos24, device):
    """qpos24 [N,24] -> chain [N,F,4,3] (wrist-local, /hand_length), tip_labels."""
    loader = RobotLoader(URDF, device=device)
    config = _load_hand_config(HAND_CONFIG)
    hand_length = loader._get_hand_length(config)
    q = torch.from_numpy(np.asarray(qpos24)).float().to(device)
    with torch.no_grad():
        fk = loader.run_fk(q)
        _, _, meta = _dong_run_stage2(fk, config)
        tip_labels = list(meta["tip_labels"])
        F = len(tip_labels)
        chain = torch.zeros((q.shape[0], F, 4, 3), device=device)
        for fi, f in enumerate(tip_labels):
            chain[:, fi] = meta["chain_positions"][f] / hand_length
    return chain.cpu(), tip_labels


def angle_between(u, v):  # [...,3] -> deg
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.rad2deg(torch.acos((u * v).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))


def parent_rel(chain, f):
    # chain[:,f] = [mcp,pip,dip,tip] positions, wrist=origin
    mcp = chain[:, f, 0]; pip = chain[:, f, 1]; dip = chain[:, f, 2]
    mcp_flex = angle_between(mcp - 0.0, pip - mcp)   # metacarpal(wrist->mcp) vs phalanx(mcp->pip)
    pip_flex = angle_between(pip - mcp, dip - pip)   # phalanx vs middle (comparable to human PIP)
    return mcp_flex.mean().item(), pip_flex.mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clean-mcp] device={device}")

    # --- ROBOT: GOOD fist/open from synthetic qpos (real fist by construction) ---
    sc = np.load(f"{DATA}/synthetic_close_hand_shadow_qpos.npz")
    so = np.load(f"{DATA}/synthetic_open_hand_shadow_qpos.npz")
    rc_idx = np.random.choice(sc["qpos24"].shape[0], N, replace=False)
    ro_idx = np.random.choice(so["qpos24"].shape[0], N, replace=False)
    chr_c, tlab = fk_chain(sc["qpos24"][rc_idx], device)
    chr_o, _    = fk_chain(so["qpos24"][ro_idx], device)
    print(f"[clean-mcp] robot tip_labels={tlab}")

    # --- HUMAN: fist (class 29) / open (class 28) chains ---
    hl = StaticHumanAnchorLoader(f"{DATA}/hagrid_dong.csv")
    print(f"[clean-mcp] human classes available: {hl._classes}")

    def hch(c):
        i = hl._class_indices[c]
        i = i[torch.randint(0, len(i), (N,))]
        return hl._chain[i].cpu()

    chh_c, chh_o = hch(29), hch(28)

    print("=" * 78)
    print("CONTROL FIRST (validity gate). Robot PIP closed must be ~90 deg.")
    print("If robot PIP closed is small (~24), the fist is FAKE -> test invalid.")
    print("-" * 78)
    print(f"{'finger':8} | {'PIP robot-clo':>13} {'PIP human-clo':>13} | {'PIP robot-open':>14} {'PIP human-open':>14}")
    for f, nm in [(1, "INDEX"), (2, "MIDDLE")]:
        _, rpip_c = parent_rel(chr_c, f); _, hpip_c = parent_rel(chh_c, f)
        _, rpip_o = parent_rel(chr_o, f); _, hpip_o = parent_rel(chh_o, f)
        print(f"{nm:8} | {rpip_c:13.1f} {hpip_c:13.1f} | {rpip_o:14.1f} {hpip_o:14.1f}")

    print("=" * 78)
    print("THE QUESTION. PARENT-RELATIVE MCP flexion (deg). MCP = bend wrist->mcp vs mcp->pip.")
    print("-" * 78)
    print(f"{'finger':8} | {'MCP robot-clo':>13} {'MCP human-clo':>13} | {'MCP robot-open':>14} {'MCP human-open':>14}")
    for f, nm in [(1, "INDEX"), (2, "MIDDLE")]:
        rmcp_c, _ = parent_rel(chr_c, f); hmcp_c, _ = parent_rel(chh_c, f)
        rmcp_o, _ = parent_rel(chr_o, f); hmcp_o, _ = parent_rel(chh_o, f)
        print(f"{nm:8} | {rmcp_c:13.1f} {hmcp_c:13.1f} | {rmcp_o:14.1f} {hmcp_o:14.1f}")
    print("-" * 78)
    print("Reference: wrist-frame Dong MCP gave robot~135 vs human~40 (the mismatch).")
    print("Converges (robot~human, closed) -> frame WAS the cause. Else -> claim FALSE.")


if __name__ == "__main__":
    main()
