"""Diagnostic: measure Xin S_k vs D_R magnitudes on one real batch.

Run 27B calibration. Loads real sampler, draws one batch, computes both
metrics on the same anchor/cand_a/cand_b triplet set, reports per-term
mean/std/min/max so we can pick lam_dr such that lam_dr*D_R is on the
same order as Xin S_k mean (or whatever scale the user prefers).

CPU-only. Loads from local data cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/yareeez/AIST-hand")
DEX_ROOT  = Path("/home/yareeez/dex-urdf")
DATA_ROOT = Path("/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed")

sys.path.insert(0, str(REPO_ROOT / "models/latent-retargeting/src"))

from cross_emb.loaders.sampler import CrossEmbodimentSampler
from cross_emb.training.losses import d_r_yan, xin_sk_full


def stats(name: str, t: torch.Tensor) -> None:
    print(f"  {name:24s} mean={t.mean().item():.4f}  std={t.std().item():.4f}  "
          f"min={t.min().item():.4f}  max={t.max().item():.4f}")


def main(B: int = 2000) -> None:
    csv_path  = DATA_ROOT / "hograspnet_abl11.csv"
    urdf_path = DEX_ROOT  / "robots/hands/shadow_hand/shadow_hand_right.urdf"
    yaml_path = REPO_ROOT / "robot/hands/shadow_hand/shadow_hand_right.yaml"
    npz_path  = DATA_ROOT / "valid_robot_poses_eigengrasp_dong.npz"

    print(f"[diag] csv  = {csv_path}")
    print(f"[diag] npz  = {npz_path}")
    print(f"[diag] B    = {B}")
    print()

    sampler = CrossEmbodimentSampler(
        csv_path         = csv_path,
        urdf_path        = urdf_path,
        hand_config_path = yaml_path,
        split            = "train",
        device           = "cpu",
        valid_poses_path = npz_path,
        extra_human_csv  = None,
        extra_human_ratio= 0.0,
    )

    batch = sampler.get_batch_temporal(B, seed=42)
    quats_h_sub = batch["quats_h_sub"]
    quats_r_sub = batch["quats_r_sub"]
    tips_h_sub  = batch["tips_h_sub"]
    tips_r_sub  = batch["tips_r_sub"]
    chain_h_sub = batch["chain_h_sub"]
    chain_r_sub = batch["chain_r_sub"]
    common_labels = batch["common_labels"]

    print(f"[diag] common_labels (J={len(common_labels)}): {common_labels}")
    print(f"[diag] quats_h_sub shape: {tuple(quats_h_sub.shape)}")
    print(f"[diag] tips_h_sub  shape: {tuple(tips_h_sub.shape)}")
    print(f"[diag] chain_h_sub shape: {tuple(chain_h_sub.shape)}")
    print()

    tips_all  = torch.cat([tips_h_sub,  tips_r_sub],  dim=0)
    chain_all = torch.cat([chain_h_sub, chain_r_sub], dim=0)
    quats_all = torch.cat([quats_h_sub, quats_r_sub], dim=0)
    B2 = tips_all.shape[0]
    n  = B2

    torch.manual_seed(0)
    anchors = torch.randperm(B2)[:n]
    cand_a  = torch.randint(0, B2 - 1, (n,))
    cand_a  = cand_a + (cand_a >= anchors).long()
    cand_b  = torch.randint(0, B2 - 1, (n,))
    cand_b  = cand_b + (cand_b >= anchors).long()
    same = cand_b == cand_a
    if same.any():
        cand_b[same] = (cand_b[same] + 1) % B2
        cand_b[same] += (cand_b[same] == anchors[same]).long()
        cand_b[same] %= B2

    tips_a   = tips_all[anchors]
    tips_ca  = tips_all[cand_a]
    tips_cb  = tips_all[cand_b]
    chain_a  = chain_all[anchors]
    chain_ca = chain_all[cand_a]
    chain_cb = chain_all[cand_b]
    quats_a  = quats_all[anchors]
    quats_ca = quats_all[cand_a]
    quats_cb = quats_all[cand_b]

    lam_fp, lam_pinch, lam_fr, lam_mid = 1.0, 10.0, 10.0, 1.0

    S_xin_a = xin_sk_full(tips_a, tips_ca, chain_a, chain_ca,
                          lam_fp=lam_fp, lam_pinch=lam_pinch,
                          lam_fr=lam_fr, lam_mid=lam_mid)
    S_xin_b = xin_sk_full(tips_a, tips_cb, chain_a, chain_cb,
                          lam_fp=lam_fp, lam_pinch=lam_pinch,
                          lam_fr=lam_fr, lam_mid=lam_mid)

    D_R_a = d_r_yan(quats_a, quats_ca)
    D_R_b = d_r_yan(quats_a, quats_cb)

    S_xin_pairs = torch.cat([S_xin_a, S_xin_b])
    D_R_pairs   = torch.cat([D_R_a, D_R_b])

    print(f"[diag] Xin S_k (lam_fp={lam_fp}, lam_pinch={lam_pinch}, "
          f"lam_fr={lam_fr}, lam_mid={lam_mid}):")
    stats("S_xin", S_xin_pairs)
    print()

    print(f"[diag] D_R (uniform sum over J={len(common_labels)} joints, no weights):")
    stats("D_R_raw", D_R_pairs)
    print()

    print("[diag] lam_dr suggestions (target = lam_dr*D_R relative to Xin S_k mean):")
    xin_mean = S_xin_pairs.mean().item()
    dr_mean  = D_R_pairs.mean().item()
    for ratio in (0.1, 0.3, 0.5, 1.0, 2.0):
        target = ratio * xin_mean
        lam_dr = target / dr_mean
        print(f"  contribute {ratio*100:5.1f}% of Xin S_k mean -> lam_dr = {lam_dr:.3f}  "
              f"(lam_dr*D_R_mean = {lam_dr*dr_mean:.3f})")


if __name__ == "__main__":
    main()
