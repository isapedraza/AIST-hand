"""
diagnose_latent_alignment.py

Pregunta central: ¿esta z_H(puno) cerca de z_R(puno) en el espacio latente?

Si la distancia es grande -> el alignment fallo.
  Causa posible: margin=0.1 satura L_cont antes de que z_H llegue a z_R.
  Causa posible: E_h no tiene capacidad para codificar info de puno.

Si la distancia es pequena -> z_H y z_R estan alineados pero D_r no generaliza
  desde la distribucion del camino humano.

Metricas reportadas:
  d_pos      : ||z_H(puno) - z_R(puno)||_2      (deberia ser pequena)
  d_neg      : ||z_H(puno) - z_R(abierta)||_2   (deberia ser grande)
  d_gap      : d_neg - d_pos                     (vs margin=0.1)
  d_ref_r    : ||z_R(puno1) - z_R(puno2)||_2    (escala natural: fist-fist robot)
  d_ref_rr   : ||z_R(puno) - z_R(abierta)||_2   (escala: fist-open robot)
  d_intra_H  : ||z_H(puno1) - z_H(puno2)||_2    (discrimina E_h poses de puno?)

Uso:
  python scripts/diagnose_latent_alignment.py \
      --ckpt ~/Downloads/stage1_best_run20.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cross_emb.robot_modules import RobotEncoder_E_r
from cross_emb.shared_modules import SharedEncoder_E_X
from cross_emb.human_modules import HumanEncoder_E_h
from human_loader import _QUAT_COLS

HAGRID_CSV = Path("/media/yareeez/94649A33649A1856/DATOS DE TESIS/"
                  "AIST-hand_grasp-model_data_cache/processed/hagrid_dong.csv")
ROBOT_NPZ  = Path("/media/yareeez/94649A33649A1856/DATOS DE TESIS/"
                  "AIST-hand_grasp-model_data_cache/processed/"
                  "valid_robot_poses_eigengrasp_dong.npz")

MCP_FLEX_IDX = [3, 7, 11, 16]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("config", ck.get("args", {}))
    z_dim = cfg.get("z_dim", 16) if isinstance(cfg, dict) else getattr(cfg, "z_dim", 16)
    shared_dim = cfg.get("shared_dim", 1024) if isinstance(cfg, dict) else getattr(cfg, "shared_dim", 1024)
    n_joints = ck["E_r"]["fc.weight"].shape[1]

    E_r = RobotEncoder_E_r(n_joints, shared_dim).to(device)
    E_X = SharedEncoder_E_X(shared_dim, z_dim).to(device)
    E_h = HumanEncoder_E_h(z_dim=z_dim).to(device)

    E_r.load_state_dict(ck["E_r"])
    E_X.load_state_dict(ck["E_X"])
    E_h.load_state_dict(ck["E_h"])

    for m in [E_r, E_X, E_h]:
        m.eval()

    step = ck.get("step", "?")
    print(f"  Checkpoint: step={step}, z_dim={z_dim}, n_joints={n_joints}")
    return E_r, E_X, E_h, z_dim


def load_human(grasp_type, n):
    df = pd.read_csv(HAGRID_CSV)
    sub = df[df["grasp_type"] == grasp_type]
    idx = np.random.choice(len(sub), min(n, len(sub)), replace=False)
    quats = sub[_QUAT_COLS].values[idx].astype(np.float32).reshape(-1, 20, 4)
    return torch.tensor(quats)  # [N, 20, 4]


def load_robot(mcp_min=None, mcp_max=None, n=200):
    npz = np.load(ROBOT_NPZ)
    q = npz["q"]  # [10M, 24] radians
    mcp = q[:, MCP_FLEX_IDX].mean(axis=1)
    mask = np.ones(len(q), dtype=bool)
    if mcp_min is not None:
        mask &= (mcp >= mcp_min)
    if mcp_max is not None:
        mask &= (mcp <= mcp_max)
    idx_pool = np.where(mask)[0]
    idx = np.random.choice(idx_pool, min(n, len(idx_pool)), replace=False)
    q_sel = q[idx].copy().astype(np.float32)
    q_sel[:, 0:2] = 0.0  # zero WRJ2, WRJ1
    mcp_sel = mcp[idx]
    return torch.tensor(q_sel), mcp_sel  # [N, 24]


@torch.no_grad()
def get_z_human(quats, E_h):
    """z_H = E_h(quats): [N, z_dim*5]"""
    return E_h(quats)


@torch.no_grad()
def get_z_robot(q_r, E_r, E_X):
    """z_R = E_X(E_r(q_r)): [N, z_dim*5]"""
    return E_X(E_r(q_r))


def mean_l2(A, B):
    """Mean pairwise L2: for each row in A, average distance to all rows in B."""
    # A: [Na, D], B: [Nb, D]
    # returns scalar (mean over Na of min or mean over Nb)
    diffs = A.unsqueeze(1) - B.unsqueeze(0)  # [Na, Nb, D]
    dists = diffs.norm(dim=-1)               # [Na, Nb]
    return dists.mean().item(), dists.min(dim=1).values.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.1,
                    help="Margin used in training (for reference)")
    args = ap.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")

    print("\n=== Latent alignment diagnostic ===\n")
    print("Loading checkpoint...")
    E_r, E_X, E_h, z_dim = load_model(args.ckpt, device)
    Z_DIM_TOTAL = z_dim * 5

    print("\nLoading human poses...")
    q_h_fist = load_human(grasp_type=29, n=args.n)
    q_h_open = load_human(grasp_type=28, n=args.n)
    print(f"  Fist (class=29): {len(q_h_fist)}")
    print(f"  Open (class=28): {len(q_h_open)}")

    print("\nLoading robot poses...")
    q_r_fist, mcp_fist = load_robot(mcp_min=1.0, n=args.n)
    q_r_open, mcp_open = load_robot(mcp_max=0.3, n=args.n)
    print(f"  Robot fist (MCP>=1.0): {len(q_r_fist)}, mean MCP={mcp_fist.mean()*57.3:.1f} deg")
    print(f"  Robot open (MCP<=0.3): {len(q_r_open)}, mean MCP={mcp_open.mean()*57.3:.1f} deg")

    print("\nComputing latent embeddings...")
    z_H_fist = get_z_human(q_h_fist, E_h)   # [N, 80]
    z_H_open = get_z_human(q_h_open, E_h)
    z_R_fist = get_z_robot(q_r_fist, E_r, E_X)
    z_R_open = get_z_robot(q_r_open, E_r, E_X)

    print(f"  z dim total: {Z_DIM_TOTAL}")
    print(f"  z_H_fist norm (mean): {z_H_fist.norm(dim=-1).mean():.3f}")
    print(f"  z_R_fist norm (mean): {z_R_fist.norm(dim=-1).mean():.3f}")

    print("\n" + "="*55)
    print("DISTANCES (L2 in latent space, dim={})".format(Z_DIM_TOTAL))
    print("="*55)

    d_pos_mean, d_pos_min = mean_l2(z_H_fist, z_R_fist)
    d_neg_mean, d_neg_min = mean_l2(z_H_fist, z_R_open)
    d_ref_rr_mean, _      = mean_l2(z_R_fist, z_R_fist)   # intra-robot fist
    d_ref_ro_mean, _      = mean_l2(z_R_fist, z_R_open)   # robot fist vs open
    d_intra_H, _          = mean_l2(z_H_fist, z_H_fist)   # intra-human fist

    gap = d_neg_mean - d_pos_mean

    print(f"\n  d_pos  (z_H_fist vs z_R_fist)  : {d_pos_mean:.4f}  [mean], {d_pos_min:.4f} [mean-of-min]")
    print(f"  d_neg  (z_H_fist vs z_R_open)  : {d_neg_mean:.4f}  [mean], {d_neg_min:.4f} [mean-of-min]")
    print(f"  d_gap  = d_neg - d_pos          : {gap:+.4f}  (margin in training = {args.margin})")
    print(f"\n  d_ref_rr (z_R_fist vs z_R_fist): {d_ref_rr_mean:.4f}  <- escala natural robot fist")
    print(f"  d_ref_ro (z_R_fist vs z_R_open): {d_ref_ro_mean:.4f}  <- separacion robot fist/open")
    print(f"  d_intra_H (z_H_fist vs z_H_fist): {d_intra_H:.4f}  <- varianza interna E_h fist")

    print("\n" + "="*55)
    print("INTERPRETATION")
    print("="*55)
    print(f"\n  d_pos / d_ref_rr = {d_pos_mean/d_ref_rr_mean:.2f}x")
    print(f"    -> 1.0x: z_H_fist tan cerca de z_R_fist como dos robots fist entre si")
    print(f"    -> >>1x: z_H_fist LEJOS de z_R_fist -- alignment fallo")
    print(f"\n  d_gap ({gap:.4f}) vs margin ({args.margin}):")
    if gap > args.margin * 5:
        print(f"    -> d_gap >> margin: L_cont gradient = 0 desde pasos tempranos.")
        print(f"       margin={args.margin} trivialmente satisfecho.")
        print(f"       Subir margin a ~{gap*0.5:.2f} mantendria L_cont activo.")
    elif gap > args.margin:
        print(f"    -> d_gap > margin: L_cont satisfecho (gradient=0 para estos pares).")
    else:
        print(f"    -> d_gap <= margin: L_cont activo pero no logra separar.")


if __name__ == "__main__":
    main()
