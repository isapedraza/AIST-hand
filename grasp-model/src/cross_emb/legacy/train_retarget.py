"""
Cross-embodiment retargeting training script.
Architecture: E_h (GAT) + E_X + D_X (MLP) + E_r + D_r (linear) + ShadowFK.
Losses: L_contrastive + L_rec + L_ltc + L_temporal (Yan & Lee 2026).
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from grasp_gcn.network.human_encoder import HumanEncoder
from grasp_gcn.network.cross_embodiment import CrossEmbodimentEncoder, CrossEmbodimentDecoder
from grasp_gcn.network.robot_embedding import make_shadow_hand_embedding
from grasp_gcn.network.shadow_fk import ShadowFK
from grasp_gcn.network.losses import (
    loss_ltc, loss_rec, loss_contrastive, loss_temporal
)
from grasp_gcn.dataset.retarget_dataset import RetargetDataset

# Loss weights from Yan & Lee 2026
LAMBDA_C    = 10.0
LAMBDA_REC  = 5.0
LAMBDA_LTC  = 1.0
LAMBDA_TEMP = 0.1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",       default="grasp-model/data/processed/hograspnet_retarget.csv")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch",     type=int,   default=256)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--z_dim",     type=int,   default=32)
    p.add_argument("--shared",    type=int,   default=256)
    p.add_argument("--workers",   type=int,   default=0)
    p.add_argument("--ckpt_dir",  default="grasp-model/checkpoints/retarget")
    p.add_argument("--save_every",type=int,   default=5)
    p.add_argument("--omega",     type=float, default=1.0,  help="D_ee weight in similarity")
    p.add_argument("--alpha",     type=float, default=0.05, help="triplet margin")
    return p.parse_args()


def normalize_human_tips(xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: [B, 21, 3] raw world coords
    Returns normalized fingertip positions [B, 5, 3]:
      wrist-relative, divided by hand_length = ||MCP_middle - wrist||
    Landmark indices: wrist=0, tips=[4,8,12,16,20], MCP_middle=9
    """
    wrist    = xyz[:, 0, :]                        # [B, 3]
    tips     = xyz[:, [4, 8, 12, 16, 20], :]      # [B, 5, 3]
    mcp_mid  = xyz[:, 9, :]                        # [B, 3]
    hand_len = (mcp_mid - wrist).norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, 1]
    tips_rel = tips - wrist.unsqueeze(1)           # [B, 5, 3]
    return tips_rel / hand_len.unsqueeze(1)        # [B, 5, 3]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Models
    E_h  = HumanEncoder(in_dim=4, hidden_dim=32, heads=4, z_dim=args.z_dim).to(device)
    E_X  = CrossEmbodimentEncoder(shared_dim=args.shared, z_dim=args.z_dim).to(device)
    D_X  = CrossEmbodimentDecoder(z_dim=args.z_dim, shared_dim=args.shared).to(device)
    E_r  = make_shadow_hand_embedding(shared_dim=args.shared).to(device)

    params = list(E_h.parameters()) + list(E_X.parameters()) + \
             list(D_X.parameters()) + list(E_r.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    fk = ShadowFK(device=device)

    # Dataset
    print("Building dataset...")
    dataset = RetargetDataset(args.csv, return_temporal=True)
    loader  = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, follow_batch=[]
    )
    print(f"  {len(dataset):,} pairs, {len(loader)} batches/epoch")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = total_lc = total_rec = total_ltc = total_temp = 0.0
        n_batches = 0

        for batch in loader:
            # --- Unpack batch ---
            graph_t  = batch["graph_t"].to(device)
            graph_t1 = batch["graph_t1"].to(device)
            xyz_t    = batch["xyz_t"].to(device)     # [B, 21, 3]
            xyz_t1   = batch["xyz_t1"].to(device)
            quats_t  = batch["quats_t"].to(device)   # [B, 20, 4]

            B = xyz_t.shape[0]

            # --- Human forward (frame t) ---
            z_h = E_h(graph_t.x, graph_t.edge_index, graph_t.batch)  # [B, z_dim]

            # --- L_ltc ---
            shared_h = D_X(z_h)                      # [B, shared_dim]
            z_rt     = E_X(shared_h)                 # [B, z_dim]
            l_ltc    = loss_ltc(z_h, z_rt)

            # --- Robot random sample ---
            qpos    = fk.sample(B)                   # [B, 24]
            r_tips_raw, r_quats = fk.forward(qpos)   # [B,5,3], [B,24,4]
            r_tips  = fk.normalize_tips(r_tips_raw)  # [B,5,3]

            # --- Robot encode ---
            shared_r = E_r.encode(qpos)              # [B, shared_dim]
            z_r      = E_X(shared_r)                 # [B, z_dim]

            # --- L_rec ---
            shared_r2 = D_X(z_r)                     # [B, shared_dim]
            qpos_hat  = E_r.decode(shared_r2)         # [B, 24]
            l_rec     = loss_rec(qpos, qpos_hat)

            # --- L_contrastive ---
            z_all   = torch.cat([z_h, z_r], dim=0)   # [2B, z_dim]
            h_tips  = normalize_human_tips(xyz_t)     # [B, 5, 3]
            l_c     = loss_contrastive(
                z_all, quats_t, r_quats, h_tips, r_tips,
                alpha=args.alpha, omega=args.omega
            )

            # --- L_temporal ---
            z_h_t1   = E_h(graph_t1.x, graph_t1.edge_index, graph_t1.batch)
            shared_h1 = D_X(z_h_t1)
            qpos_t1   = E_r.decode(shared_h1)         # retargeted robot pose at t+1
            tips_t_fk,  _ = fk.forward(E_r.decode(D_X(z_h)).detach())
            tips_t1_fk, _ = fk.forward(qpos_t1.detach())
            ee_t  = fk.normalize_tips(tips_t_fk)[:, 1, :]   # mftip
            ee_t1 = fk.normalize_tips(tips_t1_fk)[:, 1, :]
            l_temp = loss_temporal(xyz_t, xyz_t1, ee_t, ee_t1)

            # --- Total loss ---
            loss = LAMBDA_C * l_c + LAMBDA_REC * l_rec + \
                   LAMBDA_LTC * l_ltc + LAMBDA_TEMP * l_temp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_lc   += l_c.item()
            total_rec  += l_rec.item()
            total_ltc  += l_ltc.item()
            total_temp += l_temp.item()
            n_batches  += 1

        avg = lambda x: x / n_batches
        print(f"Epoch {epoch:03d} | loss={avg(total_loss):.4f} "
              f"Lc={avg(total_lc):.4f} Lrec={avg(total_rec):.4f} "
              f"Lltc={avg(total_ltc):.4f} Ltemp={avg(total_temp):.4f}")

        if epoch % args.save_every == 0:
            ckpt = ckpt_dir / f"retarget_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "E_h":  E_h.state_dict(),
                "E_X":  E_X.state_dict(),
                "D_X":  D_X.state_dict(),
                "E_r":  E_r.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt)
            print(f"  Saved {ckpt}")


if __name__ == "__main__":
    train(parse_args())
