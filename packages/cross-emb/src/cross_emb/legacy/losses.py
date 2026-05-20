import torch
import torch.nn.functional as F

# Mapping: Dong node index (0=wrist, 1-20=q1-q20) → Shadow quat index (0-23)
# Dong order: thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
# Shadow order per JOINT_NAMES: WRJ2(0),WRJ1(1),FF(2-5),MF(6-9),RF(10-13),LF(14-18),TH(19-23)
# THJ5(19) and LFJ5(14) have no Dong equivalent → skipped
HUMAN_QUAT_IDX = [  # indices into human quats [B, 20, 4] (Dong nodes 1-20, 0-indexed)
    0, 1, 2, 3,    # thumb  MCP,PIP,DIP,TIP → THJ4,THJ3,THJ2,THJ1
    4, 5, 6, 7,    # index  MCP,PIP,DIP,TIP → FFJ4,FFJ3,FFJ2,FFJ1
    8, 9, 10, 11,  # middle MCP,PIP,DIP,TIP → MFJ4,MFJ3,MFJ2,MFJ1
    12, 13, 14, 15,# ring   MCP,PIP,DIP,TIP → RFJ4,RFJ3,RFJ2,RFJ1
    16, 17, 18, 19,# pinky  MCP,PIP,DIP,TIP → LFJ4,LFJ3,LFJ2,LFJ1
]
ROBOT_QUAT_IDX = [  # indices into robot quats [B, 24, 4]
    20, 21, 22, 23,  # THJ4,THJ3,THJ2,THJ1
    2, 3, 4, 5,      # FFJ4,FFJ3,FFJ2,FFJ1
    6, 7, 8, 9,      # MFJ4,MFJ3,MFJ2,MFJ1
    10, 11, 12, 13,  # RFJ4,RFJ3,RFJ2,RFJ1
    15, 16, 17, 18,  # LFJ4,LFJ3,LFJ2,LFJ1 (skip LFJ5=14)
]


def loss_ltc(z_h: torch.Tensor, z_rt: torch.Tensor) -> torch.Tensor:
    """
    Latent consistency loss (Yan & Lee 2026, Eq. 7).

    Measures discrepancy between original human latent z_h and
    the re-encoded version after round-trip through D_X -> E_X.

    Args:
        z_h  : E_h(x_H)              [B, z_dim]
        z_rt : E_X(D_X(E_h(x_H)))   [B, z_dim]

    Usage:
        z_h  = E_h(x, edge_index, batch)
        z_rt = E_X(D_X(z_h))
        loss = loss_ltc(z_h, z_rt)
    """
    return (z_h - z_rt).norm(dim=-1).mean()


def loss_rec(qpos: torch.Tensor, qpos_hat: torch.Tensor) -> torch.Tensor:
    """
    Robot reconstruction loss (Yan & Lee 2026, Eq. 6).

    Ensures the robot autoencoder (E_r -> E_X -> D_X -> D_r)
    faithfully reconstructs the input joint configuration.

    Args:
        qpos     : original robot joint angles  [B, n_joints]
        qpos_hat : reconstructed joint angles   [B, n_joints]

    Usage:
        shared   = E_r.encode(qpos)
        z_r      = E_X(shared)
        shared2  = D_X(z_r)
        qpos_hat = E_r.decode(shared2)
        loss     = loss_rec(qpos, qpos_hat)
    """
    return (qpos - qpos_hat).norm(dim=-1).mean()


def _similarity(
    h_quats: torch.Tensor,   # [Bh, 20, 4] Dong quats, wrist-local
    r_quats: torch.Tensor,   # [Br, 24, 4] Shadow FK quats, palm-relative
    h_tips: torch.Tensor,    # [Bh, 5, 3]  human fingertips, wrist-relative normalized
    r_tips: torch.Tensor,    # [Br, 5, 3]  robot fingertips, palm-relative normalized
    omega: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise similarity S = D_R + omega * D_ee for all (human, robot) pairs.
    Lower = more similar.

    Returns S [Bh + Br, Bh + Br] — full pairwise matrix over mixed batch.
    """
    Bh = h_quats.shape[0]
    Br = r_quats.shape[0]
    B = Bh + Br

    # --- D_R: select 20 comparable joints ---
    hq = h_quats[:, HUMAN_QUAT_IDX, :]                    # [Bh, 20, 4]
    rq = r_quats[:, ROBOT_QUAT_IDX, :].to(hq.device)     # [Br, 20, 4]
    all_q = torch.cat([hq, rq], dim=0)                    # [B, 20, 4]

    # D_R(i,j) = sum_k (1 - <q_i^k, q_j^k>^2)
    dot = (all_q.unsqueeze(1) * all_q.unsqueeze(0)).sum(-1)  # [B, B, 20]
    D_R = (1 - dot ** 2).sum(-1)                             # [B, B]

    # --- D_ee ---
    h_t = h_tips.view(Bh, -1)                    # [Bh, 15]
    r_t = r_tips.view(Br, -1).to(h_t.device)    # [Br, 15]
    all_t = torch.cat([h_t, r_t], dim=0)         # [B, 15]
    # D_ee(i,j) = ||tips_i - tips_j||
    diff = all_t.unsqueeze(1) - all_t.unsqueeze(0)  # [B, B, 15]
    D_ee = diff.norm(dim=-1)                         # [B, B]

    return D_R + omega * D_ee  # [B, B]


def loss_contrastive(
    z: torch.Tensor,          # [B, z_dim]  all latents (human + robot concatenated)
    h_quats: torch.Tensor,    # [Bh, 20, 4]
    r_quats: torch.Tensor,    # [Br, 24, 4]
    h_tips: torch.Tensor,     # [Bh, 5, 3]
    r_tips: torch.Tensor,     # [Br, 5, 3]
    alpha: float = 0.05,
    omega: float = 1.0,
) -> torch.Tensor:
    """
    Triplet loss in latent space with similarity-based triplet mining.

    For each sample i (anchor), find:
      positive j* = argmin_{j≠i} S(i,j)
      negative k* = argmax_{j≠i} S(i,j)  [hardest negative in batch]

    Loss = mean over anchors of max(||z_i - z_j*|| - ||z_i - z_k*|| + alpha, 0)

    Args:
        z       : [Bh+Br, z_dim] latents, human first then robot
        h_quats : [Bh, 20, 4] Dong quats (nodes 1-20, wrist-local)
        r_quats : [Br, 24, 4] Shadow FK quats (palm-relative)
        h_tips  : [Bh, 5, 3] human fingertip positions (wrist-relative, normalized)
        r_tips  : [Br, 5, 3] robot fingertip positions (palm-relative, normalized)
    """
    B = z.shape[0]
    S = _similarity(h_quats, r_quats, h_tips, r_tips, omega)  # [B, B]

    # Mask diagonal (self-similarity)
    mask = torch.eye(B, dtype=torch.bool, device=z.device)
    S_masked = S.masked_fill(mask, float('inf'))

    # Positive: most similar (smallest S)
    pos_idx = S_masked.argmin(dim=1)   # [B]
    # Negative: most dissimilar (largest S), but not inf
    S_neg = S.masked_fill(mask, float('-inf'))
    neg_idx = S_neg.argmax(dim=1)      # [B]

    # Distances in latent space
    z_anchor = z                        # [B, z_dim]
    z_pos    = z[pos_idx]               # [B, z_dim]
    z_neg    = z[neg_idx]               # [B, z_dim]

    d_pos = (z_anchor - z_pos).norm(dim=-1)   # [B]
    d_neg = (z_anchor - z_neg).norm(dim=-1)   # [B]

    return F.relu(d_pos - d_neg + alpha).mean()


def loss_temporal(
    xyz_t: torch.Tensor,   # [B, 21, 3] human raw XYZ at frame t
    xyz_t1: torch.Tensor,  # [B, 21, 3] human raw XYZ at frame t+1
    ee_t: torch.Tensor,    # [B, 3] robot middle fingertip at t  (palm-relative, normalized)
    ee_t1: torch.Tensor,   # [B, 3] robot middle fingertip at t+1
) -> torch.Tensor:
    """
    Temporal consistency loss (Yan & Lee 2026, Eq. 9).

    Aligns velocity of human middle fingertip with robot EE velocity.
    Both normalized by hand_length for scale consistency.

    Args:
        xyz_t, xyz_t1 : consecutive human frames [B, 21, 3], raw world coords
        ee_t, ee_t1   : robot middle fingertip (mftip) positions [B, 3],
                        palm-relative and normalized by robot hand_length
                        → from ShadowFK.forward() + normalize_tips()

    Usage:
        tips_t,  _ = fk.forward(qpos_t)
        tips_t1, _ = fk.forward(qpos_t1)
        ee_t  = fk.normalize_tips(tips_t,  qpos_t)[:, 1, :]   # mftip
        ee_t1 = fk.normalize_tips(tips_t1, qpos_t1)[:, 1, :]
        loss  = loss_temporal(xyz_t, xyz_t1, ee_t, ee_t1)
    """
    # Human middle fingertip velocity (landmark 12 = middle tip), wrist-relative
    wrist_t  = xyz_t[:, 0, :]           # [B, 3]
    wrist_t1 = xyz_t1[:, 0, :]
    mcp_mid  = xyz_t[:, 9, :]           # landmark 9 = middle MCP

    hand_len = (mcp_mid - wrist_t).norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, 1]

    tip_t_rel  = (xyz_t[:, 12, :]  - wrist_t)  / hand_len   # [B, 3]
    tip_t1_rel = (xyz_t1[:, 12, :] - wrist_t1) / hand_len   # [B, 3]

    v_human = tip_t1_rel - tip_t_rel    # [B, 3]
    v_robot = (ee_t1 - ee_t).to(v_human.device)  # [B, 3]  already normalized

    return (v_human - v_robot).norm(dim=-1).mean()
