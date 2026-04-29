import torch
import torch.nn as nn
import torch.nn.functional as F

# Node indices in the 21-node graph (0=wrist identity, 1-20=Dong joints per finger)
# Wrist (node 0) propagates info to all via CAM but has no projection head.
SUBSPACE_NODES: dict[str, list[int]] = {
    "thumb":     [1, 2, 3, 4],
    "precision": [5, 6, 7, 8, 9, 10, 11, 12],   # index + middle
    "support":   [13, 14, 15, 16, 17, 18, 19, 20],  # ring + pinky
}
SUBSPACE_FINGERS: dict[str, list[str]] = {
    "thumb":     ["thumb"],
    "precision": ["index", "middle"],
    "support":   ["ring", "pinky"],
}
SUBSPACE_LABEL_PREFIX: dict[str, tuple[str, ...]] = {
    "thumb":     ("thumb_",),
    "precision": ("index_", "middle_"),
    "support":   ("ring_", "pinky_"),
}


class CAMLayer(nn.Module):
    """
    Single CAM-GNN layer (Leng et al., IEEE VR 2021).

    Aggregation: f_agg[k] = sum_j( CAM[k,j] * x[j] )   (Eq. 3)
    Update:      x_out[k] = ELU( Linear( cat(f_agg[k], x[k]) ) )  (Eq. 2)

    x expected as [B*N, F]. Reshaped internally to [B, N, F] for einsum.
    """

    def __init__(self, in_features: int, out_features: int, n_joints: int):
        super().__init__()
        self.n_joints = n_joints
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        B_N, feat = x.shape
        N = self.n_joints
        B = B_N // N
        x_3d = x.view(B, N, feat)
        x_agg = torch.einsum('ij,bjf->bif', cam, x_3d)             # [B, N, feat]
        x_cat = torch.cat([x_agg, x_3d], dim=-1)                   # [B, N, 2*feat]
        return F.elu(self.linear(x_cat.view(B_N, 2 * feat)))       # [B*N, out]


class HumanEncoder_E_h(nn.Module):
    """
    E_h: encodes hand pose to decoupled latent subspaces via CAM-GNN.

    Input : quats [B, 20, 4]   Dong quaternions (joints 1-20, wrist prepended as identity)
    Output: z     [B, 3*z_dim] concat of [z_thumb, z_precision, z_support], each [-1,1]

    One shared CAM-GNN processes all 21 nodes. Three projection heads pool over
    their respective node subsets: thumb (1-4), precision/index+middle (5-12),
    support/ring+pinky (13-20). Wrist node 0 participates in CAM propagation only.
    """

    N_JOINTS = 21

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, z_dim: int = 64):
        super().__init__()
        self.cam    = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS).uniform_(-1, 1))
        self.layer1 = CAMLayer(in_dim,     hidden_dim, self.N_JOINTS)
        self.layer2 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.layer3 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.proj_thumb     = nn.Linear(hidden_dim, z_dim)
        self.proj_precision = nn.Linear(hidden_dim, z_dim)
        self.proj_support   = nn.Linear(hidden_dim, z_dim)
        self.out_act = nn.Tanh()

    def forward(self, quats: torch.Tensor) -> torch.Tensor:
        # quats: [B, 20, 4] -- prepend identity wrist node -> [B, 21, 4]
        B = quats.shape[0]
        wrist = torch.zeros(B, 1, 4, dtype=quats.dtype, device=quats.device)
        wrist[:, 0, 0] = 1.0
        quats = torch.cat([wrist, quats], dim=1)           # [B, 21, 4]
        B, N, in_f = quats.shape
        x = quats.reshape(B * N, in_f)
        x = self.layer1(x, self.cam)
        x = self.layer2(x, self.cam)
        x = self.layer3(x, self.cam)
        x = x.view(B, N, -1)                              # [B, 21, hidden]
        z_thumb = self.out_act(self.proj_thumb(
            x[:, SUBSPACE_NODES["thumb"], :].max(dim=1).values))
        z_prec  = self.out_act(self.proj_precision(
            x[:, SUBSPACE_NODES["precision"], :].max(dim=1).values))
        z_supp  = self.out_act(self.proj_support(
            x[:, SUBSPACE_NODES["support"], :].max(dim=1).values))
        return torch.cat([z_thumb, z_prec, z_supp], dim=-1)  # [B, 3*z_dim]
