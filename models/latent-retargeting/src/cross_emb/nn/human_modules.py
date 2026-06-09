import torch
import torch.nn as nn
import torch.nn.functional as F

# Node indices in the 21-node graph (0=wrist identity, 1-20=Dong joints per finger)
# Wrist (node 0) propagates info to all via CAM but has no projection head.
SUBSPACE_NODES: dict[str, list[int]] = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
SUBSPACE_FINGERS: dict[str, list[str]] = {
    "thumb":  ["thumb"],
    "index":  ["index"],
    "middle": ["middle"],
    "ring":   ["ring"],
    "pinky":  ["pinky"],
}
SUBSPACE_LABEL_PREFIX: dict[str, tuple[str, ...]] = {
    "thumb":  ("thumb_",),
    "index":  ("index_",),
    "middle": ("middle_",),
    "ring":   ("ring_",),
    "pinky":  ("pinky_",),
}


def _identity_wrist_feature(B: int, in_f: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Identity wrist node for quaternion or R6 hand-pose features."""
    wrist = torch.zeros(B, 1, in_f, dtype=dtype, device=device)
    if in_f == 4:
        wrist[:, 0, 0] = 1.0
    elif in_f == 6:
        wrist[:, 0, 0] = 1.0
        wrist[:, 0, 4] = 1.0
    else:
        raise ValueError(f"Unsupported human pose feature dim: {in_f}")
    return wrist


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

    Input : pose  [B, 20, F]   Dong quats (F=4) or R6 rotations (F=6)
    Output: z     [B, 5*z_dim] concat of one latent per finger, each [-1,1]

    One shared CAM-GNN processes all 21 nodes. Five projection heads pool over
    their respective finger node subsets. Wrist node 0 participates in CAM
    propagation only.
    """

    N_JOINTS = 21

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, z_dim: int = 64):
        super().__init__()
        if in_dim not in (4, 6):
            raise ValueError(f"in_dim must be 4 (quat) or 6 (R6), got {in_dim}")
        self.cam    = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS).uniform_(-1, 1))
        self.layer1 = CAMLayer(in_dim,     hidden_dim, self.N_JOINTS)
        self.layer2 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.layer3 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.proj_thumb  = nn.Linear(hidden_dim, z_dim)
        self.proj_index  = nn.Linear(hidden_dim, z_dim)
        self.proj_middle = nn.Linear(hidden_dim, z_dim)
        self.proj_ring   = nn.Linear(hidden_dim, z_dim)
        self.proj_pinky  = nn.Linear(hidden_dim, z_dim)
        self.out_act = nn.Tanh()

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        # pose: [B, 20, F] -- prepend identity wrist node -> [B, 21, F]
        B = pose.shape[0]
        wrist = _identity_wrist_feature(B, pose.shape[-1], pose.dtype, pose.device)
        pose = torch.cat([wrist, pose], dim=1)
        B, N, in_f = pose.shape
        x = pose.reshape(B * N, in_f)
        x = self.layer1(x, self.cam)
        x = self.layer2(x, self.cam)
        x = self.layer3(x, self.cam)
        x = x.view(B, N, -1)                              # [B, 21, hidden]
        z_thumb  = self.out_act(self.proj_thumb( x[:, SUBSPACE_NODES["thumb"],  :].max(dim=1).values))
        z_index  = self.out_act(self.proj_index( x[:, SUBSPACE_NODES["index"],  :].max(dim=1).values))
        z_middle = self.out_act(self.proj_middle(x[:, SUBSPACE_NODES["middle"], :].max(dim=1).values))
        z_ring   = self.out_act(self.proj_ring(  x[:, SUBSPACE_NODES["ring"],   :].max(dim=1).values))
        z_pinky  = self.out_act(self.proj_pinky( x[:, SUBSPACE_NODES["pinky"],  :].max(dim=1).values))
        return torch.cat([z_thumb, z_index, z_middle, z_ring, z_pinky], dim=-1)  # [B, 5*z_dim]


class HumanEncoder_E_h_single(nn.Module):
    """E_h variant with a SINGLE projection head over the whole hand (Run 25+).

    Same 3x CAM-GNN backbone as HumanEncoder_E_h. Replaces the 5 per-finger
    projection heads with a single Sequential(Linear -> Tanh) that operates on
    a global max-pool of nodes 1-20 (joints, wrist excluded). Output is a
    unified latent of size z_dim_total instead of the 5-way decoupled
    concatenation.
    """

    N_JOINTS = 21

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, z_dim_total: int = 320):
        super().__init__()
        if in_dim not in (4, 6):
            raise ValueError(f"in_dim must be 4 (quat) or 6 (R6), got {in_dim}")
        self.cam    = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS).uniform_(-1, 1))
        self.layer1 = CAMLayer(in_dim,     hidden_dim, self.N_JOINTS)
        self.layer2 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.layer3 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.proj_hand = nn.Sequential(
            nn.Linear(hidden_dim, z_dim_total),
            nn.Tanh(),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        B = pose.shape[0]
        wrist = _identity_wrist_feature(B, pose.shape[-1], pose.dtype, pose.device)
        pose = torch.cat([wrist, pose], dim=1)
        B, N, in_f = pose.shape
        x = pose.reshape(B * N, in_f)
        x = self.layer1(x, self.cam)
        x = self.layer2(x, self.cam)
        x = self.layer3(x, self.cam)
        x = x.view(B, N, -1)                              # [B, 21, hidden]
        x_pool = x[:, 1:, :].max(dim=1).values             # [B, hidden]
        return self.proj_hand(x_pool)                      # [B, z_dim_total]
