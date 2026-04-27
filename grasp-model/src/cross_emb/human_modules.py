import torch
import torch.nn as nn
import torch.nn.functional as F


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
    E_h: encodes hand pose to latent z using CAM-GNN (Leng et al. 2021).

    Input : quats [B, 21, 4]  Dong quaternions (wrist node = identity)
    Output: z     [B, z_dim]  bounded in [-1, 1] via Tanh

    Architecture:
      Shared CAM [21, 21] -- learned adjacency, init uniform(-1, 1), no normalization
      3 CAMLayer: in_dim -> hidden_dim -> hidden_dim -> hidden_dim
      global max pool over joints -> Linear -> Tanh -> z
    """

    N_JOINTS = 21

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, z_dim: int = 16):
        super().__init__()
        self.cam = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS).uniform_(-1, 1))
        self.layer1 = CAMLayer(in_dim,     hidden_dim, self.N_JOINTS)
        self.layer2 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.layer3 = CAMLayer(hidden_dim, hidden_dim, self.N_JOINTS)
        self.proj    = nn.Linear(hidden_dim, z_dim)
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
        x = x.view(B, N, -1).max(dim=1).values   # global max pool: [B, hidden]
        return self.out_act(self.proj(x))
