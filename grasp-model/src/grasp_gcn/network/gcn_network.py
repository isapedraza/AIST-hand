import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool


# =============================================================================
# CAM-GNN (Leng et al., IEEE VR 2021)
# =============================================================================

class CAMLayer(nn.Module):
    """
    Single CAM-GNN message passing layer.

    Aggregation: f_agg_k = sum_j( CAM_kj * x_j )   -- dense, learned topology
    Update:      x_out   = ELU( Linear( cat(f_agg, x_in) ) )

    x is expected in batched PyG format [B*N, F]. It is reshaped internally
    to [B, N, F] for the einsum, then flattened back.
    """

    def __init__(self, in_features: int, out_features: int, n_joints: int = 21):
        super().__init__()
        self.n_joints = n_joints
        # Input to linear is cat(aggregated, original) -> 2 * in_features
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, cam: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        B_N, feat_dim = x.shape
        N = self.n_joints
        B = B_N // N

        x_3d = x.view(B, N, feat_dim)                         # [B, N, F]
        if mask is not None:
            x_3d = x_3d * mask.view(B, N, 1)                  # zero out missing joints
        x_agg = torch.einsum('ij,bjf->bif', cam, x_3d)        # [B, N, F]
        x_cat = torch.cat([x_agg, x_3d], dim=-1)              # [B, N, 2F]
        x_out = self.linear(x_cat.view(B_N, 2 * feat_dim))    # [B*N, F_out]
        return F.elu(x_out)


class GCN_CAM_8_8_16_16_32(nn.Module):
    """
    GCN with a learnable Constraint Adjacency Matrix (CAM) replacing the fixed
    hand-skeleton edge_index.

    Architecture mirrors GCN_8_8_16_16_32 (5 layers: 8-8-16-16-32, ELU,
    mean+max readout, optional θ_CMC concatenation) but message passing uses
    a single shared CAM ∈ R^(21×21) instead of GCNConv with a fixed skeleton.

    CAM is initialized uniform(-1, 1) and optimized end-to-end with Adam.
    Negative entries are inhibitory; positive are excitatory. No normalization
    is applied to CAM, matching the original formulation. A single shared matrix
    is used across all 5 layers for interpretability and parameter efficiency
    (441 parameters vs 2205 for per-layer CAMs).

    The hand skeleton edge_index is still present in the Data object (generated
    by ToGraph) but is not used in the forward pass -- it is preserved for
    visualization and potential future use.

    Reference:
        Leng et al. "Stable Hand Pose Estimation under Tremor via Graph Neural
        Network." IEEE VR 2021. https://doi.org/10.1109/VR50410.2021.00044
    """

    N_JOINTS = 21

    def __init__(self, numFeatures: int, numClasses: int,
                 use_cmc_angle: bool = False):
        super().__init__()
        self.numClasses    = numClasses
        self.use_cmc_angle = use_cmc_angle

        # Shared learnable adjacency matrix [21, 21]
        self.cam = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS))
        nn.init.uniform_(self.cam, -1.0, 1.0)

        # 5 CAM layers mirroring GCN_8_8_16_16_32
        self.layer1 = CAMLayer(numFeatures, 8,  self.N_JOINTS)
        self.layer2 = CAMLayer(8,           8,  self.N_JOINTS)
        self.layer3 = CAMLayer(8,           16, self.N_JOINTS)
        self.layer4 = CAMLayer(16,          16, self.N_JOINTS)
        self.layer5 = CAMLayer(16,          32, self.N_JOINTS)

        # mean+max readout -> 64; +1 if θ_CMC -> 65
        readout_dim = 64 + (1 if use_cmc_angle else 0)
        self.fc1 = nn.Linear(readout_dim, 128)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, data):
        x     = data.x                                          # [B*21, F]
        batch = getattr(data, 'batch',
                        torch.zeros(x.size(0), dtype=torch.long,
                                    device=x.device))
        mask  = getattr(data, 'mask', None)                    # [B*21, 1]

        x = self.layer1(x, self.cam, mask)
        x = self.layer2(x, self.cam, mask)
        x = self.layer3(x, self.cam, mask)
        x = self.layer4(x, self.cam, mask)
        x = self.layer5(x, self.cam, mask)                     # [B*21, 32]

        h = masked_readout(x, batch, mask)                     # [B, 64]

        if self.use_cmc_angle:
            theta = data.theta_cmc.view(-1, 1)                 # [B, 1]
            h = torch.cat([h, theta], dim=1)                   # [B, 65]

        h   = F.elu(self.fc1(h))                               # [B, 128]
        out = self.fc2(h)                                      # [B, numClasses]
        return F.log_softmax(out, dim=1)

def masked_readout(x, batch, mask=None):
    """
    x:     [N, C]
    batch: [N]  (índices de grafo)
    mask:  [N, 1] con 1.0 si el nodo es válido, 0.0 si faltante (opcional)
    Devuelve [B, 2C] = concat(mean, max)
    """
    if mask is None:
        mean = global_mean_pool(x, batch)      # [B, C]
        mx   = global_max_pool(x, batch)       # [B, C]
        return torch.cat([mean, mx], dim=1)    # [B, 2C]

    # Enmascarado: apaga nodos faltantes y normaliza mean por #nodos válidos
    x_masked = x * mask                        # [N, C]
    sum_x = global_add_pool(x_masked, batch)   # [B, C]
    cnt   = global_add_pool(mask, batch)       # [B, 1]
    mean  = sum_x / cnt.clamp(min=1.0)         # [B, C]
    mx    = global_max_pool(x_masked, batch)   # [B, C]
    return torch.cat([mean, mx], dim=1)        # [B, 2C]

# =============================================================================
# CAM-GAT (CAM-GNN + Graph Attention Network hybrid)
# CAM-GNN: Leng et al., IEEE VR 2021
# GAT:     Veličković et al., ICLR 2018
# =============================================================================

class CAMGATLayer(nn.Module):
    """
    CAM-GAT hybrid message passing layer.

    Combines CAM-GNN (Leng et al., 2021) with GAT (Veličković et al., 2018):

        h_tilde_i = W1 * x_i                                     [GAT Eq. 1]
        e_ij      = LeakyReLU( a^T [h_tilde_i || h_tilde_j] )   [GAT Eq. 3]
        beta_ij   = softmax_j(e_ij)                               [GAT Eq. 2]
        alpha_ij  = CAM_ij * beta_ij                              [CAM-GNN Eq. 3]
        x'_i      = ELU( W2 [ sum_j alpha_ij * h_tilde_j || x_i ] )

    CAM provides the global structural prior (signed, learned, shared across layers).
    GAT provides per-sample dynamic modulation of attention weights.
    Skip connection [x_agg || x_in] follows CAM-GNN update rule and guards
    against over-smoothing over 5 layers.

    proj_dim (Fp): attention projection dimension, defaults to out_features.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_joints: int = 21, proj_dim: int = None):
        super().__init__()
        self.n_joints = n_joints
        Fp = proj_dim if proj_dim is not None else out_features
        self.Fp = Fp

        # W1: project to attention space [GAT Eq. 1]
        self.W1 = nn.Linear(in_features, Fp, bias=False)
        # a: attention vector [GAT Eq. 3]
        self.a  = nn.Parameter(torch.empty(2 * Fp))
        nn.init.xavier_uniform_(self.a.view(1, -1))
        # W2: final transform with skip connection [CAM-GNN Eq. 2]
        self.W2 = nn.Linear(Fp + in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, cam: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        B_N, in_feat = x.shape
        N = self.n_joints
        B = B_N // N

        x_3d = x.view(B, N, in_feat)                      # [B, N, F]
        if mask is not None:
            x_3d = x_3d * mask.view(B, N, 1)

        # Step 1: project to attention space [GAT Eq. 1]
        h = self.W1(x_3d)                                  # [B, N, Fp]

        # Step 2: attention scores -- dense over all N*N pairs [GAT Eq. 3]
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)        # [B, N, N, Fp]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)        # [B, N, N, Fp]
        e = self.leaky_relu(
            torch.cat([h_i, h_j], dim=-1) @ self.a        # [B, N, N]
        )

        # Step 3: softmax normalization [GAT Eq. 2]
        beta = torch.softmax(e, dim=-1)                    # [B, N, N]

        # Step 4: CAM modulation -- preserves signed inhibitory semantics
        alpha = cam.unsqueeze(0) * beta                    # [B, N, N]

        # Step 5: aggregate
        x_agg = torch.einsum('bij,bjf->bif', alpha, h)    # [B, N, Fp]

        # Step 6: skip connection + transform
        x_cat = torch.cat([x_agg, x_3d], dim=-1)          # [B, N, Fp+F]
        x_out = self.W2(x_cat.view(B_N, self.Fp + in_feat))
        return F.elu(x_out)                                # [B*N, F']


class GCN_CAMGAT_8_8_16_16_32(nn.Module):
    """
    CAM-GAT variant of GCN_CAM_8_8_16_16_32.

    Replaces CAMLayer with CAMGATLayer throughout. All other design decisions
    (shared CAM, 5 layers, mean+max readout, ELU, FC 128) are preserved so
    results are directly comparable to GCN_CAM_8_8_16_16_32.

    Additional parameters per layer vs CAMLayer:
        W1 in R^(Fp x F)   and   a in R^(2Fp)
    CAM is still shared across all 5 layers.
    """

    N_JOINTS = 21

    def __init__(self, numFeatures: int, numClasses: int,
                 use_cmc_angle: bool = False):
        super().__init__()
        self.numClasses    = numClasses
        self.use_cmc_angle = use_cmc_angle

        self.cam = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS))
        nn.init.uniform_(self.cam, -1.0, 1.0)

        self.layer1 = CAMGATLayer(numFeatures, 8,  self.N_JOINTS)
        self.layer2 = CAMGATLayer(8,           8,  self.N_JOINTS)
        self.layer3 = CAMGATLayer(8,           16, self.N_JOINTS)
        self.layer4 = CAMGATLayer(16,          16, self.N_JOINTS)
        self.layer5 = CAMGATLayer(16,          32, self.N_JOINTS)

        readout_dim = 64 + (1 if use_cmc_angle else 0)
        self.fc1 = nn.Linear(readout_dim, 128)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, data):
        x     = data.x
        batch = getattr(data, 'batch',
                        torch.zeros(x.size(0), dtype=torch.long,
                                    device=x.device))
        mask  = getattr(data, 'mask', None)

        x = self.layer1(x, self.cam, mask)
        x = self.layer2(x, self.cam, mask)
        x = self.layer3(x, self.cam, mask)
        x = self.layer4(x, self.cam, mask)
        x = self.layer5(x, self.cam, mask)

        h = masked_readout(x, batch, mask)                 # [B, 64]

        if self.use_cmc_angle:
            theta = data.theta_cmc.view(-1, 1)
            h = torch.cat([h, theta], dim=1)

        h   = F.elu(self.fc1(h))
        out = self.fc2(h)
        return F.log_softmax(out, dim=1)


class GCN_CAM_32_32_64_64_128(nn.Module):
    """
    GCN_CAM_8_8_16_16_32 scaled x4: hidden dims 32-32-64-64-128.

    Same architecture as GCN_CAM_8_8_16_16_32 (shared CAM, 5 CAMLayers,
    mean+max readout, ELU) but with wider layers (~70k params vs ~15k).
    Intended for capacity ablation on top of the abl04 feature set.
    """

    N_JOINTS = 21

    def __init__(self, numFeatures: int, numClasses: int,
                 use_cmc_angle: bool = False):
        super().__init__()
        self.numClasses    = numClasses
        self.use_cmc_angle = use_cmc_angle

        self.cam = nn.Parameter(torch.empty(self.N_JOINTS, self.N_JOINTS))
        nn.init.uniform_(self.cam, -1.0, 1.0)

        self.layer1 = CAMLayer(numFeatures, 32,  self.N_JOINTS)
        self.layer2 = CAMLayer(32,          32,  self.N_JOINTS)
        self.layer3 = CAMLayer(32,          64,  self.N_JOINTS)
        self.layer4 = CAMLayer(64,          64,  self.N_JOINTS)
        self.layer5 = CAMLayer(64,          128, self.N_JOINTS)

        readout_dim = 256 + (1 if use_cmc_angle else 0)
        self.fc1 = nn.Linear(readout_dim, 128)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, data):
        x     = data.x
        batch = getattr(data, 'batch',
                        torch.zeros(x.size(0), dtype=torch.long,
                                    device=x.device))
        mask  = getattr(data, 'mask', None)

        x = self.layer1(x, self.cam, mask)
        x = self.layer2(x, self.cam, mask)
        x = self.layer3(x, self.cam, mask)
        x = self.layer4(x, self.cam, mask)
        x = self.layer5(x, self.cam, mask)

        h = masked_readout(x, batch, mask)                     # [B, 256]

        if self.use_cmc_angle:
            theta = data.theta_cmc.view(-1, 1)
            h = torch.cat([h, theta], dim=1)

        h   = F.elu(self.fc1(h))
        out = self.fc2(h)
        return F.log_softmax(out, dim=1)


class GCN_8_8_16_16_32(nn.Module):
    def __init__(self, numFeatures, numClasses, use_cmc_angle: bool = False):
        super().__init__()
        self.numClasses = numClasses
        self.use_cmc_angle = use_cmc_angle

        self.conv1 = GCNConv(numFeatures, 8)
        self.conv2 = GCNConv(8, 8)
        self.conv3 = GCNConv(8, 16)
        self.conv4 = GCNConv(16, 16)
        self.conv5 = GCNConv(16, 32)

        # mean+max → 2*32 = 64; +1 if θ_CMC is concatenated → 65
        readout_dim = 64 + (1 if use_cmc_angle else 0)
        self.fc1 = nn.Linear(readout_dim, 128)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        mask  = getattr(data, "mask", None)  # [N,1]

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv5(x, edge_index))   # [N, 32]

        h = masked_readout(x, batch, mask)      # [B, 64]

        if self.use_cmc_angle:
            theta = data.theta_cmc.view(-1, 1)  # [B, 1]
            h = torch.cat([h, theta], dim=1)    # [B, 65]

        h = F.elu(self.fc1(h))                  # [B, 128]
        out = self.fc2(h)                       # [B, numClasses]
        return F.log_softmax(out, dim=1)
