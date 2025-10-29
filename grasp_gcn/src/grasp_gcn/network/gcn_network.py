import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool

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

class GCN_8_8_16_16_32(nn.Module):
    def __init__(self, numFeatures, numClasses):
        super().__init__()
        self.numClasses = numClasses

        self.conv1 = GCNConv(numFeatures, 8)
        self.conv2 = GCNConv(8, 8)
        self.conv3 = GCNConv(8, 16)
        self.conv4 = GCNConv(16, 16)
        self.conv5 = GCNConv(16, 32)

        # mean+max → 2*32 = 64
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Si llamas al modelo con un único Data (sin DataLoader), añade antes:
        # data.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        mask  = getattr(data, "mask", None)  # [N,1] si tu ToGraph la provee

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv5(x, edge_index))   # [N, 32]

        h = masked_readout(x, batch, mask)     # [B, 64]
        h = F.elu(self.fc1(h))                 # [B, 128]
        out = self.fc2(h)                      # [B, numClasses]
        return F.log_softmax(out, dim=1)
