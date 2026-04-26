import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool


class HumanEncoder_E_h(nn.Module):
    """
    E_h: encodes hand pose graph to latent z.

    Input : PyG Data with x=[N, 4] (Dong quaternions, wrist=identity)
    Output: z [B, z_dim], bounded in [-1, 1] via Tanh.

    Architecture (SAME-style, adapted for fixed 21-node hand graph):
      3 GATConv layers, hidden_dim=32, heads=4 (last layer heads=1)
      global_max_pool -> Linear -> Tanh -> z
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, heads: int = 4, z_dim: int = 16):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, add_self_loops=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, add_self_loops=True)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, add_self_loops=True)

        self.proj = nn.Linear(hidden_dim, z_dim)
        self.act = nn.ReLU()
        self.out_act = nn.Tanh()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.proj(x)
        return self.out_act(x)
