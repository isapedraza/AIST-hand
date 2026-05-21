import torch.nn as nn


def _mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


N_SUBSPACES = 5  # thumb / index / middle / ring / pinky


class SharedEncoder_E_X(nn.Module):
    """
    E_X: shared MLP encoder from robot embedding to the latent space.

    Input : [B, shared_dim]    (output of RobotEncoder_E_r)
    Output: z [B, z_dim_total] where z_dim_total = N_SUBSPACES*z_dim by default,
            or explicit z_dim_total when --single_latent is in effect.

    8-layer MLP with 256-neuron hidden layers (Yan et al. 2026).
    Shared across all robots.
    """

    def __init__(self, shared_dim: int = 1024, z_dim: int = 64, z_dim_total: int | None = None):
        super().__init__()
        if z_dim_total is None:
            z_dim_total = N_SUBSPACES * z_dim
        self.net = _mlp([shared_dim, 256, 256, 256, 256, 256, 256, 256, z_dim_total])

    def forward(self, x):
        """x: [B, shared_dim] -> z: [B, z_dim_total]"""
        return self.net(x)


class SharedDecoder_D_X(nn.Module):
    """
    D_X: shared MLP decoder from the latent space to robot embedding.

    Input : z [B, z_dim_total]    where z_dim_total = N_SUBSPACES*z_dim by default,
            or explicit z_dim_total when --single_latent is in effect.
    Output: [B, shared_dim],       bounded [-1, 1] via Tanh

    8-layer MLP with 256-neuron hidden layers (Yan et al. 2026).
    Shared across all robots.
    """

    def __init__(self, z_dim: int = 64, shared_dim: int = 1024, z_dim_total: int | None = None):
        super().__init__()
        if z_dim_total is None:
            z_dim_total = N_SUBSPACES * z_dim
        self.net = _mlp([z_dim_total, 256, 256, 256, 256, 256, 256, 256, shared_dim])

    def forward(self, z):
        """z: [B, z_dim_total] -> [B, shared_dim]"""
        return self.net(z)
