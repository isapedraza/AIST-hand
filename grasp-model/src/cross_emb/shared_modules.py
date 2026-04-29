import torch.nn as nn


def _mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


N_SUBSPACES = 3  # thumb / precision / support


class SharedEncoder_E_X(nn.Module):
    """
    E_X: shared MLP encoder from robot embedding to decoupled latent subspaces.

    Input : [B, shared_dim]       (output of RobotEncoder_E_r)
    Output: z [B, N_SUBSPACES*z_dim]  concat of [z_thumb, z_precision, z_support]

    8-layer MLP with 256-neuron hidden layers (Yan et al. 2026).
    Shared across all robots.
    """

    def __init__(self, shared_dim: int = 1024, z_dim: int = 64):
        super().__init__()
        self.net = _mlp([shared_dim, 256, 256, 256, 256, 256, 256, 256, N_SUBSPACES * z_dim])

    def forward(self, x):
        """x: [B, shared_dim] -> z: [B, N_SUBSPACES*z_dim]"""
        return self.net(x)


class SharedDecoder_D_X(nn.Module):
    """
    D_X: shared MLP decoder from decoupled latent subspaces to robot embedding.

    Input : z [B, N_SUBSPACES*z_dim]  concat of all subspace vectors
    Output: [B, shared_dim],          bounded [-1, 1] via Tanh

    8-layer MLP with 256-neuron hidden layers (Yan et al. 2026).
    Shared across all robots.
    """

    def __init__(self, z_dim: int = 64, shared_dim: int = 1024):
        super().__init__()
        self.net = _mlp([N_SUBSPACES * z_dim, 256, 256, 256, 256, 256, 256, 256, shared_dim])

    def forward(self, z):
        """z: [B, N_SUBSPACES*z_dim] -> [B, shared_dim]"""
        return self.net(z)
