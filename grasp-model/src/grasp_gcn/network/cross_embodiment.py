import torch
import torch.nn as nn


class CrossEmbodimentDecoder(nn.Module):
    """
    D_X: decodes latent z to shared embedding space.

    Input : z [B, z_dim]       (output of E_h or E_X)
    Output: [B, shared_dim],   bounded in [-1, 1] via Tanh

    Used in:
      - Training: z -> D_X -> shared_dim -> D_r -> qpos
      - L_ltc:    E_h(x_H) -> D_X -> E_X -> z  (round-trip consistency)
    """

    def __init__(self, z_dim: int = 32, shared_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ELU(),
            nn.Linear(128, shared_dim),
            nn.ELU(),
            nn.Linear(shared_dim, shared_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CrossEmbodimentEncoder(nn.Module):
    """
    E_X: encodes shared embedding to latent z.

    Input : [B, shared_dim]    (output of E_r)
    Output: z [B, z_dim],      bounded in [-1, 1] via Tanh

    Used in:
      - Training: x_r -> E_r -> shared_dim -> E_X -> z
      - L_ltc:    D_X(z) -> E_X -> z  (round-trip consistency)
    """

    def __init__(self, shared_dim: int = 256, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, z_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
