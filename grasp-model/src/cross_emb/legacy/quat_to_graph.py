import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

# Same hand skeleton as ToGraph (grasp_gcn) — 21-node MediaPipe topology.
_EDGE_ORIGINS = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9,
                 5, 9, 10, 10, 11, 11, 12, 13, 9, 13, 14, 14, 15, 15,
                 16, 17, 13, 17, 18, 17, 18, 19, 19, 20]
_EDGE_ENDS    = [1, 5, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 5, 7, 6, 8, 7, 5,
                 9, 10, 9, 11, 10, 12, 11, 9, 13, 14, 13, 15, 14, 16,
                 15, 13, 17, 18, 17, 0, 19, 18, 20, 19]


class QuatToGraph:
    """
    Converts Dong quaternion batch to PyG Batch for HumanEncoder (E_h).

    Input : quats [B, 20, 4]  wxyz, w>=0, ordered thumb_mcp..pinky_tip
    Output: PyG Batch, 21-node hand graph per sample

    Node 0  = WRIST, identity [1,0,0,0] — structural hub, reference frame.
    Nodes 1-20 = Dong finger quats (same finger order as ToGraph node indices).
    Edge topology identical to ToGraph (grasp_gcn).
    """

    def __init__(self, make_undirected: bool = True):
        ei = torch.tensor([_EDGE_ORIGINS, _EDGE_ENDS], dtype=torch.long)
        if make_undirected:
            ei = to_undirected(ei)
        self._edge_index = ei  # [2, E], reused across all graphs in batch

    def __call__(self, quats: Tensor) -> Batch:
        """
        Args:
            quats: [B, 20, 4] Dong quaternions
        Returns:
            PyG Batch with x [B*21, 4]
        """
        B = quats.shape[0]
        wrist = torch.zeros(B, 1, 4, dtype=quats.dtype, device=quats.device)
        wrist[:, 0, 0] = 1.0                           # identity [1,0,0,0]
        x_all = torch.cat([wrist, quats], dim=1)        # [B, 21, 4]

        ei = self._edge_index.to(quats.device)
        return Batch.from_data_list([
            Data(x=x_all[i], edge_index=ei) for i in range(B)
        ])
