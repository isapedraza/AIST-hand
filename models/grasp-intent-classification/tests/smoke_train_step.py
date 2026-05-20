import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grasp_gcn.dataset.grasps import GraspsClass
from grasp_gcn.network.utils import get_network


def main():
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("Loading train split...")
    dataset = GraspsClass(root=str(ROOT / "data"), split="train", collapse=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Building model...")
    model = get_network(
        "GCN_CAM_8_8_16_16_32",
        dataset.num_features,
        dataset.num_classes,
        use_cmc_angle=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    print("Running one train step...")
    model.train()
    batch = next(iter(loader)).to(device)
    optimizer.zero_grad()
    pred = model(batch)
    label = batch.y.view(-1)
    loss = F.nll_loss(pred, label)
    loss.backward()
    optimizer.step()

    print(f"OK - batch graphs: {batch.num_graphs}")
    print(f"OK - pred shape: {tuple(pred.shape)}")
    print(f"OK - loss: {loss.item():.6f}")
    print("Smoke train step passed.")


if __name__ == "__main__":
    main()
