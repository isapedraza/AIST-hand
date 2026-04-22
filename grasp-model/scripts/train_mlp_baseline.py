"""
MLP baseline for abl04 comparison.
Input: xyz+AHG+flex features (F=24, 21 nodes) → flat 504-dim
Output: 28 Feix classes
Same split/loss as abl04 (NLLLoss, log_softmax).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np

PROCESSED = Path("grasp-model/data/processed")
TRAIN_PT = PROCESSED / "hograspnet_train_c28_cmc_nocmc_ahga_ahgd.pt"
VAL_PT   = PROCESSED / "hograspnet_val_c28_cmc_nocmc_ahga_ahgd.pt"
TEST_PT  = PROCESSED / "hograspnet_test_c28_cmc_nocmc_ahga_ahgd.pt"

EPOCHS     = 50
BATCH_SIZE = 512
LR         = 1e-3
N_NODES    = 21
IN_DIM     = 24
N_CLASSES  = 28


def load_split(path):
    data, _ = torch.load(path, weights_only=False)
    N     = data.y.shape[0]
    x_all = data.x.view(N, N_NODES * IN_DIM).float()  # [N, 504]
    y_all = data.y.long()                               # [N]
    return TensorDataset(x_all, y_all)


class MLP(nn.Module):
    def __init__(self, in_dim=504, n_classes=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_ds = load_split(TRAIN_PT)
    val_ds   = load_split(VAL_PT)
    test_ds  = load_split(TEST_PT)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MLP(in_dim=N_NODES * IN_DIM, n_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MLP params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)

        train_acc  = correct / total
        train_loss = total_loss / total

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out  = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total   += len(y)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "grasp-model/checkpoints/mlp_baseline_best.pt")

        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Test evaluation
    print("\n--- Test ---")
    model.load_state_dict(torch.load("grasp-model/checkpoints/mlp_baseline_best.pt", weights_only=True))
    model.eval()
    all_preds = []
    all_true  = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu()
            all_preds.append(preds)
            all_true.append(y)
    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()
    test_acc  = (all_preds == all_true).mean()
    macro_f1  = f1_score(all_true, all_preds, average="macro")
    print(f"Test Acc:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Macro F1:  {macro_f1:.3f}")
    print(f"\nabl04 (GNN): 71.07% acc, F1=0.657")
    print(f"MLP baseline: {test_acc*100:.2f}% acc, F1={macro_f1:.3f}")


if __name__ == "__main__":
    main()
