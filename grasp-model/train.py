# ==============================================================
# train.py — Training script for GCN_8_8_16_16_32
# with train/val/test splits, validation metrics, and checkpoints
# ==============================================================

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# ✅ IMPORTS CORREGIDOS (paquete)
from grasp_gcn.dataset.grasps import GraspsClass
from grasp_gcn.transforms.tograph import ToGraph
from grasp_gcn.network.utils import get_network

print('--------------------------------')
print('🚀 Import libraries: OK')

# ====================== Hyperparameters =========================


lr = 1e-3
normalize = True
network_type = "GCN_8_8_16_16_32"
weight_decay = 5e-4
seed = 42
BATCH_SIZE = 256
NUM_WORKERS = 0
num_epochs = 40
EARLY_STOPPING_PATIENCE = 10

# Reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

print(f"Batch size: {BATCH_SIZE}, Epochs: {num_epochs}, LR: {lr}")
print('--------------------------------')

# ==================== TensorBoard ================================
writer = SummaryWriter(log_dir='experiments/runs/hograspnet_16cls_run004b')

# ==================== Datasets ===================================
print("📦 Loading datasets...")

datasetTrain = GraspsClass(root='data/', split='train', normalize=normalize, collapse=True)
datasetVal   = GraspsClass(root='data/', split='val',   normalize=normalize, collapse=True)
datasetTest  = GraspsClass(root='data/', split='test',  normalize=normalize, collapse=True)

print(f"Train: {len(datasetTrain)}, Val: {len(datasetVal)}, Test: {len(datasetTest)}")
print(f"✅ Num features per node: {datasetTrain.num_features}")
print(f"✅ Num classes: {datasetTrain.num_classes}")

print(f"✅ Features per node: {datasetTrain.num_features} (xyz + joint angle)")

# DataLoaders
train_loader = DataLoader(datasetTrain, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(datasetVal,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(datasetTest,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==================== Device =====================================
device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device_}")
print('--------------------------------')

# ==================== Model ======================================
model_ = get_network(network_type, datasetTrain.num_features, datasetTrain.num_classes).to(device_)

def reset_weights(m):
    """Reset model weights to avoid leakage between runs."""
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

model_.apply(reset_weights)

optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr, weight_decay=weight_decay)


# ==================== Evaluation Function ========================
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device_)
            out = model(batch)
            label = batch.y.view(-1)
            loss = F.nll_loss(out, label)
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += batch.num_graphs
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# ==================== Training Loop ==============================
def train(model_, train_loader_, val_loader_, writer):
    best_val_acc = 0.0
    epochs_no_improve = 0
    time_start_ = timer()
    print("🎯 Starting training...")

    for epoch in range(num_epochs):
        model_.train()
        epoch_loss = 0.0

        for batch in train_loader_:
            batch = batch.to(device_)
            optimizer_.zero_grad()
            pred = model_(batch)
            label = batch.y.view(-1)
            loss = F.nll_loss(pred, label)
            loss.backward()
            optimizer_.step()
            epoch_loss += loss.item() * batch.num_graphs

        epoch_loss /= len(train_loader_.dataset)
        writer.add_scalar("Train/Loss", epoch_loss, epoch)

        # Validation phase
        val_loss, val_acc = evaluate(model_, val_loader_)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f}")

        # Save best model + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            os.makedirs("experiments", exist_ok=True)
            torch.save(model_.state_dict(), "experiments/best_model.pth")
            print(f"💾 Saved best model (Val Acc = {val_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    total_time = timer() - time_start_
    print(f"\n⏱️  Training completed in {total_time/60:.2f} min.")
    return model_

# ==================== Run Training ===============================
trained_model = train(model_, train_loader, val_loader, writer)

# ==================== Final Evaluation ===========================
test_loss, test_acc = evaluate(trained_model, test_loader)
print(f"\n✅ Test set → Loss: {test_loss:.4f}, Accuracy: {test_acc:.3f}")

# Save final model
os.makedirs("experiments", exist_ok=True)
torch.save(trained_model.state_dict(), "experiments/final_model.pth")
print("💾 Saved final model → experiments/final_model.pth")

# ==================== Classification Report ======================
y_true, y_pred = [], []
trained_model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device_)
        preds = trained_model(batch).argmax(1).cpu().numpy()
        labels = batch.y.view(-1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=3)

print("\n" + "="*60)
print("📊 CONFUSION MATRIX")
print("="*60)
print(cm)
print("\n" + "="*60)
print("📈 CLASSIFICATION REPORT")
print("="*60)
print(report)
print("="*60)


print("\n🎉 Training process complete. Ready for analysis!")
