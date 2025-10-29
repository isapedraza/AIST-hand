# ==========================
#  train_gcn.py
#  Entrenamiento didáctico para GCN_8_8_16_16_32 con ToGraph
# ==========================

import os, random, numpy as np, torch
from timeit import default_timer as timer

# ---------- Reproducibilidad ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- Device ----------
device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device_}")

# ---------- Dataset & Transform ----------
from grasp_gcn.transforms.tograph import ToGraph
from grasp_gcn.dataset.grasps import GraspsClass
from torch_geometric.loader import DataLoader

CSV_PATH = "data/grasps_sample_train.csv"  # ajusta a tu ruta real

# Activamos normalización por muestra aquí (¡clave!)
transform = ToGraph(features='xyz', normalize='per_sample_meanstd', make_undirected=True)

dataset = GraspsClass(
    root='data/',
    split=None,
    transform=transform,
    csvs=CSV_PATH
)

print(f"[INFO] Total samples: {len(dataset)}")

# Split 80/20
indices = np.arange(len(dataset))
np.random.shuffle(indices)
split = int(0.8 * len(dataset))
train_idx, val_idx = indices[:split], indices[split:]

from torch.utils.data import Subset
train_set = Subset(dataset, train_idx)
val_set   = Subset(dataset, val_idx)

BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ---------- Modelo ----------
from grasp_gcn.network.utils import get_network

model_ = get_network(
    "GCN_8_8_16_16_32",
    dataset.num_features,   # debe ser 3 si features='xyz'
    dataset.num_classes     # p.ej. 3
).to(device_)

# Reset opcional de parámetros (una sola vez)
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

model_.apply(reset_weights)

# ---------- Loss: Class Weights ----------
# Calculamos distribución de clases en el TRAIN
ys = []
with torch.no_grad():
    for batch in train_loader:
        ys.append(batch.y)
ys = torch.cat(ys, dim=0)
counts = torch.bincount(ys)
# inversa de frecuencia (normalizada para no desbalancear el loss global)
class_weights = counts.sum().float() / counts.float().clamp_min(1)
class_weights = class_weights / class_weights.mean()
class_weights = class_weights.to(device_)

print(f"[INFO] Class counts: {counts.tolist()}")
print(f"[INFO] Class weights: {class_weights.tolist()}")

# ---------- Optimizador & Scheduler ----------
import torch.nn.functional as F
optimizer = torch.optim.Adam(model_.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# ---------- TensorBoard ----------
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/gcn_baseline")

# ---------- Helpers ----------
def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)             # log_probs [B, C]
        target = batch.y.long()        # [B]
        loss = F.nll_loss(out, target, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total   += batch.num_graphs

    return running_loss / len(loader.dataset), correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)             # log_probs
        pred = out.argmax(dim=1)
        target = batch.y.long()
        correct += (pred == target).sum().item()
        total   += batch.num_graphs

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    acc = correct / total if total > 0 else 0.0
    all_preds   = torch.cat(all_preds, dim=0).numpy() if all_preds else np.array([])
    all_targets = torch.cat(all_targets, dim=0).numpy() if all_targets else np.array([])
    return acc, all_preds, all_targets

# ---------- Loop principal ----------
EPOCHS = 50
best_val_acc, best_state = 0.0, None
t0 = timer()

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model_, train_loader, optimizer, device_, class_weights)
    val_acc, y_pred, y_true = evaluate(model_, val_loader, device_)

    writer.add_scalar('loss/train', train_loss, epoch)
    writer.add_scalar('acc/train',  train_acc,  epoch)
    writer.add_scalar('acc/val',    val_acc,    epoch)

    # checkpoint del mejor
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            'model': model_.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'num_features': dataset.num_features,
            'num_classes': dataset.num_classes
        }

    scheduler.step(val_acc)
    print(f"[{epoch:03d}] loss {train_loss:.4f} | acc {train_acc:.3f} | val_acc {val_acc:.3f}")

# ---------- Guardar mejor modelo ----------
os.makedirs('checkpoints', exist_ok=True)
save_path = 'checkpoints/gcn_best.pt'
torch.save(best_state, save_path)
writer.close()

t1 = timer()
print(f"[DONE] Best val acc: {best_val_acc:.3f}")
print(f"[DONE] Saved: {save_path}")
print(f"[TIME] {t1 - t0:.1f}s")

# ---------- (Opcional) Reportes de evaluación ----------
try:
    from sklearn.metrics import confusion_matrix, classification_report
    if len(val_set) > 0:
        # Re-evaluamos para imprimir la matriz con el mejor estado
        state = torch.load(save_path, map_location='cpu')
        model_.load_state_dict(state['model'])
        val_acc, y_pred, y_true = evaluate(model_, val_loader, device_)
        print("\n[CONFUSION MATRIX]")
        print(confusion_matrix(y_true, y_pred))
        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(y_true, y_pred, digits=3))
except Exception as e:
    print(f"[WARN] No se pudo generar reporte sklearn: {e}")
