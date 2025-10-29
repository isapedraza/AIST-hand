# ==========================
# quick_kfold_demo.py
# K-Fold "solo con TEST" para demo rápida
# (Entrena y valida dentro del mismo CSV de test)
# ==========================

import os, random, numpy as np, torch
from sklearn.model_selection import StratifiedKFold
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

# ---------- Dataset ----------
from dataset.grasps import GraspsClass
from torch_geometric.loader import DataLoader

# ⚠️ Ruta al CSV de test (lo leerá desde root/raw si es relativo)
CSV_TEST = "data/grasps_sample_test.csv"

# Nota: ToGraph se aplica dentro de dataset.process(); aquí NO pasamos transform
dataset = GraspsClass(
    root='data/',
    split=None,
    transform=None,       # <- clave para evitar doble ToGraph
    csvs=CSV_TEST
)

N = len(dataset)
print(f"[INFO] TEST samples leídos: {N}")
assert N > 0, "El CSV de test está vacío o la ruta es incorrecta."

# Vector de etiquetas completo (para StratifiedKFold y baseline)
y_all = np.array([dataset[i].y.item() for i in range(N)], dtype=int)
classes, counts_all = np.unique(y_all, return_counts=True)
print(f"[INFO] Distribución total de clases: {dict(zip(classes, counts_all))}")

# ---------- Modelo ----------
from network.utils import get_network

def build_model(num_features, num_classes):
    model_ = get_network("GCN_8_8_16_16_32", num_features, num_classes).to(device_)
    # reset de pesos por limpieza
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    model_.apply(reset_weights)
    return model_

# ---------- Helpers ----------
import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_accum, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)

        # --- AUGMENT: jitter suave sobre nodos válidos (solo en train) ---
        if hasattr(batch, "x"):
            sigma = 0.02  # acorde a z-score per-sample
            noise = torch.randn_like(batch.x) * sigma
            if hasattr(batch, "mask") and batch.mask is not None:
                m = batch.mask
                if m.dim() == 2 and m.size(1) == 1:
                    m = m.expand_as(batch.x)
                batch.x = batch.x + noise * m
            else:
                batch.x = batch.x + noise

        optimizer.zero_grad()
        out = model(batch)                 # log_probs [B, C]
        target = batch.y.long()
        loss = F.nll_loss(out, target)     # sin class weights (folds balanceados)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_accum += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total   += batch.num_graphs
    return loss_accum / len(loader.dataset), correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1)
        target = batch.y.long()
        correct += (pred == target).sum().item()
        total   += batch.num_graphs
    return correct / total if total > 0 else 0.0

# ---------- K-Fold (estratificado) sobre TEST (demo) ----------
K = 5
EPOCHS = 30
BATCH_SIZE = 8

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
fold_accs = []
fold_sizes = []
t0 = timer()

# Baseline: clase mayoritaria
maj = np.bincount(y_all).argmax()
baseline = (y_all == maj).mean()
print(f"[BASELINE] Accuracy clase mayoritaria: {baseline:.3f}  (clase={maj})")

for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), y_all), start=1):
    print(f"\n========== FOLD {fold}/{K} ==========")
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    fold_sizes.append(len(val_set))

    # Distribución por fold
    y_train = y_all[train_idx]
    y_val   = y_all[val_idx]
    classes_tr, counts_tr = np.unique(y_train, return_counts=True)
    classes_va, counts_va = np.unique(y_val,   return_counts=True)
    print(f"[INFO][Fold {fold}] Train: {dict(zip(classes_tr, counts_tr))} | Val: {dict(zip(classes_va, counts_va))}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device_.type == 'cuda')
    )
    val_loader   = DataLoader(
        val_set,   batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device_.type == 'cuda')
    )

    model_ = build_model(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model_.parameters(), lr=1e-3, weight_decay=5e-4)

    best_val = 0.0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model_, train_loader, optimizer, device_)
        val_acc = evaluate(model_, val_loader, device_)
        best_val = max(best_val, val_acc)
        print(f"[Fold {fold}][{epoch:02d}] loss {tr_loss:.4f} | acc {tr_acc:.3f} | val_acc {val_acc:.3f}")

    print(f"[Fold {fold}] Best val_acc: {best_val:.3f}")
    fold_accs.append(best_val)

t1 = timer()

# ---------- Resumen ----------
mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
std_acc  = float(np.std(fold_accs))  if fold_accs else 0.0

print("\n========== RESUMEN ==========")
print(f"Conjunto: {CSV_TEST}  | N={N}  | Clases={len(classes)}  | Semilla={SEED}")
print(f"K={K} | Épocas={EPOCHS} | BATCH={BATCH_SIZE} | Device={device_}")
print(f"Baseline mayoritaria: {baseline:.3f} (clase {maj})")
print("Accuracies por fold:", [f"{a:.3f}" for a in fold_accs])
print(f"Media val_acc: {mean_acc:.3f} ± {std_acc:.3f}")
print(f"Muestras de validación por fold:", fold_sizes)
print(f"[TIEMPO] {t1 - t0:.1f}s")
