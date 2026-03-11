"""
analyze_class_separability.py -- Analisis de separabilidad entre las 28 clases.

1. PCA global con varimax rotation (Jarque-Bou et al., 2019)
2. Centroides por clase en espacio PCA
3. Clustering jerarquico de los 28 centroides -> dendrograma
4. MLP con 28 clases originales -> matriz de confusion
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ALL_ANGLE_NAMES, CLASS_NAMES, PCA_VARIANCE_TARGET,
    MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_LR, MLP_EPOCHS,
    MLP_BATCH_SIZE, MLP_PATIENCE, MLP_SEED, RESULTS_DIR,
)
from data import load_and_split
from features import compute_joint_angles, trial_median

np.random.seed(MLP_SEED)
torch.manual_seed(MLP_SEED)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Varimax Rotation
# =============================================================================

def varimax(loadings, max_iter=100, tol=1e-6):
    """
    Varimax rotation (Kaiser, 1958).
    Input: loadings matrix (n_features, n_components).
    Output: rotated loadings, rotation matrix.
    """
    p, k = loadings.shape
    R = np.eye(k)
    d = 0

    for _ in range(max_iter):
        L = loadings @ R
        # Varimax criterion
        u, s, vt = np.linalg.svd(
            loadings.T @ (L**3 - (1.0 / p) * L @ np.diag(np.sum(L**2, axis=0)))
        )
        R = u @ vt
        d_new = np.sum(s)
        if abs(d_new - d) < tol:
            break
        d = d_new

    return loadings @ R, R


# =============================================================================
# Main Pipeline
# =============================================================================

# ── Paso 1: Cargar datos ────────────────────────────────────────────────────
print("=" * 60)
print("Carga y split")
print("=" * 60)
train_df, val_df, test_df = load_and_split()


# ── Paso 2: Angulos + medianas ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Angulos + Punto de Trial")
print("=" * 60)

splits = {}
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    xyz = df.iloc[:, 5:].values.astype("float32").reshape(-1, 21, 3)
    angles = compute_joint_angles(xyz)
    medians = trial_median(df, angles)
    splits[name] = {"df": df, "angles": angles, "medians": medians}
    print(f"  {name}: {len(df):,} frames -> {len(medians):,} trials")


# ── Paso 3: PCA con varimax ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PCA + Varimax Rotation")
print("=" * 60)

scaler = StandardScaler()
X_train = scaler.fit_transform(splits["train"]["medians"][ALL_ANGLE_NAMES].values)

pca = PCA()
Z_train_raw = pca.fit_transform(X_train)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)

print(f"Componentes para {PCA_VARIANCE_TARGET*100:.0f}% varianza: {n_comp}")
print(f"Varianza acumulada: {cumvar[:n_comp].round(4)}")

# Varimax rotation on the retained components
loadings_raw = pca.components_[:n_comp].T  # (20, n_comp)
loadings_rotated, rotation_matrix = varimax(loadings_raw)

# Reproject training data with rotated loadings
Z_train = X_train @ loadings_rotated  # (n_trials, n_comp)

print(f"\nLoadings rotados (top 3 por componente):")
for pc_i in range(n_comp):
    col = loadings_rotated[:, pc_i]
    top_idx = np.argsort(np.abs(col))[::-1][:3]
    desc = ", ".join(f"{ALL_ANGLE_NAMES[j]}({col[j]:+.3f})" for j in top_idx)
    print(f"  RC{pc_i+1}: {desc}")

train_medians = splits["train"]["medians"]
grasp_types = train_medians["grasp_type"].values
unique_classes = sorted(np.unique(grasp_types))
n_classes = len(unique_classes)


# ── Paso 4: Centroides por clase ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Centroides por clase en espacio PCA rotado")
print("=" * 60)

centroids = np.zeros((n_classes, n_comp))

for i, cls in enumerate(unique_classes):
    mask = grasp_types == cls
    cls_trials = Z_train[mask]
    centroids[i] = cls_trials.mean(axis=0)
    name = CLASS_NAMES.get(int(cls), f"class_{cls}")
    print(f"  [{cls:2d}] {name:<28s} n={mask.sum():4d}  "
          f"RC1={centroids[i, 0]:+.2f}  RC2={centroids[i, 1]:+.2f}  "
          f"RC3={centroids[i, 2]:+.2f}")


# ── Paso 5: Clustering jerarquico de centroides ─────────────────────────────
print("\n" + "=" * 60)
print("Clustering jerarquico de centroides")
print("=" * 60)

dist_matrix = squareform(pdist(centroids, metric="euclidean"))
Z_link = linkage(centroids, method="ward")

class_labels = [f"[{cls}] {CLASS_NAMES.get(int(cls), '')[:15]}" for cls in unique_classes]

fig, ax = plt.subplots(figsize=(18, 8))
dendrogram(Z_link, labels=class_labels, ax=ax,
           leaf_rotation=90, leaf_font_size=8, color_threshold=0)
ax.set_title("Dendrograma Ward -- Centroides de 28 clases en espacio PCA rotado (varimax)")
ax.set_ylabel("Distancia Ward")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "dendrogram_classes_varimax.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Dendrograma guardado en {RESULTS_DIR / 'dendrogram_classes_varimax.png'}")

# Top pares mas cercanos
print("\nTop 15 pares mas cercanos (distancia euclidiana de centroides):")
pairs = []
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        pairs.append((unique_classes[i], unique_classes[j], dist_matrix[i, j]))
pairs.sort(key=lambda x: x[2])

for rank, (ci, cj, d) in enumerate(pairs[:15], 1):
    ni = CLASS_NAMES.get(int(ci), f"class_{ci}")
    nj = CLASS_NAMES.get(int(cj), f"class_{cj}")
    print(f"  {rank:2d}. [{ci:2d}] {ni:<25s} <-> [{cj:2d}] {nj:<25s}  dist={d:.3f}")


# ── Paso 6: Scatter RC1 vs RC2 ──────────────────────────────────────────────
cmap = plt.cm.get_cmap("tab20")

fig, ax = plt.subplots(figsize=(14, 10))
for i, cls in enumerate(unique_classes):
    mask = grasp_types == cls
    ax.scatter(Z_train[mask, 0], Z_train[mask, 1], alpha=0.15, s=8,
               color=cmap(i % 20))
    ax.scatter(centroids[i, 0], centroids[i, 1], s=80, color=cmap(i % 20),
               edgecolors="black", linewidth=0.5, zorder=5)
    ax.annotate(f"{cls}", (centroids[i, 0], centroids[i, 1]),
                fontsize=6, ha="center", va="bottom", fontweight="bold")

ax.set_xlabel("RC1")
ax.set_ylabel("RC2")
ax.set_title("Distribuciones de clase en RC1 vs RC2 (varimax)")
fig.savefig(RESULTS_DIR / "scatter_rc1_rc2_varimax.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nScatter RC1-RC2 guardado en {RESULTS_DIR / 'scatter_rc1_rc2_varimax.png'}")


# ── Paso 7: MLP con 28 clases originales ────────────────────────────────────
print("\n" + "=" * 60)
print("MLP -- 28 clases originales")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def make_loader(df, shuffle=False):
    X = torch.from_numpy(df.iloc[:, 5:].values.astype("float32"))
    y = torch.from_numpy(df["grasp_type"].values.astype("int64"))
    return DataLoader(TensorDataset(X, y), batch_size=MLP_BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(train_df, shuffle=True)
val_loader = make_loader(val_df)
test_loader = make_loader(test_df)

num_classes = 28

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)

model = MLP(63, MLP_HIDDEN_DIMS, num_classes, MLP_DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)

best_val_acc = 0.0
patience_counter = 0

for epoch in range(MLP_EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= MLP_PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1}/{MLP_EPOCHS} | Val Acc: {val_acc:.3f}")

# Test
model.load_state_dict(best_state)
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        y_true.extend(yb.numpy())
        y_pred.extend(model(xb).argmax(1).cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

test_acc = (y_true == y_pred).mean()
cm = confusion_matrix(y_true, y_pred, labels=list(range(28)))
report = classification_report(y_true, y_pred, labels=list(range(28)),
                                target_names=[CLASS_NAMES[i] for i in range(28)],
                                digits=3, output_dict=True, zero_division=0)

print(f"\n  Test Accuracy: {test_acc:.3f}")
print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")

# Confusiones significativas
print("\n  Confusiones significativas (>10% en cualquier direccion):")
confusions = []
for i in range(28):
    row_sum = cm[i].sum()
    if row_sum == 0:
        continue
    for j in range(28):
        if i == j:
            continue
        conf_rate = cm[i, j] / row_sum
        if conf_rate > 0.10:
            confusions.append((i, j, conf_rate))

confusions.sort(key=lambda x: -x[2])
for ci, cj, rate in confusions[:30]:
    ni = CLASS_NAMES[ci]
    nj = CLASS_NAMES[cj]
    rev_rate = cm[cj, ci] / cm[cj].sum() if cm[cj].sum() > 0 else 0
    print(f"    [{ci:2d}] {ni:<25s} -> [{cj:2d}] {nj:<25s}: {rate:.1%}  (reversa: {rev_rate:.1%})")

# Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(16, 14))
cm_norm = cm.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm = cm_norm / row_sums

im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.5)
ax.set_xticks(range(28))
ax.set_yticks(range(28))
short_names = [CLASS_NAMES[i][:12] for i in range(28)]
ax.set_xticklabels(short_names, rotation=90, fontsize=7)
ax.set_yticklabels(short_names, fontsize=7)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("MLP 28 clases -- Confusion Matrix (normalizada por fila)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(RESULTS_DIR / "confusion_28classes.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Confusion matrix guardada en {RESULTS_DIR / 'confusion_28classes.png'}")


# ── Guardar resultados ──────────────────────────────────────────────────────
results = {
    "pca": {
        "n_components": n_comp,
        "explained_variance_ratio": pca.explained_variance_ratio_[:n_comp].tolist(),
        "cumulative_variance": float(cumvar[n_comp - 1]),
        "varimax_applied": True,
        "loadings_rotated": loadings_rotated.tolist(),
        "angle_names": ALL_ANGLE_NAMES,
    },
    "class_centroids_pca": {
        int(cls): centroids[i].tolist()
        for i, cls in enumerate(unique_classes)
    },
    "top_15_closest_pairs": [
        {
            "class_a": int(ci), "name_a": CLASS_NAMES.get(int(ci), ""),
            "class_b": int(cj), "name_b": CLASS_NAMES.get(int(cj), ""),
            "distance": float(d),
        }
        for ci, cj, d in pairs[:15]
    ],
    "mlp_28": {
        "test_accuracy": float(test_acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": {
            CLASS_NAMES[i]: float(report.get(CLASS_NAMES[i], {}).get("f1-score", 0))
            for i in range(28)
        },
        "top_confusions": [
            {"from": int(ci), "to": int(cj), "rate": float(r)}
            for ci, cj, r in confusions[:30]
        ],
    },
}

with open(RESULTS_DIR / "class_separability.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResultados guardados en {RESULTS_DIR / 'class_separability.json'}")
