"""
analyze_clusters.py -- Analisis detallado de composicion de clusters.
Muestra TODAS las clases por cluster, no solo top 3.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from config import ALL_ANGLE_NAMES, CLASS_NAMES, PCA_VARIANCE_TARGET, MLP_SEED
from data import load_and_split
from features import compute_joint_angles, trial_median

np.random.seed(MLP_SEED)

# Cargar datos
train_df, val_df, test_df = load_and_split()

# Calcular angulos y medianas (train)
xyz = train_df.iloc[:, 5:].values.astype("float32").reshape(-1, 21, 3)
angles = compute_joint_angles(xyz)
medians = trial_median(train_df, angles)

# PCA
scaler = StandardScaler()
X = scaler.fit_transform(medians[ALL_ANGLE_NAMES].values)
pca = PCA()
Z = pca.fit_transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)
Z = Z[:, :n_comp]

# Clustering con k=5
km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=MLP_SEED)
labels = km.fit_predict(Z)
grasp_types = medians["grasp_type"].values

print("=" * 70)
print("COMPOSICION DETALLADA DE CLUSTERS (k=5)")
print("=" * 70)

for c in range(5):
    mask = labels == c
    original = grasp_types[mask]
    unique, counts = np.unique(original, return_counts=True)
    total = counts.sum()

    print(f"\n--- Cluster {c} ({total} trials) ---")
    sorted_idx = np.argsort(-counts)
    for idx in sorted_idx:
        cls = int(unique[idx])
        cnt = counts[idx]
        pct = cnt / total * 100
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        print(f"  {pct:5.1f}%  ({cnt:4d})  [{cls:2d}] {name}")

# Tabla resumen: que cluster domina cada clase
print("\n" + "=" * 70)
print("ASIGNACION POR CLASE: donde cae cada grasp type")
print("=" * 70)
print(f"{'Clase':<30s} {'Cluster':>8s} {'% en cluster':>12s} {'Total trials':>12s}")
print("-" * 70)

for cls_id in sorted(CLASS_NAMES.keys()):
    cls_mask = grasp_types == cls_id
    if cls_mask.sum() == 0:
        continue
    cls_clusters = labels[cls_mask]
    unique_c, counts_c = np.unique(cls_clusters, return_counts=True)
    dominant = unique_c[np.argmax(counts_c)]
    pct = counts_c.max() / counts_c.sum() * 100
    total = counts_c.sum()
    name = CLASS_NAMES[cls_id]
    # Show distribution across clusters
    dist = ", ".join(f"C{uc}:{cc}" for uc, cc in zip(unique_c, counts_c))
    print(f"  [{cls_id:2d}] {name:<25s} C{dominant}  {pct:5.1f}%  (n={total:4d})  [{dist}]")
