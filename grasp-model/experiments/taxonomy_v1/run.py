"""
run.py -- Synergy-Taxonomy Analysis v1
Entry point: python run.py

Paso 3: PCA sobre vectores medianos de angulos (train)
Paso 4: Clustering ensemble (k-means++, GMM, Jerarquico)
Paso 5: MLP con XYZ aplanado + etiquetas de cluster
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score,
)
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from config import (
    RESULTS_DIR, ALL_ANGLE_NAMES, CLASS_NAMES,
    PCA_VARIANCE_TARGET, CLUSTER_K_RANGE,
    MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_LR, MLP_EPOCHS,
    MLP_BATCH_SIZE, MLP_PATIENCE, MLP_SEED,
)
from data import load_and_split, get_xyz, get_labels, get_sequence_ids
from features import compute_joint_angles, trial_median


# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(MLP_SEED)
np.random.seed(MLP_SEED)


# =============================================================================
# Paso 3 -- PCA
# =============================================================================

def run_pca(train_medians, val_medians, test_medians):
    """
    Normaliza angulos (z-score fit en train) y aplica PCA.
    Retorna proyecciones, el objeto PCA, y el scaler.
    """
    angle_cols = ALL_ANGLE_NAMES

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_medians[angle_cols].values)
    X_val   = scaler.transform(val_medians[angle_cols].values)
    X_test  = scaler.transform(test_medians[angle_cols].values)

    pca = PCA()
    Z_train = pca.fit_transform(X_train)

    # Componentes que explican PCA_VARIANCE_TARGET
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)

    print(f"\n=== PCA ===")
    print(f"Varianza explicada por componente: {pca.explained_variance_ratio_[:6].round(4)}")
    print(f"Varianza acumulada: {cumvar[:6].round(4)}")
    print(f"Componentes para {PCA_VARIANCE_TARGET*100:.0f}% varianza: {n_components}")

    # Loadings: que angulos cargan mas en cada PC (hasta 6)
    n_show = min(6, pca.n_components_)
    print(f"\n  Loadings (top 3 angulos por PC):")
    for pc_i in range(n_show):
        loadings = pca.components_[pc_i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:3]
        desc = ", ".join(f"{angle_cols[j]}({loadings[j]:+.3f})" for j in top_idx)
        print(f"    PC{pc_i+1}: {desc}")

    Z_train = Z_train[:, :n_components]
    Z_val   = pca.transform(X_val)[:, :n_components]
    Z_test  = pca.transform(X_test)[:, :n_components]

    return Z_train, Z_val, Z_test, pca, scaler, n_components


# =============================================================================
# Paso 4 -- Clustering Ensemble
# =============================================================================

def run_clustering(Z_train, train_labels):
    """
    Ejecuta k-means++, GMM y Aglomerativo sobre Z_train.
    Retorna el mejor modelo y las etiquetas de cluster.
    """
    print(f"\n=== Clustering ===")

    best_score = -1
    best_result = None

    for k in CLUSTER_K_RANGE:
        results = {}

        # k-means++
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=MLP_SEED)
        km_labels = km.fit_predict(Z_train)
        km_sil = silhouette_score(Z_train, km_labels)
        results["kmeans"] = {"model": km, "labels": km_labels, "silhouette": km_sil}

        # GMM
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=MLP_SEED)
        gmm_labels = gmm.fit_predict(Z_train)
        gmm_sil = silhouette_score(Z_train, gmm_labels)
        results["gmm"] = {"model": gmm, "labels": gmm_labels, "silhouette": gmm_sil}

        # Aglomerativo
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        agg_labels = agg.fit_predict(Z_train)
        agg_sil = silhouette_score(Z_train, agg_labels)
        results["agglomerative"] = {"model": agg, "labels": agg_labels, "silhouette": agg_sil}

        # ARI entre los tres para medir consenso
        ari_km_gmm = adjusted_rand_score(km_labels, gmm_labels)
        ari_km_agg = adjusted_rand_score(km_labels, agg_labels)
        ari_gmm_agg = adjusted_rand_score(gmm_labels, agg_labels)
        mean_ari = (ari_km_gmm + ari_km_agg + ari_gmm_agg) / 3

        print(f"  k={k}: sil(km={km_sil:.3f}, gmm={gmm_sil:.3f}, agg={agg_sil:.3f}) "
              f"| ARI medio={mean_ari:.3f}")

        # Seleccionar por mejor silhouette promedio
        avg_sil = (km_sil + gmm_sil + agg_sil) / 3
        if avg_sil > best_score:
            best_score = avg_sil
            best_result = {
                "k": k,
                "results": results,
                "mean_ari": mean_ari,
                "avg_silhouette": avg_sil,
            }

    best_k = best_result["k"]
    # Usar k-means como etiqueta primaria (es el mas estable para asignar a nuevos puntos)
    cluster_labels = best_result["results"]["kmeans"]["labels"]
    cluster_model  = best_result["results"]["kmeans"]["model"]

    print(f"\n  Mejor k={best_k} (silhouette promedio={best_score:.3f})")

    # Mapeo: que clases originales caen en cada cluster
    print(f"\n  Composicion de clusters:")
    for c in range(best_k):
        mask = cluster_labels == c
        original = train_labels[mask]
        unique, counts = np.unique(original, return_counts=True)
        top3 = sorted(zip(unique, counts), key=lambda x: -x[1])[:3]
        desc = ", ".join(f"{CLASS_NAMES.get(int(cls), cls)}({cnt})" for cls, cnt in top3)
        print(f"    Cluster {c}: {mask.sum()} trials — {desc}")

    # Dendrograma con Ward linkage
    Z_link = linkage(Z_train, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z_link, truncate_mode="lastp", p=best_k * 3, ax=ax,
               leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f"Dendrograma Ward (k={best_k})")
    ax.set_ylabel("Distancia")
    dend_path = RESULTS_DIR / "dendrogram.png"
    fig.savefig(dend_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Dendrograma guardado en {dend_path}")

    return cluster_model, cluster_labels, best_result


# =============================================================================
# Paso 5 -- MLP Validation
# =============================================================================

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


def propagate_cluster_labels(df, trial_medians, cluster_labels):
    """
    Propaga la etiqueta de cluster de cada trial a todos sus frames.
    Retorna un array (N_frames,) con la etiqueta de cluster de cada frame.
    """
    trial_to_cluster = dict(zip(
        trial_medians["sequence_id"].values,
        cluster_labels
    ))
    return df["sequence_id"].map(trial_to_cluster).values


def run_mlp(train_df, val_df, test_df, train_clusters, val_clusters, test_clusters):
    """
    Entrena MLP con XYZ aplanado -> etiquetas de cluster.
    Evalua en test.
    """
    print(f"\n=== MLP ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = int(max(train_clusters.max(), val_clusters.max(), test_clusters.max()) + 1)

    # Datasets
    X_train = torch.from_numpy(train_df.iloc[:, 5:].values.astype("float32"))
    y_train = torch.from_numpy(train_clusters.astype("int64"))
    X_val   = torch.from_numpy(val_df.iloc[:, 5:].values.astype("float32"))
    y_val   = torch.from_numpy(val_clusters.astype("int64"))
    X_test  = torch.from_numpy(test_df.iloc[:, 5:].values.astype("float32"))
    y_test  = torch.from_numpy(test_clusters.astype("int64"))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=MLP_BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=MLP_BATCH_SIZE)

    model = MLP(63, MLP_HIDDEN_DIMS, num_classes, MLP_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)

    # Training loop
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

        # Validation
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
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{MLP_EPOCHS} | Val Acc: {val_acc:.3f}")

    # Test evaluation with best model
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
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)

    print(f"\n  Test Accuracy: {test_acc:.3f}")
    print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"\n  Confusion Matrix:\n{cm}")

    # Criterio de colapso: confusion reciproca > 50%
    print(f"\n  Confusiones reciprocas > 50%:")
    n_clusters = cm.shape[0]
    collapses = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            row_i = cm[i].sum()
            row_j = cm[j].sum()
            if row_i > 0 and row_j > 0:
                conf_ij = cm[i, j] / row_i
                conf_ji = cm[j, i] / row_j
                if conf_ij > 0.5 or conf_ji > 0.5:
                    collapses.append((i, j, conf_ij, conf_ji))
                    print(f"    Cluster {i} <-> {j}: {conf_ij:.1%} / {conf_ji:.1%}")

    if not collapses:
        print("    Ninguna.")

    return {
        "test_accuracy": float(test_acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "collapses": collapses,
        "best_val_acc": float(best_val_acc),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = timer()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Paso 1: cargar datos ──────────────────────────────────────────────
    print("=" * 60)
    print("Paso 1 -- Carga y split")
    print("=" * 60)
    train_df, val_df, test_df = load_and_split()

    # ── Paso 2: angulos + El Punto de Trial ───────────────────────────────
    print("\n" + "=" * 60)
    print("Paso 2 -- Angulos articulares + Punto de Trial")
    print("=" * 60)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        xyz = df.iloc[:, 5:].values.astype("float32").reshape(-1, 21, 3)
        angles = compute_joint_angles(xyz)
        medians = trial_median(df, angles)
        print(f"  {name}: {len(df):,} frames -> {len(medians):,} trials")
        if name == "Train":
            train_angles, train_medians = angles, medians
        elif name == "Val":
            val_angles, val_medians = angles, medians
        else:
            test_angles, test_medians = angles, medians

    # ── Paso 3: PCA ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Paso 3 -- PCA")
    print("=" * 60)
    Z_train, Z_val, Z_test, pca, scaler, n_comp = run_pca(
        train_medians, val_medians, test_medians
    )

    # ── Paso 4: Clustering ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Paso 4 -- Clustering Ensemble")
    print("=" * 60)
    cluster_model, train_cluster_labels, cluster_result = run_clustering(
        Z_train, train_medians["grasp_type"].values
    )

    # Asignar clusters a val y test via el modelo k-means
    val_cluster_labels  = cluster_model.predict(Z_val)
    test_cluster_labels = cluster_model.predict(Z_test)

    # Propagar etiquetas de cluster a frames individuales
    train_frame_clusters = propagate_cluster_labels(train_df, train_medians, train_cluster_labels)
    val_frame_clusters   = propagate_cluster_labels(val_df, val_medians, val_cluster_labels)
    test_frame_clusters  = propagate_cluster_labels(test_df, test_medians, test_cluster_labels)

    # Filtrar frames sin cluster asignado (secuencias que no sobrevivieron el filtro)
    train_valid = ~np.isnan(train_frame_clusters.astype(float))
    val_valid   = ~np.isnan(val_frame_clusters.astype(float))
    test_valid  = ~np.isnan(test_frame_clusters.astype(float))

    print(f"\n  Frames con cluster: train={train_valid.sum():,} "
          f"val={val_valid.sum():,} test={test_valid.sum():,}")

    # ── Paso 5: MLP ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Paso 5 -- MLP Validation")
    print("=" * 60)
    mlp_results = run_mlp(
        train_df[train_valid], val_df[val_valid], test_df[test_valid],
        train_frame_clusters[train_valid].astype(int),
        val_frame_clusters[val_valid].astype(int),
        test_frame_clusters[test_valid].astype(int),
    )

    # ── Guardar resultados ────────────────────────────────────────────────
    results = {
        "pca": {
            "n_components": n_comp,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "loadings": pca.components_[:6].tolist(),
            "angle_names": ALL_ANGLE_NAMES,
        },
        "clustering": {
            "best_k": cluster_result["k"],
            "avg_silhouette": cluster_result["avg_silhouette"],
            "mean_ari": cluster_result["mean_ari"],
        },
        "mlp": mlp_results,
        "elapsed_seconds": timer() - t0,
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {results_path}")
    print(f"Tiempo total: {results['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    main()
