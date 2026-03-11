"""
analyze_collapse_candidates.py -- Cross-reference Feix taxonomy cells with
empirical separability to propose principled collapse candidates.

Three criteria per pair:
  C1 (Theoretical): Same Feix cell -- explanatory, not decisional.
  C2 (Effect size):  Centroid distance / within-class spread < 1.0
  C3 (Empirical):    MLP confusion > 2x chance (7.1% for 28 classes)

Four cases:
  Case 1: Same cell + indistinguishable   -> Collapse (contact-dependent, Feix predicted)
  Case 2: Same cell + distinguishable     -> Keep (object shape forces different posture)
  Case 3: Diff cell + indistinguishable   -> Collapse (sensor limitation, novel finding)
  Case 4: Diff cell + distinguishable     -> Keep (expected)

References:
  - Feix et al. (2016), Fig. 4 and Section IV-A
  - Cohen (1988) for effect size interpretation
"""

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, inconsistent
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ALL_ANGLE_NAMES, CLASS_NAMES, PCA_VARIANCE_TARGET, RESULTS_DIR, MLP_SEED,
)
from data import load_and_split
from features import compute_joint_angles, trial_median

np.random.seed(MLP_SEED)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Feix Taxonomy Cell Mapping (from Feix et al., 2016, Fig. 4)
# =============================================================================

# Each cell is defined by: (Type, Opposition, VF, Thumb position)
# Grasps in the same cell differ mainly in object shape/contact, not hand config.

FEIX_CELL_DEFS = {
    "Pow_Palm_VF35_Abd":  ("Power", "Palm", "3-5", "Abducted"),
    "Pow_Palm_VF25_Abd":  ("Power", "Palm", "2-5", "Abducted"),
    "Pow_Palm_VF35_Add":  ("Power", "Palm", "3-5", "Adducted"),
    "Pow_Palm_VF25_Add":  ("Power", "Palm", "2-5", "Adducted"),
    "Pow_Pad_VF2_Abd":    ("Power", "Pad", "2", "Abducted"),
    "Pow_Pad_VF23_Abd":   ("Power", "Pad", "2-3", "Abducted"),
    "Pow_Pad_VF24_Abd":   ("Power", "Pad", "2-4", "Abducted"),
    "Pow_Pad_VF25_Abd":   ("Power", "Pad", "2-5", "Abducted"),
    "Int_Side_VF2_Abd":   ("Intermediate", "Side", "2", "Abducted"),
    "Int_Side_VF2_Add":   ("Intermediate", "Side", "2", "Adducted"),
    "Int_Side_VF34_Add":  ("Intermediate", "Side", "3-4", "Adducted"),
    "Pre_Pad_VF2_Abd":    ("Precision", "Pad", "2", "Abducted"),
    "Pre_Pad_VF23_Abd":   ("Precision", "Pad", "2-3", "Abducted"),
    "Pre_Pad_VF24_Abd":   ("Precision", "Pad", "2-4", "Abducted"),
    "Pre_Pad_VF25_Abd":   ("Precision", "Pad", "2-5", "Abducted"),
    "Pre_Pad_VF25_Add":   ("Precision", "Pad", "2-5", "Adducted"),
    "Pre_Side_VF3_Abd":   ("Precision", "Side", "3", "Abducted"),
}

# HOGraspNet class ID -> Feix cell ID
# Mapping derived from Feix et al. (2016), Fig. 4
CLASS_TO_CELL = {
    0:  "Pow_Palm_VF35_Abd",   # Large_Diameter (Feix #1)
    1:  "Pow_Palm_VF25_Abd",   # Small_Diameter (Feix #2)
    2:  "Pow_Palm_VF35_Add",   # Index_Finger_Extension (Feix #17)
    3:  "Pow_Pad_VF24_Abd",    # Extension_Type (Feix #18)
    4:  "Pre_Pad_VF25_Add",    # Parallel_Extension (Feix #22)
    5:  "Pow_Palm_VF25_Add",   # Palmar (Feix #30)
    6:  "Pow_Palm_VF25_Abd",   # Medium_Wrap (Feix #3)
    7:  "Pow_Palm_VF25_Add",   # Adducted_Thumb (Feix #4)
    8:  "Pow_Palm_VF25_Add",   # Light_Tool (Feix #5)
    9:  "Pow_Pad_VF25_Abd",    # Distal (Feix #19)
    10: "Pow_Pad_VF2_Abd",     # Ring (Feix #31)
    11: "Pow_Palm_VF25_Abd",   # Power_Disk (Feix #10)
    12: "Pow_Palm_VF25_Abd",   # Power_Sphere (Feix #11)
    13: "Pow_Pad_VF24_Abd",    # Sphere_4_Finger (Feix #26)
    14: "Pow_Pad_VF23_Abd",    # Sphere_3_Finger (Feix #28)
    15: "Int_Side_VF2_Add",    # Lateral (Feix #16)
    16: "Int_Side_VF2_Add",    # Stick (Feix #29)
    17: "Int_Side_VF2_Abd",    # Adduction_Grip (Feix #23)
    18: "Pre_Side_VF3_Abd",    # Writing_Tripod (Feix #20)
    19: "Int_Side_VF34_Add",   # Lateral_Tripod (Feix #25)
    20: "Pre_Pad_VF2_Abd",     # Palmar_Pinch (Feix #9)
    21: "Pre_Pad_VF2_Abd",     # Tip_Pinch (Feix #24)
    22: "Pre_Pad_VF2_Abd",     # Inferior_Pincer (Feix #33)
    23: "Pre_Pad_VF24_Abd",    # Prismatic_3_Finger (Feix #7)
    24: "Pre_Pad_VF25_Abd",    # Precision_Disk (Feix #12)
    25: "Pre_Pad_VF25_Abd",    # Precision_Sphere (Feix #13)
    26: "Pre_Pad_VF24_Abd",    # Quadpod (Feix #27)
    27: "Pre_Pad_VF23_Abd",    # Tripod (Feix #14)
}


def varimax(loadings, max_iter=100, tol=1e-6):
    """Varimax rotation (Kaiser, 1958)."""
    p, k = loadings.shape
    R = np.eye(k)
    d = 0
    for _ in range(max_iter):
        L = loadings @ R
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
# Step 1: Load data and compute PCA projections
# =============================================================================

print("=" * 70)
print("Step 1: Data + PCA + Varimax")
print("=" * 70)

train_df, val_df, test_df = load_and_split()

xyz = train_df.iloc[:, 5:].values.astype("float32").reshape(-1, 21, 3)
angles = compute_joint_angles(xyz)
medians = trial_median(train_df, angles)

scaler = StandardScaler()
X_train = scaler.fit_transform(medians[ALL_ANGLE_NAMES].values)

pca = PCA()
pca.fit(X_train)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)

loadings_raw = pca.components_[:n_comp].T
loadings_rotated, _ = varimax(loadings_raw)
Z_train = X_train @ loadings_rotated

grasp_types = medians["grasp_type"].values
unique_classes = sorted(np.unique(grasp_types))
n_classes = len(unique_classes)

print(f"  {len(medians):,} trials, {n_comp} RCs ({cumvar[n_comp-1]*100:.1f}% variance)")


# =============================================================================
# Step 2: Per-class statistics in PCA space
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 2: Per-class centroids and within-class spread")
print("=" * 70)

centroids = np.zeros((n_classes, n_comp))
spreads = np.zeros(n_classes)  # mean euclidean dist to own centroid

for i, cls in enumerate(unique_classes):
    mask = grasp_types == cls
    cls_trials = Z_train[mask]
    centroids[i] = cls_trials.mean(axis=0)
    # Within-class spread: mean distance from each trial to its centroid
    diffs = cls_trials - centroids[i]
    dists_to_centroid = np.sqrt((diffs ** 2).sum(axis=1))
    spreads[i] = dists_to_centroid.mean()

    name = CLASS_NAMES.get(int(cls), f"class_{cls}")
    cell = CLASS_TO_CELL[int(cls)]
    print(f"  [{cls:2d}] {name:<28s} n={mask.sum():4d}  "
          f"spread={spreads[i]:.2f}  cell={cell}")

mean_spread = spreads.mean()
print(f"\n  Global mean within-class spread: {mean_spread:.3f}")


# =============================================================================
# Step 3: Load MLP confusion matrix from previous run
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 3: Load MLP confusion matrix")
print("=" * 70)

json_path = RESULTS_DIR / "class_separability.json"
with open(json_path) as f:
    prev_results = json.load(f)

cm = np.array(prev_results["mlp_28"]["confusion_matrix"])
# Normalize by row to get confusion rates
cm_norm = cm.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm = cm_norm / row_sums

mlp_acc = prev_results["mlp_28"]["test_accuracy"]
mlp_f1 = prev_results["mlp_28"]["macro_f1"]
print(f"  MLP: {mlp_acc:.1%} accuracy, {mlp_f1:.3f} macro F1")
print(f"  Chance level: {1/28:.1%}, threshold (2x chance): {2/28:.1%}")

CHANCE = 1.0 / 28
CONFUSION_THRESHOLD = 2 * CHANCE  # 7.14%


# =============================================================================
# Step 4: Pairwise analysis -- all 378 pairs
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 4: Pairwise analysis (378 pairs)")
print("=" * 70)

dist_matrix = squareform(pdist(centroids, metric="euclidean"))

pair_results = []
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        ci = int(unique_classes[i])
        cj = int(unique_classes[j])

        # C1: Same Feix cell?
        cell_i = CLASS_TO_CELL[ci]
        cell_j = CLASS_TO_CELL[cj]
        same_cell = cell_i == cell_j

        # C2: Effect size (centroid distance / pooled within-class spread)
        d = dist_matrix[i, j]
        pooled_spread = (spreads[i] + spreads[j]) / 2
        effect_size = d / pooled_spread if pooled_spread > 0 else float("inf")

        # C3: MLP confusion (max of both directions)
        conf_ij = cm_norm[ci, cj]  # true=ci, pred=cj
        conf_ji = cm_norm[cj, ci]  # true=cj, pred=ci
        max_conf = max(conf_ij, conf_ji)

        # Classify into 4 cases
        # Decision driven by C3 (MLP confusion) as the primary criterion.
        # C2 (effect size) provides supporting evidence but is not sufficient
        # alone -- the MLP sees the full 63D XYZ space and is the stronger test.
        empirically_indistinguishable = max_conf > CONFUSION_THRESHOLD

        if same_cell and empirically_indistinguishable:
            case = 1  # Same cell, indistinguishable -> collapse
        elif same_cell and not empirically_indistinguishable:
            case = 2  # Same cell, distinguishable -> keep (interesting!)
        elif not same_cell and empirically_indistinguishable:
            case = 3  # Diff cell, indistinguishable -> collapse (novel!)
        else:
            case = 4  # Diff cell, distinguishable -> keep

        pair_results.append({
            "class_a": ci, "class_b": cj,
            "name_a": CLASS_NAMES[ci], "name_b": CLASS_NAMES[cj],
            "cell_a": cell_i, "cell_b": cell_j,
            "same_cell": same_cell,
            "centroid_dist": float(d),
            "pooled_spread": float(pooled_spread),
            "effect_size": float(effect_size),
            "conf_a_to_b": float(conf_ij),
            "conf_b_to_a": float(conf_ji),
            "max_conf": float(max_conf),
            "c2_overlap": effect_size < 1.0,
            "c3_confused": max_conf > CONFUSION_THRESHOLD,
            "case": case,
        })


# =============================================================================
# Step 5: Report by case
# =============================================================================

def print_pair(r, rank=None):
    prefix = f"  {rank:3d}. " if rank else "  "
    c2_flag = "ES<1" if r["c2_overlap"] else f"ES={r['effect_size']:.2f}"
    c3_flag = f"conf={r['max_conf']:.1%}" if r["c3_confused"] else f"conf={r['max_conf']:.1%}"
    print(f"{prefix}[{r['class_a']:2d}] {r['name_a']:<25s} <-> "
          f"[{r['class_b']:2d}] {r['name_b']:<25s}  "
          f"{c2_flag:<10s} {c3_flag:<12s} "
          f"cell={'SAME' if r['same_cell'] else 'DIFF'}")


# --- Case 1: Same cell + indistinguishable (collapse, expected) ---
case1 = [r for r in pair_results if r["case"] == 1]
case1.sort(key=lambda x: x["effect_size"])

print(f"\n{'=' * 70}")
print(f"CASE 1: Same Feix cell + empirically indistinguishable ({len(case1)} pairs)")
print(f"  -> COLLAPSE (contact-dependent, Feix Section IV-A predicted this)")
print("=" * 70)
for i, r in enumerate(case1, 1):
    print_pair(r, i)

# --- Case 2: Same cell + distinguishable (keep, surprising) ---
case2 = [r for r in pair_results if r["case"] == 2]
case2.sort(key=lambda x: x["effect_size"], reverse=True)

print(f"\n{'=' * 70}")
print(f"CASE 2: Same Feix cell + empirically distinguishable ({len(case2)} pairs)")
print(f"  -> KEEP SEPARATE (object shape forces different posture)")
print("=" * 70)
if case2:
    for i, r in enumerate(case2, 1):
        print_pair(r, i)
else:
    print("  None found.")

# --- Case 3: Different cell + indistinguishable (collapse, novel) ---
case3 = [r for r in pair_results if r["case"] == 3]
case3.sort(key=lambda x: x["effect_size"])

print(f"\n{'=' * 70}")
print(f"CASE 3: Different Feix cell + empirically indistinguishable ({len(case3)} pairs)")
print(f"  -> COLLAPSE (sensor limitation, novel finding)")
print("=" * 70)
for i, r in enumerate(case3, 1):
    print_pair(r, i)

# --- Case 4: Different cell + distinguishable (keep, expected) ---
case4 = [r for r in pair_results if r["case"] == 4]

print(f"\n{'=' * 70}")
print(f"CASE 4: Different Feix cell + empirically distinguishable ({len(case4)} pairs)")
print(f"  -> KEEP SEPARATE (expected)")
print("=" * 70)
print(f"  {len(case4)} pairs (not listed, this is the majority)")


# =============================================================================
# Step 6: Intra-cell analysis (all cells with 2+ HOGraspNet classes)
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 6: Intra-cell analysis (cells with 2+ classes)")
print("=" * 70)

from collections import defaultdict
cell_to_classes = defaultdict(list)
for cls_id, cell in CLASS_TO_CELL.items():
    cell_to_classes[cell].append(cls_id)

for cell, classes in sorted(cell_to_classes.items(), key=lambda x: -len(x[1])):
    if len(classes) < 2:
        continue

    cell_def = FEIX_CELL_DEFS[cell]
    print(f"\n  Cell: {cell}")
    print(f"    {cell_def[0]} / {cell_def[1]} / VF {cell_def[2]} / Thumb {cell_def[3]}")
    print(f"    Classes: {', '.join(f'[{c}] {CLASS_NAMES[c]}' for c in sorted(classes))}")

    # All intra-cell pairs
    for ii in range(len(classes)):
        for jj in range(ii + 1, len(classes)):
            ci, cj = sorted(classes)[ii], sorted(classes)[jj]
            # Find this pair in results
            match = [r for r in pair_results
                     if (r["class_a"] == ci and r["class_b"] == cj)
                     or (r["class_a"] == cj and r["class_b"] == ci)]
            if match:
                r = match[0]
                verdict = "COLLAPSE" if r["case"] in (1, 3) else "KEEP"
                c2_str = f"ES={r['effect_size']:.2f}"
                c3_str = f"conf={r['max_conf']:.1%}"
                print(f"    [{ci:2d}] {CLASS_NAMES[ci]:<22s} <-> "
                      f"[{cj:2d}] {CLASS_NAMES[cj]:<22s}  "
                      f"{c2_str:<10s} {c3_str:<12s} -> {verdict}")


# =============================================================================
# Step 7: Dendrogram with Feix cell annotations
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 7: Annotated dendrogram")
print("=" * 70)

Z_link = linkage(centroids, method="ward")

# Inconsistency coefficients
inc = inconsistent(Z_link, d=2)
print(f"  Inconsistency stats (depth=2):")
print(f"    Mean: {inc[:, 0].mean():.3f}")
print(f"    Std:  {inc[:, 1].mean():.3f}")
print(f"    Max inconsistency coeff: {inc[:, 3].max():.3f}")
print(f"    Median inconsistency coeff: {np.median(inc[:, 3]):.3f}")

# Color labels by cell membership
cell_colors = {}
color_palette = plt.cm.get_cmap("tab20")
unique_cells = sorted(set(CLASS_TO_CELL.values()))
for idx, cell in enumerate(unique_cells):
    cell_colors[cell] = color_palette(idx % 20)

class_labels = []
for cls in unique_classes:
    cell = CLASS_TO_CELL[int(cls)]
    short_cell = cell.replace("Pow_", "P/").replace("Pre_", "Pr/").replace("Int_", "I/")
    class_labels.append(f"[{cls}] {CLASS_NAMES[int(cls)][:15]}\n{short_cell}")

fig, ax = plt.subplots(figsize=(20, 10))
dendrogram(Z_link, labels=class_labels, ax=ax,
           leaf_rotation=90, leaf_font_size=7, color_threshold=0)
ax.set_title("Ward Dendrogram -- 28 class centroids in rotated PCA space\n"
             "(labels show Feix cell assignment)")
ax.set_ylabel("Ward distance")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "dendrogram_feix_annotated.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {RESULTS_DIR / 'dendrogram_feix_annotated.png'}")


# =============================================================================
# Step 8: Summary table for the report
# =============================================================================

print(f"\n{'=' * 70}")
print("SUMMARY: Collapse candidates (Cases 1 + 3)")
print("=" * 70)

# All collapse candidates (C3 > threshold), sorted by confusion rate
all_collapse = [r for r in pair_results if r["c3_confused"]]
all_collapse.sort(key=lambda x: -x["max_conf"])

print(f"\n  All collapse candidates (MLP confusion > {CONFUSION_THRESHOLD:.1%}):")
print(f"  {'#':<4s} {'Class A':<25s} {'Class B':<25s} {'ES':<7s} {'Conf':<8s} "
      f"{'Cell':<6s} {'Case':<6s} {'Evidence'}")
print(f"  {'-'*4} {'-'*25} {'-'*25} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*20}")
for i, r in enumerate(all_collapse, 1):
    cell_str = "SAME" if r["same_cell"] else "DIFF"
    # Evidence strength: C2+C3 = strong, C3 only = moderate
    if r["c2_overlap"] and r["c3_confused"]:
        evidence = "C2+C3 (strong)"
    else:
        evidence = "C3 only (moderate)"
    print(f"  {i:<4d} {r['name_a']:<25s} {r['name_b']:<25s} "
          f"{r['effect_size']:<7.2f} {r['max_conf']:<8.1%} {cell_str:<6s} "
          f"{r['case']:<6d} {evidence}")


# =============================================================================
# Step 9: Confusion-based dendrogram (Option B)
# =============================================================================
# Instead of clustering on PCA centroids, cluster directly on the MLP confusion
# matrix. This uses the primary criterion (empirical distinguishability) to
# determine which classes should merge.
#
# Distance: d(i,j) = 1 - max(conf_norm[i,j], conf_norm[j,i])
#   - High mutual confusion -> small distance -> merge early
#   - No confusion -> distance ~ 1.0 -> merge late
#
# The dendrogram is cut using inconsistency coefficients (natural breaks)
# rather than an arbitrary threshold.

print(f"\n{'=' * 70}")
print("Step 9: Confusion-based dendrogram")
print("=" * 70)

# Build symmetric confusion-based distance matrix
conf_sim = np.zeros((28, 28))
for i in range(28):
    for j in range(28):
        if i == j:
            conf_sim[i, j] = 1.0  # self-similarity = 1
        else:
            conf_sim[i, j] = max(cm_norm[i, j], cm_norm[j, i])

conf_dist = 1.0 - conf_sim
# Ensure diagonal is 0 and matrix is symmetric
np.fill_diagonal(conf_dist, 0.0)
conf_dist = (conf_dist + conf_dist.T) / 2  # enforce symmetry

# Convert to condensed form for linkage
conf_dist_condensed = squareform(conf_dist, checks=False)

# Ward linkage on confusion-based distances
Z_conf = linkage(conf_dist_condensed, method="ward")

# Inconsistency analysis to find natural cuts
for depth in [2, 3, 4]:
    inc_conf = inconsistent(Z_conf, d=depth)
    max_inc = inc_conf[:, 3].max()
    median_inc = np.median(inc_conf[:, 3])
    print(f"  Inconsistency (depth={depth}): median={median_inc:.3f}, max={max_inc:.3f}")

# Use depth=2 for the cut analysis
inc_conf = inconsistent(Z_conf, d=2)

# Find natural cut: where inconsistency jumps above threshold
# Standard threshold: inconsistency > 1.0 means the merge is > 1 std above
# the mean of its children's merge distances
INC_THRESHOLD = 1.0

# Walk the linkage matrix and find which merges are "inconsistent"
n_leaves = 28
print(f"\n  Merge order (last 15 merges, showing inconsistency):")
print(f"  {'Merge':<6s} {'Cluster A':<30s} {'Cluster B':<30s} "
      f"{'Dist':<8s} {'Inc':<8s} {'Cut?'}")
print(f"  {'-'*6} {'-'*30} {'-'*30} {'-'*8} {'-'*8} {'-'*5}")

# Build cluster membership as we go
cluster_members = {i: [i] for i in range(28)}

for merge_idx in range(len(Z_conf)):
    c1_idx = int(Z_conf[merge_idx, 0])
    c2_idx = int(Z_conf[merge_idx, 1])
    merge_dist = Z_conf[merge_idx, 2]
    inc_coeff = inc_conf[merge_idx, 3]
    new_cluster_id = n_leaves + merge_idx

    members_1 = cluster_members.get(c1_idx, [c1_idx])
    members_2 = cluster_members.get(c2_idx, [c2_idx])
    cluster_members[new_cluster_id] = members_1 + members_2

    # Only print last 15 merges
    if merge_idx >= len(Z_conf) - 15:
        def fmt_cluster(members):
            if len(members) <= 3:
                return ", ".join(f"{CLASS_NAMES[m][:12]}" for m in sorted(members))
            return (", ".join(f"{CLASS_NAMES[m][:12]}" for m in sorted(members)[:2])
                    + f" +{len(members)-2} more")

        cut_str = "<-- CUT" if inc_coeff > INC_THRESHOLD else ""
        print(f"  {merge_idx+1:<6d} {fmt_cluster(members_1):<30s} "
              f"{fmt_cluster(members_2):<30s} "
              f"{merge_dist:<8.3f} {inc_coeff:<8.3f} {cut_str}")


# Explore multiple cut levels using maxclust criterion
from scipy.cluster.hierarchy import fcluster

print(f"\n  Exploring cut levels (k = number of groups):")
print(f"  {'k':<4s} {'Singletons':<12s} {'Merged':<8s} {'Largest group':<15s} "
      f"{'Avg intra-group conf'}")
print(f"  {'-'*4} {'-'*12} {'-'*8} {'-'*15} {'-'*25}")

cut_results = {}
for k in range(5, 25):
    labels_k = fcluster(Z_conf, t=k, criterion="maxclust")
    groups_k = defaultdict(list)
    for cls_idx, label in enumerate(labels_k):
        groups_k[label].append(cls_idx)

    n_singletons = sum(1 for g in groups_k.values() if len(g) == 1)
    n_merged = sum(1 for g in groups_k.values() if len(g) > 1)
    largest = max(len(g) for g in groups_k.values())

    # Average intra-group confusion for merged groups
    intra_confs = []
    for members in groups_k.values():
        if len(members) < 2:
            continue
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                ci, cj = members[ii], members[jj]
                intra_confs.append(max(cm_norm[ci, cj], cm_norm[cj, ci]))
    avg_intra = np.mean(intra_confs) if intra_confs else 0

    cut_results[k] = {
        "labels": labels_k, "groups": dict(groups_k),
        "n_singletons": n_singletons, "largest": largest,
        "avg_intra_conf": avg_intra,
    }
    print(f"  {k:<4d} {n_singletons:<12d} {n_merged:<8d} {largest:<15d} {avg_intra:.1%}")

# Select the cut where the largest group is still reasonable (<=4)
# and average intra-group confusion is maximized (merging confused classes)
best_k = None
for k in sorted(cut_results.keys()):
    if cut_results[k]["largest"] <= 4:
        best_k = k
        break

if best_k is None:
    best_k = max(cut_results.keys())

print(f"\n  Selected k={best_k} (largest group <= 4 members)")

# Display the selected taxonomy
labels_best = cut_results[best_k]["labels"]
groups_best = defaultdict(list)
for cls_idx, label in enumerate(labels_best):
    groups_best[label].append(cls_idx)

sorted_groups = sorted(groups_best.items(), key=lambda x: -len(x[1]))
n_groups = best_k

print(f"\n  Proposed taxonomy ({n_groups} classes):")
print(f"  {'Group':<6s} {'Size':<5s} {'Members':<65s} {'Feix cells'}")
print(f"  {'-'*6} {'-'*5} {'-'*65} {'-'*30}")

proposed_taxonomy = {}
for group_id, (label, members) in enumerate(sorted_groups):
    members_sorted = sorted(members)
    class_str = ", ".join(f"[{m}]{CLASS_NAMES[m][:12]}" for m in members_sorted)
    cells = set(CLASS_TO_CELL[m] for m in members_sorted)
    cell_str = ", ".join(sorted(cells))

    if len(members_sorted) == 1:
        merge_type = "singleton"
    elif len(cells) == 1:
        merge_type = "same-cell"
    else:
        merge_type = "cross-cell"

    proposed_taxonomy[group_id] = {
        "members": members_sorted,
        "names": [CLASS_NAMES[m] for m in members_sorted],
        "cells": list(cells),
        "merge_type": merge_type,
    }

    size = len(members_sorted)
    print(f"  {group_id+1:<6d} {size:<5d} {class_str:<65s} {cell_str}")

# Validate: for each multi-class group, show all intra-group confusion rates
print(f"\n  Validation -- intra-group confusion rates:")
for group_id, info in proposed_taxonomy.items():
    members = info["members"]
    if len(members) < 2:
        continue
    print(f"\n  Group {group_id+1} ({info['merge_type']}): "
          f"{', '.join(info['names'])}")
    for ii in range(len(members)):
        for jj in range(ii + 1, len(members)):
            ci, cj = members[ii], members[jj]
            c_ij = cm_norm[ci, cj]
            c_ji = cm_norm[cj, ci]
            same = "SAME" if CLASS_TO_CELL[ci] == CLASS_TO_CELL[cj] else "DIFF"
            print(f"    [{ci:2d}]{CLASS_NAMES[ci]:<20s} <-> [{cj:2d}]{CLASS_NAMES[cj]:<20s}  "
                  f"{c_ij:.1%} / {c_ji:.1%}  cell={same}")

# Confusion-based dendrogram plot with cut line
fig, ax = plt.subplots(figsize=(20, 10))
class_labels_conf = [f"[{i}] {CLASS_NAMES[i][:15]}" for i in range(28)]

# Color threshold: the merge distance that produces n_groups clusters
cut_dist = Z_conf[-(n_groups - 1), 2] if n_groups > 1 else 0

dendrogram(Z_conf, labels=class_labels_conf, ax=ax,
           leaf_rotation=90, leaf_font_size=7,
           color_threshold=cut_dist)
ax.set_title("Confusion-based Ward Dendrogram\n"
             "d(i,j) = 1 - max(conf(i->j), conf(j->i))")
ax.set_ylabel("Ward distance (confusion-based)")
ax.axhline(y=cut_dist, color="red", linestyle="--", alpha=0.7,
           label=f"Cut -> {n_groups} groups")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "dendrogram_confusion_based.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {RESULTS_DIR / 'dendrogram_confusion_based.png'}")


# =============================================================================
# Save results
# =============================================================================

output = {
    "criteria": {
        "c2_threshold": "effect_size < 1.0 (centroid dist < within-class spread)",
        "c3_threshold": f"max confusion > {CONFUSION_THRESHOLD:.4f} (2x chance for 28 classes)",
        "c1_role": "explanatory (same Feix cell), not decisional",
    },
    "global_stats": {
        "n_classes": n_classes,
        "n_pairs": len(pair_results),
        "n_pca_components": n_comp,
        "pca_variance": float(cumvar[n_comp - 1]),
        "mean_within_class_spread": float(mean_spread),
        "chance_level": float(CHANCE),
    },
    "case_counts": {
        "case_1_same_cell_indistinguishable": len(case1),
        "case_2_same_cell_distinguishable": len(case2),
        "case_3_diff_cell_indistinguishable": len(case3),
        "case_4_diff_cell_distinguishable": len(case4),
    },
    "feix_cell_mapping": {
        int(k): v for k, v in CLASS_TO_CELL.items()
    },
    "per_class_spread": {
        int(unique_classes[i]): {
            "name": CLASS_NAMES[int(unique_classes[i])],
            "spread": float(spreads[i]),
            "cell": CLASS_TO_CELL[int(unique_classes[i])],
        }
        for i in range(n_classes)
    },
    "collapse_candidates": [
        {**r, "evidence": "C2+C3" if r["c2_overlap"] and r["c3_confused"] else "C3_only"}
        for r in all_collapse
    ],
    "all_pairs": pair_results,
    "confusion_dendrogram": {
        "method": "Ward linkage on d(i,j) = 1 - max(conf_ij, conf_ji)",
        "selected_k": best_k,
        "selection_criterion": "smallest k where largest group <= 4 members",
        "n_groups": n_groups,
        "proposed_taxonomy": proposed_taxonomy,
        "cut_sweep": {
            str(k): {
                "n_singletons": v["n_singletons"],
                "largest_group": v["largest"],
                "avg_intra_conf": float(v["avg_intra_conf"]),
            }
            for k, v in cut_results.items()
        },
    },
}

def sanitize(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

out_path = RESULTS_DIR / "collapse_candidates.json"
with open(out_path, "w") as f:
    json.dump(sanitize(output), f, indent=2)
print(f"\nResults saved to {out_path}")
