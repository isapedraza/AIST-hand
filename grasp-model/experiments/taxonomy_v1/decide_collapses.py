"""
decide_collapses.py
-------------------
Two-step collapse decision methodology:

  Step 1 — GCN confusion validation
      Max(confusion(A->B), confusion(B->A)) from the trained GCN.
      A pair collapses if and only if this value exceeds gcn_thresh.
      Source: confusion_matrix_norm_gcn.csv

  Step 2 — Synergy space description (descriptive only, no threshold)
      Pairwise centroid distances in PCA-varimax space (model-independent).
      Used to describe each collapsed pair -- not to gate the collapse decision.
      Source: class_separability.json -> class_centroids_pca

Outputs (all in results/):
  collapse_decisions.csv       -- all pairs, sorted by c_gcn
  proposed_taxonomy.json       -- final grouped classes
  scatter_synergy_vs_gcn.png   -- 2D decision plot
  four_case_summary.txt        -- breakdown by Feix cell agreement
"""

import json
import itertools
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# -- Class metadata -------------------------------------------------------------

CLASS_NAMES = {
    0:  "Large_Diameter",     1:  "Small_Diameter",
    2:  "Index_Finger_Ext",   3:  "Extension_Type",
    4:  "Parallel_Ext",       5:  "Palmar",
    6:  "Medium_Wrap",        7:  "Adducted_Thumb",
    8:  "Light_Tool",         9:  "Distal",
    10: "Ring",               11: "Power_Disk",
    12: "Power_Sphere",       13: "Sphere_4_Finger",
    14: "Sphere_3_Finger",    15: "Lateral",
    16: "Stick",              17: "Adduction_Grip",
    18: "Writing_Tripod",     19: "Lateral_Tripod",
    20: "Palmar_Pinch",       21: "Tip_Pinch",
    22: "Inferior_Pincer",    23: "Prismatic_3F",
    24: "Precision_Disk",     25: "Precision_Sphere",
    26: "Quadpod",            27: "Tripod",
}

FEIX_CELLS = {
    0: [0, 1, 6, 11, 12],
    1: [5, 7, 8],
    2: [3, 13],
    3: [15, 16],
    4: [20, 21, 22],
    5: [23, 26],
    6: [24, 25],
}
CLASS_TO_CELL = {}
for cell, members in FEIX_CELLS.items():
    for m in members:
        CLASS_TO_CELL[m] = cell
for m in [2, 4, 9, 10, 14, 17, 18, 19, 27]:
    CLASS_TO_CELL[m] = -1  # singleton

N = 28

# -- Union-Find -----------------------------------------------------------------

def make_uf(n):
    return list(range(n))

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, x, y):
    px, py = find(parent, x), find(parent, y)
    if px != py:
        parent[px] = py

# -- Main -----------------------------------------------------------------------

def main(gcn_thresh: float, results_dir: Path):

    # Load centroids (descriptive only)
    sep_path = results_dir / "class_separability.json"
    with open(sep_path) as f:
        sep = json.load(f)
    centroids = np.array([sep["class_centroids_pca"][str(i)] for i in range(N)])

    # Pairwise centroid distances [28, 28] -- descriptive, no threshold applied
    dist_matrix = cdist(centroids, centroids, metric="euclidean")

    # Load GCN confusion matrix
    gcn_cm = pd.read_csv(results_dir / "confusion_matrix_norm_gcn.csv",
                         index_col=0).values.astype(float)

    # Print GCN confusion distribution to document the gap that justifies gcn_thresh
    upper_gcn = sorted(
        [max(gcn_cm[i, j], gcn_cm[j, i]) for i, j in itertools.combinations(range(N), 2)],
        reverse=True
    )
    print("-- GCN max-confusion distribution (378 pairs) ----------------------")
    for pct in [97, 95, 90, 75, 50]:
        idx = int(len(upper_gcn) * (100 - pct) / 100)
        print(f"  p{pct}: {upper_gcn[idx]:.4f}")
    print(f"  pairs above gcn_thresh ({gcn_thresh}): "
          f"{sum(1 for c in upper_gcn if c > gcn_thresh)}")
    print(f"  gap: [{upper_gcn[sum(1 for c in upper_gcn if c > gcn_thresh)-1]:.4f}, "
          f"{upper_gcn[sum(1 for c in upper_gcn if c > gcn_thresh)]:.4f}]")
    print()

    # -- Build pairwise decision table ------------------------------------------
    rows = []
    for i, j in itertools.combinations(range(N), 2):
        d_syn  = dist_matrix[i, j]
        c_gcn  = max(gcn_cm[i, j], gcn_cm[j, i])
        same_cell = (CLASS_TO_CELL[i] == CLASS_TO_CELL[j]
                     and CLASS_TO_CELL[i] != -1)
        collapse = c_gcn > gcn_thresh

        rows.append(dict(
            class_a=i, class_b=j,
            name_a=CLASS_NAMES[i], name_b=CLASS_NAMES[j],
            d_synergy=round(d_syn, 4), c_gcn=round(c_gcn, 4),
            same_feix=same_cell, collapse=collapse,
        ))

    df = pd.DataFrame(rows)
    df_collapse = (df[df["collapse"]]
                   .sort_values("c_gcn", ascending=False)
                   .reset_index(drop=True))

    df_collapse.to_csv(results_dir / "collapse_decisions.csv", index=False)
    print("-- Collapse candidates ---------------------------------------------")
    print(df_collapse[["name_a", "name_b", "d_synergy", "c_gcn",
                        "same_feix"]].to_string(index=False))
    print()

    # -- Four-case summary ------------------------------------------------------
    lines = ["Four-case framework\n" + "="*60 + "\n"]

    case_labels = {
        ("same", True):  "Case A -- Same Feix cell, GCN confused    -> collapse (Feix and data agree)",
        ("same", False): "Case B -- Same Feix cell, GCN separates   -> KEEP SEPARATE (Feix wrong)",
        ("diff", True):  "Case C -- Diff Feix cell, GCN confused    -> data-driven collapse",
        ("diff", False): "Case D -- Diff Feix cell, GCN separates   -> keep separate (correct)",
    }

    cases = defaultdict(list)
    for _, row in df.iterrows():
        key = ("same" if row["same_feix"] else "diff", row["collapse"])
        cases[key].append(f"  {row['name_a']} + {row['name_b']} "
                          f"(d_syn={row['d_synergy']}, c_gcn={row['c_gcn']})")

    for key, label in case_labels.items():
        lines.append(f"\n{label}")
        lines.append(f"  ({len(cases[key])} pairs)")
        for entry in cases[key][:10]:
            lines.append(entry)
        if len(cases[key]) > 10:
            lines.append(f"  ... and {len(cases[key])-10} more")

    summary = "\n".join(lines)
    print(summary)
    (results_dir / "four_case_summary.txt").write_text(summary)

    # -- Scatter plot -----------------------------------------------------------
    collapse_color = "#d62728"
    keep_color     = "#aec7e8"

    fig, ax = plt.subplots(figsize=(13, 8))

    df_keep = df[~df["collapse"]]
    ax.scatter(df_keep["d_synergy"], df_keep["c_gcn"],
               c=keep_color, alpha=0.45, s=22, label="keep", zorder=2)
    ax.scatter(df_collapse["d_synergy"], df_collapse["c_gcn"],
               c=collapse_color, alpha=0.75, s=36, label="collapse", zorder=3)

    for _, row in df_collapse.iterrows():
        ax.annotate(
            f"{row['name_a'][:8]}\n{row['name_b'][:8]}",
            (row["d_synergy"], row["c_gcn"]),
            fontsize=5.5, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
            color=collapse_color,
        )

    ax.axhline(gcn_thresh, color="gray", linestyle="--", linewidth=1,
               label=f"gcn_thresh = {gcn_thresh}")

    ax.set_xlabel("Synergy space centroid distance (PCA-varimax, descriptive)", fontsize=11)
    ax.set_ylabel("GCN max confusion rate  [max(CM[i,j], CM[j,i])]", fontsize=11)
    ax.set_title("Collapse decision space -- criterion: GCN confusion only", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(results_dir / "scatter_synergy_vs_gcn.png", dpi=150)
    plt.close()
    print("Scatter saved.")

    # -- Build proposed taxonomy via union-find ---------------------------------
    parent = make_uf(N)
    for _, row in df_collapse.iterrows():
        union(parent, int(row["class_a"]), int(row["class_b"]))

    groups = defaultdict(list)
    for i in range(N):
        groups[find(parent, i)].append(i)

    taxonomy = []
    for members in sorted(groups.values(), key=lambda m: -len(m)):
        taxonomy.append(dict(
            new_id=len(taxonomy),
            members=members,
            names=[CLASS_NAMES[m] for m in members],
            size=len(members),
            type="collapsed" if len(members) > 1 else "singleton",
        ))

    with open(results_dir / "proposed_taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    n_classes = len(taxonomy)
    print(f"\n-- Proposed taxonomy: {N} -> {n_classes} classes ------------------")
    for t in taxonomy:
        tag = f"[{t['size']}]" if t["size"] > 1 else "   "
        print(f"  {tag} {t['new_id']:2d}: {', '.join(t['names'])}")

    print(f"\nResults saved to {results_dir}/")
    return taxonomy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcn_thresh", type=float, default=0.15,
                        help="Max GCN confusion rate above which classes are collapsed")
    parser.add_argument("--results_dir", type=str,
                        default="experiments/taxonomy_v1/results")
    args = parser.parse_args()
    main(args.gcn_thresh, Path(args.results_dir))
