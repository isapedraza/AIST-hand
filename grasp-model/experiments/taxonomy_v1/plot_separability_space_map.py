"""
plot_separability_space_map.py -- RC1/RC2 centroid map for taxonomy_v1.

Builds a publication-friendly 2D map from class centroids stored in:
  results/class_separability.json

Run from AIST-hand/:
  .venv/bin/python grasp-model/experiments/taxonomy_v1/plot_separability_space_map.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from config import CLASS_NAMES, RESULTS_DIR

matplotlib.use("Agg")

JSON_PATH = RESULTS_DIR / "class_separability.json"
OUT_PATH = RESULTS_DIR / "separability_space_map_rc1_rc2.png"

GROUPS = {
    "Power": ["Large_Diameter", "Small_Diameter", "Medium_Wrap", "Light_Tool", "Power_Disk", "Power_Sphere"],
    "Precision": [
        "Precision_Disk",
        "Precision_Sphere",
        "Prismatic_3_Finger",
        "Tip_Pinch",
        "Palmar_Pinch",
        "Writing_Tripod",
        "Lateral_Tripod",
        "Tripod",
        "Quadpod",
        "Sphere_3_Finger",
        "Sphere_4_Finger",
    ],
    "Lateral": ["Lateral", "Adduction_Grip", "Adducted_Thumb"],
    "Extension": ["Index_Finger_Extension", "Extension_Type", "Parallel_Extension"],
    "Pinch": ["Distal", "Ring", "Palmar", "Inferior_Pincer", "Stick"],
}

GROUP_COLORS = {
    "Power": "#e07070",
    "Precision": "#70aae0",
    "Lateral": "#70c870",
    "Extension": "#e0c060",
    "Pinch": "#c070e0",
    "Other": "#aaaaaa",
}


def class_group(name: str) -> str:
    for group_name, members in GROUPS.items():
        if name in members:
            return group_name
    return "Other"


def pretty(name: str) -> str:
    return name.replace("_", " ")


def main() -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    var = data["pca"]["explained_variance_ratio"]
    centroids = data["class_centroids_pca"]

    class_ids = sorted(int(k) for k in centroids.keys())
    xy = np.array([[centroids[str(cid)][0], centroids[str(cid)][1]] for cid in class_ids], dtype=np.float64)
    names = [CLASS_NAMES[cid] for cid in class_ids]
    groups = [class_group(name) for name in names]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Soft reference cloud from centroid locations to keep visual continuity.
    ax.scatter(xy[:, 0], xy[:, 1], s=500, alpha=0.06, color="#555577", linewidths=0, zorder=1)

    y_span = max(1e-6, float(np.max(xy[:, 1]) - np.min(xy[:, 1])))
    y_off = 0.016 * y_span

    for (x, y), name, color in zip(xy, names, colors):
        ax.scatter(x, y, s=130, color=color, edgecolors="white", linewidths=0.7, zorder=3)
        text = ax.text(
            x,
            y + y_off,
            pretty(name),
            fontsize=8.5,
            color="white",
            ha="center",
            va="bottom",
            zorder=4,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.6, foreground="#0f0f1a")])

    for group_name in ["Power", "Precision", "Lateral", "Extension", "Pinch"]:
        ax.scatter(
            [],
            [],
            s=90,
            color=GROUP_COLORS[group_name],
            edgecolors="white",
            linewidths=0.7,
            label=group_name,
        )
    ax.legend(
        framealpha=0.2,
        facecolor="#1a1a2e",
        edgecolor="#555577",
        labelcolor="white",
        fontsize=10,
        loc="upper right",
    )

    ax.set_xlabel(f"RC1 ({var[0]:.1%} variance)", color="#aaaacc", fontsize=12)
    ax.set_ylabel(f"RC2 ({var[1]:.1%} variance)", color="#aaaacc", fontsize=12)
    ax.tick_params(colors="#555577", labelsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.set_title(
        "Landmark class separability space  --  28 Feix classes (HOGraspNet, S1)\n"
        "Class centroids in PCA-varimax space (RC1 vs RC2)",
        color="white",
        fontsize=14,
        pad=14,
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=170, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

