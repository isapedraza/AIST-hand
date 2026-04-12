"""
Robot postural space map: 28 Feix anchors projected into PC1-PC2.

Fits PCA on all Dexonomy squeeze_qpos (all classes), then projects the
28 pose_close anchors from the YAML into the first two PCs. Output is
the "robot postural space map" -- analogous to Segil & Weir (2014) Fig. 3.

Run from AIST-hand/:
    .venv/bin/python grasp-robot/scripts/postural_space_map.py
"""

from __future__ import annotations
import pathlib
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "grasp-robot/grasp_configs/shadow_hand_canonical_v5_grasp.yaml"
DEX_DIR   = pathlib.Path("/media/yareeez/94649A33649A1856/dexonomy/Dexonomy_GRASP_shadow/succ_collect")
OUT_PATH  = ROOT / "grasp-robot/experiments/postural_space_map.png"

DEX2MJC_FINGER = slice(7, 29)
MAX_FILES_PER_CLASS = 50
N_PCS = 7   # 90% variance

# ── Feix taxonomy grouping (for color coding) ─────────────────────────────────
# Broad grasp type groups following Feix et al. (2016)
GROUPS = {
    "Power":     ["Large Diameter", "Small Diameter", "Medium Wrap", "Light Tool",
                  "Power Disk", "Power Sphere"],
    "Precision": ["Precision Disk", "Precision Sphere", "Prismatic 3 Finger",
                  "Tip Pinch", "Palmar Pinch", "Writing Tripod", "Lateral Tripod",
                  "Tripod", "Quadpod", "Sphere 3-Finger", "Sphere 4-Finger"],
    "Lateral":   ["Lateral", "Adduction Grip", "Adducted Thumb"],
    "Extension": ["Index Finger Extension", "Extension Type", "Parallel Extension"],
    "Pinch":     ["Distal", "Ring", "Palmar", "Inferior Pincer", "Stick"],
}
GROUP_COLORS = {
    "Power":     "#e07070",
    "Precision": "#70aae0",
    "Lateral":   "#70c870",
    "Extension": "#e0c060",
    "Pinch":     "#c070e0",
}

def class_group(name: str) -> str:
    for g, members in GROUPS.items():
        if name in members:
            return g
    return "Other"

# ── Load Dexonomy for PCA ─────────────────────────────────────────────────────
print("Loading Dexonomy poses...")
qpos_all = []
for class_dir in sorted(DEX_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    count = 0
    for grasp_dir in class_dir.iterdir():
        if count >= MAX_FILES_PER_CLASS:
            break
        floating = grasp_dir / "floating"
        if not floating.exists():
            continue
        for npy in sorted(floating.glob("*.npy")):
            try:
                d = np.load(npy, allow_pickle=True).item()
                qpos_all.append(d["squeeze_qpos"][:, DEX2MJC_FINGER])
                count += 1
                break
            except Exception:
                continue

Q = np.vstack(qpos_all)
print(f"  {Q.shape[0]} poses from {len(list(DEX_DIR.iterdir()))} classes")

# ── Fit PCA ───────────────────────────────────────────────────────────────────
print(f"Fitting PCA ({N_PCS} components)...")
pca = PCA(n_components=N_PCS, random_state=42)
pca.fit(Q)
var = pca.explained_variance_ratio_
print(f"  PC1={var[0]:.1%}  PC2={var[1]:.1%}  total ({N_PCS} PCs)={var.sum():.1%}")

# ── Load YAML anchors and project ─────────────────────────────────────────────
print("Projecting anchors...")
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

names, coords, colors = [], [], []
for entry in cfg.values():
    if not isinstance(entry, dict) or "class_name" not in entry:
        continue
    name = entry["class_name"]
    q24  = np.array(entry["pose_close"], dtype=np.float64)
    f22  = q24[2:]          # skip wrist (indices 0,1)
    pc   = pca.transform(f22.reshape(1, -1))[0]
    g    = class_group(name)
    names.append(name)
    coords.append(pc[:2])   # PC1, PC2 for 2D map
    colors.append(GROUP_COLORS.get(g, "#aaaaaa"))
    print(f"  {name:<25s}  PC1={pc[0]:+.3f}  PC2={pc[1]:+.3f}  [{g}]")

coords = np.array(coords)   # [28, 2]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")

# Background scatter of all Dexonomy poses (density cloud)
Q_proj = pca.transform(Q)
ax.scatter(Q_proj[:, 0], Q_proj[:, 1],
           s=1, alpha=0.06, color="#555577", linewidths=0, zorder=1)

# Anchor points
for (x, y), name, color in zip(coords, names, colors):
    ax.scatter(x, y, s=120, color=color, edgecolors="white",
               linewidths=0.6, zorder=3)
    label = ax.text(x, y + 0.012 * (coords[:, 1].max() - coords[:, 1].min()),
                    name, fontsize=6.5, color="white", ha="center", va="bottom",
                    zorder=4)
    label.set_path_effects([pe.withStroke(linewidth=1.5, foreground="#0f0f1a")])

# Legend
for g, color in GROUP_COLORS.items():
    ax.scatter([], [], s=80, color=color, edgecolors="white",
               linewidths=0.6, label=g)
ax.legend(framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
          labelcolor="white", fontsize=8, loc="upper right")

ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)", color="#aaaacc", fontsize=10)
ax.set_ylabel(f"PC2  ({var[1]:.1%} variance)", color="#aaaacc", fontsize=10)
ax.tick_params(colors="#555577")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

ax.set_title(
    f"Shadow Hand robot postural space  --  28 Feix anchors (Dexonomy)\n"
    f"PCA over {Q.shape[0]:,} squeeze_qpos  |  {N_PCS} PCs = {var.sum():.0%} variance",
    color="white", fontsize=10, pad=10
)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
print(f"\nSaved: {OUT_PATH}")
