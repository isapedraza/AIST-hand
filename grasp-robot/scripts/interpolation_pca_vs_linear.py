"""
Visual comparison: linear interpolation vs PCA interpolation between two Shadow Hand poses.

Experiment:
  - Anchor A: Power Disk  (class 11, feix_id 10) -- pose_close from v5 YAML
  - Anchor B: Power Sphere (class 12, feix_id 11) -- pose_close from v5 YAML
  - Interpolation alphas: [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]  (7 steps)

Two methods:
  1. Linear (24D joint space): q(a) = (1-a)*A + a*B
  2. PCA (2-PC robot postural space):
       - Fit PCA on all Dexonomy squeeze_qpos (all classes)
       - Project A, B into PC space
       - Interpolate in PC space, project back

Output:
  grasp-robot/experiments/interpolation_pca_vs_linear.png
  (2 rows x 7 columns; top=linear, bottom=PCA)

Argument: PCA intermediate poses respect joint co-activation patterns learned from real
grasps. Linear interpolation ignores covariance -- joints move independently, potentially
producing unnatural configurations.

Run from AIST-hand/:
    .venv/bin/python grasp-robot/scripts/interpolation_pca_vs_linear.py
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import yaml
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "grasp-robot/grasp_configs/shadow_hand_canonical_v5_grasp.yaml"
XML_PATH  = ROOT / "third_party/mujoco_menagerie/shadow_hand/right_hand.xml"
DEX_DIR   = pathlib.Path("/media/yareeez/94649A33649A1856/dexonomy/Dexonomy_GRASP_shadow/succ_collect")
OUT_PATH  = ROOT / "grasp-robot/experiments/interpolation_pca_vs_linear.png"

# Dexonomy qpos[7:29] -> MuJoCo Menagerie [2:24] (skip wrist indices 0,1)
DEX2MJC_FINGER = slice(7, 29)   # 22 finger joints from Dexonomy
MJC_FINGER     = slice(2, 24)   # same 22 joints in Menagerie (indices 2-23)

N_PCS   = 7   # 90% variance
ALPHAS  = np.linspace(0, 1, 7)
MAX_FILES_PER_CLASS = 50   # ~550 poses per class for PCA -- fast, sufficient

IMG_H, IMG_W = 320, 300

# ── Camera ────────────────────────────────────────────────────────────────────
# Dorsal (top-down) view -- shows finger spread and curl simultaneously.
# lookat at center of palm; hand spans x=[0.21, 0.42] in model frame.
CAM_AZIMUTH   = 90.0
CAM_ELEVATION = -20.0
CAM_DISTANCE  = 0.32
CAM_LOOKAT    = np.array([0.35, 0.0, 0.01])

# ── Load YAML anchors ─────────────────────────────────────────────────────────
print("Loading YAML anchors...")
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

# Find Power Disk (feix_id=10) and Power Sphere (feix_id=11)
anchor_A = anchor_B = None
for entry in cfg.values():
    if not isinstance(entry, dict) or 'feix_id' not in entry:
        continue
    if entry['feix_id'] == 10:
        anchor_A = np.array(entry['pose_close'], dtype=np.float64)
        name_A   = entry['class_name']
    elif entry['feix_id'] == 11:
        anchor_B = np.array(entry['pose_close'], dtype=np.float64)
        name_B   = entry['class_name']

assert anchor_A is not None and anchor_B is not None, "Anchors not found in YAML"
print(f"  A: {name_A}  (feix 10)")
print(f"  B: {name_B}  (feix 11)")

# ── Load Dexonomy poses for PCA ───────────────────────────────────────────────
print("Loading Dexonomy poses for PCA...")
qpos_all: list[np.ndarray] = []

for class_dir in sorted(DEX_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    count = 0
    for grasp_dir in class_dir.iterdir():
        if count >= MAX_FILES_PER_CLASS:
            break
        floating_dir = grasp_dir / "floating"
        if not floating_dir.exists():
            continue
        for npy_file in sorted(floating_dir.glob("*.npy")):
            try:
                data = np.load(npy_file, allow_pickle=True).item()
                sq = data["squeeze_qpos"]          # [11, 29]
                finger = sq[:, DEX2MJC_FINGER]     # [11, 22]
                qpos_all.append(finger)
                count += 1
                break   # one file per grasp folder is enough
            except Exception:
                continue

qpos_matrix = np.vstack(qpos_all)   # [N, 22]
print(f"  Loaded {qpos_matrix.shape[0]} poses from {len(list(DEX_DIR.iterdir()))} classes")

# ── Fit PCA ───────────────────────────────────────────────────────────────────
print(f"Fitting PCA ({N_PCS} components)...")
pca = PCA(n_components=N_PCS, random_state=42)
pca.fit(qpos_matrix)
var_explained = pca.explained_variance_ratio_
print(f"  Variance explained: PC1={var_explained[0]:.1%}  PC2={var_explained[1]:.1%}  "
      f"total={var_explained.sum():.1%}")

# ── Interpolation functions ────────────────────────────────────────────────────
def interp_linear(a_24: np.ndarray, b_24: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation in 24D joint space."""
    return (1.0 - alpha) * a_24 + alpha * b_24

def interp_pca(a_24: np.ndarray, b_24: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate in PCA space, project back to 24D."""
    # Extract finger joints only (indices 2-24) for PCA
    a_f = a_24[MJC_FINGER]
    b_f = b_24[MJC_FINGER]
    # Project to PC space
    pc_a = pca.transform(a_f.reshape(1, -1))[0]
    pc_b = pca.transform(b_f.reshape(1, -1))[0]
    # Interpolate in PC space
    pc_mid = (1.0 - alpha) * pc_a + alpha * pc_b
    # Inverse transform back to 22 finger DOFs
    finger_mid = pca.inverse_transform(pc_mid.reshape(1, -1))[0]
    # Reconstruct full 24D qpos (wrist stays at 0)
    q = np.zeros(24)
    q[MJC_FINGER] = finger_mid
    return q

# ── MuJoCo renderer ───────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data  = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

_cam = mujoco.MjvCamera()
_cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
_cam.azimuth   = CAM_AZIMUTH
_cam.elevation = CAM_ELEVATION
_cam.distance  = CAM_DISTANCE
_cam.lookat    = CAM_LOOKAT

_scene_opt = mujoco.MjvOption()
_scene_opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

# Enable headlight for better visibility
model.vis.headlight.ambient  = np.array([0.4, 0.4, 0.4])
model.vis.headlight.diffuse  = np.array([0.8, 0.8, 0.8])
model.vis.headlight.specular = np.array([0.3, 0.3, 0.3])

def render_pose(q24: np.ndarray) -> np.ndarray:
    """Set qpos, forward, render and return RGB array [H, W, 3]."""
    data.qpos[:] = q24
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=_cam, scene_option=_scene_opt)
    return renderer.render().copy()

# ── Generate images ────────────────────────────────────────────────────────────
print("Rendering poses...")
imgs_linear = []
imgs_pca    = []

for alpha in ALPHAS:
    q_lin = interp_linear(anchor_A, anchor_B, alpha)
    q_pca = interp_pca(anchor_A, anchor_B, alpha)
    imgs_linear.append(render_pose(q_lin))
    imgs_pca.append(render_pose(q_pca))
    print(f"  alpha={alpha:.2f} done")

renderer.close()

# ── Plot ───────────────────────────────────────────────────────────────────────
print("Saving figure...")
n_steps = len(ALPHAS)
fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 2.5, 6.5))
fig.patch.set_facecolor('#1a1a2e')

row_labels = ["Linear (24D joint space)", f"PCA ({N_PCS} components, {var_explained.sum():.0%} var)"]
row_colors = ["#e8b4b8", "#a8d8ea"]

for row, (imgs, label, color) in enumerate(zip([imgs_linear, imgs_pca], row_labels, row_colors)):
    for col, (img, alpha) in enumerate(zip(imgs, ALPHAS)):
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        if row == 0:
            if col == 0:
                ax.set_title(f"A\n{name_A}", fontsize=7.5, color='white', pad=4)
            elif col == n_steps - 1:
                ax.set_title(f"B\n{name_B}", fontsize=7.5, color='white', pad=4)
            else:
                ax.set_title(f"α={alpha:.2f}", fontsize=8, color='#aaaaaa', pad=4)
    # Row label on the left
    axes[row, 0].set_ylabel(label, fontsize=8.5, color=color, labelpad=6)

fig.suptitle(
    f"Shadow Hand interpolation: {name_A}  →  {name_B}\n"
    f"Linear (top) vs PCA-space (bottom, PC1+PC2={var_explained.sum():.0%} var)",
    fontsize=10, color='white', y=1.01
)
plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"\nSaved: {OUT_PATH}")

# ── Print joint-space diff stats ───────────────────────────────────────────────
print("\n-- Midpoint comparison (alpha=0.5) --")
q_lin_mid = interp_linear(anchor_A, anchor_B, 0.5)
q_pca_mid = interp_pca(anchor_A, anchor_B, 0.5)
diff = np.abs(q_lin_mid - q_pca_mid)
joint_names = [
    'WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1',
    'MFJ4','MFJ3','MFJ2','MFJ1',
    'RFJ4','RFJ3','RFJ2','RFJ1',
    'LFJ5','LFJ4','LFJ3','LFJ2','LFJ1',
    'THJ5','THJ4','THJ3','THJ2','THJ1'
]
print(f"{'Joint':<8}  {'Linear':>8}  {'PCA':>8}  {'|diff|':>8}  (rad)")
for name, ql, qp, d in zip(joint_names, q_lin_mid, q_pca_mid, diff):
    marker = " <--" if d > 0.1 else ""
    print(f"{name:<8}  {ql:8.4f}  {qp:8.4f}  {d:8.4f}{marker}")
print(f"\nMean |diff|: {diff.mean():.4f} rad  |  Max |diff|: {diff.max():.4f} rad")
