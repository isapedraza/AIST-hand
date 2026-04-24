"""
Embedding visualization for Power Disk (class 11) test samples.

Extracts the 128-D pre-fc2 embedding from GCN_CAM_8_8_16_16_32 (abl04)
via a forward hook, reduces to 2D with t-SNE, and plots two views:
  - left : colored by hand aperture (mean fingertip-to-wrist distance)
  - right: colored by subject ID

Usage (from repo root, with .venv active):
    python grasp-model/scripts/embedding_tsne_power_disk.py

Output:
    grasp-model/experiments/run_abl04/embedding_tsne_power_disk.png
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grasp-model" / "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from sklearn.manifold import TSNE

from grasp_gcn.network.gcn_network import GCN_CAM_8_8_16_16_32
from grasp_gcn.transforms.tograph import ToGraph
from grasp_gcn.dataset.grasps import XYZ_COLS

# ── Config ────────────────────────────────────────────────────────────────────
POWER_DISK_CLASS  = 11
TEST_SUBJECTS     = list(range(74, 100))          # S74-S99 (S1 split)
FINGERTIP_IDX     = [4, 8, 12, 16, 20]           # thumb/index/middle/ring/pinky tips
WRIST_IDX         = 0

MODEL_PATH  = ROOT / "grasp-app/models/best_model_run_abl04_xyz_ahg_flex.pth"
CSV_PATH    = ROOT / "grasp-model/data/raw/hograspnet.csv"
OUT_PATH    = ROOT / "grasp-model/experiments/run_abl04/embedding_tsne_power_disk.png"

NUM_FEATURES = 24   # xyz(3) + flex(1) + ahg_angles(10) + ahg_distances(10)
NUM_CLASSES  = 28
BATCH_SIZE   = 512
TSNE_PERP    = 40   # perplexity; good for ~5k points

# ── Load CSV and filter ────────────────────────────────────────────────────────
print("Loading CSV...")
usecols = ['subject_id', 'sequence_id', 'grasp_type'] + XYZ_COLS
df = pd.read_csv(CSV_PATH, usecols=usecols)
df = df[
    df['subject_id'].isin(TEST_SUBJECTS) &
    (df['grasp_type'] == POWER_DISK_CLASS)
].reset_index(drop=True)
print(f"  Power Disk test samples: {len(df):,}")

# ── Pre-compute apertura from raw XYZ (wrist-relative, already normalized) ───
xyz = df[XYZ_COLS].values.astype(np.float32).reshape(-1, 21, 3)   # [N, 21, 3]
wrist      = xyz[:, WRIST_IDX, :]                                   # [N, 3]
fingertips = xyz[:, FINGERTIP_IDX, :]                               # [N, 5, 3]
apertura   = np.linalg.norm(fingertips - wrist[:, None, :], axis=2).mean(axis=1)  # [N]

# ── Build graphs ──────────────────────────────────────────────────────────────
print("Building graphs...")
tograph = ToGraph(
    add_joint_angles=True,
    add_cmc_angle=False,
    add_bone_vectors=False,
    add_velocity=False,
    add_mano_pose=False,
    add_global_swing=False,
    add_ahg_angles=True,
    add_ahg_distances=True,
)

graphs = []
for i in range(len(df)):
    row = df.iloc[i]
    vals = row[XYZ_COLS].values.astype(np.float32).reshape(21, 3)
    sample = {'grasp_type': POWER_DISK_CLASS}
    from grasp_gcn.dataset.grasps import JOINT_NAMES
    for j, name in enumerate(JOINT_NAMES):
        sample[name] = vals[j]
    graphs.append(tograph(sample))

# ── Load model and register hook ─────────────────────────────────────────────
print("Loading model...")
device = torch.device('cpu')
model = GCN_CAM_8_8_16_16_32(
    numFeatures=NUM_FEATURES,
    numClasses=NUM_CLASSES,
    use_cmc_angle=False,
)
ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
model.load_state_dict(sd)
model.eval()

embeddings_list = []

def _hook(module, input, output):
    # input[0] is the 128-D tensor entering fc2
    embeddings_list.append(input[0].detach().cpu())

hook_handle = model.fc2.register_forward_hook(_hook)

# ── Inference in batches ──────────────────────────────────────────────────────
print("Running inference...")
with torch.no_grad():
    for start in range(0, len(graphs), BATCH_SIZE):
        batch = Batch.from_data_list(graphs[start:start + BATCH_SIZE]).to(device)
        model(batch)
        if start % 2000 == 0 and start > 0:
            print(f"  {start}/{len(graphs)}")

hook_handle.remove()

embeddings = torch.cat(embeddings_list, dim=0).numpy()   # [N, 128]
assert embeddings.shape[0] == len(df), "embedding count mismatch"
print(f"  Embeddings shape: {embeddings.shape}")

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print(f"Running t-SNE (perplexity={TSNE_PERP})...")
tsne = TSNE(n_components=2, perplexity=TSNE_PERP, random_state=42,
            max_iter=1000, verbose=1)
coords = tsne.fit_transform(embeddings)   # [N, 2]
print("  Done.")

# ── Plot ──────────────────────────────────────────────────────────────────────
subjects = df['subject_id'].values
n_subjects = len(np.unique(subjects))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Power Disk (class 11) -- 128-D embedding t-SNE  [abl04, test set]",
             fontsize=12)

# Left: apertura
sc0 = axes[0].scatter(coords[:, 0], coords[:, 1],
                      c=apertura, cmap='plasma', s=4, alpha=0.6, linewidths=0)
plt.colorbar(sc0, ax=axes[0], label='apertura (mean fingertip-wrist dist, normalized)')
axes[0].set_title('Colored by hand apertura')
axes[0].set_xticks([]); axes[0].set_yticks([])

# Right: subject ID (categorical)
cmap_cat = plt.get_cmap('tab20', n_subjects)
unique_subjects = np.sort(np.unique(subjects))
subj_idx = np.searchsorted(unique_subjects, subjects)
sc1 = axes[1].scatter(coords[:, 0], coords[:, 1],
                      c=subj_idx, cmap=cmap_cat,
                      s=4, alpha=0.6, linewidths=0,
                      vmin=0, vmax=n_subjects - 1)
plt.colorbar(sc1, ax=axes[1],
             label=f'subject index  (S{unique_subjects[0]:02d}–S{unique_subjects[-1]:02d})')
axes[1].set_title(f'Colored by subject ({n_subjects} subjects)')
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")
