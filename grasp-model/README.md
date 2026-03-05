# grasp-model — GraphGrasp

Graph-based grasp classification module for **GraphGrasp: A Framework for Grasp Intent-Driven Teleoperation**.

> **Thesis:** "Graph Neural Networks for Intention-Oriented Grasp Recognition in Vision-Based Robotic Hand Teleoperation"

The system uses Graph Convolutional Networks (GCN) to classify human hand grasps from 3D keypoints into semantic grasp types (GRASP taxonomy, Feix et al. 2016), acting as an intermediate semantic layer between human intent and robot execution.

---

## System Overview

The full system is split across three repositories:

| Repository | Role |
|------------|------|
| **`grasp-model`** (this repo) | Installable Python package: GCN model, graph construction, GraspToken, VotingWindow |
| `grasp-app` | Real-time perception app: camera capture, MediaPipe, GUI |
| `grasp-ros` | Robot adapters: ROS Noetic, Shadow Hand, etc. |

The `.pth` model weights live in `experiments/` during development. For inference, they are copied into `grasp-app/models/` (not versioned in git — downloaded from GitHub Releases).

---

## Architecture

```
TRAINING (offline)
──────────────────
HOGraspNet JSONs → [ingestion] → CSV (21 joints + label)
                              → [graph/tograph] → PyG Data
                              → [training/train] → model.pth

INFERENCE (online)
──────────────────
RGB Camera → [perception/mediapipe] → 21 keypoints 3D        ← grasp-app
                                    → [graph/tograph] → PyG Data
                                    → [model/gcn] → logits        ← grasp-model
                                    → [token/grasp_token] → GraspToken
                                    → YAMLRobotAdapter → ROS / Robot  ← grasp-ros
```

---

## Public API

Install this package once, use it from any other repo:

```bash
pip install -e .
```

```python
from grasp_gcn import get_network, ToGraph, GraspToken, VotingWindow
```

| Symbol | Description |
|--------|-------------|
| `get_network(name, num_classes)` | Instantiate the GCN model |
| `ToGraph` | Convert a dict of 21 joints to a PyG `Data` object |
| `GraspToken` | Dataclass `{class_id, class_name, confidence}` — contract between model and robot |
| `VotingWindow(n=5)` | Temporal smoothing: emits a `GraspToken` only after N consecutive frames agree |

---

## GraspToken

The `GraspToken` is the semantic output of the classification layer. It decouples the model from any specific robot:

```python
@dataclass
class GraspToken:
    class_id:   int    # grasp type index (GRASP taxonomy)
    class_name: str    # human-readable name (e.g. "Tripod")
    confidence: float  # model confidence [0, 1]
```

---

## Robot Extensibility

Any robot hand can be supported without modifying this package. The adapter pattern (Strategy) is used in `grasp-ros`:

```python
adapter = YAMLRobotAdapter("grasp_configs/shadow_hand.yaml")
adapter.execute(grasp_token)
```

Each robot has its own YAML file mapping grasp class IDs to joint configurations:

```yaml
# grasp_configs/shadow_hand.yaml
0:  # Large Diameter
  rh_FFJ1: 90
  rh_FFJ2: 45
  ...
```

To support a new robot hand, only a new YAML file is needed — no code changes.

**Shadow Hand joint angles** are sourced from [Dexonomy (RSS 2025)](https://arxiv.org/abs/2504.18829), which provides 9.5M physically-validated grasps covering 31 GRASP taxonomy types for the Shadow Hand (`Dexonomy_GRASP_shadow.tar.gz`).

---

## Temporal Stability (Voting Window)

The system emits a `GraspToken` only after N consecutive frames agree on the same class. This filters frame-to-frame noise without competing with the natural pace of intentional grasps (operators hold a grasp configuration for several seconds).

- Default: N=5 frames at 30fps ≈ 167ms latency
- Named in literature as *temporal smoothing* / *voting window*

```python
window = VotingWindow(n=5)
token = window.update(class_id, class_name, confidence)
# returns GraspToken if consensus reached, None otherwise
```

---

## Repository Structure

```
grasp-model/
├── src/grasp_gcn/
│   ├── __init__.py            # public API: get_network, ToGraph, GraspToken, VotingWindow
│   ├── dataset/
│   │   └── grasps.py          # GraspsClass — InMemoryDataset (PyG)
│   ├── transforms/
│   │   └── tograph.py         # ToGraph — dict of joints → PyG Data
│   ├── network/
│   │   ├── gcn_network.py     # GCN_8_8_16_16_32 model
│   │   └── utils.py           # get_network() registry
│   ├── token/
│   │   ├── grasp_token.py     # GraspToken dataclass
│   │   └── voting_window.py   # VotingWindow
│   ├── utils/
│   │   ├── evaluation.py
│   │   ├── plotaccuracies.py
│   │   ├── plotlosses.py
│   │   └── plotconfusionmatrix.py
│   └── legacy/                # deprecated, do not use
│
├── scripts/
│   ├── ingestion/             # dataset ingestion scripts (HOGraspNet → CSV)
│   ├── build_dataset_mp.py    # images → CSV via MediaPipe (3-class sample)
│   ├── app.py                 # Gradio demo
│   ├── quick_kfold_demo.py    # K-Fold cross-validation demo
│   └── normtest.py            # 3D landmark visualization
│
├── tests/
│   ├── test_gcn_network.py
│   └── test_tograph_from_csv.py
│
├── train.py                   # main training script
├── data/
│   ├── raw/                   # CSVs + source images
│   └── processed/             # cached .pt graphs + train_stats.npz
│
└── experiments/
    ├── best_model.pth
    ├── final_model.pth
    └── runs/                  # TensorBoard logs
```

---

## Dataset

### Current (3-class sample)

A small hand-curated sample using images processed with MediaPipe:

| Class | Name | Train | Val | Test |
|-------|------|-------|-----|------|
| 0 | Large Diameter | 437 | 54 | 56 |
| 1 | Parallel Extension | 310 | 38 | 40 |
| 2 | Precision Sphere | 176 | 22 | 22 |

### Target: HOGraspNet (ECCV 2024)

The full training dataset is [HOGraspNet](https://hograspnet2024.github.io/), which provides:
- 28 grasp classes from the full GRASP taxonomy (indexed 1–33)
- ~1.5M annotated RGB-D frames, 99 subjects, 30 YCB objects
- **3D hand keypoints already extracted** (21 joints, MANO fitting) — no MediaPipe needed
- JSON annotations in `labeling_data/` with fields:
  - `annotations[0].class_id` / `class_name` — grasp label
  - `hand` — 21 joint 3D poses in world coordinates

Joint ordering (OpenPose) is identical to MediaPipe's — `ToGraph` works without modification.

**To request access:** https://hograspnet2024.github.io/

---

## Installation

Requires Python 3.11 (recommended) and a CUDA-capable GPU (recommended).

```bash
# Install pyenv (if needed)
curl https://pyenv.run | bash
pyenv install 3.11.9
pyenv local 3.11.9

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -e .
pip install -r requirements.txt
```

---

## Usage

### Build dataset from images (3-class sample)

```bash
python scripts/build_dataset_mp.py
```

Outputs: `data/raw/grasps_sample_{train,val,test}.csv`

### Train

```bash
python train.py
```

Outputs:
- `experiments/best_model.pth` — best validation accuracy checkpoint
- `experiments/final_model.pth` — final epoch checkpoint
- `experiments/runs/` — TensorBoard logs

Monitor training:
```bash
tensorboard --logdir experiments/runs/
```

### Run tests

```bash
python tests/test_gcn_network.py
python tests/test_tograph_from_csv.py
```

---

## Model

**GCN_8_8_16_16_32** — 5-layer Graph Convolutional Network:

```
Input [N=21 nodes, F=3 features (x,y,z)]
  → GCNConv(3→8)   + ELU
  → GCNConv(8→8)   + ELU
  → GCNConv(8→16)  + ELU
  → GCNConv(16→16) + ELU
  → GCNConv(16→32) + ELU
  → masked mean+max readout → [B, 64]
  → Linear(64→128) + ELU
  → Linear(128→num_classes)
  → log_softmax
```

The `mask` attribute handles missing or low-confidence joints — masked nodes are excluded from mean pooling.

### Graph structure

Each hand is a graph of 21 nodes (joints) with edges defined by hand kinematics: each finger is a linear chain, MCP joints are connected across the palm, and the wrist connects to all finger bases.

---

## Design Notes

**No data leakage:** normalization statistics (mean, std) are computed only on the training split and saved to `data/processed/train_stats.npz`. Val and test splits load these same statistics. The same stats must be applied at inference time.

**Domain gap:** the model is trained on HOGraspNet data (multi-view RGBD + MANO fitting, world coordinates) but runs inference on MediaPipe output (single-view RGB, normalized [0,1] coordinates). This gap is mitigated by z-score normalization but should be discussed when interpreting real-time performance.

**Hand mirroring:** left-hand samples are horizontally mirrored (`x' = 1 - x`) during dataset construction to normalize all samples to a right-hand reference frame.

---

## Pending / Future Work

- [ ] HOGraspNet ingestion script (JSON → CSV, replaces MediaPipe for training)
- [ ] Real-time inference script (`grasp-app`)
- [ ] Shadow Hand YAML configuration from Dexonomy dataset
- [ ] ROS integration (`grasp-ros`)

---

## Citation

If you use this code, please cite the associated thesis (citation forthcoming) and:

**GRASP Taxonomy:**
> Feix, T., Romero, J., Schmiedmayer, H. B., Dollar, A. M., & Kragic, D. (2016). The GRASP taxonomy of human grasp types. IEEE Transactions on Human-Machine Systems, 46(1), 66-77.

**HOGraspNet:**
> Lee, et al. "Dense Hand-Object (HO) GraspNet with Full Grasping Taxonomy and Dynamics." ECCV 2024.

**Dexonomy:**
> Chen, et al. "Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy." RSS 2025.

---

## License

Academic use only, consistent with the HOGraspNet dataset license terms.
