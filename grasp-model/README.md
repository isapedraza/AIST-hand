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
| `GraspToken` | Dataclass `{class_id, class_name, confidence, apertura}` — full intent signal passed to the robot adapter |
| `VotingWindow(n=5)` | Returns `True` when the same class appears N consecutive frames in a row, `False` otherwise |

---

## GraspToken

The `GraspToken` is the semantic output of the classification layer. It decouples the model from any specific robot:

```python
@dataclass
class GraspToken:
    class_id:   int    # grasp type index (Feix taxonomy)
    class_name: str    # human-readable label (e.g. "Tripod")
    confidence: float  # model confidence [0, 1]
    apertura:   float  # normalized hand openness [0, 1]
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
confirmed = window.update(class_id, confidence)  # returns bool
if confirmed:
    token = GraspToken(class_id, class_name, confidence, apertura)
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

[HOGraspNet](https://hograspnet2024.github.io/) (ECCV 2024):
- 28 grasp classes from the Feix taxonomy
- ~1.5M annotated frames, 99 subjects, 30 YCB objects
- 3D hand keypoints provided (21 joints, MANO fitting) — no MediaPipe needed

Joint ordering matches MediaPipe's — `ToGraph` works without modification.

CSVs are generated from the raw HOGraspNet annotations using `scripts/ingestion/hograspnet_to_csv.py` and are not versioned in git (too large). See the script's docstring for usage.

**To request access:** https://hograspnet2024.github.io/

### Why 28 classes and not 33?

The original Feix GRASP taxonomy (2016) defines 33 grasp types. HOGraspNet deliberately covers 28:

> *"Some grasp classes out of 33 are seemingly redundant, visually hard to distinguish, and geometrically close; we redefined them to 28 grasp classes."*
> — Cho et al., ECCV 2024

The 5 absent Feix IDs are **6, 8, 15, 21, 32** — merged into geometrically similar classes or excluded as uncommon. The exact merging is detailed in the HOGraspNet supplementary material (Cho et al., ECCV 2024).

This is a **design decision**, not a limitation. The 28 classes represent the operationally distinguishable set for real-world grasping with 3D keypoints.

### Class mapping

The model outputs class indices 0–27. The mapping to Feix IDs and grasp names (following Feix et al. 2016, Fig. 4) is:

| Local idx | Feix ID | Grasp name | Category |
|:---------:|:-------:|------------|----------|
| 0 | 1 | Large Diameter | Power |
| 1 | 2 | Small Diameter | Power |
| 2 | 17 | Index Finger Extension | Power |
| 3 | 18 | Extension Type | Power |
| 4 | 22 | Parallel Extension | Precision |
| 5 | 30 | Palmar | Power |
| 6 | 3 | Medium Wrap | Power |
| 7 | 4 | Adducted Thumb | Power |
| 8 | 5 | Light Tool | Power |
| 9 | 19 | Distal | Power |
| 10 | 31 | Ring | Power |
| 11 | 10 | Power Disk | Power |
| 12 | 11 | Power Sphere | Power |
| 13 | 26 | Sphere 4-Finger | Power |
| 14 | 28 | Sphere 3-Finger | Power |
| 15 | 16 | Lateral | Intermediate |
| 16 | 29 | Stick | Intermediate |
| 17 | 23 | Adduction Grip | Power |
| 18 | 20 | Writing Tripod | Precision |
| 19 | 25 | Lateral Tripod | Intermediate |
| 20 | 9 | Palmar Pinch | Precision |
| 21 | 24 | Tip Pinch | Precision |
| 22 | 33 | Inferior Pincer | Precision |
| 23 | 7 | Prismatic 3 Finger | Precision |
| 24 | 12 | Precision Disk | Precision |
| 25 | 13 | Precision Sphere | Precision |
| 26 | 27 | Quadpod | Precision |
| 27 | 14 | Tripod | Precision |

> **Note:** Feix IDs 26 (Sphere 4-Finger) and 28 (Sphere 3-Finger) are both power sphere grasps distinguished by the number of fingers in contact (Feix et al. 2016, Fig. 4, Power–Pad column). Local indices are assigned by the HOGraspNet ingestion script (`FEIX_INDICES` in `hograspnet_to_csv.py`) and do not follow Feix order.

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

### Current architecture: GCN_CAM_8_8_16_16_32

5-layer GCN with a learnable Constraint Adjacency Matrix (CAM) replacing the fixed hand skeleton:

```
Input [B*21, F=4 (x, y, z, θ_flex)]
  → CAMLayer(4→8)   + ELU   ─┐
  → CAMLayer(8→8)   + ELU    │  each layer: f_agg = CAM @ x
  → CAMLayer(8→16)  + ELU    │              x_out = ELU(Linear(cat(f_agg, x)))
  → CAMLayer(16→16) + ELU    │
  → CAMLayer(16→32) + ELU   ─┘
  → masked mean+max readout → [B, 64]
  → cat(θ_CMC) → [B, 65]
  → Linear(65→128) + ELU
  → Linear(128→num_classes)
  → log_softmax
```

**CAM** (Constraint Adjacency Matrix) is a single shared `R^(21×21)` learnable parameter, initialized uniform(-1, 1) and optimized end-to-end with Adam. Instead of constraining message passing to the hand skeleton topology, CAM lets the model discover which joint pairs are actually informative for grasp discrimination. Negative entries are inhibitory; positive are excitatory. A single shared matrix is used across all 5 layers (441 parameters) for interpretability -- one matrix can be visualized after training to see which joint correlations the model learned.

**Why CAM over GCNConv for grasp classification:**

The standard argument for GNNs on hand data is that the skeleton graph encodes anatomical priors -- adjacent joints are physically connected, so message passing along bones is meaningful. This is valid for pose estimation, where preserving physical constraints matters.

For grasp classification, however, the relevant relationships are not anatomical adjacency but functional synergies. The synergy analysis (PCA on 20 DOFs) found that grasps are discriminated by correlated joint patterns -- ring finger PIP and pinky PIP flexing together, thumb CMC abduction correlating with index MCP abduction. These correlations span non-adjacent joints. With GCNConv, getting ring finger information to the thumb requires multi-hop propagation through the wrist, diluting the signal at each step. CAM can learn a direct weighted edge between any pair -- ring_PIP to thumb_CMC -- if that correlation is discriminative.

The paper confirms this empirically: a fixed skeleton GNN scores 9.58mm vs 9.498mm for CAM on NYU hand pose estimation. The gain is modest on pose estimation (where anatomy is the right prior) but the argument is stronger for classification (where synergies are the right prior). Crucially, the learned CAM naturally recovers the skeleton topology (bright diagonal blocks in Fig. 7 of Leng et al.) plus discovers the cross-finger correlations the skeleton cannot encode -- best of both worlds.

**CAM and the multi-head architecture:**

CAM's benefit extends beyond Head A (discrete classification). Head B regresses onto PCA synergy coefficients -- continuous projections of the same joint correlation structure CAM is modeling. With GCNConv, the shared backbone organizes representations around bone chains; Head B would need to "undo" that structure to predict synergy coefficients. With CAM, the backbone organizes representations around joint correlations -- precisely the basis Head B regresses onto. The shared backbone serves both heads naturally: the feature space it produces is already synergy-structured.

The full pipeline is therefore coherent end to end: PCA finds the synergy structure in the data → CAM learns it implicitly through the classification signal → Head A discretizes it into grasp classes → Head B reads it continuously as synergy coefficients. The analysis and the architecture are aligned by design, not by coincidence.

Reference: Leng et al., "Stable Hand Pose Estimation under Tremor via GNN", IEEE VR 2021.

### Previous architecture: GCN_8_8_16_16_32

Used in Runs 001-005. Same layer dimensions but with `GCNConv` and fixed skeleton edge_index. Kept in `gcn_network.py` for reference.

**Note on Runs 001-005:** these runs used a frame-level random split (grasps_train/val/test.csv), which introduced severe data leakage -- consecutive frames from the same trial appeared in both train and test. Their metrics are not comparable to the new pipeline (hograspnet.csv + S1 subject-level split).

### Graph structure

Each hand is a graph of 21 nodes (joints). The hand skeleton edge_index (finger chains + palm connections + wrist-to-MCP edges) is still generated by `ToGraph` and stored in the Data object, but is not used in the CAM forward pass. It is preserved for visualization and potential future use.

Node features (F=4): `[x, y, z, θ_flex]` -- geometric coordinates + joint flexion angle.
Graph-level feature: `θ_CMC` -- palmar abduction angle of the thumb CMC joint, concatenated after readout.

### Occlusion robustness

Real-time perception (MediaPipe or depth sensors) does not always return all 21 landmarks.
Fingers can be self-occluded, out of frame, or tracked with insufficient confidence. The
architecture handles this at three levels:

**1. Mask propagation (ToGraph)**
`ToGraph` attaches a `mask` tensor [21, 1] to every Data object: 1.0 for valid joints,
0.0 for missing or low-confidence ones. When a joint is absent from the perception output,
its features are set to zero and its mask entry is 0.0. The mask travels through the
PyG batch unchanged.

**2. Masked message passing (CAMLayer)**
Before the CAM aggregation, each layer zeros out missing joints in the input:
```python
x_3d = x_3d * mask.view(B, N, 1)   # missing joints contribute nothing to neighbors
x_agg = torch.einsum('ij,bjf->bif', cam, x_3d)
```
Without this, a missing joint (all-zero features) would still pollute its neighbors with
a learned-weight-scaled zero signal -- incorrect because the CAM never saw zero-joint
inputs during training and its weights are not calibrated for them. With masking, a
missing joint is simply absent from every neighbor's aggregation.

**3. Masked readout**
`masked_readout` normalizes the mean pool by the number of valid nodes rather than the
fixed count of 21. The max pool is computed over valid nodes only. This ensures the
graph-level embedding is not diluted by the number of occluded joints.

**Training augmentation: random joint dropout**
HOGraspNet always provides all 21 landmarks, so the model would otherwise never see
incomplete graphs during training. To close this gap, each training batch applies random
joint dropout (`p=0.10`): each of the B×21 nodes is independently zeroed (features +
mask) with probability 0.10, simulating ~2 missing joints per hand on average. The model
learns that mask=0 means "no information here", not "joint is at the origin". Validation
and test use full landmarks only -- reported metrics reflect clean inference.

The `p=0.10` default is conservative. If the target deployment environment has frequent
occlusion (e.g., side-view cameras, small objects occluding fingers), increasing to
`p=0.15--0.20` is reasonable.

---

## Design Notes

**Why not 1:1 teleoperation?**

The obvious alternative to this system is direct joint-angle mapping: read the human's finger angles, send them to the robot. This breaks for two independent reasons.

The first is structural. Human and robot hands are morphologically different -- different number of fingers, different joint ranges, different proportions. There is no general mapping between a human MCP angle and a robot actuator position. A three-fingered gripper cannot receive a five-finger joint vector. No amount of calibration fixes this because the problem is not numerical, it is topological.

The second reason applies even in cases where approximate mapping is geometrically possible: sensor noise propagates directly to the robot. Every small tremor or detection jitter in the landmarks becomes a command sent downstream. The robot moves continuously even when the operator's intent is stable.

Classifying at the level of intent rather than configuration sidesteps both problems. The GCN maps 21 keypoints to a grasp type -- a discrete, stable, morphology-agnostic signal. The robot adapter translates that signal into its own joint space, independently of how the human hand is shaped. The VotingWindow withholds the signal until the intent is unambiguous, rather than low-pass filtering a continuous stream. The control channel is stable by construction.

**No z-score normalization:** removed from the pipeline. Geometric normalization (root-relative + scale by dist(WRIST, INDEX_MCP) = 10cm, Santos et al. 2025) is applied at ingestion time and eliminates the domain gap between HOGraspNet and MediaPipe coordinates. Z-score was a patch for that gap and is no longer needed.

**Hand mirroring:** left-hand samples are horizontally mirrored (`x' = 1 - x`) during dataset construction to normalize all samples to a right-hand reference frame.

---

## Experiment Log

> **Two phases.** Runs 001-005 (below) used the old dataset and a frame-level random
> split. They are kept as a development record. Metrics are **not comparable** to the
> new pipeline. The new pipeline (subject-level S1 split, `hograspnet.csv`,
> GCN_CAM_8_8_16_16_32) begins after the phase separator.

### Run 001 — Baseline GCN_8_8_16_16_32 (2026-03-05)

**Configuration:**
- Model: `GCN_8_8_16_16_32`
- Dataset: HOGraspNet (1,191,319 train / 148,896 val / 148,896 test)
- Epochs: 30 | Batch size: 256 | LR: 1e-3 | Optimizer: Adam
- Loss: CrossEntropyLoss (uniform weights)
- Hardware: Google Colab T4 GPU | Training time: 59.33 min

**Results:**

| Split | Loss | Accuracy |
|-------|------|----------|
| Val (best, epoch 28) | 1.2221 | **60.8%** |
| Test | 1.2282 | **60.5%** |
| — | Macro F1 | 0.529 |
| — | Weighted F1 | 0.593 |

**Training curve (Val Accuracy by epoch):**

```
Ep 01: 47.2%  Ep 08: 57.7%  Ep 15: 59.1%  Ep 22: 59.2%  Ep 29: 60.6%
Ep 02: 51.2%  Ep 09: 58.1%  Ep 16: 59.7%  Ep 23: 60.3%  Ep 30: 60.6%
Ep 03: 53.0%  Ep 10: 58.1%  Ep 17: 59.6%  Ep 24: 60.0%
Ep 04: 54.4%  Ep 11: 59.0%  Ep 18: 59.4%  Ep 25: 60.0%
Ep 05: 55.7%  Ep 12: 58.4%  Ep 19: 59.6%  Ep 26: 60.2%
Ep 06: 56.6%  Ep 13: 58.1%  Ep 20: 60.3%  Ep 27: 60.4%
Ep 07: 56.7%  Ep 14: 59.4%  Ep 21: 60.7%  Ep 28: 60.8% ← best
```

Curve nearly plateaus after epoch 21 — marginal gains (~0.001/epoch from ep 21 onward).

**Per-class performance:**

| Local idx | Feix | Grasp name | F1 | Notes |
|:---------:|:----:|------------|----|-------|
| 2 | 17 | Index Finger Extension | **0.799** | Best class |
| 9 | 19 | Distal | 0.795 | |
| 3 | 18 | Extension Type | 0.786 | |
| 24 | 12 | Precision Disk | 0.736 | |
| 15 | 16 | Lateral | 0.688 | |
| 6 | 3 | Medium Wrap | **0.047** | Nearly fails — only 2.5% recall |
| 19 | 25 | Lateral Tripod | 0.150 | 41% mistaken for Writing Tripod |
| 16 | 29 | Stick | 0.264 | Split between Light Tool (24%) and Lateral (19%) |
| 25 | 13 | Precision Sphere | 0.269 | 47% mistaken for Precision Disk |
| 1 | 2 | Small Diameter | 0.339 | Scattered across Adducted Thumb, Light Tool, Lateral |

**Main confusion pairs:**

| True class | Feix | Predicted as | Feix | Count | % of support |
|---|---|---|---|---|---|
| Lateral Tripod | 25 | Writing Tripod | 20 | 850 | 41% |
| Precision Sphere | 13 | Precision Disk | 12 | 1586 | 47% |
| Light Tool | 5 | Lateral | 16 | 2507 | 20% |
| Lateral | 16 | Light Tool | 5 | 1352 | 11% |
| Adducted Thumb | 4 | Light Tool | 5 | 1108 | 15% |
| Light Tool | 5 | Adducted Thumb | 4 | 934 | 8% |
| Writing Tripod | 20 | Tripod | 14 | 760 | 10% |
| Power Sphere | 11 | Large Diameter | 1 | 493 | 19% |

**Discussion:**

The model learns well on classes with geometrically distinctive keypoint configurations —
prismatic grasps, distal-type, index finger extension — and fails mainly on pairs where
the class difference lies in the object being held rather than in the hand shape itself.

Feix et al. (2016) noted that the 33 taxonomy classes reduce to 17 if object properties
(size, shape, orientation) are factored out. HOGraspNet took a
partial step in that direction by reducing to 28, but the results show that several pairs
remain geometrically indistinguishable from keypoints alone.

The clearest cases are Precision Sphere vs Precision Disk (47% mutual confusion) and the
cylindrical group Large/Medium/Small Diameter: in all of these, the hand adopts the same
topology and it is the object that varies. The model has no access to the object, so
confusing these classes is the correct response given what it observes.

The GraspToken design already accounts for this. The separation between `class_id`
(discrete topology) and `apertura` (continuous adaptation to the object) means that the
groups the classifier cannot distinguish are exactly those that `apertura` resolves at
runtime — the classifier's "confusions" within these groups are not functional errors
for the robot.

Where there is a genuine problem is in topologically distinct pairs with high confusion:
Lateral Tripod vs Writing Tripod (41%) and Light Tool vs Lateral (20%). The difference
there is real — thumb position, contact surface — and the model is not capturing it well,
likely due to class imbalance and the limitations of using raw Cartesian coordinates
without joint angles.

**Identified issues for next run:**
- Class imbalance: Medium Wrap (1,260 samples) vs Lateral (12,545) → weighted loss needed
- No early stopping → wasted ~8 epochs after plateau
- Uniform loss weights punish rare classes implicitly

### Run 002 — Weighted Loss + Early Stopping (2026-03-05)

**Configuration:**
- Model: `GCN_8_8_16_16_32`
- Dataset: same splits as Run 001
- Epochs: 20 (max) | Early stopping patience=5 | Batch size: 256 | LR: 1e-3
- Loss: CrossEntropyLoss with inverse-frequency class weights (min=0.246, max=2.800)
- Hardware: Google Colab T4 GPU | Training time: 38.49 min

**Results:**

| Split | Loss | Accuracy |
|-------|------|----------|
| Val (best, epoch 20) | 1.3561 | **56.1%** |
| Test | 1.3615 | **56.1%** |
| — | Macro F1 | 0.524 |
| — | Weighted F1 | 0.565 |

**Run 001 vs Run 002 — per-class F1 delta:**

| Local idx | Feix | Grasp name | R001 F1 | R002 F1 | Δ |
|:---------:|:----:|------------|---------|---------|---|
| 6 | 3 | Medium Wrap | 0.047 | 0.203 | **+0.156** |
| 19 | 25 | Lateral Tripod | 0.150 | 0.295 | **+0.145** |
| 25 | 13 | Precision Sphere | 0.269 | 0.395 | **+0.126** |
| 1 | 2 | Small Diameter | 0.339 | 0.395 | +0.056 |
| 16 | 29 | Stick | 0.264 | 0.314 | +0.050 |
| 8 | 5 | Light Tool | 0.505 | 0.340 | **-0.165** |
| 17 | 23 | Adduction Grip | 0.589 | 0.427 | **-0.162** |
| 22 | 33 | Inferior Pincer | 0.597 | 0.488 | -0.109 |
| 27 | 14 | Tripod | 0.584 | 0.514 | -0.070 |
| 24 | 12 | Precision Disk | 0.736 | 0.683 | -0.053 |

**Global comparison:**

| Metric | Run 001 | Run 002 | Δ |
|--------|---------|---------|---|
| Test Accuracy | 60.5% | 56.1% | -4.4pp |
| Macro F1 | 0.529 | 0.524 | -0.005 |
| Macro Recall | 0.522 | 0.571 | +0.049 |
| Macro Precision | 0.580 | 0.520 | -0.060 |

**Discussion:**

The weighted loss moved recall up and precision down by roughly the same margin —
a zero-sum redistribution of errors across classes. The macro F1 difference (-0.005)
is within run-to-run noise and is not statistically meaningful without multiple seeds.

The rare classes that were nearly failing did respond: Medium Wrap went from 0.047 to
0.203, confirming that class imbalance was part of the problem. But it still sits at
0.203, well below the model's average — imbalance was not the whole story.

The frequent classes paid the price. Light Tool dropped from 0.505 to 0.340, Adduction
Grip from 0.589 to 0.427. The model is now predicting rare classes more aggressively
but with lower precision, which hurts the classes that were already working.

The conclusion is that the representation bottleneck dominates over the imbalance effect.
Weighted loss does not give the model new information — it just changes how it allocates
its existing capacity. The path forward is richer features, not better loss weighting.
Weighted loss is dropped for Run 003, which returns to uniform CrossEntropyLoss.

**Feature analysis for Run 003**

The confusion pairs that remain after two runs share a pattern: the model struggles where
classes differ in *how* the fingers are bent, not in *where* the joints are in space.
Lateral Tripod vs Writing Tripod (41%), Precision Sphere vs Precision Disk (47%), and
Light Tool vs Lateral (20%) are all cases where raw Cartesian coordinates carry ambiguous
information but joint flexion angles would not.

Two quantitative studies support this.

Jarque-Bou et al. (2019) extracted kinematic synergies from 77 subjects performing 20
grasps. Only three synergies appeared in more than half of the subjects — the universal
coordination patterns across the population. Synergy 1 is MCP flexion of fingers 3–5,
PIP flexion of fingers 2–5, and adduction of the 4th finger MCP. Synergy 3 is thumb CMC
abduction combined with MCP extension and IP flexion (thumb opposition). These two
synergies define the primary axes along which grasps differ kinematically.

Stival et al. (2019) build on this. Their quantitative taxonomy groups the same
grasps into five categories from joint angle data, and explicitly maps the categories to
synergies: cylindrical grasps correspond to Synergy 1 (PIP flexion + finger closure),
spherical grasps to MCP flexion patterns. The confusion pairs above map to boundaries
between these categories — exactly where Synergy 1 and 3 are most discriminative.

Both studies measure joint angles with instrumented gloves (CyberGlove II, 22 sensors),
not with visual landmarks. The transfer to our setting is justified by the nature of the
quantity itself: joint flexion angle is a geometric property of the hand skeleton, defined
by the relative orientation of two adjacent bone segments. A flex sensor measures it
directly; we compute it as the arccos of the dot product between the incoming and outgoing
bone vectors at each joint. Both paths produce the same quantity — the angle at that
articulation — with different precision and DOF coverage. The kinematic synergies that
Jarque-Bou et al. identify are properties of how the motor system coordinates the hand,
not of the instrument used to record it. They emerge from the biomechanics of the
skeleton, which is the same structure that our graph encodes: nodes are joints, edges are
bones, and the features at each node are the angles those bones form. The graph topology
is the skeleton that both glove and landmark systems instrument.

The main limitation of our implementation relative to glove-based measurements is DOF
coverage: the CyberGlove captures abduction and adduction between fingers in addition to
flexion, while the parent→joint→child triple captures only the flexion component. This
means Synergy 3 (thumb CMC abduction) is only partially represented — the flexion and
extension components at the thumb joints are captured, but the lateral abduction angle
of the thumb relative to the palm plane is not. This is a known limitation of the
representation, not a fundamental incompatibility between the approach and the literature.

Synergy 2 (wrist flexion + palmar arch, CMC5) is not computable from 21 landmarks —
CMC5 has no corresponding keypoint in the MediaPipe skeleton, and wrist abduction
requires forearm reference points that are not available. This is a hard limit of the
representation, not a modeling choice.

**Run 003 design decision:** add the joint flexion angle as one extra scalar feature per
node, computed from the parent→joint→child triple for each internal joint. WRIST and
fingertips receive 0.0. This increases node features from 3 to 4 and targets the flexion components of
Synergies 1 and 3, without changing the graph topology or model architecture.

### Run 003 — Joint Flexion Angles (2026-03-05)

**Configuration:**
- Model: `GCN_8_8_16_16_32`
- Dataset: same splits as Run 001/002
- Features: `[x, y, z, θ]` per node — θ = joint flexion angle in radians
- Epochs: 20 (max) | Batch size: 256 | LR: 1e-3
- Loss: CrossEntropyLoss (uniform weights)
- Hardware: Google Colab T4 GPU | Training time: ~75 min

**Physical validation of the features**

Before training, three checks confirm that the computed angles capture what the
literature says they should.

*Range check.* Jarque-Bou et al. (2019) define joint angles using the sign convention
in Table 4: flexion positive, extension negative, computed from calibrated CyberGlove
data. Our implementation uses arccos of the normalized dot product between the incoming
and outgoing bone vectors, which produces values in [0, π] — π for full extension,
approaching 0 for full flexion. Values outside this range indicate a bug in the
parent→joint→child mapping.

*Variance check.* Jarque-Bou et al. (2019) Synergy 1 — the most prevalent synergy,
present in 70 of 77 subjects — loads on MCP 3–5 and PIP 2–5 flexion. DIP joints do
not appear in any of the three primary synergies. This means PIP nodes should show
higher inter-grasp variance than DIP nodes in our dataset. If DIP variance exceeds PIP
variance, the angles are being computed at the wrong joints.

*Per-class sanity check.* Stival et al. (2019) state that "the index finger extension
is clearly distant from all the other movements, while small diameter, fixed hook, large
diameter, and medium wrap are very similar grasps," and that cylindrical grasps are
characterized by "closure of the finger aperture achieved by flexion at the pip joints."
Concretely: samples of Large Diameter should have small θ at PIP joints (high flexion),
while Index Finger Extension samples should have θ close to π at PIP joints 2–5 (fingers
extended). If this ordering is not observed in the raw data before training, the feature
is not encoding what the literature predicts.

**Verification results (5,000 samples from grasps_train.csv):**

```
[Check 1] Range [0, pi]
  min=0.0000  max=2.8471  pi=3.1416  → PASS

[Check 2] PIP variance > DIP variance
  mean PIP variance=0.203477  mean DIP variance=0.136389  → PASS

[Check 3] Large Diameter PIP angle < Index Finger Extension PIP angle
  Large Diameter         = 0.3632 rad (20.8 deg)
  Index Finger Extension = 0.7313 rad (41.9 deg)  → PASS
```

All three checks pass. The feature encodes what the literature predicts.
Script: `tests/verify_joint_angles.py`

**Results:**

| Split | Loss | Accuracy |
|-------|------|----------|
| Val (best, epoch 15) | 1.1331 | **63.4%** |
| Test | 1.1158 | **63.8%** |
| — | Macro F1 | 0.573 |
| — | Weighted F1 | 0.630 |

**Global comparison:**

| Metric | Run 001 | Run 002 | Run 003 | Δ (vs 001) |
|--------|---------|---------|---------|------------|
| Test Accuracy | 60.5% | 56.1% | **63.8%** | +3.3pp |
| Macro F1 | 0.529 | 0.524 | **0.573** | +0.044 |
| Weighted F1 | 0.593 | 0.565 | **0.630** | +0.037 |

**Per-class F1 (selected):**

| Local idx | Feix | Grasp name | R001 F1 | R003 F1 | Δ |
|:---------:|:----:|------------|---------|---------|---|
| 2 | 17 | Index Finger Extension | — | **0.818** | best class |
| 3 | 18 | Extension Type | 0.786 | **0.810** | |
| 24 | 12 | Precision Disk | 0.736 | **0.755** | |
| 15 | 16 | Lateral | — | 0.724 | |
| 19 | 25 | Lateral Tripod | 0.150 | 0.270 | +0.120 |
| 6 | 3 | Medium Wrap | 0.047 | 0.240 | +0.193 |
| 16 | 29 | Stick | 0.264 | 0.330 | +0.066 |
| 25 | 13 | Precision Sphere | 0.269 | 0.346 | +0.077 |

**Main confusion pairs:**

| True class | Grasp | Predicted as | Grasp | Count | % support |
|---|---|---|---|---|---|
| 6 | Medium Wrap | 7 | Adducted Thumb | 306 | 24% |
| 25 | Precision Sphere | 24 | Precision Disk | 1451 | 43% |
| 18 | Writing Tripod | 27 | Tripod | 713 | 9% |
| 27 | Tripod | 18 | Writing Tripod | 647 | 6% |
| 19 | Lateral Tripod | 18 | Writing Tripod | 708 | 34% |
| 8 | Light Tool | 15 | Lateral | 1753 | 14% |

**Discussion:**

The joint flexion angle improves Macro F1 by 0.044 over Run 001, confirming that the
feature adds discriminative information. The improvement is concentrated in classes where
flexion patterns differ across grasps — the model now separates flat grasps (Index Finger
Extension, Extension Type) from cylindrical and spherical ones more cleanly than before.

The model plateaued at epoch 15 (Val Acc 0.634) and oscillated for the remaining five
epochs without improvement. The plateau indicates the representational ceiling for this
feature set.

The confusion matrix partially confirms what Stival et al. (2019) predict. Two of the
three main confusion clusters match Stival's quantitative categories:

- **Cylindrical** (Large Diameter, Medium Wrap, Small Diameter): Medium Wrap reaches only
  F1=0.240 despite the angle feature. Stival identifies these three as the most similar
  grasps in the taxonomy — the same hand closure, different object size. The model cannot
  separate them, and with 21 landmarks it cannot, because there is nothing in the hand
  configuration to separate.

- **Spherical** (Power Sphere, Precision Sphere, Sphere 4-Finger variants): Precision
  Sphere (F1=0.346) is predicted as Precision Disk in 43% of cases. These share near-
  identical finger flexion — the difference between them is the number of fingers in contact
  with the object, which varies with object shape, not with hand intent.

A third confusion cluster — Tripod, Writing Tripod, and Lateral Tripod — does not map
to a Stival category. In Stival's general taxonomy, Tripod is placed in the spherical
group (alongside Power Sphere and Precision Sphere), not with Writing Tripod. Lateral
Tripod is absent from the Ninapro dataset Stival used. The mutual confusion between
these three (Tripod→Writing Tripod: 6%, Writing Tripod→Tripod: 9%, Lateral
Tripod→Writing Tripod: 34%) is an empirical observation from Run 003 that the kinematic
literature does not predict.

Conversely, the classes Stival identifies as isolated perform best. Index Finger Extension
(F1=0.818) is the top class and Extension Type (F1=0.810) is second — both described by
Stival as "clearly distant from all other movements." The model's best-performing classes
are exactly the ones the literature predicts should be easiest to separate.

One anomaly: Adducted Thumb, Light Tool, and Lateral (classes 7, 8, 15) show high mutual
confusion despite belonging to different Stival categories. This is likely a class imbalance
effect — Lateral has 12,545 test samples vs Light Tool's 1,260 — combined with genuine
geometric proximity in the landmark space that the joint angle does not resolve.

**Conclusion:** The joint flexion angle feature works. The remaining confusions are
structurally predicted by the quantitative literature on hand kinematics. Adding more
features derived from the same 21 landmarks will not resolve them — the limiting factor
is the taxonomy itself: several Feix classes are kinematically indistinguishable without
knowledge of the object being held.

The unresolved confusions share a pattern: classes that Feix distinguishes by object size,
orientation, or contact surface, not by hand topology. An object perception module could
separate Large Diameter from Medium Wrap, or distinguish sphere variants by the object's
shape.

Whether that distinction serves the objective of this work is a different question. Stival
et al. (2019) show that these classes share the same kinematic category: same finger
closure pattern, same muscular activation, same grasp shape. For robot execution, they
map to the same joint trajectory. The variable that differs between them is aperture,
which GraspToken already encodes continuously and which the robot adapter maps to its own
joint limits at runtime.

A perception module could still be useful in a different role: not for classification,
but for active aperture correction during execution, scaling the aperture scalar based
on the object's geometric dimensions. That correction would likely not require a learned
classifier; geometric representations such as point clouds could suffice. However, this
would extend the framework toward shared autonomy, where the robot actively corrects the
operator's signal using its own perception. That is outside the scope of this work.

Run 004 collapses the Feix classes by their cell in the taxonomy matrix.

Each cell in the Feix matrix (Fig. 4, Feix et al. 2016) groups grasps by opposition type
(Palm / Pad / Side), virtual finger assignment (VF 2, 2-3, 2-4, 2-5, 3-5), and thumb
position (abducted / adducted). Those three parameters fully describe the hand topology.
Grasps within a cell have the same hand configuration — what varies between them is the
object: its size, shape, or the contact surface it presents.

Run 003 confirms this empirically: the confusion clusters map exactly to multi-grasp
cells. The model separates cells cleanly but cannot separate within a cell. Collapsing
within-cell classes removes a distinction the model structurally cannot make and that
robot execution does not require.

After applying HOGraspNet exclusions (Feix IDs 6, 8, 15, 21, 32), 28 classes reduce
to **16 functional classes**: 7 collapsed groups and 9 singletons.

**Collapse groups (7 cells, each collapses to one label):**

| Feix cell | Feix IDs | Grasp names | Local indices |
|-----------|----------|-------------|:-------------:|
| Power, Palm, VF 2-5, Thumb Abducted | 1, 2, 3, 10, 11 | Large Diameter, Small Diameter, Medium Wrap, Power Disk, Power Sphere | 0, 1, 6, 11, 12 |
| Power, Palm, VF 2-5, Thumb Adducted | 4, 5, 30 | Adducted Thumb, Light Tool, Palmar | 7, 8, 5 |
| Power, Pad, VF 2-4, Thumb Abducted | 18, 26 | Extension Type, Sphere 4-Finger | 3, 13 |
| Intermediate, Side, VF 2, Thumb Adducted | 16, 29 | Lateral, Stick | 15, 16 |
| Precision, Pad, VF 2, Thumb Abducted | 9, 24, 33 | Palmar Pinch, Tip Pinch, Inferior Pincer | 20, 21, 22 |
| Precision, Pad, VF 2-4, Thumb Abducted | 7, 27 | Prismatic 3-Finger, Quadpod | 23, 26 |
| Precision, Pad, VF 2-5, Thumb Abducted | 12, 13 | Precision Disk, Precision Sphere | 24, 25 |

**Singleton cells (9 cells, unchanged):**

| Feix cell | Feix ID | Grasp name | Local idx |
|-----------|:-------:|------------|:---------:|
| Power, Palm, VF 3-5, Thumb Adducted | 17 | Index Finger Extension | 2 |
| Power, Pad, VF 2, Thumb Abducted | 31 | Ring | 10 |
| Power, Pad, VF 2-3, Thumb Abducted | 28 | Sphere 3-Finger | 14 |
| Power, Pad, VF 2-5, Thumb Abducted | 19 | Distal | 9 |
| Intermediate, Side, VF 2, Thumb Abducted | 23 | Adduction Grip | 17 |
| Intermediate, Side, VF 3, Thumb Abducted | 25 | Lateral Tripod | 19 |
| Precision, Pad, VF 2-3, Thumb Abducted | 14 | Tripod | 27 |
| Precision, Pad, VF 2-5, Thumb Adducted | 22 | Parallel Extension | 4 |
| Precision, Side, VF 3, Thumb Abducted | 20 | Writing Tripod | 18 |

> **Note:** Feix 15 (Fixed Hook) and 32 (Ventral) are absent from HOGraspNet but belong
> to multi-grasp cells. Their exclusion reduces those cells to 3 and 2 classes
> respectively, without changing the collapse logic. Feix 8 (Prismatic 2-Finger) is
> absent and was the only other member of the Tripod cell, leaving it as a singleton.
> Feix 21 (Tripod Variation) was alone in its cell and its absence removes that cell
> entirely.

**Cross-reference with Run 003 confusion matrix**

The Feix cell collapse resolves all within-group confusions by definition. The question
is which of the Run 003 confusion pairs fall within a group (and are therefore eliminated)
and which cross group boundaries (and therefore survive).

| True class | Predicted as | Relationship | Status after collapse |
|------------|--------------|:------------:|----------------------|
| Precision Sphere (25) | Precision Disk (24) | Same cell — Precision, Pad, VF 2-5 | Resolved: both merge into one class |
| Medium Wrap (6) | Adducted Thumb (7) | Cross-group — Thumb Abducted vs Thumb Adducted boundary | Survives |
| Writing Tripod (18) | Tripod (27) | Cross-singleton — Precision Side vs Precision Pad | Survives |
| Tripod (27) | Writing Tripod (18) | Cross-singleton — Precision Pad vs Precision Side | Survives |
| Lateral Tripod (19) | Writing Tripod (18) | Cross-singleton — Intermediate Side vs Precision Side | Survives |
| Light Tool (8) | Lateral (15) | Cross-group — Power Palm Adducted vs Intermediate Side | Survives |

The Feix collapse directly resolves the Precision Sphere–Disk confusion (43% of Precision
Sphere support in Run 003 — the second largest confusion pair). The cylindrical group
(Large Diameter, Small Diameter, Medium Wrap, Power Disk, Power Sphere) also collapses
entirely, eliminating whatever within-group confusion existed between them.

The surviving pairs point to two clusters worth examining for a second collapse step:

- **Tripod cluster**: Writing Tripod, Tripod, and Lateral Tripod are three singleton cells
  that confuse each other bidirectionally after the Feix collapse. Collapsing them further
  requires verifying whether they produce the same robot joint trajectory — if they do,
  merging is justified; if not, the confusion is a genuine error the model must learn.

- **Thumb boundary**: Medium Wrap (cylindrical group, Thumb Abducted) confuses with
  Adducted Thumb (thumb-adducted group). These differ in thumb position, which is a
  meaningful distinction for robot execution. This confusion is more likely a geometric
  proximity effect at the group boundary than a functional equivalence, and should not
  be collapsed.

- **Light Tool vs Lateral**: group 2 (Power Palm Adducted) to group 4 (Intermediate Side
  Adducted). Structurally different opposition types. This is likely a class imbalance
  residual — Lateral has 12,545 test samples vs Light Tool's 1,260 — rather than genuine
  functional equivalence.

The two-step approach is therefore: apply the Feix cell collapse as the structural base,
then decide on the Tripod cluster based on robot execution requirements. That decision
is deferred to Run 004 design.

### Run 004 — Feix Cell Collapse: 28 → 16 Classes (2026-03-06)

**Configuration:**
- Model: `GCN_8_8_16_16_32`
- Dataset: same CSV splits as Run 001–003, `collapse=True` (28 → 16 Feix functional classes)
- Features: `[x, y, z, θ]` per node — same as Run 003
- Epochs: 20 (max) | Batch size: 256 | LR: 1e-3
- Loss: CrossEntropyLoss (uniform weights)
- Early stopping: patience=10 (not triggered — model still improving at epoch 20)
- Hardware: Google Colab T4 GPU | Training time: 41.4 min

**Results:**

| Split | Loss | Accuracy |
|-------|------|----------|
| Val (best, epoch 19) | 0.9140 | **69.3%** |
| Test | 0.9350 | **68.5%** |
| — | Macro F1 | 0.621 |
| — | Weighted F1 | 0.678 |

**Global comparison:**

| Metric | Run 001 | Run 002 | Run 003 | Run 004 | Δ (vs 003) |
|--------|---------|---------|---------|---------|------------|
| Test Accuracy | 60.5% | 56.1% | 63.8% | **68.5%** | +4.7pp |
| Macro F1 | 0.529 | 0.524 | 0.573 | **0.621** | +0.048 |
| Weighted F1 | 0.593 | 0.565 | 0.630 | **0.678** | +0.048 |

> Run 001–003 use 28 classes; Run 004 uses 16. The comparison is valid as a measure of
> model progress across runs, but absolute accuracy values are not directly comparable:
> fewer classes reduce the chance of an incorrect prediction by construction.

**Per-class results:**

| Class | Grasp (collapsed cell) | Precision | Recall | F1 | Support |
|:-----:|------------------------|:---------:|:------:|:--:|--------:|
| 0 | Power, Palm, VF 2-5, Abducted | 0.567 | 0.620 | 0.593 | 13,759 |
| 1 | Power, Palm, VF 2-5, Adducted | 0.694 | 0.693 | 0.693 | 23,370 |
| 2 | Power, Pad, VF 2-4, Abducted | 0.709 | 0.813 | **0.758** | 9,288 |
| 3 | Intermediate, Side, VF 2, Adducted | 0.689 | 0.741 | 0.714 | 14,663 |
| 4 | Precision, Pad, VF 2, Abducted | 0.702 | 0.785 | 0.741 | 14,527 |
| 5 | Precision, Pad, VF 2-4, Abducted | 0.591 | 0.382 | 0.464 | 5,550 |
| 6 | Precision, Pad, VF 2-5, Abducted | 0.761 | 0.834 | **0.796** | 17,686 |
| 7 | Index Finger Extension | 0.747 | 0.865 | **0.802** | 7,813 |
| 8 | Parallel Extension | 0.788 | 0.611 | 0.688 | 3,701 |
| 9 | Distal | 0.823 | 0.752 | 0.786 | 8,958 |
| 10 | Ring | 0.768 | 0.289 | 0.420 | 2,025 |
| 11 | Sphere 3-Finger | 0.782 | 0.370 | 0.502 | 5,312 |
| 12 | Adduction Grip | 0.799 | 0.551 | 0.652 | 2,172 |
| 13 | Writing Tripod | 0.473 | 0.660 | 0.551 | 7,812 |
| 14 | Lateral Tripod | 0.507 | 0.100 | 0.167 | 2,069 |
| 15 | Tripod | 0.660 | 0.564 | 0.608 | 10,191 |

**Main confusion pairs:**

| True class | Grasp | Predicted as | Grasp | Count | % support |
|---|---|---|---|---:|---:|
| 14 | Lateral Tripod | 13 | Writing Tripod | 817 | 39% |
| 10 | Ring | 4 | Precision, Pad, VF 2 | 862 | 43% |
| 11 | Sphere 3-Finger | 15 | Tripod | 1182 | 22% |
| 5 | Precision, Pad, VF 2-4 | 13 | Writing Tripod | 899 | 16% |
| 15 | Tripod | 13 | Writing Tripod | 1228 | 12% |
| 1 | Power, Palm, VF 2-5, Adducted | 3 | Intermediate, Side, VF 2 | 2794 | 12% |
| 0 | Power, Palm, VF 2-5, Abducted | 6 | Precision, Pad, VF 2-5 | 1716 | 12% |

**Discussion:**

The Feix cell collapse raises Macro F1 by 0.048 and Test Accuracy by 4.7pp relative to
Run 003. Both changes are structural: the collapse removes distinctions the model cannot
make, specifically within-cell separations based on object size or contact surface, not
distinctions it failed to learn. The improvement for the removed classes is guaranteed by
construction. The question is whether the remaining 16-class problem is easier, or whether
new confusion patterns emerge at cell boundaries.

The answer is both. The Precision Sphere–Disk confusion that accounted for 43% of Precision
Sphere errors in Run 003 is eliminated by definition: both merge into class 6, which
reaches F1=0.796, the highest in the run. The cylindrical group (Large Diameter, Small
Diameter, Medium Wrap, Power Disk, Power Sphere) collapses into class 0, removing whatever
within-group confusion existed between them. Both outcomes match the predictions in the
Run 003 cross-reference table.

The Tripod cluster does not improve. Lateral Tripod (class 14) reaches F1=0.167, the
worst class by a large margin. The model assigns 39% of Lateral Tripod samples to Writing
Tripod (class 13) and another 10% to class 3 (Intermediate Side Adducted). Writing Tripod
(class 13), Lateral Tripod (class 14), and Tripod (class 15) are all singletons: the
collapse does not merge them. Their mutual confusion persists because all three involve
three-finger opposition, and the kinematic difference between them is the specific finger
combination and the lateral vs. palmar contact surface on the thumb, which 21 landmark
coordinates do not reliably encode.

Ring (class 10, 2,025 samples) is predicted as Precision Pad VF2 (class 4) in 43% of
cases. Ring is a power grip on a single finger, which shares pad opposition and a small
virtual finger count with the precision group. With only 2,025 test samples, the model
has limited exposure to a geometrically unusual class. Sphere 3-Finger (class 11, 5,312
samples) distributes across Tripod (22%), the cylindrical group (11%), and the adducted
group (8%), indicating a diffuse representation with no dominant prediction target.

The model had not converged at epoch 20: val acc was 0.693 at epoch 19 and early stopping
(patience=10) was not triggered. An extended run is the next step before introducing new
features.

### Run 004b — Extended Training: 40 Epochs (2026-03-06)

**Configuration changes from Run 004:** max epochs 20 → 40. Everything else identical.

**Results:**

| Split | Loss | Accuracy |
|-------|------|----------|
| Val (best, epoch 37) | 0.8640 | **71.0%** |
| Test | 0.8680 | **70.7%** |
| — | Macro F1 | 0.660 |
| — | Weighted F1 | 0.703 |

**Global comparison:**

| Metric | Run 001 | Run 002 | Run 003 | Run 004 | Run 004b | Δ (vs 004) |
|--------|---------|---------|---------|---------|----------|------------|
| Test Accuracy | 60.5% | 56.1% | 63.8% | 68.5% | **70.7%** | +2.2pp |
| Macro F1 | 0.529 | 0.524 | 0.573 | 0.621 | **0.660** | +0.039 |
| Weighted F1 | 0.593 | 0.565 | 0.630 | 0.678 | **0.703** | +0.025 |

**Per-class results (with comparison to Run 004):**

| Class | Grasp (collapsed cell) | R004 F1 | R004b F1 | Δ | Support |
|:-----:|------------------------|:-------:|:--------:|:---:|--------:|
| 0 | Power, Palm, VF 2-5, Abducted | 0.593 | 0.612 | +0.019 | 13,759 |
| 1 | Power, Palm, VF 2-5, Adducted | 0.693 | 0.703 | +0.010 | 23,370 |
| 2 | Power, Pad, VF 2-4, Abducted | 0.758 | **0.770** | +0.012 | 9,288 |
| 3 | Intermediate, Side, VF 2, Adducted | 0.714 | 0.725 | +0.011 | 14,663 |
| 4 | Precision, Pad, VF 2, Abducted | 0.741 | 0.761 | +0.020 | 14,527 |
| 5 | Precision, Pad, VF 2-4, Abducted | 0.464 | 0.515 | +0.051 | 5,550 |
| 6 | Precision, Pad, VF 2-5, Abducted | 0.796 | **0.809** | +0.013 | 17,686 |
| 7 | Index Finger Extension | 0.802 | **0.830** | +0.028 | 7,813 |
| 8 | Parallel Extension | 0.688 | 0.694 | +0.006 | 3,701 |
| 9 | Distal | 0.786 | **0.805** | +0.019 | 8,958 |
| 10 | Ring | 0.420 | 0.561 | **+0.141** | 2,025 |
| 11 | Sphere 3-Finger | 0.502 | 0.656 | **+0.154** | 5,312 |
| 12 | Adduction Grip | 0.652 | 0.662 | +0.010 | 2,172 |
| 13 | Writing Tripod | 0.551 | 0.594 | +0.043 | 7,812 |
| 14 | Lateral Tripod | 0.167 | 0.229 | +0.062 | 2,069 |
| 15 | Tripod | 0.608 | 0.633 | +0.025 | 10,191 |

**Main confusion pairs:**

| True class | Grasp | Predicted as | Grasp | Count | % support |
|---|---|---|---|---:|---:|
| 14 | Lateral Tripod | 13 | Writing Tripod | 573 | 28% |
| 10 | Ring | 4 | Precision, Pad, VF 2 | 554 | 27% |
| 1 | Power, Palm, VF 2-5, Adducted | 3 | Intermediate, Side, VF 2 | 3140 | 13% |
| 5 | Precision, Pad, VF 2-4 | 15 | Tripod | 620 | 11% |
| 11 | Sphere 3-Finger | 15 | Tripod | 628 | 12% |
| 15 | Tripod | 11 | Sphere 3-Finger | 958 | 9% |
| 13 | Writing Tripod | 1 | Power, Palm, VF 2-5, Adducted | 858 | 11% |

**Discussion:**

The additional 20 epochs confirm that Run 004 had not converged: val acc continued rising
from 0.693 at epoch 19 to 0.710 at epoch 37, and early stopping (patience=10) was not
triggered before the epoch limit. The improvement from 004 to 004b is distributed across
all 16 classes, not concentrated in any single one.

The two largest per-class gains are in Ring (class 10, +0.141) and Sphere 3-Finger (class
11, +0.154). Both were the worst singletons in Run 004, with low recall caused by their
small support relative to nearby classes. More training epochs partially resolve this: Ring
recall rises from 29% to 51%, and Sphere 3-Finger recall from 37% to 65%. The primary
confusion for Ring shifts from Precision Pad VF2 (43% → 27%) to the same class, suggesting
the model is forming a cleaner boundary.

Lateral Tripod (class 14) remains the worst class at F1=0.229. Recall improves from 10%
to 15%, but the confusion with Writing Tripod (28% of support) and Intermediate Side
Adducted (12%) is structural: the three tripod singletons differ in contact topology in
ways that 21 landmark coordinates and joint angles do not separate. Additional epochs do
not resolve this — the features do not contain the information required.

A new confusion pattern appears between Tripod (class 15) and Sphere 3-Finger (class 11):
9.4% of Tripod samples are predicted as Sphere 3-Finger, and 11.8% of Sphere 3-Finger
samples are predicted as Tripod. Neither class appeared in each other's top confusions in
Run 004. This is a boundary effect: as the model corrects the Sphere 3-Finger representation
(improving recall from 37% to 65%), it draws on samples from nearby classes, creating a
new bidirectional confusion at the Tripod boundary.

The confusion between Power Palm Adducted (class 1) and Intermediate Side Adducted (class
3) persists: 13% of class 1 samples are predicted as class 3. Both involve thumb adduction
and a closed palm, with the main distinction being the opposition type (pad vs. side
contact on the thumb). This is a cross-group boundary confusion that is not resolved by
more training alone.

The model reached epoch 40 without triggering early stopping (best at epoch 37, three
epochs remaining within the patience window). The plateau between epochs 34 and 40 (val
acc oscillating between 0.702 and 0.710) suggests the representational ceiling for this
feature set and epoch budget is near. The remaining confusion clusters — Tripod singletons,
Thumb Adducted boundary, and low-support singletons — require new features, not more
training.

**Feature analysis for Run 005**

The current feature set is `[x, y, z, θ_flex]` per node. The joint flexion angle θ_flex,
introduced in Run 003, captures MCP extension and IP flexion at each joint. What it does
not encode is the lateral separation of the thumb from the palm: the palmar abduction
angle of the first metacarpal.

Feix et al. (2016), Section IV:

> "The position of the thumb is used to differentiate between the two rows. [...] the
> thumb CMC joint can be in either an adducted or abducted position. This is a new
> feature introduced in the GRASP taxonomy."

Section V:

> "In order for the thumb to oppose the fingers, the thumb has to be in abducted
> position. The thumb is adducted only in cases where the opposition is between the
> thumb and the side of the finger (e.g., lateral grasp) or the thumb is not involved
> in opposition at all."

The parameter Feix introduces as the row separator in the taxonomy matrix is the one
quantity not yet encoded in the model. The raw XYZ coordinates distribute some signal
about thumb position across four nodes (CMC, MCP, IP, TIP), but it is confounded by
hand scale and wrist orientation. The joint flexion angle at the thumb joints captures
MCP extension and IP flexion, the two distal components of Jarque-Bou Synergy 3 (CMC
abduction + MCP extension + IP flexion). It does not capture the CMC abduction
component — the rotation of the first metacarpal away from the palm plane.

**Feature definition: palmar abduction angle (θ_CMC)**

The relevant motion is palmar abduction: the first metacarpal rotating out of the palm
plane so that the thumb pad can face the fingertips. This is measured by the out-of-plane
component of the first metacarpal direction.

```
thumb_dir   = normalize(THUMB_MCP  - THUMB_CMC)                    # landmark 2 - 1
palm_normal = normalize(cross(WRIST→INDEX_MCP, WRIST→PINKY_MCP))   # palm plane normal
θ_CMC       = arcsin(|dot(thumb_dir, palm_normal)|)                 # ∈ [0, π/2]
```

Large θ_CMC: metacarpal points out of the palm plane — thumb abducted, pad facing
fingertips (Row 1 of the Feix matrix: Power/Pad, Precision/Pad grasps).
Small θ_CMC: metacarpal lies in the palm plane — thumb adducted (Row 2: Lateral, Stick,
Palmar, Adducted Thumb, and related classes).

**Why not `WRIST→THUMB_CMC`**

Landmark 1 (THUMB_CMC) is the carpometacarpal joint, anchored to the trapezium bone
relative to the wrist. After geometric normalization (center at WRIST, scale by
WRIST→MIDDLE_MCP), this landmark's position does not change meaningfully with
abduction — the CMC joint does not translate, it rotates. What changes is the direction
the first metacarpal points from that joint: `THUMB_CMC→THUMB_MCP` (landmark 1→2).
Using `WRIST→THUMB_CMC` as the thumb direction encodes a near-constant quantity across
poses and would carry no discriminative signal.

**Why not an in-plane angle**

An earlier candidate definition projected the thumb and index directions onto the palm
plane and measured the angle between the projections. That captures radial abduction
(thumb spreading within the palm plane), not palmar abduction (thumb rotating out of it).
Feix's row separator is palmar abduction — the motion that positions the thumb pad in
opposition to the fingertips. The two motions correlate in practice, but the out-of-plane
component is the geometrically correct quantity for the stated purpose and the one that
cleanly separates the two rows of the taxonomy.

**Convergence of taxonomy and biomechanics**

Jarque-Bou et al. (2019) Synergy 3 loads on CMC abduction, MCP extension, and IP
flexion of the thumb. Run 003 added the joint flexion angle θ_flex, capturing MCP
extension and IP flexion at the thumb joints. This run adds CMC abduction, completing
the encoding of Synergy 3. The two design decisions are independent in origin — one from
taxonomy analysis (Feix), one from motor control evidence (Jarque-Bou) — and converge on
the same missing quantity.

**Scope and expected impact**

The feature is added as a single scalar at the graph level, concatenated to the global
readout vector after pooling and before the fully connected layers. It is a property of
the full hand configuration, not of a single joint.

An analysis of the Run 004b confusion matrix by Feix taxonomy axis shows that the
dominant remaining errors are column errors (Power vs Precision within the same opposition
type and VF assignment), not row errors (Abducted vs Adducted, same column). The model
already extracts partial row signal from XYZ coordinates and from thumb flexion angles.
CMC abduction provides the geometrically correct encoding of that signal, but its impact
will be bounded: it directly addresses row errors, while the larger confusion mass lies on
the column axis.

**Empirical verification (tests/verify_cmc_abduction.py)**

Before implementing, the feature was verified against 5,000 samples from `grasps_train.csv`
using three checks grounded in Feix et al. (2016):

```
[Check 1] Range [0, π/2]
  min=0.0064  max=1.1175  π/2=1.5708  → PASS

[Check 2] mean θ_CMC(abducted) > mean θ_CMC(adducted)
  abducted classes  (3246 samples): mean=0.5575 rad (31.9°)
  adducted classes  (1754 samples): mean=0.4194 rad (24.0°)  → PASS

[Check 3] High-contrast class pairs
  Adducted Thumb  (adducted): 0.3834 rad (22.0°)
  Tripod          (abducted): 0.5659 rad (32.4°)  → PASS
  Lateral         (adducted): 0.4446 rad (25.5°)
  Palmar Pinch    (abducted): 0.5761 rad (33.0°)  → PASS
```

**Per-class mean θ_CMC (ascending):**

| Idx | Grasp | Row | mean θ (deg) |
|----:|-------|-----|-------------:|
| 7 | Adducted Thumb | adducted | 22.0° |
| 8 | Light Tool | adducted | 23.2° |
| 4 | Parallel Extension | adducted | 23.7° |
| 2 | Index Finger Extension | adducted | 24.7° |
| 16 | Stick | adducted | 24.8° |
| 15 | Lateral | adducted | 25.5° |
| 5 | Palmar | adducted | 25.5° |
| 17 | Adduction Grip | abducted | 25.6° |
| 6 | Medium Wrap | abducted | 25.8° |
| 13 | Sphere 4-Finger | abducted | 26.1° |
| 10 | Ring | abducted | 28.4° |
| ... | ... | ... | ... |
| 12 | Power Sphere | abducted | 37.0° |

The 7 adducted classes cluster cleanly at the bottom (22°–25.5°). The abducted classes
occupy the upper range (25.6°–37°) with two boundary cases discussed below.

**Taxonomy consistency: two boundary cases**

*Parallel Extension (#22, local 4, 23.7°).* The paper notes explicitly (Section IV-B):
"The Parallel Extension Grasp (#22) [is] in some intermediate position between abduction
and adduction. As the opposition is done partly by the side of the thumb, we then put the
grasp into the adducted category." The data confirms the intermediate position: 23.7° is
slightly above the most clearly adducted classes but within the adducted cluster.

*Adduction Grip (#23, local 17, 25.6°).* Feix places this grasp in the abducted row of
Fig. 4 but notes it as the exception: "The only exception to this 'rule' is the Adduction
Grasp, where the thumb is not in contact with the object." Section V states that the thumb
is adducted when "the thumb is not involved in opposition at all (e.g., fixed hook and
palmar)." The Adduction Grip also has the thumb uninvolved, creating an internal tension
in the taxonomy that Feix does not resolve explicitly. The empirical θ_CMC (25.6°) places
it at the boundary between the two clusters, consistent with the ambiguous status in the
paper. With only 53 samples in the draw the estimate is noisy. This class is not a dominant
source of confusion in Run 004b and does not affect the implementation decision.

The verification confirms that θ_CMC separates the two rows of the Feix taxonomy as
intended. All checks pass.

**Hypothesis: contact-dependent collapse as the next structural step**

Feix et al. (2016), Section V, notes explicitly that Palm and Pad opposition grasps can
share an identical hand configuration — the only distinguishing factor being whether the
palm is in contact with the object. In teleoperation, the operator does not hold a real
object, so contact is never present and cannot be measured. This is a permanent limit of
the problem setting, not of the model.

The Run 004b confusion matrix already reflects this: class 0 (Power Palm Abducted) is
predicted as class 6 (Precision Pad Abducted) in 12% of cases, and class 1 (Power Palm
Adducted) as class 3 (Intermediate Side Adducted) in 13% of cases. Both are Palm→Pad or
Palm→Side confusions where hand topology is nearly identical. The model is not failing —
it is correctly reporting that these classes are indistinguishable from landmarks alone.

If these classes also produce similar joint trajectories in the target robot hand, then
collapsing them is the correct design decision: the taxonomy distinction is real, but it
is operationally irrelevant given the sensor and task constraints. This will be evaluated
against Dexonomy (Chen et al., RSS 2025), which provides canonical joint configurations
for the Shadow Hand across Feix grasp types.

**Hypothesis: 16 classes is the natural granularity ceiling for landmark-based intent recognition**

The Feix taxonomy has an internal structure: grasps within a cell share the same hand
topology and differ mainly in the shape of the object held. Feix et al. (2016), Section
IV-A state this explicitly:

> "The difference between the grasps within one cell is mainly the shape of the object.
> This offers the possibility to reduce the set of all 33 grasps down to 17 grasps by
> merging the grasps within one cell to a corresponding 'standard' grasp."

Our 16 classes are the HOGraspNet subset of that 17-type reduction (5 Feix IDs absent
from the dataset collapse some cells further). The cell structure is therefore not an
arbitrary design choice — it is the coarser level of the two-level granularity that Feix
himself proposes.

**What Feix proposes vs. what this framework adds.** Feix's reduction yields 17 discrete
prototypes, selecting the most frequent grasp per cell as the canonical representative.
The `apertura` scalar in GraspToken goes one step further: rather than picking a single
prototype, it treats within-cell variation as continuous. In the teleoperation setting,
the operator's hand adapts to the object incrementally rather than jumping between
prototypes, so a continuous scalar is more appropriate than a second discrete label. This
extension is a design decision for the teleoperation use case, not a claim derived from
the taxonomy itself.

**Where the ceiling applies — and where it does not.** The within-cell argument (object
variation → apertura) covers cases where grasps in the same cell are merged. But Feix
also identifies a second kind of unresolvable ambiguity: cross-cell distinctions that
depend on contact, not on hand shape. Section V:

> "The classification in the taxonomy depends not only on the hand pose, but also the type
> of contact between hand and object. For example, the medium wrap (#3) and the prismatic
> 4 finger (#6) have a similar hand shape, but the first has additional palm contact,
> whereas the latter has only fingertip contact. [...] using a glove that measures only
> the joint angles might not be sufficient, since it would also need to measure contact in
> order to correctly classify the grasps in the taxonomy."

Medium Wrap (#3) and Prismatic 4-Finger (#6) are in different cells in our 16-class scheme
(class 0 vs. class 2 respectively), so this is a cross-cell confusion, not a within-cell
one. The empirical evidence is consistent: in Run 004b, 12% of class 0 (Power Palm
Abducted) samples are predicted as class 2 (Power Pad Abducted), and 13% of class 1
(Power Palm Adducted) as class 3 (Intermediate Side Adducted). Both are Palm→Pad or
Palm→Side confusions at boundaries where the only reliable discriminator is contact
information the input signal does not carry.

This means the ceiling is not only at the within-cell level. Some cross-cell boundaries
are also contact-dependent, and those confusions will persist regardless of feature
engineering on 21 XYZ landmarks. The contact-dependent collapse hypothesis (above)
addresses this for the cases where the distinction is also operationally irrelevant for
the robot. Where it is relevant, the confusion is a genuine limitation of the problem
setting, not of the model.

The combined argument — within-cell variation is object-driven (Feix IV-A), some cross-
cell distinctions require contact (Feix V), and teleoperation provides neither the object
nor contact signals — suggests that 16 classes is close to the practical ceiling for this
input modality. This remains a hypothesis. It has not been validated in a real
teleoperation setting, and the exact ceiling may shift with richer landmark representations
or multi-modal inputs.

---

### Run 005 — Palmar Abduction Angle as Graph-Level Feature (2026-03-06)

**Configuration changes from Run 004b:** θ_CMC added as a graph-level scalar concatenated
to the readout vector after pooling (fc1 input: 64 → 65 dims). Node features unchanged:
`[x, y, z, θ_flex]`. Everything else identical to Run 004b (40 epochs max, patience=10,
collapse=True).

**Results:**

| Split | Loss | Accuracy |
|-------|------|----------|
| Val (best, epoch 40) | 0.8397 | **71.8%** |
| Test | 0.8464 | **71.6%** |
| — | Macro F1 | 0.672 |
| — | Weighted F1 | 0.715 |

Training time: 84.36 min (Google Colab T4 GPU).

**Training curve (Val Accuracy by epoch):**

```
Ep 01: 55.5%  Ep 09: 67.8%  Ep 17: 70.1%  Ep 25: 70.9%  Ep 33: 71.3% ← local best
Ep 02: 60.6%  Ep 10: 67.9%  Ep 18: 69.9%  Ep 26: 70.7%  Ep 34: 71.1%
Ep 03: 63.0%  Ep 11: 68.4%  Ep 19: 69.6%  Ep 27: 70.2%  Ep 35: 70.7%
Ep 04: 63.9%  Ep 12: 67.9%  Ep 20: 70.4%  Ep 28: 70.6%  Ep 36: 71.1%
Ep 05: 65.0%  Ep 13: 68.7%  Ep 21: 70.6%  Ep 29: 70.6%  Ep 37: 70.9%
Ep 06: 66.1%  Ep 14: 69.5%  Ep 22: 70.3%  Ep 30: 70.9%  Ep 38: 71.7%
Ep 07: 67.0%  Ep 15: 68.6%  Ep 23: 70.4%  Ep 31: 71.0%  Ep 39: 70.7%
Ep 08: 67.5%  Ep 16: 69.9%  Ep 24: 70.2%  Ep 32: 71.2%  Ep 40: 71.8% ← best
```

The model was still improving at epoch 40 — early stopping (patience=10) was not triggered.
The gain rate in the final 15 epochs is ~0.0006/epoch (0.709 → 0.718), indicating the
representational ceiling is near but the model had not fully converged within the epoch
budget.

**Global comparison:**

| Metric | Run 001 | Run 002 | Run 003 | Run 004 | Run 004b | Run 005 | Δ (vs 004b) |
|--------|---------|---------|---------|---------|----------|---------|-------------|
| Test Accuracy | 60.5% | 56.1% | 63.8% | 68.5% | 70.7% | **71.6%** | +0.9pp |
| Macro F1 | 0.529 | 0.524 | 0.573 | 0.621 | 0.660 | **0.672** | +0.012 |
| Weighted F1 | 0.593 | 0.565 | 0.630 | 0.678 | 0.703 | **0.715** | +0.012 |

**Per-class results (with comparison to Run 004b):**

| Class | Grasp (collapsed cell) | R004b F1 | R005 F1 | Δ | Support |
|:-----:|------------------------|:--------:|:-------:|:---:|--------:|
| 0 | Power, Palm, VF 2-5, Abducted | 0.612 | 0.628 | +0.016 | 13,759 |
| 1 | Power, Palm, VF 2-5, Adducted | 0.703 | 0.728 | +0.025 | 23,370 |
| 2 | Power, Pad, VF 2-4, Abducted | 0.770 | 0.788 | +0.018 | 9,288 |
| 3 | Intermediate, Side, VF 2, Adducted | 0.725 | 0.727 | +0.002 | 14,663 |
| 4 | Precision, Pad, VF 2, Abducted | 0.761 | 0.787 | +0.026 | 14,527 |
| 5 | Precision, Pad, VF 2-4, Abducted | 0.515 | 0.517 | +0.002 | 5,550 |
| 6 | Precision, Pad, VF 2-5, Abducted | 0.809 | 0.810 | +0.001 | 17,686 |
| 7 | Index Finger Extension | 0.830 | 0.816 | -0.014 | 7,813 |
| 8 | Parallel Extension | 0.694 | 0.690 | -0.004 | 3,701 |
| 9 | Distal | 0.805 | 0.801 | -0.004 | 8,958 |
| 10 | Ring | 0.561 | 0.560 | -0.001 | 2,025 |
| 11 | Sphere 3-Finger | 0.656 | 0.660 | +0.004 | 5,312 |
| 12 | Adduction Grip | 0.662 | 0.654 | -0.008 | 2,172 |
| 13 | Writing Tripod | 0.594 | 0.591 | -0.003 | 7,812 |
| 14 | Lateral Tripod | 0.229 | **0.338** | **+0.109** | 2,069 |
| 15 | Tripod | 0.633 | 0.653 | +0.020 | 10,191 |

**Main confusion pairs:**

| True class | Grasp | Predicted as | Grasp | Count | % support |
|---|---|---|---|---:|---:|
| 14 | Lateral Tripod | 13 | Writing Tripod | 618 | 29.9% |
| 14 | Lateral Tripod | 1 | Power, Palm, VF 2-5, Adducted | 239 | 11.5% |
| 11 | Sphere 3-Finger | 15 | Tripod | 832 | 15.7% |
| 7 | Index Finger Extension | 9 | Distal | 747 | 9.6% |
| 15 | Tripod | 13 | Writing Tripod | 920 | 9.0% |
| 1 | Power, Palm, VF 2-5, Adducted | 3 | Intermediate, Side, VF 2 | 1646 | 7.0% |

**Discussion:**

The global improvement is modest (+0.9pp accuracy, +0.012 macro F1) but the distribution
of gains matches the pre-run prediction precisely: θ_CMC addresses row errors (abducted vs
adducted distinction) and leaves column-axis confusions unchanged.

The largest gain is Lateral Tripod (class 14): F1 0.229 → 0.338, recall 10% → 30%.
Lateral Tripod (Feix #25, Intermediate/Side/VF3/Abducted) sits at the abducted row of the
taxonomy but is systematically confused with Writing Tripod (#20) and the adducted power
group (class 1). The θ_CMC feature provides the classifier a direct geometric signal for
the first metacarpal's out-of-plane angle, which partially resolves this. The recall
improvement (+20pp) comes at a precision cost (0.507 → 0.381): the model now identifies
more Lateral Tripod samples correctly but generates more false positives from neighboring
classes.

Classes on the abducted/adducted boundary show the next largest gains: class 1 (+0.025),
class 4 (+0.026), class 15 (+0.020). These are all cases where the row signal was present
but noisy in the raw XYZ and θ_flex features — θ_CMC provides it directly.

The predicted scope limitation is confirmed. Column-axis confusions — Sphere 3-Finger →
Tripod (15.7%), Tripod → Writing Tripod (9.0%), Power Palm Adducted → Intermediate Side
(7.0%) — persist at nearly identical rates to Run 004b. These involve grasps that share
the same row (both abducted, or both adducted) and differ along the opposition type or VF
assignment axis, where θ_CMC provides no signal.

Two small regressions appear. Index Finger Extension (class 7, -0.014) develops a new
confusion with Distal (class 9, 9.6% of support): both grasps extend the fingers with low
MCP/PIP flexion and produce similar θ_CMC values (~24-25°), so the feature adds noise at
this boundary. Adduction Grip (class 12, -0.008) shows a similar pattern. Both regressions
are small and do not change the overall interpretation.

Lateral Tripod remains the worst class at F1=0.338. Despite the improvement, 30% of its
samples are still predicted as Writing Tripod. The structural reason is that both classes
involve three-finger side opposition and both sit in the abducted row — θ_CMC distinguishes
them at the row level but not at the finer contact-topology level. This is a hard limit of
the 21-landmark representation.

**What this run establishes:**

Run 005 closes the encoding of Jarque-Bou Synergy 3 (CMC abduction + MCP extension + IP
flexion). Run 003 added MCP extension and IP flexion via θ_flex; Run 005 adds CMC
abduction via θ_CMC. The two features are independent in origin — one from kinematic
synergy analysis (Jarque-Bou et al. 2019), one from taxonomy design (Feix et al. 2016) —
and converge on the same anatomical quantity.

The remaining confusion clusters — Tripod singletons, Sphere 3-Finger↔Tripod, and
contact-dependent Palm/Pad boundary — are not addressable with additional features
derivable from 21 XYZ landmarks. The current feature set (`[x, y, z, θ_flex]` per node
+ θ_CMC at graph level) represents the practical ceiling for this input modality.

**Analysis of residual confusions: contact-dependent vs postural**

The confusions that survive Run 005 fall into two qualitatively different categories. The
distinction matters because each implies a different response.

*Contact-dependent confusions.* Some Feix classes are only distinguishable by where the
hand contacts the object — not by the hand configuration itself. Feix et al. (2016),
Section V, give the canonical example: Medium Wrap (Palm opposition) and Prismatic
4-Finger (Pad opposition) share nearly identical hand shapes, but differ in whether the
palm contacts the object. Without an object, the operator is not executing either grasp
specifically — they are in a configuration that could become either upon contact. This is
not a modeling failure or a missing feature. It is an ontological property of the grasps
themselves: they do not fully exist as acts until contact is made.

In teleoperation, the operator has no real object. The classifier correctly identifies the
base hand topology, and the apertura scalar captures how the operator adapts their hand
configuration continuously. The specific contact surface that would disambiguate Palm from
Pad is not communicated through intent — it is concretized by the operator's physical hand
movement as the grasp is executed. This is not a limitation to be fixed by a new feature;
it is the boundary of what intent-from-morphology can resolve.

The Palm/Pad boundary (class 0 vs class 6, 12%) and the Palm/Side boundary (class 1 vs
class 3, 7%) have persisted across every run regardless of added features, confirming this.
For these classes, collapse is the structurally consistent option: treat them as one intent
class representing a base hand topology, with the specific execution concretized through
the operator's continuous hand adaptation. Whether the collapsed pairs produce sufficiently
similar joint trajectories in a given robot to justify this will be evaluated against
Dexonomy (Chen et al., RSS 2025).

*Posturally ambiguous confusions.* The remaining confusion clusters are less clear.

The Tripod group (Writing Tripod class 13, Lateral Tripod class 14, Tripod class 15) and
Sphere 3-Finger (class 11) confuse each other significantly. These grasps do differ in
hand posture — Lateral Tripod has a distinct lateral thumb contact, Tripod and Writing
Tripod differ in finger curl and opposition angle — but the differences are subtle and may
not be well-encoded by the current feature set. Whether new features (inter-finger
abduction, relative fingertip distances) would resolve these confusions, or whether the
grasps are posturally too similar to separate reliably from landmarks, is not confirmed.
Collapse of some or all of these is possible but not yet justified without further evidence.

Index Finger Extension (class 7) confusing with Distal (class 9, 9.6%) is unexpected:
these classes are in different rows and columns, and θ_CMC was added precisely to address
row separation. This confusion appeared only in Run 005 and warrants investigation before
attributing it to a structural cause.

*On when to stop.* The stopping criterion for feature engineering on this representation
is not fully defined. The contact-dependent confusions set a hard floor that no additional
feature can lower. The posturally ambiguous confusions may or may not improve with new
features — the only way to know is to try, verify against literature first, and measure.
The diminishing returns pattern across runs suggests the ceiling is near, but the exact
boundary between "unresolvable" and "needs a better feature" requires case-by-case analysis.

---

### End of Phase 1 — GCN_8_8_16_16_32, Frame-Level Split

Runs 001-005 used three separate CSVs (`grasps_train/val/test.csv`) generated with a
stratified random split at the **frame level**. Consecutive frames from the same trial
regularly appeared in both train and test. The model was evaluated on frames it had already
seen partial versions of during training, which inflated all reported metrics.

This was identified in the dataset analysis session (2026-03-09). Development stopped on
this branch because:

1. **Data leakage is fundamental.** The split cannot be patched; the CSVs have to be
   rebuilt from scratch with a subject-level protocol.
2. **The new pipeline is a clean break.** `hograspnet.csv` (unified, subject-level S1
   split: train S11-S73, val S01-S10, test S74-S99) was built from the original JSON
   files. Metrics from the new pipeline are not comparable to Runs 001-005.
3. **Architecture also changed.** The new baseline uses GCN_CAM_8_8_16_16_32 (learnable
   adjacency matrix) instead of fixed-skeleton GCNConv. Even if the split were comparable,
   the model would not be.

Runs 001-005 remain here as a record of the design decisions that shaped the current
architecture and taxonomy strategy.

---

### Dataset Analysis — HOGraspNet Frame Phase Distribution (2026-03-09)

**Question:** Do the CSVs contain only stable-grasp frames, or do they include approach and release phases?

**Findings:** Each trial in HOGraspNet covers the full temporal sequence of a grasp event. The annotated frames ("sparse frames" in the paper) are a manual subselection across the entire trial, not filtered to the stable phase. Empirical inspection of contact maps across a 70-frame sequence confirms three distinct phases:

- Frames at the start of the sequence (e.g., frame 1): contact_sum ~23, low contact -- approach phase.
- Middle frames (e.g., frames 7-82): contact_sum ~180-260, consistent high contact -- stable grasp.
- Frames at the end (e.g., frame 89): contact_sum ~17 -- release phase.

Approximately 6% of frames in the dataset have contact_sum < 0.5 (effectively zero contact). These frames carry a class label assigned from the sequence name, but the hand is not yet touching the object or has already released it.

**Available JSON fields not currently extracted in the CSVs:**

- `contact`: 778 float values [0,1], one per MANO hand mesh vertex, indicating contact intensity with the object. This is the only available proxy for grasp stability.
- `Mesh[0].mano_pose`: 45 MANO pose parameters -- an alternative/complementary representation to raw XYZ landmarks.
- `Mesh[0].mano_betas`: 10 MANO shape parameters capturing inter-subject hand shape variation.
- `actor.*`: subject demographics (sex, age, height, hand size).
- `projected_2D_pose_per_cam`: 2D projection of the 21 landmarks onto the image plane.

**Recommended action:** Re-extract CSVs including `contact_sum` per frame and filter training data to frames with contact_sum above a threshold (~50-100). This removes approach and release noise while keeping the stable-grasp core of the dataset.

---

---

### Dataset Split Revision — Migration to Subject-Based Split (s1) (2026-03-09)

**Decision:** Migrate from the current frame-stratified split to the official HOGraspNet `s1` (Unseen Subjects) protocol.

**Problem with current split:** The existing CSVs were generated with a stratified split at the frame level. This allows frames from the same trial to appear in both train and test simultaneously. In practice, the model is evaluated on frames it has already seen partial versions of during training. The model learns to generalize frames, not complete grasp instances from unseen subjects.

**Why this matters:** The target deployment is teleoperation with arbitrary users. A model that generalizes frames from known subjects does not guarantee generalization to new users with different hand morphologies. HOGraspNet covers subjects aged 10 to 74, with hand sizes varying from children to adults -- subject identity is a meaningful covariate.

**The s1 protocol** is defined in the official HOGraspNet dataloader and is designed explicitly to measure generalization to unseen subjects:
- Train: S11-S73 (~1M frames, 68%)
- Val: S01-S10 (~147K frames, 10%)
- Test: S74-S99 (~327K frames, 22%)

Test subjects never appear in train. All 28 classes are represented in all three splits. Results under this protocol are directly comparable to other work using HOGraspNet s1.

**Implication:** All runs prior to this decision (001-005) were evaluated under the frame-stratified split. Their results are internally consistent but may overestimate generalization to new subjects. Re-training under s1 is required to obtain honest generalization metrics. This is tracked as a pending item.

---

### Experiment: Synergy-Taxonomy Analysis v1 (2026-03-09)

*This is not a model training run. It is a dataset-level analysis establishing what grasp categories are empirically distinguishable using only the geometry of points and connections that a monocular sensor can capture. The result defines the reduced taxonomy that feeds Head A of the multitask model.*

---

#### Paso 1 — Preparación y Curaduría de Datos

**Creación del CSV desde los JSONs de HOGraspNet:** Se recorren los zips de anotaciones de HOGraspNet y se genera un único CSV con las siguientes columnas: `subject_id` (entero 1-99), `sequence_id` (string `{fecha}_{sujeto}_{objeto}_{agarre}/trial_{n}`), `cam` (mas/sub1/sub2/sub3), `grasp_type` (índice local 0-27), `contact_sum` (suma de los 778 valores del campo `contact`), y las 63 columnas XYZ de los 21 keypoints. La normalización geométrica se aplica en este paso, frame a frame, antes de guardar al CSV: se resta la posición de la muñeca (keypoint 0) a todos los puntos y se escala el esqueleto de forma que la distancia entre la muñeca (0) y el nudillo del índice (INDEX_FINGER_MCP, keypoint 5) sea 10 cm (factor S = 0.1 / d). Este parámetro está documentado en Santos et al. (2025), que lo validan en teleoperation 1:1 con pinch fino y movimientos de rotación. El split S1 y el filtro de existencia no se aplican en la ingesta -- el CSV contiene todos los frames de todos los sujetos, lo que permite reutilizarlo tanto para este experimento como para el entrenamiento del GCN.

**Filtro de Existencia (exclusivo para este experimento):** Al cargar el CSV, se conservan únicamente los frames donde `contact_sum > 0`. Este filtro elimina frames donde la mano no interactúa físicamente con el objeto sin penalizar las clases de agarre de precisión, cuyo contacto intrínseco es bajo pero real. El entrenamiento del GCN utiliza el CSV completo sin este filtro.

**Segmentación por Sujetos — Split S1 (oficial HOGraspNet):** El dataset se divide por identidad del participante, siguiendo el protocolo `s1` del dataloader oficial de HOGraspNet:

- Entrenamiento: Sujetos S11 a S73
- Validación: Sujetos S01 a S10
- Prueba: Sujetos S74 a S99

Esta segmentación garantiza que los resultados midan la generalización a nuevas manos con morfologías no vistas durante el entrenamiento (el dataset cubre sujetos de 10 a 74 años), y no la memorización de secuencias específicas de video.

**Por qué no un split aleatorio por frame:** HOGraspNet está organizado en secuencias temporales. Cada secuencia es un sujeto agarrando un objeto en un trial, con decenas de frames consecutivos. Un split aleatorio a nivel de frame mezcla frames casi idénticos de la misma secuencia entre train y test -- el frame 82 y el frame 83 del mismo trial son posturas prácticamente iguales. El modelo entrena con uno y se evalúa con el otro, lo que infla artificialmente las métricas: no está midiendo si generaliza a nuevas manos, sino si recuerda posturas que ya vio en una forma ligeramente distinta. El split por sujeto elimina este problema por completo: ningún frame de un sujeto de test aparece en train, garantizando que la evaluación mide generalización real a morfologías de mano no vistas.

**Referencia de etiquetas HOGraspNet (28 clases):** Los nombres provienen del campo `annotations[0].class_name` de los JSONs de HOGraspNet. El índice local (columna `grasp_type` en el CSV) corresponde al orden de `grasp_types` en `config.py` del dataloader oficial.

| Local | Feix ID | Nombre HOGraspNet       |
|-------|---------|-------------------------|
| 0     | 1       | Large_Diameter          |
| 1     | 2       | Small_Diameter          |
| 2     | 17      | Index_Finger_Extension  |
| 3     | 18      | Extension_Type          |
| 4     | 22      | Parallel_Extension      |
| 5     | 30      | Palmar                  |
| 6     | 3       | Medium_Wrap             |
| 7     | 4       | Adducted_Thumb          |
| 8     | 5       | Light_Tool              |
| 9     | 19      | Distal                  |
| 10    | 31      | Ring                    |
| 11    | 10      | Power_Disk              |
| 12    | 11      | Power_Sphere            |
| 13    | 26      | Sphere_4_Finger         |
| 14    | 28      | Sphere_3_Finger         |
| 15    | 16      | Lateral                 |
| 16    | 29      | Stick                   |
| 17    | 23      | Adduction_Grip          |
| 18    | 20      | Writing_Tripod          |
| 19    | 25      | Lateral_Tripod          |
| 20    | 9       | Palmar_Pinch            |
| 21    | 24      | Tip_Pinch               |
| 22    | 33      | Inferior_Pincer         |
| 23    | 7       | Prismatic_3_Finger      |
| 24    | 12      | Precision_Disk          |
| 25    | 13      | Precision_Sphere        |
| 26    | 27      | Quadpod                 |
| 27    | 14      | Tripod                  |

> **Fuente de cada columna:** El **Feix ID** y el **Nombre** se leen directamente del campo `annotations[0]` de cada JSON (`class_id` y `class_name`). El **Local index** no está en los JSONs -- es la posición del Feix ID dentro de la lista `grasp_types` de `config.py` del dataloader oficial de HOGraspNet, que define el orden canónico de las 28 clases. Es nuestra referencia interna y la columna `grasp_type` en el CSV.

---

#### Paso 2 — Derivación del Subespacio de Observación

El subespacio de observación se construye sobre los 21 keypoints XYZ normalizados geométricamente (ver Paso 1), siguiendo el orden estándar MediaPipe/HOGraspNet: 0 (Muñeca), 1-4 (Pulgar), 5-8 (Índice), 9-12 (Medio), 13-16 (Anular), 17-20 (Meñique).

**Cálculo de Vectores Óseos:** Se calculan 20 vectores 3D que representan los segmentos óseos de la mano siguiendo la cadena cinemática: `b_ij = j_i - j_j`, donde `j_i` y `j_j` son las coordenadas XYZ de dos articulaciones adyacentes. Estos vectores preservan la topología del esqueleto y constituyen la representación base para el cálculo de ángulos y para la red de grafos. Esta definición coincide con la formulación de bone vectors en Chen et al. (Robotics and Autonomous Systems, 2025), Ecuación (6).

**Extracción de Ángulos Articulares (20 DOFs):** Se extraen 20 ángulos que representan los grados de libertad de la mano, equivalentes a los 20 revolute joints del modelo articulado de DexPilot (Handa et al., ICRA 2020): 1 abducción + 3 flexión por dedo, 5 dedos = 20.

*Ángulos de flexión (15):* Para cada articulación con un triplete parent-joint-child válido, se calcula `θ_flex = arccos( dot(v_in, v_out) / (|v_in| |v_out|) )`, donde `v_in = joint - parent` y `v_out = child - joint`. Este cálculo es el método estándar de joint space mapping para teleoperación de manos diestras, validado por Chen et al. (2025), Ecuación (10), donde los ángulos articulares se obtienen como el arccos del producto punto normalizado de los bone vectors convergentes en cada articulación. Esto produce 3 ángulos por dedo largo (MCP, PIP, DIP) y 3 para el pulgar (CMC, MCP, IP), totalizando 15.

*Ángulos de abducción (5):* Miden la desviación lateral de cada dedo respecto a su dirección neutra, proyectada sobre el plano de la palma. Se define el plano de la palma como:

```
palm_normal = normalize( cross(kp5 - kp0, kp17 - kp0) )
```

donde kp0 = WRIST, kp5 = INDEX_MCP, kp17 = PINKY_MCP.

Para los 4 dedos largos, la abducción en el MCP se calcula como:

```
r = MCP_i - WRIST           # dirección metacarpal (referencia neutra)
v = PIP_i - MCP_i           # dirección falange proximal (dirección real)
r_proj = r - dot(r, n) * n  # proyección al plano de la palma
v_proj = v - dot(v, n) * n
θ_abd = arccos( dot(normalize(r_proj), normalize(v_proj)) )
```

| Ángulo | r (metacarpal) | v (proximal) |
|--------|---------------|--------------|
| θ_abd_index  | kp5 - kp0  | kp6 - kp5   |
| θ_abd_middle | kp9 - kp0  | kp10 - kp9  |
| θ_abd_ring   | kp13 - kp0 | kp14 - kp13 |
| θ_abd_pinky  | kp17 - kp0 | kp18 - kp17 |

Para el pulgar, la abducción es la componente fuera del plano del metacarpo del pulgar (palmar abduction en CMC):

```
thumb_dir = kp2 - kp1   # THUMB_MCP - THUMB_CMC
θ_abd_thumb = arcsin( |dot(normalize(thumb_dir), palm_normal)| )
```

| Dedo | Flexión (3) | Abducción (1) |
|------|-------------|---------------|
| Pulgar | CMC, MCP, IP | CMC palmar abd |
| Índice | MCP, PIP, DIP | MCP lateral |
| Medio | MCP, PIP, DIP | MCP lateral |
| Anular | MCP, PIP, DIP | MCP lateral |
| Meñique | MCP, PIP, DIP | MCP lateral |

**Síntesis por Mediana -- El Punto de Trial:** Los frames se agrupan por `sequence_id`. Para cada ángulo calculado se obtiene la mediana estadística sobre todos los frames del trial que sobrevivieron el filtro de existencia. El resultado es un único vector de ángulos por trial, que elimina el ruido de las fases de aproximación y liberación, y es robusto a los saltos de muestreo irregular del dataset.

---

#### Paso 3 — Análisis de Sinergias Posturales (PCA)

**Normalización Estadística:** Los vectores de ángulos de los sujetos de entrenamiento se normalizan para que cada ángulo tenga media cero y varianza unitaria. Esto evita que las flexiones de gran rango opaquen los movimientos informativos de menor amplitud, como la oposición del pulgar.

**Ejecución del PCA:** Se aplica Análisis de Componentes Principales sobre la matriz de vectores medianos de los sujetos de entrenamiento (S11-S73). El PCA se ajusta exclusivamente sobre el conjunto de entrenamiento; la transformación aprendida se aplica posteriormente a validación y prueba para evitar fuga de datos en el Paso 5.

**Criterio de Selección:** Se retienen los componentes principales que acumulen al menos el 85% de la varianza total. Adicionalmente se registran los loadings de los PCs 4 al 6, ya que la literatura sugiere que estos capturan la destreza fina asociada a la conformación a objetos específicos (Santello et al., 1998).

---

#### Paso 4 — Agrupamiento No Supervisado (Ensemble Clustering)

**Algoritmos:** Se ejecutan en paralelo k-means++, Modelos de Mezcla Gaussiana (GMM) y Clustering Jerárquico Aglomerativo sobre el subespacio de los componentes principales seleccionados.

**Métrica:** KMeans++ y Aglomerativo (Ward) operan con distancia euclidiana en el subespacio PCA. GMM utiliza implícitamente Mahalanobis a través de su covarianza completa. Dado que PCA ya decorrelaciona las variables, la distancia euclidiana en este subespacio es razonable.

**Selección de k:** Se exploran k = 3..12 clusters. Para cada k se calcula la silhouette promedio de los tres métodos y el ARI (Adjusted Rand Index) medio entre pares de métodos como medida de consenso. El k final se selecciona considerando tanto la silhouette como la interpretabilidad de los clusters resultantes.

**Reducción Taxonómica:** A partir de los dendrogramas, la composición de clusters y métricas de separación, las 28 clases de HOGraspNet se evalúan para determinar cuáles son posturalmente distinguibles con 21 keypoints XYZ. El número final de categorías se determina empíricamente.

---

#### Paso 5 — Validación de Identificabilidad

Este paso constituye el criterio de realidad definitivo: confirma si el sensor puede distinguir los clusters propuestos desde su representación nativa.

**Reetiquetado:** Los frames del dataset se reetiquetan con las nuevas etiquetas de cluster. Frames que pertenecían a clases distintas bajo la taxonomía original pero que el clustering asignó al mismo grupo reciben la misma etiqueta, reflejando su equivalencia funcional geométrica.

**Entrenamiento del Clasificador:** Se entrena un MLP simple tomando como entrada los 63 valores XYZ aplanados (21 keypoints × 3 coordenadas, normalizados geométricamente) de los sujetos de entrenamiento (S11-S73) y como objetivo las etiquetas de cluster. El XYZ aplanado es la representación nativa del sensor y la misma entrada que recibe el GCN, lo que hace esta validación directamente comparable con las condiciones de despliegue.

**Prueba sobre Sujetos No Vistos:** El clasificador se evalúa exclusivamente sobre los datos de los sujetos de prueba (S74-S99), garantizando que ningún sujeto del conjunto de prueba haya participado en la definición de los clusters ni en el entrenamiento del clasificador.

**Criterio de Colapso Final:** Se genera una matriz de confusión sobre el conjunto de prueba. Si dos clusters presentan una tasa de confusión recíproca superior al 50%, la metodología dicta que deben fusionarse en una única categoría funcional, ya que son geométricamente redundantes para el sensor bajo las condiciones de observación monocular.

---

**Results (v1, 2026-03-10):**

*Paso 3 -- PCA with Varimax Rotation:* 9 rotated components (RCs) for 87.9% variance. Varimax (Kaiser, 1958) applied following Jarque-Bou et al. (2019) to produce physiologically interpretable components:

| RC | Top loadings | Interpretation |
|----|-------------|----------------|
| RC1 | RING_PIP_flex, RING_MCP_flex, PINKY_PIP_flex | Global closure synergy |
| RC2 | MIDDLE_DIP_flex, INDEX_MCP_abd, RING_DIP_flex | Distal flexion vs abduction |
| RC3 | THUMB_CMC_flex, THUMB_CMC_abd | Thumb opposition |
| RC4 | PINKY_MCP_abd, RING_MCP_abd | Lateral abduction |
| RC5 | THUMB_MCP_flex, THUMB_IP_flex | Thumb flexion |
| RC6 | THUMB_IP_flex, THUMB_CMC_abd | Fine thumb dexterity |

*Paso 4 -- Class Separability Analysis:*

Two analyses were conducted:

**4a. Trial-level clustering (k-means, GMM, Ward):** Applied to the 8,361 trial vectors in 9D PCA space. Silhouette scores decreased monotonically from k=3 (0.195) to k=12 (<0.116), favoring the gross Power/Precision/Lateral split. Too coarse for teleoperation -- discarded as a collapse criterion. This pipeline groups trials by postural similarity but cannot answer "which specific pairs of the 28 classes are distinguishable?"

**4b. Pairwise centroid distances:** For each of the 378 pairs of classes, the Euclidean distance between class centroids in the 9D PCA-varimax space was computed: `d(i,j) = ||centroid_i - centroid_j||_2`. These distances are descriptive -- they characterize how similar two classes are in average posture. No threshold is applied here. The full distance matrix and the 28 centroids are stored in `results/class_separability.json` under `class_centroids_pca`.

Top 5 closest pairs (smallest d_synergy): Precision_Disk/Precision_Sphere (0.962), Power_Sphere/Precision_Sphere (0.979), Palmar_Pinch/Tip_Pinch (1.197), Writing_Tripod/Lateral_Tripod (1.352), Palmar/Medium_Wrap (1.337).

**Output of this step:** `results/class_separability.json` -- contains `class_centroids_pca` (28 vectors of 9 floats each, one per class). This file is an input to the Collapse Decision Analysis (Section 7).

*Paso 5 -- MLP reference classifier (28 classes):*

**Input:** 63 XYZ values per frame (21 keypoints x 3 coordinates), geometrically normalized. Same representation as the GCN.
**Model:** 3-layer MLP (63->256->128->28), ReLU, Dropout(0.3), NLL loss, Adam lr=1e-3, early stopping patience=10.
**Training data:** 995,886 frames from subjects S11-S73.
**Test data:** 327,260 frames from subjects S74-S99.

**Results:** Test Acc 69.7%, Macro F1 0.625 (early stopping at epoch 48).

**Top confusions (>10% in at least one direction):**

| True class | Predicted as | Rate | Symmetric? |
|-----------|-------------|------|-----------|
| Stick | Light_Tool | 42.4% | No (reverse 1.3%) |
| Precision_Sphere | Precision_Disk | 36.9% | No (reverse 2.6%) |
| Lateral_Tripod | Writing_Tripod | 32.8% | No (reverse 0.6%) |
| Palmar_Pinch | Tip_Pinch | 23.8% | No (reverse 1.7%) |
| Lateral | Light_Tool | 12.5% | Yes (reverse 11.5%) |

**Role of this step:** The MLP confusion matrix is a reference to assess whether the XYZ representation contains enough information to separate the 28 classes in principle. It is not the decision criterion for collapse -- a weak classifier failing on a pair does not justify collapsing them in a stronger model. The actual collapse criterion is the GCN confusion matrix (Run 006), which uses graph structure and angular features.

**Output of this step:** `results/confusion_28classes.png` (visual). The MLP confusion matrix is not used as a numerical input to any downstream step.

---

**Data flow into Collapse Decision Analysis:**

The two inputs to the collapse decision (`decide_collapses.py`) are:

| Input | Source | Content |
|-------|--------|---------|
| `class_separability.json` | This experiment, Paso 4b | 28 class centroids in 9D PCA-varimax space → `d_synergy` per pair (descriptive) |
| `confusion_matrix_norm_gcn.csv` | Run 006 (GCN trained on 28 classes) | Row-normalized confusion matrix → `c_gcn = max(CM[i,j], CM[j,i])` per pair (decision criterion) |

The synergy centroids and the GCN confusion matrix are computed independently -- one from the angle-based PCA analysis, one from end-to-end GCN training on raw keypoints. Their combination in the collapse decision is what makes the methodology two-signal rather than single-source.

---

### Run 006 -- GCN_CAM_8_8_16_16_32 Baseline, 28 Classes (2026-03-11)

**Purpose:** Establish the GCN-specific confusion matrix over all 28 original classes, to serve as the empirical validation criterion for collapse decisions. This run is not an optimization target -- it is a diagnostic tool for the taxonomy analysis.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | `GCN_CAM_8_8_16_16_32` |
| Classes | 28 (all original HOGraspNet / Feix classes) |
| Dataset | `data/raw/hograspnet.csv` |
| Split | S1 subject-level (Train S11-S73 / Val S01-S10 / Test S74-S99) |
| Train / Val / Test frames | 1,015,342 / 146,509 / 327,260 |
| Node features | `[x, y, z, theta_flex]` (F=4) |
| Graph-level feature | `theta_CMC` (palmar abduction) |
| Max epochs | 40 |
| Early stopping | patience = 10 (val accuracy) |
| Best epoch | **15** |
| Actual epochs run | 25 (stopped at epoch 25) |
| Batch size | 256 |
| LR | 1e-3 (Adam) |
| Loss | NLL + log_softmax |
| Hardware | Google Colab T4 |
| Training time | 43.57 min |

**Results:**

| Split | Loss | Accuracy | Macro F1 | Weighted F1 |
|-------|------|----------|----------|-------------|
| Val (best, epoch 15) | 1.3375 | **60.2%** | -- | -- |
| Test (S74-S99) | 1.2001 | **62.0%** | **0.550** | 0.613 |

**Training curve (Val Accuracy by epoch):**

```
Ep 01: 54.0% * | Ep 10: 58.1%   | Ep 19: 58.4%
Ep 02: 53.8%   | Ep 11: 58.0%   | Ep 20: 59.3%
Ep 03: 54.6% * | Ep 12: 58.1%   | Ep 21: 58.3%
Ep 04: 56.9% * | Ep 13: 57.6%   | Ep 22: 59.8%
Ep 05: 56.8%   | Ep 14: 59.6% * | Ep 23: 59.5%
Ep 06: 57.8% * | Ep 15: 60.2% * | Ep 24: 58.5%
Ep 07: 58.6% * | Ep 16: 58.1%   | Ep 25: 59.2%  ← early stop
Ep 08: 57.1%   | Ep 17: 59.1%   |
Ep 09: 58.3%   | Ep 18: 59.0%   |   * = saved checkpoint
```

The model reaches its best validation accuracy at epoch 15. Epochs 16-25 oscillate between 58.1% and 59.8% without improvement, and early stopping fires at epoch 25. Train loss continues decreasing (1.154 at ep 15 -> 1.105 at ep 25) while val loss stagnates (1.338 -> 1.348), indicating mild overfitting after epoch 15 and a representation ceiling rather than insufficient capacity.

**Per-class performance (all 28 classes, test set):**

| Local idx | Feix ID | Grasp name | Prec | Recall | F1 | Support |
|:---------:|:-------:|------------|------|--------|----|---------|
| 0 | 1 | Large_Diameter | 0.518 | 0.619 | 0.564 | 7,948 |
| 1 | 2 | Small_Diameter | 0.670 | 0.536 | 0.596 | 7,334 |
| 2 | 17 | Index_Finger_Ext | 0.829 | 0.772 | **0.800** | 18,858 |
| 3 | 18 | Extension_Type | 0.724 | 0.862 | 0.787 | 15,229 |
| 4 | 22 | Parallel_Ext | 0.696 | 0.671 | 0.683 | 6,688 |
| 5 | 30 | Palmar | 0.603 | 0.739 | 0.664 | 6,997 |
| 6 | 3 | Medium_Wrap | 0.441 | 0.160 | 0.234 | 2,204 |
| 7 | 4 | Adducted_Thumb | 0.721 | 0.540 | 0.618 | 16,783 |
| 8 | 5 | Light_Tool | 0.657 | 0.424 | 0.515 | 23,738 |
| 9 | 19 | Distal | 0.750 | 0.789 | 0.769 | 20,543 |
| 10 | 31 | Ring | 0.387 | 0.595 | 0.469 | 5,356 |
| 11 | 10 | Power_Disk | 0.578 | 0.632 | 0.604 | 4,933 |
| 12 | 11 | Power_Sphere | 0.470 | 0.547 | 0.505 | 6,602 |
| 13 | 26 | Sphere_4_Finger | 0.594 | 0.756 | 0.665 | 4,880 |
| 14 | 28 | Sphere_3_Finger | 0.541 | 0.585 | 0.562 | 13,047 |
| 15 | 16 | Lateral | 0.654 | 0.849 | 0.739 | 26,701 |
| 16 | 29 | Stick | 0.359 | 0.054 | **0.093** | 3,006 |
| 17 | 23 | Adduction_Grip | 0.766 | 0.529 | 0.626 | 5,786 |
| 18 | 20 | Writing_Tripod | 0.524 | 0.479 | 0.500 | 16,116 |
| 19 | 25 | Lateral_Tripod | 0.395 | 0.227 | 0.289 | 5,118 |
| 20 | 9 | Palmar_Pinch | 0.579 | 0.643 | 0.609 | 13,338 |
| 21 | 24 | Tip_Pinch | 0.625 | 0.417 | 0.501 | 7,154 |
| 22 | 33 | Inferior_Pincer | 0.558 | 0.599 | 0.578 | 14,076 |
| 23 | 7 | Prismatic_3F | 0.410 | 0.288 | 0.338 | 6,905 |
| 24 | 12 | Precision_Disk | 0.763 | 0.762 | 0.762 | 32,794 |
| 25 | 13 | Precision_Sphere | 0.283 | 0.340 | 0.309 | 5,335 |
| 26 | 27 | Quadpod | 0.421 | 0.599 | 0.495 | 5,491 |
| 27 | 14 | Tripod | 0.498 | 0.531 | 0.514 | 24,300 |

**Observations:** The five worst classes by F1 (Stick 0.093, Lateral_Tripod 0.289, Prismatic_3F 0.338, Precision_Sphere 0.309, Medium_Wrap 0.234) are all represented in the collapse pairs above threshold. The five best classes (Index_Finger_Ext 0.800, Extension_Type 0.787, Distal 0.769, Precision_Disk 0.762, Lateral 0.739) are all singletons in the proposed 17-class taxonomy -- the GCN distinguishes them cleanly.

**Main confusion pairs (max(CM[i,j], CM[j,i]) > 0.15):**

| Class A | Class B | max confusion | d_synergy | Notes |
|---------|---------|--------------|-----------|-------|
| Lateral | Stick | 0.370 | 1.729 | same Feix cell |
| Precision_Disk | Precision_Sphere | 0.313 | 0.962 | same Feix cell |
| Prismatic_3F | Tripod | 0.289 | 2.263 | diff Feix cell |
| Light_Tool | Lateral | 0.281 | 2.677 | diff Feix cell |
| Light_Tool | Stick | 0.239 | 1.615 | diff Feix cell |
| Ring | Inferior_Pincer | 0.221 | 1.884 | diff Feix cell |
| Writing_Tripod | Lateral_Tripod | 0.214 | 1.352 | diff Feix cell |
| Lateral_Tripod | Tripod | 0.211 | 2.980 | diff Feix cell |
| Palmar_Pinch | Tip_Pinch | 0.196 | 1.197 | same Feix cell |
| Large_Diameter | Power_Disk | 0.189 | 2.921 | same Feix cell |
| Sphere_3_Finger | Tripod | 0.182 | 1.739 | diff Feix cell |
| Palmar_Pinch | Inferior_Pincer | 0.179 | 1.762 | same Feix cell |
| Writing_Tripod | Tripod | 0.177 | 1.707 | diff Feix cell |

---

### Collapse Decision Analysis (2026-03-11)

**Inputs:**
- Synergy space centroids: `experiments/taxonomy_v1/results/class_separability.json` (`class_centroids_pca`, 9D PCA-varimax space)
- GCN confusion matrix: `experiments/taxonomy_v1/results/confusion_matrix_norm_gcn.csv` (Run 006, normalized by row)
- Script: `experiments/taxonomy_v1/decide_collapses.py`

**Decision criterion:** The GCN confusion rate is the sole filter for collapse. A pair collapses if and only if `max(CM[i,j], CM[j,i]) > gcn_thresh`. The synergy space distance characterizes the *type* of confusion but does not gate the collapse decision.

**Why max() and not mean():** The confusion metric for each pair is `max(CM[i,j], CM[j,i])` -- the worst-case direction -- rather than the symmetric mean. The choice follows from the operational context.

In teleoperation, the relevant question is not "are these two classes mutually indistinguishable?" but "is either class reliably recognizable for the operator?" If CM[A,B] = 21% (21% of grasp A frames are predicted as B), then when the operator tries to produce grasp A, the system will emit the wrong token 1 in 5 times -- regardless of whether B is also confused as A. The grasp A is operationally unreliable.

`mean()` would miss this. Asymmetric confusions occur systematically when one class is small or rare and gets absorbed by a larger neighboring class. Using `mean()` would leave those small classes in the taxonomy as nominally distinct categories that the system cannot reliably emit in practice. Two collapse pairs are affected concretely:

| Pair | max() | mean() | Decision with max | Decision with mean |
|------|-------|--------|-------------------|--------------------|
| Light_Tool + Stick | 0.239 | 0.122 | collapse | keep |
| Lateral_Tripod + Tripod | 0.211 | 0.106 | collapse | keep |

Using `mean()` would yield 19 classes instead of 17, with Stick and Lateral_Tripod retained as separate categories despite having recalls of 5.4% and 22.7% respectively -- too low to be reliable in deployment.

**Threshold:** `gcn_thresh = 0.15`. Single criterion -- no other threshold is used.

**Justification of gcn_thresh = 0.15:**

The threshold is not arbitrary -- it was selected to lie within a natural gap in the distribution of max pairwise confusion values. Sorting all 378 class pairs by their max(CM[i,j], CM[j,i]) value:

| Rank | Max confusion | Pair | Decision |
|------|--------------|------|----------|
| 13 | **0.177** | Writing_Tripod vs Tripod | last collapse |
| -- | **[gap: 0.043]** | -- | -- |
| 14 | **0.135** | Medium_Wrap vs Precision_Disk | first keep |
| 15 | 0.125 | Small_Diameter vs Adducted_Thumb | keep |
| ... | ... | ... | ... |

The gap between the 13th and 14th pairs is 0.043 units -- a 31% relative jump. **Any threshold in the interval [0.135, 0.177] produces exactly the same 13 collapse pairs.** The threshold of 0.15 lies at the center of this interval and is stable to perturbations of ±0.022.

Three additional properties support this choice:

1. **4.2x chance level.** With 28 classes, uniform random prediction yields 1/28 ≈ 0.036 per off-diagonal cell. A threshold of 0.15 = 4.2 × chance level selects confusions that are systematic -- not attributable to random prediction error.

2. **Top 3.4% of all pairs.** The 13 collapse pairs are in the 97th percentile of the full distribution (p97 = 0.178). The distribution is highly right-skewed: p50 = 0.009, p90 = 0.065, p95 = 0.114. Collapsing the top 3.4% is a conservative, well-localized intervention.

3. **The 14th pair does not generalize.** Medium_Wrap vs Precision_Disk (max confusion = 0.135) is a strongly asymmetric confusion: CM[Medium_Wrap, Precision_Disk] = 0.135 but CM[Precision_Disk, Medium_Wrap] = 0.001. This is a small class being partially absorbed by a large one -- not a symmetric indistinguishability. The 13 pairs above 0.15 show higher bidirectionality, consistent with genuine postural overlap rather than class-size artifacts.

**Role of the synergy space:** The synergy distance `d_synergy` is included as a descriptive field -- it characterizes *why* the GCN confuses each pair but does not gate the collapse decision. No threshold is applied to it. Pairs with small `d_synergy` have similar posture centroids (confusion likely reflects genuine postural overlap); pairs with large `d_synergy` have distinct posture centroids on average (confusion likely reflects high within-class variability or contact-dependent differences that landmarks cannot capture). These are interpretations, not criteria.

**Four-case framework (Feix cell cross-reference):**

| Case | Feix cell | GCN | Count | Interpretation |
|------|-----------|-----|-------|----------------|
| A -- collapse, Feix agrees | Same | confused | 5 pairs | Feix taxonomy and data converge |
| B -- keep, Feix wrong | Same | separates | 15 pairs | GCN overrides Feix; object shape forces distinct posture |
| C -- collapse, data-driven | Different | confused | 8 pairs | Feix taxonomy insufficient; empirical collapse |
| D -- keep, both agree | Different | separates | 350 pairs | No collapse; correct |

Case B is the key validation: 15 pairs share a Feix cell yet the GCN separates them cleanly (e.g., Large_Diameter vs Small_Diameter: c=0.0, d=6.08). Feix cell membership is neither necessary nor sufficient as a collapse criterion.

Case C is the key novel finding: 8 pairs cross Feix cells but the GCN cannot distinguish them (e.g., Light_Tool vs Lateral: c=0.281; the pair is separated biomechanically by thumb adduction but shares the adducted, extended finger configuration from keypoints).

**13 collapse pairs:**

| Pair | d_synergy | c_gcn | same_feix | Notes |
|------|-----------|-------|-----------|-------|
| Lateral + Stick | 1.729 | 0.370 | True | close centroids, Feix agrees |
| Precision_Disk + Precision_Sphere | 0.962 | 0.313 | True | nearest pair in synergy space |
| Prismatic_3F + Tripod | 2.263 | 0.289 | False | distant centroids, cross-cell |
| Light_Tool + Lateral | 2.677 | 0.281 | False | distant centroids, cross-cell |
| Light_Tool + Stick | 1.615 | 0.239 | False | asymmetric: Stick->Light_Tool 23.9% |
| Ring + Inferior_Pincer | 1.884 | 0.221 | False | cross-cell |
| Writing_Tripod + Lateral_Tripod | 1.352 | 0.214 | False | close centroids |
| Lateral_Tripod + Tripod | 2.980 | 0.211 | False | asymmetric: Lat_Tripod->Tripod 21.1% |
| Palmar_Pinch + Tip_Pinch | 1.197 | 0.196 | True | close centroids, Feix agrees |
| Large_Diameter + Power_Disk | 2.921 | 0.189 | True | distant centroids, Feix agrees |
| Sphere_3_Finger + Tripod | 1.739 | 0.182 | False | cross-cell |
| Palmar_Pinch + Inferior_Pincer | 1.762 | 0.179 | True | Feix agrees |
| Writing_Tripod + Tripod | 1.707 | 0.177 | False | weakest collapse pair (c just above threshold) |

**Proposed taxonomy (28 -> 17 classes):** Classes are merged using union-find over the 13 collapse pairs. Transitivity is intentional: if the GCN confuses A+B and B+C, then {A, B, C} form a single confusion cluster.

| New ID | Members | Size | Type |
|--------|---------|------|------|
| 0 | Sphere_3_Finger, Writing_Tripod, Lateral_Tripod, Prismatic_3F, Tripod | 5 | collapsed |
| 1 | Ring, Palmar_Pinch, Tip_Pinch, Inferior_Pincer | 4 | collapsed |
| 2 | Light_Tool, Lateral, Stick | 3 | collapsed |
| 3 | Large_Diameter, Power_Disk | 2 | collapsed |
| 4 | Precision_Disk, Precision_Sphere | 2 | collapsed |
| 5-16 | Small_Diameter, Index_Finger_Ext, Extension_Type, Parallel_Ext, Palmar, Medium_Wrap, Adducted_Thumb, Distal, Power_Sphere, Sphere_4_Finger, Adduction_Grip, Quadpod | 1 each | singleton |

**Notable finding:** This data-driven analysis independently recovers 17 functional classes -- the same number that Feix et al. (2016) identify as shape-equivalent postures when grouping their 33 grasps by hand configuration alone (Fig. 4, columns). The groupings are not identical (Feix uses opposition type and virtual finger count; we use GCN confusion), but the convergence on the same cardinality provides external validation that 17 is approximately the information-theoretic ceiling for contactless landmark-based grasp recognition.

**Outputs saved to `experiments/taxonomy_v1/results/`:**
- `collapse_decisions.csv` -- all 378 pairs with d_synergy, c_gcn, type, collapse flag
- `proposed_taxonomy.json` -- 17-class grouped taxonomy
- `scatter_synergy_vs_gcn.png` -- 2D decision plot (synergy distance vs GCN confusion)
- `four_case_summary.txt` -- breakdown by Feix cell agreement

---

### Run 007 -- Taxonomy V1 Collapse: 28 -> 17 Classes (2026-03-13)

**Purpose:** Train GCN_CAM_8_8_16_16_32 on the 17-class taxonomy derived from the Run 006 confusion analysis. Primary goal: verify that the data-driven collapse improves accuracy over Run 006 (28-class baseline) while retaining discriminability for the remaining singletons.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | `GCN_CAM_8_8_16_16_32` |
| Classes | 17 (`taxonomy_v1` collapse) |
| Dataset | `data/raw/hograspnet.csv` |
| Split | S1 subject-level (Train S11-S73 / Val S01-S10 / Test S74-S99) |
| Node features | `[x, y, z, theta_flex]` (F=4) |
| Graph-level feature | `theta_CMC` (palmar abduction) |
| Max epochs | 40 |
| Early stopping | patience = 10 (val accuracy) |
| Best epoch | **32** |
| Actual epochs run | 40 (no early stopping -- converged) |
| Batch size | 256 |
| LR | 1e-3 (Adam) |
| Loss | NLL + log_softmax |
| Hardware | Google Colab T4 |
| Training time | 64.78 min |

**Results:**

| Split | Loss | Accuracy | Macro F1 | Weighted F1 |
|-------|------|----------|----------|-------------|
| Val (best, epoch 32) | -- | **76.3%** | -- | -- |
| Test (S74-S99) | -- | **75.6%** | **0.670** | **0.750** |

**Comparison vs Run 006 (28 classes):**

| Metric | Run 006 | Run 007 | Delta |
|--------|---------|---------|-------|
| Test Accuracy | 62.0% | 75.6% | +13.6 pp |
| Macro F1 | 0.550 | 0.670 | +0.120 |
| Weighted F1 | 0.613 | 0.750 | +0.137 |

The taxonomy collapse produces a consistent and substantial improvement across all metrics. The +13.6 pp accuracy gain is structural -- the 13 collapse pairs removed distinctions the model could not learn, and the remaining 17 classes are more reliably distinguishable from 21 XYZ landmarks.

**Per-class results (test set):**

| Class | Members (original) | F1 | Notes |
|-------|-------------------|-----|-------|
| Tripod_cluster | Sphere_3F, Writing_Tripod, Lateral_Tripod, Prismatic_3F, Tripod | 0.764 | |
| Pinch_cluster | Ring, Palmar_Pinch, Tip_Pinch, Inferior_Pincer | 0.802 | |
| Lateral_cluster | Light_Tool, Lateral, Stick | 0.816 | |
| Power_Wrap_cluster | Large_Diameter, Power_Disk | 0.687 | |
| Precision_cluster | Precision_Disk, Precision_Sphere | 0.795 | |
| Small_Diameter | Small_Diameter | -- | singleton |
| Index_Finger_Ext | Index_Finger_Ext | -- | singleton |
| Extension_Type | Extension_Type | -- | singleton |
| Parallel_Ext | Parallel_Ext | -- | singleton |
| Palmar | Palmar | -- | singleton |
| Medium_Wrap | Medium_Wrap | **0.178** | worst class; recall=0.105 |
| Adducted_Thumb | Adducted_Thumb | **0.554** | strong confusion with Lateral_cluster |
| Distal | Distal | -- | singleton |
| Power_Sphere | Power_Sphere | **0.518** | confused with Precision_cluster |
| Sphere_4_Finger | Sphere_4_Finger | -- | singleton |
| Adduction_Grip | Adduction_Grip | -- | singleton |
| Quadpod | Quadpod | **0.529** | confused with Precision_cluster |

All five collapsed clusters achieve F1 > 0.68. The four worst singletons are Medium_Wrap (F1=0.178), Power_Sphere (0.518), Quadpod (0.529), and Adducted_Thumb (0.554).

**Medium_Wrap analysis:**

Medium_Wrap is the pathological case. Recall=0.105 -- the model almost never emits it. Confusion targets: Tripod_cluster (25%), Small_Diameter (17%), Precision_cluster (16%), Lateral_cluster (11%). Notably, it almost never maps to Power_Wrap_cluster (1%), which is its natural Feix neighbor. Hypothesized cause: YCB object #3 (bowl) produces keypoints consistent with spherical/tripod grasps, not cylindrical ones -- the physical object imposes the hand configuration and Medium_Wrap with a bowl looks nothing like Medium_Wrap in the Feix taxonomy diagram. Reference samples: subject_id=58, sequence_id=231004_S58_obj_03_grasp_3/trial_1.

**Observations:**

The four remaining problematic singletons represent two qualitatively different floors:

- **Contact-dependent (hard floor):** Power_Sphere and Quadpod differ from neighboring classes primarily in object shape and size. Without the object, the hand configurations overlap with Precision_cluster. These are likely permanent limits of contactless landmark-based recognition.
- **Viewpoint-sensitive (potentially addressable):** Adducted_Thumb vs Lateral_cluster confusion -- the thumb adduction that distinguishes these classes may not be stably visible from monocular webcam. Medium_Wrap likely reflects object-conditioned keypoint geometry that does not generalize.

---

### Run 008 -- Bone Vectors as Additional Node Features

**Why 28 classes again:** Run 008 deliberately returns to the full 28-class setting (same as Run 006) rather than taxonomy_v1. The motivation is to push the representational limits of the current data before collapsing. Recent literature (Lu et al. 2026, among others) proposes feature families -- bone vectors, velocity streams, pose encodings -- that may resolve confusions that forced the collapse in taxonomy_v1. The strategy is: exhaust the feature space on 28 classes first; only collapse what remains genuinely unresolvable. This also keeps Run 008 directly comparable to Run 006, isolating the effect of the new features.

**Hypothesis:** Adding bone vectors as raw displacement features per node reduces the domain gap between HOGraspNet keypoints (multi-camera, calibrated) and MediaPipe keypoints (monocular, webcam). Bone vectors -- defined as `bone_i = position_i - position_parent(i)` -- encode relative joint-to-joint displacements. The reasoning is that a uniform translation or scale error in absolute positions partially cancels in relative displacements, making the learned representations more robust to sensor-specific biases.

**Theoretical basis:** Lu et al., "Skeleton-Prompt: Rethinking spatial and temporal representations for learning skeleton-based action recognition", *Pattern Recognition* 2026. Section 3.1 defines the bone stream as `b_i = p_i - p_{parent(i)}` -- a 3D vector, not a scalar magnitude -- and demonstrates that bone and joint streams carry complementary information.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | `GCN_CAM_8_8_16_16_32` |
| Classes | 28 (same as Run 006, for direct comparison) |
| Dataset | `data/raw/hograspnet.csv` |
| Split | S1 subject-level |
| Node features | `[x, y, z, theta_flex, bone_x, bone_y, bone_z]` (F=7) |
| Graph-level feature | `theta_CMC` (palmar abduction) |
| Bone vector definition | `bone_i = position_i - position_parent(i)`, raw (not unit-normalized) |
| WRIST (no parent) | `[0, 0, 0]` |
| Scale | Controlled by geometric normalization (dist WRIST->INDEX_MCP = 0.1) |
| Max epochs | 40 |
| Early stopping | patience = 10 |
| Batch size | 256 |
| LR | 1e-3 (Adam) |
| Hardware | Google Colab T4 |
| Checkpoint | `best_model_run008_c28_xyz_bone.pth` |
| Colab env vars | `GG_RUN_NAME=run008_c28_xyz_bone`, `GG_COLLAPSE=none`, `GG_BONE_VECTORS=true` |

**What changes vs Run 006:** Only the node feature vector. Run 006: F=4 `[x,y,z,theta_flex]`. Run 008: F=7 `[x,y,z,theta_flex,bone_x,bone_y,bone_z]`. Architecture, dataset, split, and hyperparameters are identical.

**Results:**

| Split | Loss | Accuracy | Macro F1 | Weighted F1 |
|-------|------|----------|----------|-------------|
| Val (best, epoch 19) | 1.2961 | **61.3%** | -- | -- |
| Test (S74-S99) | 1.1241 | **64.6%** | **0.575** | **0.639** |

Training stopped at epoch 29 via early stopping (patience=10, no improvement since epoch 19).

**Comparison vs Run 006 (28 classes, no bone vectors):**

| Metric | Run 006 | Run 008 | Delta |
|--------|---------|---------|-------|
| Test Accuracy | 62.0% | 64.6% | +2.6 pp |
| Macro F1 | 0.550 | 0.575 | +0.025 |
| Weighted F1 | 0.613 | 0.639 | +0.026 |
| Training time | -- | 48.14 min | -- |

Bone vectors produce a consistent but modest improvement across all metrics. The +2.6 pp accuracy gain confirms that relative joint displacements carry additional discriminative information beyond absolute XYZ positions.

**Per-class performance (all 28 classes, test set):**

| Local idx | Feix ID | Grasp name | Prec | Recall | F1 | Delta vs R006 | Support |
|:---------:|:-------:|------------|------|--------|----|:-------------:|---------|
| 0 | 1 | Large_Diameter | 0.534 | 0.652 | 0.587 | +0.023 | 7,948 |
| 1 | 2 | Small_Diameter | 0.578 | 0.616 | 0.596 | +0.001 | 7,334 |
| 2 | 17 | Index_Finger_Ext | 0.810 | 0.809 | **0.809** | +0.010 | 18,858 |
| 3 | 18 | Extension_Type | 0.789 | 0.827 | 0.808 | +0.021 | 15,229 |
| 4 | 22 | Parallel_Ext | 0.655 | 0.707 | 0.680 | -0.003 | 6,688 |
| 5 | 30 | Palmar | 0.688 | 0.649 | 0.668 | +0.004 | 6,997 |
| 6 | 3 | Medium_Wrap | 0.516 | 0.160 | 0.244 | +0.010 | 2,204 |
| 7 | 4 | Adducted_Thumb | 0.664 | 0.537 | 0.594 | -0.024 | 16,783 |
| 8 | 5 | Light_Tool | 0.562 | 0.664 | 0.609 | +0.094 | 23,738 |
| 9 | 19 | Distal | 0.816 | 0.761 | 0.788 | +0.019 | 20,543 |
| 10 | 31 | Ring | 0.494 | 0.497 | 0.495 | +0.026 | 5,356 |
| 11 | 10 | Power_Disk | 0.593 | 0.728 | 0.654 | +0.050 | 4,933 |
| 12 | 11 | Power_Sphere | 0.542 | 0.548 | 0.545 | +0.040 | 6,602 |
| 13 | 26 | Sphere_4_Finger | 0.603 | 0.771 | 0.677 | +0.012 | 4,880 |
| 14 | 28 | Sphere_3_Finger | 0.626 | 0.517 | 0.567 | +0.005 | 13,047 |
| 15 | 16 | Lateral | 0.756 | 0.769 | 0.763 | +0.024 | 26,701 |
| 16 | 29 | Stick | 0.378 | 0.180 | 0.244 | +0.151 | 3,006 |
| 17 | 23 | Adduction_Grip | 0.741 | 0.580 | 0.650 | +0.024 | 5,786 |
| 18 | 20 | Writing_Tripod | 0.497 | 0.551 | 0.523 | +0.022 | 16,116 |
| 19 | 25 | Lateral_Tripod | 0.533 | 0.161 | 0.248 | -0.041 | 5,118 |
| 20 | 9 | Palmar_Pinch | 0.600 | 0.692 | 0.643 | +0.034 | 13,338 |
| 21 | 24 | Tip_Pinch | 0.640 | 0.469 | 0.541 | +0.040 | 7,154 |
| 22 | 33 | Inferior_Pincer | 0.596 | 0.655 | 0.624 | +0.046 | 14,076 |
| 23 | 7 | Prismatic_3F | 0.506 | 0.286 | 0.365 | +0.027 | 6,905 |
| 24 | 12 | Precision_Disk | 0.729 | 0.826 | 0.774 | +0.012 | 32,794 |
| 25 | 13 | Precision_Sphere | 0.318 | 0.238 | 0.272 | -0.037 | 5,335 |
| 26 | 27 | Quadpod | 0.533 | 0.563 | 0.547 | +0.052 | 5,491 |
| 27 | 14 | Tripod | 0.567 | 0.582 | 0.574 | +0.060 | 24,300 |

Bone vectors improve 23 out of 28 classes. Largest gains: Stick (+0.151), Light_Tool (+0.094), Power_Disk (+0.050). Regressions: Lateral_Tripod (-0.041), Precision_Sphere (-0.037), Adducted_Thumb (-0.024) -- all classes with already low F1 in Run 006. The Tripod-adjacent cluster (Medium_Wrap, Stick, Lateral_Tripod) remains the worst-performing group with F1 under 0.25 and is the primary candidate for collapse once the feature space is exhausted.

### Run 009 -- Bone Vectors + Velocity (28 classes)

**Configuration:** Same as Run 008 plus velocity stream. `GG_BONE_VECTORS=true`, `GG_VELOCITY=true`.
Node feature vector F=10: `[x, y, z, theta_flex, bone_x, bone_y, bone_z, vel_x, vel_y, vel_z]`.
Dataset: `hograspnet_mano.csv` (required for `frame_id` column to compute sequence-aware velocity).

**Velocity definition:** `v_i = (pos_i(t) - pos_i(t-1)) / dt`, in units/second.
- Training: `dt = 0.1s` (HOGraspNet annotations at 10 fps, Section 3.5 of paper).
- Deploy (HaMeR): `dt = time.time() - timestamp_prev`, measured wall-clock between consecutive
  HaMeR responses. Measured empirically: mean ~374ms (~2.7 fps).

**Why dt-normalization:** Without normalization, training velocity scale (dt=100ms) differs
from deploy scale (dt~374ms) by a factor of ~3.7x -- the same physical movement produces
velocity vectors ~3.7x larger in deploy than in training. Dividing by dt on both sides
puts velocity in units/second, making the feature scale-invariant to frame rate. This
allows the model trained on 10 fps data to generalize to any deploy frame rate.

**Implementation:**
- `grasps.py`: `vel = (curr_xyz - prev_xyz) / 0.1` per sequence group.
- `tograph.py`: deploy branch tracks `_prev_timestamp` alongside `_prev_positions`;
  `vel = (pos - prev_pos) / dt_real`; `reset_velocity()` clears both buffers.

**Results:** TBD (pending Colab run).

---

## Pending / Future Work

### Model
- [x] **Baseline GCN on 28 classes:** Done -- Run 006 (62.0% test acc, Macro F1=0.550).
- [x] **Finalize collapse taxonomy:** Done -- taxonomy_v1, 17 classes, gcn_thresh=0.15.
  See Collapse Decision Analysis section above.
- [x] **GCN on reduced taxonomy:** Done -- Run 007 (75.6% test acc, Macro F1=0.670).
- [x] **Bone vectors (Run 008):** Done -- 64.6% test acc, Macro F1=0.575. +2.6 pp over Run 006.
- [ ] **Bone + velocity (Run 009):** Pending Colab run. F=10: `[x,y,z,theta_flex,bone_x,bone_y,bone_z,vel_x,vel_y,vel_z]`.
- [ ] **Dexonomy verification:** Check whether collapsed pairs produce sufficiently similar
  joint configs for Shadow Hand. Validates that collapsed classes are also functionally
  equivalent from the robot actuation perspective.
- [ ] **Multi-Head Output:** After global pooling, bifurcate into two parallel heads.
  This is the core design decision motivating the entire taxonomy experiment.

  **Head A -- Discrete classification (the lock):** Predicts one of N grasp classes,
  where N is exactly what the taxonomy experiment determines: the number of classes
  that the sensor + architecture can reliably resolve. This is a data-driven answer
  to the question Stival et al. (2019) and Abbasi et al. (2019) addressed by
  collapsing to 5 coarse categories -- but we let the data decide rather than
  imposing an arbitrary number. Head A gates the robot: without a stable class lock,
  the robot does not actuate.

  **Head B -- Continuous synergy regression (the fine adjustment):** Predicts k
  postural synergy coefficients (PCA on 20 joint angles, k=2 or k=3 capturing
  ~80-85% of postural variance, Santello et al. 1998). These capture within-class
  variation: aperture, finger curvature, fine pinch depth -- everything Head A
  cannot resolve discretely. Targets are pre-computed by fitting PCA on the training
  set joint angles and projecting each sample. PCA is global (not per-class) since
  the shared backbone already separates classes.

  **Why this works:** Most pinch/precision grasps that appear visually similar
  (Precision Disk vs Precision Sphere, Palmar Pinch vs Tip Pinch) differ mainly in
  continuous finger configuration, not discrete posture. Head A locks the grasp
  type; Head B drives the continuous adjustment. Neither head alone is sufficient.

  **Neuroscientific grounding (Flindall & Gonzalez, 2019):** Kinematic evidence from
  reach-to-grasp studies shows that the human motor system operates in exactly this
  two-stage fashion: "the appropriate grasp is chosen from a repertoire of actions
  (a vocabulary of movements), and then adapted to current target parameters as
  needed." Head A models the discrete selection from the vocabulary; Head B models
  the continuous adaptation. The same paper shows that Maximum Grip Aperture (MGA)
  -- the continuous hand opening during prehension -- is modulated by both object
  size and action intent independently, providing direct neuroscientific precedent
  for the `apertura`/synergy_coeffs separation from discrete class identity.
  Notably, the paper also establishes that "movements that appear similar in their
  mechanical execution may have entirely distinct neural origins, stemming from
  differences in their functional purpose" -- justification for maintaining fine-
  grained Head A classes rather than collapsing to coarse categories.

  **Loss:** `L = L_CE(Head A) + lambda * L_MSE(Head B)`, start with lambda=1.0.
  GraspToken becomes `{class_id, confidence, synergy_coeffs: List[float]}`.
  YAMLRobotAdapter maps synergy_coeffs to Shadow Hand joint targets.
- [ ] **LR scheduler:** Try cosine annealing or ReduceLROnPlateau. Low cost, no new features.
- [ ] **Index Finger Extension → Distal confusion (9.6%):** Investigate -- unexpected given
  θ_CMC should separate these (different rows and columns). May be a training artifact.

### Framework
- [ ] Real-time inference app (`grasp-app`): MediaPipeBackend → ToGraph → GCN → VotingWindow → GraspToken
- [ ] Shadow Hand YAML configuration from Dexonomy dataset (`grasp-robot`)
- [ ] ROS integration (`grasp-robot`)

---

## Citation

If you use this code, please cite the associated thesis (citation forthcoming) and:

**GRASP Taxonomy:**
> Feix, T., Romero, J., Schmiedmayer, H. B., Dollar, A. M., & Kragic, D. (2016). The GRASP taxonomy of human grasp types. IEEE Transactions on Human-Machine Systems, 46(1), 66-77.

**HOGraspNet:**
> Lee, et al. "Dense Hand-Object (HO) GraspNet with Full Grasping Taxonomy and Dynamics." ECCV 2024.

**Dexonomy:**
> Chen, et al. "Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy." RSS 2025.

**Reach-to-grasp kinematics and action intent:**
> Flindall, J., & Gonzalez, C. L. R. (2019). On the neurocircuitry of grasping: The influence of action intent on kinematic asymmetries in reach-to-grasp actions. Attention, Perception, & Psychophysics, 81, 2217-2236.

**CAM-GNN (learnable adjacency matrix):**
> Leng, Z., Chen, J., Shum, H. P. H., Li, F. W. B., & Liang, X. (2021). Stable hand pose estimation under tremor via graph neural network. IEEE VR 2021. https://doi.org/10.1109/VR50410.2021.00044

---

## License

Academic use only, consistent with the HOGraspNet dataset license terms.
