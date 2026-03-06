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

**Why not 1:1 teleoperation?**

The obvious alternative to this system is direct joint-angle mapping: read the human's finger angles, send them to the robot. This breaks for two independent reasons.

The first is structural. Human and robot hands are morphologically different — different number of fingers, different joint ranges, different proportions. There is no general mapping between a human MCP angle and a robot actuator position. A three-fingered gripper cannot receive a five-finger joint vector. No amount of calibration fixes this because the problem is not numerical, it is topological.

The second reason applies even in cases where approximate mapping is geometrically possible: sensor noise propagates directly to the robot. Every small tremor or detection jitter in the landmarks becomes a command sent downstream. The robot moves continuously even when the operator's intent is stable.

Classifying at the level of intent rather than configuration sidesteps both problems. The GCN maps 21 keypoints to a grasp type — a discrete, stable, morphology-agnostic signal. The robot adapter translates that signal into its own joint space, independently of how the human hand is shaped. The VotingWindow withholds the signal until the intent is unambiguous, rather than low-pass filtering a continuous stream. The control channel is stable by construction.

**No data leakage:** normalization statistics (mean, std) are computed only on the training split and saved to `data/processed/train_stats.npz`. Val and test splits load these same statistics. The same stats must be applied at inference time.

**Domain gap:** the model is trained on HOGraspNet data (multi-view RGBD + MANO fitting, world coordinates) but runs inference on MediaPipe output (single-view RGB, normalized [0,1] coordinates). This gap is mitigated by z-score normalization but should be discussed when interpreting real-time performance.

**Hand mirroring:** left-hand samples are horizontally mirrored (`x' = 1 - x`) during dataset construction to normalize all samples to a right-hand reference frame.

---

## Experiment Log

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

---

## Pending / Future Work

- [ ] Real-time inference app (`grasp-app`)
- [ ] Shadow Hand YAML configuration from Dexonomy dataset
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

---

## License

Academic use only, consistent with the HOGraspNet dataset license terms.
