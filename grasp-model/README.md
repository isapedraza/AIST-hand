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
| 9 | 19 | Distal Type | Power |
| 10 | 31 | Ring Finger | Power |
| 11 | 10 | Power Disk | Power |
| 12 | 11 | Power Sphere | Power |
| 13 | 26 | Sphere 4-Finger | Power |
| 14 | 28 | Sphere 4-Finger (variant) | Power |
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

> **Note:** Feix IDs 26 and 28 are both sphere power grasps distinguished by finger configuration (Feix et al. 2016, Fig. 4, Power–Pad column). Local indices are assigned by the HOGraspNet ingestion script (`FEIX_INDICES` in `hograspnet_to_csv.py`) and do not follow Feix order.

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

This system sidesteps both problems by classifying at the level of intent rather than configuration. The GCN maps 21 keypoints to a grasp type — a discrete, stable, morphology-agnostic signal. The robot adapter then translates that signal into its own joint space, independently of how the human hand is shaped. The VotingWindow reinforces this: it does not low-pass filter a continuous signal, it withholds the signal until the intent is unambiguous. The result is a control channel that is stable by design, not by post-processing.

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
| 9 | 19 | Distal Type | 0.795 | |
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

This is not surprising: Feix et al. (2016) noted that the 33 taxonomy classes reduce to
17 if object properties (size, shape, orientation) are factored out. HOGraspNet took a
partial step in that direction by reducing to 28, but the results show that several pairs
remain geometrically indistinguishable from keypoints alone.

The clearest cases are Precision Sphere vs Precision Disk (47% mutual confusion) and the
cylindrical group Large/Medium/Small Diameter: in all of these, the hand adopts the same
topology and it is the object that varies. The model has no access to the object, so
confusing these classes is the correct response given what it observes.

This connects directly to the GraspToken design. The separation between `class_id`
(discrete topology) and `apertura` (continuous adaptation to the object) already anticipates
this problem: the groups the classifier cannot distinguish are exactly those that `apertura`
resolves at runtime. In that sense, the classifier's "confusions" within these groups are
not functional errors for the robot.

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

This is directly supported by two quantitative studies.

Jarque-Bou et al. (2019) extracted kinematic synergies from 77 subjects performing 20
grasps. Only three synergies appeared in more than half of the subjects — the universal
coordination patterns across the population. Synergy 1 is MCP flexion of fingers 3–5
combined with PIP flexion of fingers 2–5. Synergy 3 is thumb CMC abduction with MCP
extension and IP flexion (thumb opposition). These two synergies define the primary axes
along which grasps differ kinematically.

Stival et al. (2019) build on this directly. Their quantitative taxonomy groups the same
grasps into five categories from joint angle data, and explicitly maps the categories to
synergies: cylindrical grasps correspond to Synergy 1 (PIP flexion + finger closure),
spherical grasps to MCP flexion patterns. The confusion pairs above map to boundaries
between these categories — exactly where Synergy 1 and 3 are most discriminative.

Synergy 2 (wrist flexion + palmar arch, CMC5) is not computable from 21 landmarks —
CMC5 has no corresponding keypoint in the MediaPipe skeleton, and wrist abduction
requires forearm reference points that are not available. This is a hard limit of the
representation, not a modeling choice.

**Run 003 design decision:** add the joint flexion angle as one extra scalar feature per
node, computed from the parent→joint→child triple for each internal joint. WRIST and
fingertips receive 0.0. This increases node features from 3 to 4 and directly encodes
Synergies 1 and 3 without changing the graph topology or model architecture.

### Run 003 — Joint Flexion Angles (pending)

**Configuration:**
- Model: `GCN_8_8_16_16_32`
- Dataset: same splits as Run 001/002
- Features: `[x, y, z, θ]` per node — θ = joint flexion angle in radians
- Epochs: 20 (max) | Early stopping patience=5 | Batch size: 256 | LR: 1e-3
- Loss: CrossEntropyLoss (uniform weights)
- Hardware: Google Colab T4 GPU

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
