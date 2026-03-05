# GraphGrasp: A Framework for Grasp Intent-Driven Teleoperation

Graph Neural Network for intention-oriented grasp recognition in vision-based robotic hand teleoperation.

> **Thesis:** "Graph Neural Networks for Intention-Oriented Grasp Recognition in Vision-Based Robotic Hand Teleoperation"

---

## Structure

```
AIST-hand/
├── grasp-model/              # GCN model, graph construction, GraspToken, VotingWindow
│   ├── src/grasp_gcn/        # installable package
│   ├── scripts/
│   │   └── ingestion/        # HOGraspNet → CSV (pending)
│   ├── train.py
│   ├── data/                 # raw CSVs + processed graphs
│   └── experiments/          # model checkpoints + TensorBoard
├── grasp-app/                # Real-time perception app
│   ├── perception/
│   │   ├── mediapipe_backend.py  # reference implementation (RGB + MediaPipe)
│   │   └── apertura.py           # AperturaCalculator
│   ├── main.py               # inference loop entry point
│   └── models/               # place best_model.pth here
├── grasp-robot/              # Robot adapters: YAMLRobotAdapter
│   └── grasp_configs/
│       └── shadow_hand.yaml  # canonical poses per grasp class (TODO)
└── .venv/                    # shared virtual environment
```

---

## System Architecture

```
TRAINING (offline)
──────────────────
HOGraspNet JSONs → [ingestion] → CSV → [ToGraph] → PyG Data → [train.py] → model.pth


INFERENCE (online)
──────────────────
                  ┌─ grasp-app ──────────────────────────────────────────────────────────┐
Cámara → MediaPipe → landmarks XYZ
                │
                ↓
             ToGraph → GCN → class_id + confidence
                                  │
                         VotingWindow → False (no consensus, skip)
                                      → True  (class confirmed, locked)
                                                │
                                      ┌─────────┴──────────────────────┐
                                      │  per frame while class locked:  │
                                      │  AperturaCalculator → apertura  │
                                      │  GraspToken(clase, conf, ap)    │
                                      └─────────────────────────────────┘
                                                │
                  ┌─ grasp-robot ───────────────┼──────────────────────────────────────┐
                   YAMLRobotAdapter(token) → pose_base[clase] × apertura → joint angles → robot
```

---

## Module Responsibilities

| Component | Module | Description |
|---|---|---|
| `PerceptionBackend` | `grasp-model` | Abstract interface: any sensor → 21 XYZ landmarks |
| `ToGraph` | `grasp-model` | Landmark dict → PyG graph |
| `GCN` | `grasp-model` | PyG Data → logits → grasp class |
| `VotingWindow` | `grasp-model` | Filters class over N consecutive frames |
| `GraspToken` | `grasp-model` | `{class_id, class_name, confidence, apertura}` — full intent signal |
| `grasp-app` | `grasp-app` | Reference implementation of `PerceptionBackend` (RGB camera + MediaPipe) |
| `AperturaCalculator` | `grasp-app` | landmarks + class → aperture float [0,1] |
| `YAMLRobotAdapter` | `grasp-robot` | `pose_base[class] × apertura` → joint angles |
| YAML per hand | `grasp-robot` | Canonical poses + aperture joints per grasp class |

---

## Key Design Decisions

**GraspToken includes aperture.**
`GraspToken` is the complete intent signal — class (discrete, from model) + aperture (continuous, from geometry). Together they fully describe the intended grasp. The adapter only needs `GraspToken` to execute, keeping the interface clean and robot-agnostic.

**Aperture is computed from human hand geometry, not robot morphology.**
Aperture is a normalized scalar [0,1] measuring how open the human hand is for a given grasp class. It is independent of the robot — 0 means as closed as possible for that grasp, 1 means fully open. The robot interprets this scalar within its own physical limits. No explicit min/max needed in the YAML.

**Aperture depends on class — computed after VotingWindow confirms class.**
You cannot measure aperture without knowing the grasp type (different grasps measure openness differently). Therefore: landmarks → class (via GCN + VotingWindow) → aperture (via AperturaCalculator) → GraspToken.

**VotingWindow on class only. Aperture is frame-by-frame once class is locked.**
- Class: discrete, requires stability → voting window (N=5 ≈ 167ms at 30fps)
- Once class is confirmed, it locks and aperture is computed every frame continuously
- Each frame emits a new GraspToken with the same class but updated aperture
- Class unlocks when VotingWindow detects a new consensus

**GraphGrasp is sensor-agnostic.**
The framework contract starts at 21 XYZ landmarks — not at the camera. Any sensor (RGB + MediaPipe, depth camera, motion capture, haptic glove) can feed the pipeline by implementing `PerceptionBackend`. `grasp-app` is the reference implementation using MediaPipe.

**Geometric normalization is the sensor contract.**
Each `PerceptionBackend` implementation must normalize landmarks geometrically before passing them to `ToGraph`: (1) subtract wrist position (center at origin), (2) divide by distance from wrist to middle finger MCP (scale-invariant). This ensures that different sensors produce landmarks in the same coordinate space, making the model truly sensor-agnostic. The same normalization is applied during training (HOGraspNet ingestion).

**Bimanual teleoperation: left hand handled via mirroring.**
The model is trained on right-hand data only (HOGraspNet captures right hands). For bimanual use, each `PerceptionBackend` is responsible for detecting handedness and mirroring left-hand landmarks (flip X axis) before passing them to the pipeline. This keeps the model architecture simple while supporting both hands. `MediaPipeBackend` in `grasp-app` must implement this.

**New robot hand = new YAML, no code changes.**
The YAML defines: `pose_base` per class (canonical joint configuration) + which landmarks measure aperture per class (human geometry reference). Canonical poses must be defined per robot hand by the implementor (e.g. via simulation-based pose synthesis). See [Dexonomy (RSS 2025)](https://arxiv.org/abs/2504.18829) for methodology reference.

---

## Public API (`grasp-model`)

Install once, use from any module:

```bash
pip install -e grasp-model/
```

```python
from grasp_gcn import get_network, ToGraph, GraspToken, VotingWindow, PerceptionBackend
```

---

## Setup

```bash
# Install Python 3.11 via pyenv (if needed)
curl https://pyenv.run | bash
pyenv install 3.11.9

# Create venv at monorepo root
python3.11 -m venv .venv
source .venv/bin/activate

# Install grasp-model as editable package
pip install -e grasp-model/
pip install -r grasp-model/requirements.txt
```

---

## Shadow Hand Simulation (ROS Noetic)

Install via Shadow Robot Aurora one-liner:

```bash
bash <(curl -Ls https://raw.githubusercontent.com/shadow-robot/aurora/v2.2.5/bin/run-ansible.sh) docker_deploy \
  --branch=v2.2.5 \
  --inventory=local \
  product=hand_e \
  tag=noetic-v1.0.31 \
  reinstall=true \
  sim_icon=true \
  sim_hand=true \
  container_name=dexterous_hand_simulated
```

Requires Ubuntu 20.04. Robot integration lives in `grasp-robot/`.

---

## Citation

**GRASP Taxonomy:**
> Feix, T., Romero, J., Schmiedmayer, H. B., Dollar, A. M., & Kragic, D. (2016). The GRASP taxonomy of human grasp types. IEEE Transactions on Human-Machine Systems, 46(1), 66-77.

**HOGraspNet:**
> Lee, et al. "Dense Hand-Object (HO) GraspNet with Full Grasping Taxonomy and Dynamics." ECCV 2024.

**Dexonomy:**
> Chen, et al. "Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy." RSS 2025.
