# DECISIONS.md -- GraphGrasp Project Decision Log

## How to add an entry

Each entry documents a design decision, architectural change, or key insight.
Use this format:

```
---
## Entry N -- YYYY-MM-DD: Short title

**Context**: Why this decision was needed. What problem or observation motivated it.

**Decision**: What was decided. Be specific -- architecture, algorithm, justification.

**Alternatives considered**: What was ruled out and why.

**Expected impact**: What this should change in the system.

**References**: Papers, experiments, or prior entries that justify this.

**Status**: Proposed | In progress | Implemented | Validated
---
```

Rules:
- One entry per decision. Do not edit past entries -- add a new one if something changes.
- Link to experiments by run name (e.g. abl04) not by file path.
- Keep Context and Decision sections honest. If a decision was wrong, document that in a follow-up entry.
- References should be citable (author, year, or run name).

---

## Entry 0 -- 2026-03-31: System overview (baseline)

**Context**: Overview of the full system as of end of ablation study. Starting point for future architectural decisions.

### Model

- Architecture: `GCN_CAM_8_8_16_16_32` -- 5 GCNConv layers (8-8-16-16-32), ELU activation
- Readout: mean-pool + max-pool concat → [B, 64] → FC 64→128→28
- Learned adjacency: CAM [21, 21], shared across all samples and frames (static)
- Node features (24): xyz (3) + flex angle (1) + AHG angles (10) + AHG distances (10)
- Graph-level feature: theta_CMC concatenated before fc1 in some variants
- Parameters: ~14k (intentionally small, CPU-deployable)
- Output: log_softmax [B, 28] -- 28 Feix grasp classes

### Dataset

- HOGraspNet (ECCV 2024): ~1.5M frames, 99 subjects, 30 YCB objects, 28 Feix classes
- S1 subject-level split: train S11-S73, val S01-S10, test S74-S99
- No contact_sum filtering

### Best model (abl04)

- Features: xyz + AHG + flex (interaction effect: flex alone hurts, AHG alone marginal, together best)
- Test accuracy: 71.07%, Macro F1: 0.657
- Key insight from ablation: capacity is not the bottleneck (abl09 = 4.7x params, same accuracy)

### Deploy pipeline

```
Camera → MediaPipe → ToGraph (24 features) → GCN_CAM_8_8_16_16_32 → VotingWindow → FSM → Shadow Hand qpos
```

### Control strategy (current)

- FSM: argmax → keyframe lookup (8/28 classes mapped) → lerp with scalar POSE_ALPHA=0.18
- Stabilization patches: VotingWindow (N=3), CONFIDENCE_THRESHOLD=0.55, POSE_SWITCH_VOTES=4
- Problem: double discretization -- argmax discards 27 probability values, keyframe lookup discards intra-class variability
- Problem: frame-wise -- no temporal context, susceptible to single-frame noise

### Inherited baseline (Nadia Leal Reyes, CIIEC)

- 3 classes, DexYCB + lab images, static hardcoded adjacency, no temporal modeling, no control strategy
- Objective: offline know-how capture for assembly tasks
- Current work covers that objective and extends to real-time teleoperation control

**Status**: Implemented and validated (ablation complete)

---

## Entry 1 -- 2026-03-31: Multiframe CAM + ED-TCN + Postural Control

**Context**:

The system behaves erratically in deploy. The robot makes abrupt and incoherent movements.

The root cause is structural: the system is frame-wise and discrete end-to-end. Each frame is processed independently, and the FSM controller converts each prediction into a discrete canonical pose. This architecture directly amplifies any prediction fluctuation into the actuator.

Two sources of fluctuation were identified:

**Source 1 -- Occlusion**: when the user holds an object, overlaps fingers, or rotates the hand, MediaPipe produces geometrically incoherent landmarks. The model receives an observation that corresponds to no real grasp. Result: an isolated prediction completely different from the surrounding frames -- e.g., Tip Pinch in the middle of a Power Disk sequence. With a discrete FSM, that single frame sends an immediate command to the robot.

**Source 2 -- Grasp transitions**: when the user changes grasp, there is an intermediate state with no valid class. The frame-wise model is forced to predict something at every frame. Result: up to 5 different classes in 5 consecutive frames while the model searches for a stable class. With a discrete FSM, each of those predictions sends a separate command. The effect on the actuator is catastrophic.

Current patches (VotingWindow, CONFIDENCE_THRESHOLD, POSE_SWITCH_VOTES) attenuate symptoms but do not change the frame-wise discrete nature of the system. They delay or suppress outputs -- they are still discrete.

Additionally, even if the model predicted perfectly, the FSM controller would still be deficient. argmax → keyframe lookup is double discretization: the model computes 28 probabilities and the controller uses exactly one. The residual uncertainty of the model -- which in abl04 is coherent with Feix geometry -- never reaches the robot.

**Decision**:

Three coordinated architectural changes that attack the same structural defect at different levels of the pipeline.

### 1. Multiframe CAM

Replace the current static CAM [21, 21] with a temporally-aware version that conditions the adjacency matrix on a window of T consecutive frames.

Motivation: a frame with incoherent landmarks -- occlusion -- is incompatible with the joint trajectory of the preceding frames. A Multiframe CAM can suppress it implicitly because it learned which joint configurations are plausible in temporal context. This acts before the prediction.

### 2. ED-TCN (Encoder-Decoder Temporal Convolutional Network)

Add a temporal stage after the per-frame GCN embeddings:

```
T frames → [GCN per frame] → embeddings [T, 128] → ED-TCN → per-frame distribution [T, 28]
```

ED-TCN uses 1D convolutions with encoder temporal pooling and decoder upsampling. A sequence of 5 different classes in 5 consecutive frames -- whether from occlusion or transition -- is temporally incoherent. ED-TCN smooths it by construction: the architecture cannot produce abrupt jumps because every frame is connected to its temporal context.

Critically, ED-TCN does not need labeled inter-class transition data. HOGraspNet has real intra-class temporality (ordered frames within each trial). ED-TCN trains on those sequences, learns what is temporally normal within each class, and generalizes to transitions in deploy. Betthauser et al. (2020) demonstrate this exactly for EMG: transient states were never labeled for training -- smooth transition handling emerged as learned behavior. The problem is isomorphic.

This acts inside the model and replaces VotingWindow as the stabilization mechanism -- but structurally, not as a post-hoc patch.

### 3. Postural Control (PC) with top-2 JAT interpolation

Replace FSM with Postural Control at deploy time. Justified by Segil et al. (2014), who showed PC outperforms FSM in physical ADL tasks with myoelectric prostheses (SD: +8.6%, p<0.05).

Formulation (Joint Angle Transform, Segil et al.):

```
qpos = (p_k1 * C_k1 + p_k2 * C_k2) / (p_k1 + p_k2)
```

Where:
- k1, k2: top-2 predicted classes
- p_k1, p_k2: their probabilities
- C_k1, C_k2: canonical Shadow Hand qpos for each class (from shadow_hand_canonical_v5_grasp.yaml)

Any residual error that passes the two previous filters does not produce a discrete switch -- it smoothly shifts the interpolated coordinates. This is safe because the model's confusions are coherent with Feix geometry: abl04 does not confuse Power Disk with Tip Pinch, it confuses it with Power Sphere and Large Diameter. Verified against abl04 confusion matrix and Feix et al. (2016) human rater data. The top-2 is almost always a pair of geometrically adjacent classes -- interpolation between them produces a small coherent movement, not a catastrophic jump.

The only cases where PC would receive a geometrically distant pair are cases of severe occlusion or active transition -- exactly the cases that Multiframe CAM and ED-TCN already filtered.

Additional advantage: PC is robust to network latency. An irregular or delayed frame shifts the interpolation slightly rather than triggering a hard class switch. This is critical for the target remote teleoperation scenario (operator in Mexico, Shadow Hand in Japan).

**The complete system**: error is not eliminated at one point -- it degrades gracefully across the pipeline. What reaches the robot is always a smooth interpolation between adjacent classes, never a catastrophic jump between Feix extremes.

**Alternatives considered**:

- CAM-GAT (camgat01): dynamic attention on top of CAM. Tested -- does not outperform plain CAM with same features (70.46% vs 71.07%). Discarded as primary direction.
- Larger model (abl09): 4.7x parameters, same accuracy as abl04. Capacity is not the bottleneck. Discarded.
- Explicit transition class: would require labeled inter-class transition data. HOGraspNet has intra-class temporality only -- no continuous inter-class sequences. Labeling transitions is ambiguous (how many transition types exist?). Discarded. ED-TCN handles this implicitly.
- HSMM for transition modeling: considered as alternative to ED-TCN for handling transient states. ED-TCN is preferred because it operates inside the model end-to-end and does not require explicit state duration modeling.
- VotingWindow as permanent solution: external patch, does not generalize to streaming inference with ED-TCN. Replaced by temporal modeling inside the model.

**Expected impact**:

- Elimination of catastrophic class jumps (e.g., Power Disk to Tip Pinch) caused by single noisy frames
- Smooth handling of grasp transition states without labeled transition data
- More principled use of model uncertainty at deploy time
- Robust behavior under network latency in remote teleoperation

**References**:

- Betthauser et al. (2020), IEEE TBME 67(6): ED-TCN for stable EMG sequence prediction, superior performance during inter-class transitions without labeled transition data
- Segil et al. (2014): comparative study FSM vs PC for myoelectric prosthetic controllers
- Lea et al. (2016, 2017): ED-TCN original architecture for action segmentation
- Feix et al. (2016): GRASP taxonomy, human rater confusion data (validates geometric adjacency of confusions)
- abl04: best current model, confusion matrix confirms Feix-adjacent confusions
- camgat01: CAM-GAT experiment, ruled out as primary direction

**Status**: Proposed

---

## Entry 3 -- 2026-04-01: Training objective redefined -- from classification to postural positioning

**Origin of this decision**:

While ordering the implementation of Entry 2 (Multiframe CAM + ED-TCN + PC), the question arose: since PC requires a probability distribution as a continuous coordinate in postural space, should the model be trained to produce exactly that -- rather than a discrete classification? That question exposed a fundamental incoherence in the original plan: proposing PC as the control strategy while keeping NLLLoss as the training objective. This entry documents the resolution.

**Context**:

The current model (abl04) is trained with NLLLoss against one-hot labels. NLLLoss computes:

```
loss = -log(p[correct_class])
```

Only the correct class receives a gradient signal. The other 27 classes are invisible to the optimizer. This pushes the model to produce overconfident, near-degenerate distributions -- all mass on one class, near-zero everywhere else.

This is appropriate for a standalone classifier. It is incoherent for a system whose control strategy is Postural Control (Segil & Weir 2014), which uses the full probability distribution as a continuous coordinate in postural space and applies JAT interpolation:

```
qpos = (p_k1 * C_k1 + p_k2 * C_k2) / (p_k1 + p_k2)
```

A perfectly trained classifier under NLLLoss would produce the worst possible input for PC: p_k1 ≈ 1.0, p_k2 ≈ 0.0, no interpolation signal. The model is optimized for an objective that is directly opposed to what the controller needs.

The root issue is not calibration -- it is objective mismatch. The model is not being asked to classify; it is being asked to position an observation in the postural space defined by the 28 Feix classes as anchors. Classification and positioning are different problems requiring different training objectives.

**Decision**:

### 1. Redefine the training objective

The model is no longer a classifier. It is a postural positioner: given 21 XYZ landmarks, produce a probability distribution over 28 Feix classes that reflects the geometric position of the hand in postural space -- where each class is an anchor and the distribution encodes proximity to each anchor.

The architecture of Entry 2 does not change. The GCN still produces [28] via softmax. ED-TCN still produces [T, 28]. PC still receives [28] and applies JAT. What changes is what the model learns to produce within that same architecture.

The only code change required: `gcn_network.py` line 112, `F.log_softmax` → `F.softmax`. NLLLoss expects log-probabilities; cross-entropy between distributions expects raw probabilities. All three model variants (GCN_CAM_8_8_16_16_32, GCN_CAMGAT, GCN_CAM_32_32_64_64_128) share the same output layer -- the change is trivially compatible with the full ablation plan.

### 2. Construct the postural space from human grasping data

The postural space is constructed by PCA over joint angles from HOGraspNet -- the same methodology used by Segil & Weir (2014), who built their PC domain via PCA of human grasping data following Santello et al.

This has already been done in the project as `taxonomy_v1`:

- Input: 20 DOFs per frame (15 flexion angles + 5 abduction angles) from HOGraspNet
- PCA run until 85% variance captured
- Result: 9 principal components explaining 87.86% of variance

These 9 PCs define the postural space. Each of the 28 Feix classes has a centroid in this space (the mean of all samples belonging to that class projected into the 9-PC basis). Centroid distances between all 378 class pairs are already computed in `taxonomy_v1/results/collapse_candidates.json`.

The postural space is not constructed from raw XYZ (63-dimensional, noisy, sensor-dependent) but from joint angles (20-dimensional, intrinsic, computed from XYZ via FK). This is more principled: it captures the actual degrees of freedom of the hand and is invariant to global hand position and orientation.

### 3. Construct soft labels from distances in postural space

For each training sample, compute its distance to the centroid of each of the 28 Feix classes in the 9-PC postural space. Convert distances to a probability distribution via softmax over negative distances:

```
q_k = exp(-d(sample, centroid_k) / tau) / sum_j exp(-d(sample, centroid_j) / tau)
```

where tau is a temperature parameter controlling the sharpness of the distribution. A sample geometrically close to Power Disk and moderately close to Power Sphere produces a soft label with mass on both -- reflecting its actual position in postural space. This is the SORD construction from Diaz & Marathe (CVPR 2019), generalized from ordinal to metric distances in the learned postural space.

The centroids in postural space are the anchors. The soft label is a continuous coordinate -- the distribution encodes which anchors the sample is closest to and by how much.

### 4. Replace NLLLoss with cross-entropy between distributions

With soft labels q, the correct loss is cross-entropy between distributions:

```
loss = -sum_k q_k * log(p_k)
```

This is equivalent to KL divergence up to a constant. Gradients now flow proportionally to all classes with mass in q -- the model learns the relational structure between classes, not just which class is correct. Hinton et al. (2015): soft targets provide "much more information per training case than hard targets" because "the relative probabilities of incorrect answers tell us a lot about how the model tends to generalize."

### 5. Effect on the robot

PC with a geometrically meaningful distribution produces continuously morphing robot poses. Instead of switching between 8 hardcoded canonical poses, the Shadow Hand continuously interpolates between whichever two Feix anchors the model assigns the most mass to -- and that mass reflects how geometrically close the observed hand is to each anchor in the postural space constructed from human data.

This requires that all 28 Feix classes have a defined Shadow Hand qpos anchor in `shadow_hand_canonical_v5_grasp.yaml`. Currently only 8/28 are defined. Completing the 28 canonical poses is a prerequisite for this step.

**Why this is coherent with Entry 2**:

Entry 2 proposes ED-TCN receiving embeddings [T, 128] and producing [T, 28], feeding PC with [28]. None of that changes. Entry 3 changes what the [28] output means and how the model is trained to produce it. The pipeline is identical -- the semantics of the output are redefined from "class probabilities" to "postural coordinates". This makes the system coherent end-to-end: the training objective, the model output, and the control strategy all operate in the same postural space.

**Implementation decisions**:

The 2x2 ablation is only possible because the four configs share an identical interface. This is a concrete implementation constraint, not just a design preference.

The model output is `softmax([28])` in all configs -- a vector of 28 floats that sums to 1. The "discretization" in the current system is not in the shape of the output: it is in what the model was trained to produce within that shape. NLLLoss pushes the model toward one-hot distributions (one class dominates, the rest collapse to near-zero). Soft labels push the model toward geometrically meaningful distributions (mass proportional to proximity to each anchor). PC receives the same tensor interface in both cases -- what changes is whether the numbers are meaningful as postural coordinates or not.

This means no architectural change is needed between configs. The 2x2 is enabled by two independent flags:

- **`use_log_softmax` flag in the model constructor** (`gcn_network.py`): controls whether the output is `F.log_softmax` (NLLLoss configs) or `F.softmax` (soft-labels configs). All three existing model variants (GCN_CAM_8_8_16_16_32, GCN_CAMGAT, GCN_CAM_32_32_64_64_128) share the same output layer -- the flag applies to all without further modification.

- **ED-TCN as an optional wrapper module**: the GCN produces per-frame embeddings `[T, 128]` from `fc1`. ED-TCN sits between those embeddings and the final `[T, 28]` output. Whether ED-TCN is present or not is controlled in the training notebook -- the GCN itself is unchanged.

The four configs of the 2x2:

```
                         No ED-TCN          ED-TCN
use_log_softmax=True  │  abl04 (done)   │  Config C
use_log_softmax=False │  Config B       │  Config D (full system)
```

Controller (FSM vs PC) is orthogonal to both flags and lives in `grasp-app`, not in the model.

**Implementation order**:

The components of Entries 2 and 3 must be implemented in a specific order to produce a journal-quality ablation table with clean attribution of each improvement.

1. **Construct postural space** -- compute 28 class centroids in 9-PC space from taxonomy_v1. Prerequisites: already have the PCA basis and all HOGraspNet joint angles.

2. **Complete 28 canonical poses** -- define Shadow Hand qpos for all 28 Feix classes in shadow_hand_canonical_v5_grasp.yaml. Required before PC can be validated end-to-end.

3. **Retrain with soft labels + KL divergence** -- change `F.log_softmax` → `F.softmax`, construct soft label CSV, retrain abl04-equivalent. This is the baseline for everything that follows.

4. **Postural Control in deploy** -- replace FSM with JAT interpolation using the new model output. Validate that the output is geometrically meaningful as a postural coordinate.

5. **Multiframe CAM** -- temporal conditioning of the adjacency matrix, trained on the soft-labels model.

6. **ED-TCN** -- temporal stage on top of per-frame GCN embeddings, trained on the soft-labels model.

7. **Retrospective ablation** -- run Multiframe CAM and ED-TCN on the original NLLLoss model. This produces the comparison table needed for the journal: the improvement from soft labels is isolated from the improvement from temporal modeling.

Steps 5 and 6 are run on the soft-labels model first because that is the system being proposed. The retrospective ablation in step 7 is for attribution, not for the primary system.

Note: the concrete implementation plan for ED-TCN (step 6) has not been designed yet. This includes: how HOGraspNet sequences are windowed for training, the ED-TCN layer configuration, how per-frame GCN embeddings are extracted and fed as input, and how inference handles variable-length or streaming sequences in deploy. A dedicated entry should be written before implementation begins.

**Alternatives considered**:

- Temperature scaling (Guo et al., ICML 2017): post-hoc calibration of a classifier trained with NLLLoss. Rejected because the problem is not miscalibration of a classifier -- it is using the wrong objective from the start. Temperature scaling would make the output less overconfident but would not teach the model to reflect geometric proximity between classes.
- Keeping NLLLoss and using softmax output directly for PC: rejected. A classifier optimized for discrete decisions produces outputs that are not geometrically meaningful as postural coordinates. Not defensible to a reviewer.
- Constructing centroids in raw XYZ space (63-dim): rejected. XYZ is noisy, sensor-dependent, and encodes global position. The 9-PC postural space is intrinsic, computed from joint angles, and validated by prior work (Santello et al., Segil & Weir 2014) as the correct space for characterizing human hand postures.

**Expected impact**:

- Model output becomes a geometrically meaningful coordinate in postural space
- PC interpolation reflects genuine proximity between classes, not classifier confidence
- Robot receives continuously morphing poses instead of discrete class switches
- System is coherent end-to-end: training objective, model output, and control strategy operate in the same space
- Clean ablation table for journal: soft labels, Multiframe CAM, and ED-TCN are isolated contributions

**References**:

- Segil & Weir (2014), IEEE Trans Neural Syst Rehabil Eng 22(2):249-257. doi:10.1109/TNSRE.2013.2260172: PC requires continuous coordinate in postural space constructed by PCA of human grasping joint angles (Santello et al.); defines both the methodology and what the model output must be
- Santello, Flanders & Soechting (1998): PCA of human hand postures -- 2 PCs explain >80% variance; foundational methodology for postural space construction
- Hinton, Vinyals & Dean (2015): soft targets provide more information per training case than hard targets; cross-entropy between distributions is the correct loss
- Muller, Kornblith & Hinton (NeurIPS 2019): training with soft label distributions improves generalization and calibration vs. one-hot cross-entropy
- Diaz & Marathe (CVPR 2019): SORD -- constructing soft labels as softmax over negative inter-class distances; trained with categorical cross-entropy
- Dogan et al. (ECCV 2020): soft labels where p_k = f(similarity(class_t, class_k)) in embedding space
- Han et al. (PETRA 2019): precedent in prosthetics domain -- probability distribution over grasp types as model output for prosthetic hand control
- taxonomy_v1: project experiment -- PCA over 20 DOFs from HOGraspNet, 9 PCs, 87.86% variance; centroid distances for all 378 class pairs already computed

**Status**: Proposed

---

## Entry 5 -- 2026-04-01: Angle selection for postural space construction -- instrument-driven from Wu et al. (ISB 2005)

**Context**:

Entry 3 states that the postural space is built from joint angles. The question was: which angles, and why? The answer has the same logic as any motion capture study: measure everything your instrument allows. Jarque-Bou et al. (2019) measured 17 angles with a CyberGlove -- not because those 17 are biomechanically special, but because that is what their instrument could record given its sensor placement. Our instrument is a 21-keypoint XYZ skeleton. Our recorded angles are whatever that instrument can compute.

The biomechanical source for which angles anatomically exist is Wu et al. (2005), the ISB standard for joint coordinate systems of the hand and wrist. The validation that computing these angles from pose estimator keypoints is valid comes from Gionfrida et al. (2022). Jarque-Bou provides supporting evidence that these angles, when used for PCA, capture the relevant postural variation in grasping.

**Decision**:

### 1. Angles that anatomically exist (Wu et al. 2005, ISB)

Wu et al. define the joint coordinate system (JCS) for each hand joint (section 4.4.1). For all finger joints -- interphalangeal, metacarpophalangeal, and carpometacarpal:

- **e1** (Z-axis of proximal segment): rotation α = **flexion/extension** (flexion positive)
- **e2** (floating axis): at MCP joints, this axis corresponds to **abduction/adduction**
- **e3** (Y-axis of distal segment): rotation γ = pronation/supination (negligible in fingers)

The metacarpal and phalanx coordinate systems (sections 4.3.4, 4.3.5) define Y_m as the longitudinal bone axis, Z_m as perpendicular in the sagittal plane. Flexion is the angle between consecutive Y_m axes. Abduction at MCP is rotation in the plane perpendicular to the flexion axis.

### 2. Angles computable from 21 XYZ keypoints

The 21-keypoint model provides: WRIST (0), thumb chain (1-4: CMC, MCP, IP, TIP), and four finger chains each with MCP, PIP, DIP, TIP (nodes 5-20). The strategy is to compute all angles that Wu defines and that our keypoints support.

**Flexion angles** -- computed as arccos between consecutive bone vectors (Gionfrida et al. 2022, Fig 7: α = arccos(A⃗·B⃗/|A⃗||B⃗|)):

| Angle | Bone vectors | In system |
|-------|-------------|-----------|
| CMC1_F | WRIST→CMC1 vs CMC1→MCP1 | Yes (add_joint_angles) |
| MCP1_F | CMC1→MCP1 vs MCP1→IP1 | Yes |
| IP1_F | MCP1→IP1 vs IP1→TIP1 | Yes |
| MCP2_F | WRIST→MCP2 vs MCP2→PIP2 | Yes |
| PIP2_F | MCP2→PIP2 vs PIP2→DIP2 | Yes |
| MCP3_F | WRIST→MCP3 vs MCP3→PIP3 | Yes |
| PIP3_F | MCP3→PIP3 vs PIP3→DIP3 | Yes |
| MCP4_F | WRIST→MCP4 vs MCP4→PIP4 | Yes |
| PIP4_F | MCP4→PIP4 vs PIP4→DIP4 | Yes |
| MCP5_F | WRIST→MCP5 vs MCP5→PIP5 | Yes |
| PIP5_F | MCP5→PIP5 vs PIP5→DIP5 | Yes |

Note: DIP angles (DIP2_F--DIP5_F) are geometrically computable but anatomically coupled with PIP (Santello 1998). They add no independent postural information and are excluded.

**Abduction angles** -- computed as angle between adjacent finger proximal phalanx vectors (Gionfrida et al. 2022, Fig 5):

| Angle | Definition | In system |
|-------|-----------|-----------|
| CMC1_A | Thumb out-of-plane abduction at CMC | Yes (add_cmc_angle) |
| MCP1-2_A | Spread between thumb and index (MCP1→IP1 vs MCP2→PIP2) | **No** |
| MCP2-3_A | Spread between index and middle (MCP2→PIP2 vs MCP3→PIP3) | **No** |
| MCP3-4_A | Spread between middle and ring (MCP3→PIP3 vs MCP4→PIP4) | **No** |
| MCP4-5_A | Spread between ring and little (MCP4→PIP4 vs MCP5→PIP5) | **No** |

Note: MCP2-3_A (index-middle) was absent in Jarque-Bou because their CyberGlove lacked a sensor there. It is computable from XYZ and should be included.

**Not computable from 21 XYZ keypoints:**

| Angle | Reason |
|-------|--------|
| WRIST_F, WRIST_A | Requires forearm reference -- not in the 21-keypoint model |
| CMC5_F (palmar arch) | Requires CMC5 keypoint -- not in the 21-keypoint model |

**Total computable: 16 angles** (11 flexion + 5 abduction). Currently in system: 12 (11 flexion + CMC1_A). Missing: 4 inter-finger abduction angles.

### 3. Validation

Gionfrida et al. (2022) validate computing finger joint angles from OpenPose 21-keypoint output against gold-standard optical motion capture (Qualisys, <0.4mm error), for MCP flexion, PIP flexion, and inter-finger abduction. Results: RMSE below 9° for abduction, mean difference 6.82° for MCP flexion -- acceptable for biomechanical assessment. Our system uses 3D keypoints (not 2D as in Gionfrida), which removes projection errors; the Gionfrida bound is conservative for our case.

### 4. Standardization: z-score instead of mean subtraction

The postural space is built using z-score standardization (subtract mean, divide by std per angle) rather than Santello's mean-subtraction only. This is equivalent to running PCA over the correlation matrix instead of the covariance matrix.

**Three independent reasons:**

**Reason 1 -- Comparability across 99 subjects (Jarque-Bou)**: HOGraspNet has 99 subjects with varying hand sizes and ROM. Without z-score, subjects with larger ROM dominate the PCA. Jarque-Bou explicitly adopts z-score for the same reason with 77 subjects: *"to ensure that the contributions from all the joints are equally weighted inside the extracted synergies -- allows joints with lower range of motion to be adequately represented."*

**Reason 2 -- Comparability across joints with different ROM**: MCP_F has ROM ~80°, MCP3-4_A has ROM ~15°. Without z-score, MCP_F dominates the first PCs by scale alone. With z-score, each joint contributes proportionally to its actual postural variation, not its absolute magnitude. This is the same operation as Reason 1 applied across joints instead of subjects.

**Reason 3 -- Discriminability for Feix-class morphing**: The postural space serves as the basis for JAT interpolation between 28 Feix class anchors. For morphing to be correct, classes that Feix considers distinct must have distinct coordinates. Without z-score, classes that differ primarily in low-ROM angles (abduction, thumb opposition) collapse to nearby coordinates -- the morphing between their anchors degenerates. With z-score, the PCA captures co-activation patterns regardless of absolute scale. Supporting evidence: Jarque-Bou obtains 12 synergies with distinct compositions using z-score, while prior studies without standardization obtain 4-7 -- more distinct synergies implies a more discriminable space.

The third reason is our own contribution. The first two have direct citation. The benefit for Feix-class discriminability is demonstrated empirically by comparing centroid separation with and without z-score.

Note: Santello did not z-score because his goal was to describe absolute postural variation in 5 subjects performing imagined grasps -- a controlled setting with homogeneous subjects. Our goals are different: 99 subjects, 28 classes to discriminate, and a space that must support meaningful interpolation.

### 5. Immediate next step

Before taxonomy_v1 is reviewed or the postural space is built: verify exactly what tograph.py currently computes and identify the gap precisely. The expected result is that 11 flexion angles and CMC1_A are present, and 4 abduction angles (MCP1-2_A, MCP2-3_A, MCP3-4_A, MCP4-5_A) are missing. This comparison is needed to confirm what needs to be implemented.

**Alternatives considered**:

- Using Jarque-Bou's 17 angles as the target set: rejected. Those 17 are CyberGlove-instrument-driven, not universally optimal. Two of them (WRIST, palmar arch) are inaccessible from our input regardless; one (MCP2-3_A) is accessible but was missed due to sensor constraints. Our set is instrument-driven from our own input.
- Raw XYZ as postural space input: rejected (Entry 3). XYZ encodes global position, not intrinsic posture.
- AHG features for postural space: rejected. AHG was designed for grasp classification, not postural space construction. Distances between fingertips are not equivalent to anatomical joint angles.

**Expected impact**:

- Postural space built from 16 angles: the maximum computable from 21 XYZ keypoints following ISB definitions.
- 4 inter-finger abduction angles to implement in tograph.py before taxonomy_v1 review.
- The absence of WRIST and palmar arch should be acknowledged in the thesis as an instrument limitation of the 21-keypoint skeleton, not a methodological choice.

**References**:

- Wu et al. (2005), J Biomech 38:981-992: ISB standard JCS definitions for hand and wrist joints; source for which angles anatomically exist and how to define them
- Gionfrida et al. (2022), PLoS ONE 17(11):e0276799: validation of finger kinematics from OpenPose 21-keypoint output against optical motion capture; provides explicit arccos formula and validates abduction and MCP flexion computation
- Jarque-Bou et al. (2019), J NeuroEngineering Rehabil 16:63: 77 subjects, 20 grasps; z-score justification for comparability across subjects; 12 synergies with distinct compositions; confirms angles capture relevant postural variation in grasping
- Gracia-Ibáñez et al. (2017), Comput Methods Biomech Biomed Eng 20:587-597: calibration protocol [25] in Jarque-Bou -- the CyberGlove equivalent of our Gionfrida; confirms instrument-driven angle selection logic
- Santello, Flanders & Soechting (1998): DIP excluded as coupled with PIP; mean-subtraction only (no z-score) -- appropriate for their controlled 5-subject setting, insufficient for ours
- Entry 3: postural space construction plan

**Status**: Proposed

---

## Entry 4 -- 2026-04-01: Two-space architecture -- decoupling observation from control

**Context**:

Entry 3 defines the human postural space (9 PCs from HOGraspNet joint angles) and trains the model to produce coordinates in that space. Entry 1 defines PC as the control strategy using JAT interpolation between Shadow Hand canonical poses. Neither entry makes explicit that these are two distinct spaces with a specific relationship between them -- Entry 3 speaks of "postural space" as if it were one.

In practice, the human observation space and the robot control space are structurally different:

- The human hand has its own DOFs, range of motion, and postural variation patterns
- The Shadow Hand (or any target robot) has a different kinematic structure
- There is no reason to expect geometric equivalence between the two spaces -- the same grasp intent does not correspond to identical joint angle vectors in both hands

If the system assumes a single shared space, any deviation between human and robot kinematics produces a systematic error at the control level. The system needs an explicit formulation of how the two spaces relate.

**Decision**:

### 1. Two distinct spaces

The architecture operates in two separate domains:

**Human observation space**: the 9-PC postural space constructed from HOGraspNet (Entry 3). A human hand configuration x_h projects to a latent coordinate z_h ∈ R^9. Each of the 28 Feix classes has a centroid c_i^h in this space. The model produces a distribution over these centroids -- a continuous coordinate in the observation space.

**Robot control space**: the domain in which robot commands are synthesized. Each of the 28 Feix classes has a corresponding anchor c_i^r in this space. The robot adapter receives the same distribution from the model and applies JAT morphing between robot anchors:

```
u_r = (p_k1 * c_k1^r + p_k2 * c_k2^r) / (p_k1 + p_k2)
```

### 2. Structural correspondence, not geometric equivalence

The control hypothesis is not that human and robot joints are numerically equivalent. It is that the local structure of the observation space is preserved in the control space:

```
z_h between c_a^h and c_b^h  =>  u_r between c_a^r and c_b^r
```

If the observed hand is geometrically between Power Disk and Power Sphere in the human postural space, the robot should morph between its own Power Disk and Power Sphere anchors -- regardless of whether the joint angle vectors are numerically similar. What is preserved is the neighborhood relationship, not the coordinates.

### 3. Two options for the robot control space

**Option A -- Joint space anchors**: each Feix class is represented by a canonical robot joint configuration q_i^r ∈ R^Nr. Morphing is linear interpolation in joint space:

```
q_r = alpha * q_a^r + (1 - alpha) * q_b^r
```

This is the current approach (shadow_hand_canonical_v5_grasp.yaml) and is valid. Morphing occurs in raw joint space, which is higher-dimensional and less structured.

**Option B -- Robot synergy space**: the robot also has a low-dimensional postural space constructed from its own motion data (or from the canonical poses themselves via PCA):

```
q_r ≈ mu_r + U_r * s_r,   s_r ∈ R^m
```

Each Feix class maps to an anchor s_i^r in synergy space. Morphing occurs in synergy space and joints are reconstructed afterward. This is more coherent with Segil & Weir (2014), where morphing occurs in a low-dimensional postural domain.

**Current decision**: Option A for the initial implementation. The 28 canonical joint configurations are already being defined (shadow_hand_canonical_v5_grasp.yaml).

**Both options produce valid continuous morphing.** The morphing mechanism is a weighted sum of anchors: `q = Σ w_i * anchor_i`, where w_i are the class probabilities from the model. Since anchors are real grasps from Dexonomy, any weighted combination produces a coherent posture -- the correlations between joints are preserved implicitly in the anchors themselves. The distinction between Option A and Option B is not functional but architectural:

- Option A: anchors live in raw joint space (24D for Shadow Hand). Weighted sum operates in 24D.
- Option B: anchors live in a low-dimensional synergy space (2-5 PCs). Weighted sum operates in that space, then JAT decodes to joint space.

**Three reasons to prefer Option B for a formal publication**:
1. Continuity with Segil & Weir (2014) -- same methodology, directly citable extension
2. Consistency across robots -- regardless of joint count (18, 24, 30 DOF), PCA over grasping data is expected to yield a comparably low-dimensional space (hypothesis, following Santello et al. 1998 for the human hand). The navigation space remains dimensionally consistent across robots.
3. Semantic packaging -- the space captures co-activation patterns of grasping, not independent joint values. Morphing in that space has a biomechanical interpretation.

Option B can be adopted as a refinement once Option A is validated end-to-end.

### 4. Semantic communication between spaces

The message between the human observation space and the robot control space is not a geometric coordinate -- it is a distribution over the 28 Feix classes.

The model says: "I am 70% Power Disk, 20% Power Sphere, 10% Large Diameter."
The robot adapter receives that and responds: "locate me in my own space at those proportions -- 70% toward my Power Disk anchor, 20% toward my Power Sphere anchor, 10% toward my Large Diameter anchor."

The communication is semantic, not geometric. The Feix taxonomy is the shared language. The two spaces do not need the same dimensionality or the same geometry -- both speak Feix. This is what makes structural correspondence sufficient and geometric equivalence unnecessary.

### 5. Robot-agnostic design

The two spaces do not need to share the same PCs, the same dimensionality, or the same kinematic structure. What makes the system work is not that the spaces are equivalent -- it is that both are anchored to the same 28 Feix classes. The Feix taxonomy is the bridge.

The argument: certain synergies in the human postural space lead to Power Disk. A continuous path in that space between Power Disk and Power Sphere corresponds to a gradual transition between those grasps. In the robot control space, the same transition is recreated by interpolating between the robot's own Power Disk and Power Sphere canonical poses -- because those poses are also anchored to the same Feix taxonomy, each defined as what that robot's hand looks like for that grasp.

The PCs of the human space and the joints of the robot space do not need to represent the same thing. The coherence of the interpolation comes from the canonical poses being well-defined for the robot -- not from any numerical correspondence between spaces.

This makes the system genuinely robot-agnostic. A different robot (Shadow Hand, Allegro, any future effector) only requires a new set of 28 canonical configurations defined for its own kinematics. The model, the human postural space, and the soft label construction remain completely unchanged. This is a concrete property of the architecture, not a design aspiration.

**Relationship to Segil & Weir (2014)**:

Segil's JAT is: `q = PC1_vec * s1 + PC2_vec * s2` -- a linear combination of joint vectors weighted by postural coordinates. Our JAT is: `q = C_k1 * p_k1 + C_k2 * p_k2` -- a linear combination of canonical pose vectors weighted by the model's distribution. These are mathematically the same operation. The only difference is that Segil uses 2 basis vectors (the PCs of Santello et al.) and we use 28 (the Feix canonical poses). Our system is Segil's JAT generalized from 2 anchors to 28.

The reason Segil does not need two separate spaces is that his prosthesis replaces the user's own hand -- a single PCA space built from human data serves both observation and control. In our teleoperation scenario, the observed hand and the controlled hand are kinematically distinct. The two-space architecture is a direct consequence of that scenario, not an arbitrary design choice.

**Alternatives considered**:

- Single shared postural space (human and robot in same domain): requires geometric equivalence between human and robot kinematics. Not valid for hands with different DOF counts, ranges, or kinematic structure.
- Direct joint mapping (no postural space): maps human joint angles directly to robot joint angles via a learned or geometric transform. Couples the model to a specific robot, breaks the robot-agnostic property.

**Expected impact**:

- Explicit contract between the model and any robot adapter: model produces [28] postural coordinates, adapter defines 28 anchors, morphing preserves neighborhood structure
- Any new robot requires only a new set of 28 anchor configurations
- Morphing is geometrically meaningful: interpolation occurs between Feix-adjacent anchors, not arbitrary class pairs

**References**:

- Entry 1: PC formulation, JAT, argument for top-2 safety via Feix-adjacent confusions
- Entry 3: human observation space construction (taxonomy_v1 PCA, 9 PCs, centroids)
- Segil & Weir (2014): PC in low-dimensional postural domain; target poses as anchors; morphing between two nearest anchors

**Status**: Proposed

---

## Entry 6 -- 2026-04-02: Two-space architecture -- final argument for asymmetric design (PCA-joints)

**Context**: After extensive analysis of the two-space architecture, the justification for using PCA on the human side and joints directly on the robot side was unclear. Previous entries argued for PCA on both sides (following Segil), but the technical reason for PCA on the robot side was not convincing. This entry documents the final, technically honest argument.

**Decision**:

### The asymmetric architecture is correct: human side PCA, robot side joints directly.

The two sides are asymmetric by necessity, not by simplification. Each design choice responds to the nature of its data.

### Human observation space -- PCA + z-score (technically justified)

The human operator is variable. HOGraspNet captures 99 subjects with different morphologies, hand sizes, and execution styles performing the same 28 grasps. The postural space must generalize over that variability -- it is built to answer "where in the Feix map is this operator's pose, regardless of who they are."

PCA over 1.5M frames captures the structure of variance that emerges from that diversity. The z-score normalization is required because human joints have heterogeneous ROM (FFJ3 ~90 deg, FFJ4 ~20 deg) -- without it, high-ROM joints dominate the distance metric and the postural map reflects movement amplitude rather than postural similarity. After z-score, distances reflect how unusual a configuration is relative to the data distribution, which is the correct basis for soft label construction.

The soft labels derived from distances in this space are what make the model a postural locator rather than a hard classifier. The model learns to output w = [w_0, ..., w_27] as a continuous coordinate in the Feix semantic space -- not a binary decision.

### Robot control space -- joints directly (technically justified)

The robot is invariant. A single morphology, a single physical reality. There is no inter-subject variability to model. The 28 Feix anchors are specific, reproducible configurations of that robot's joints.

No metric is computed in the robot control space. The only operation is:

```
q = sum_i( w_i * anchor_i )
```

Since no distances are computed, the scale problem (ROM heterogeneity) does not exist. The co-activation patterns are preserved in the anchors themselves -- each anchor is a real Dexonomy grasp where all joints are already in their natural co-activated state. Any weighted combination of real poses produces a coherent posture.

The variance in Dexonomy (thousands of configurations per class) is variance of robotic search, not variance of execution. It does not play the same role as inter-subject human variance. Modeling it via PCA would add complexity without technical benefit.

### Why not PCA on both sides?

PCA on the robot side would be technically correct but adds nothing. The interpolation in PCA space between two anchors is mathematically equivalent to linear interpolation between the PCA-reconstructed versions of those anchors. The difference observed in practice (~0.1 rad at midpoint) comes entirely from reconstruction error introduced by using fewer than 22 PCs -- not from any co-activation property of the PCA mechanism. Confirmed experimentally: grasp-robot/experiments/interpolation_pca_vs_linear.png.

### Why not joints directly on both sides?

Joints directly on the human side, without z-score, produces a distorted postural map where high-ROM joints dominate distances. The soft labels would not reflect postural similarity but movement amplitude. This could be fixed with z-score alone (without PCA), but PCA additionally removes redundancy between correlated joints and connects the methodology to Santello et al. (1998) and Jarque-Bou et al. (2019/2021).

### The semantic bridge

The two spaces communicate only through w -- 28 numbers summing to 1. The human side outputs probabilities over Feix classes. The robot side receives those probabilities and applies them over its own geometry. The spaces are completely independent in construction, scale, and dimensionality. What makes them compatible is not numerical correspondence but shared anchoring to the same 28 Feix classes.

```
[XYZ + features] -> GNN -> w (semantic, 28D) -> sum_i(w_i * anchor_i) -> qpos (geometric, 24D)
```

The model is a postural locator. The robot is a postural executor. The Feix taxonomy is the language between them.

**Alternatives considered**:

- Both PCA: robot PCA technically unnecessary, adds implementation complexity with no functional gain.
- Both joints direct: valid if z-score is applied on human side. PCA additionally removes joint correlation redundancy and connects to established literature.
- Joints direct on human (no z-score): produces distorted soft labels -- high-ROM joints dominate distances.

**Expected impact**:

- Cleaner implementation: robot adapter requires only 28 anchor poses, no PCA fitting, no dataset needed
- Stronger journal argument: each design choice is justified by the nature of its data, not by symmetry with the other side
- Robot-agnostic property is maximally simple: new robot = new 28 anchors

**References**:

- Entry 3: human postural space construction (taxonomy_v1, z-score, KL divergence)
- Entry 4: two-space architecture, semantic communication via Feix
- Santello et al. (1998): z-score normalization before PCA over hand joint angles
- Jarque-Bou et al. (2019, 2021): z-score ("unit variance, mean=0, SD=1") for CyberGlove angles
- Segil & Weir (2014): JAT as weighted sum of anchors in postural space
- Experiment: interpolation_pca_vs_linear.png -- PCA robot interpolation vs linear, difference is reconstruction error not co-activation

**Status**: Proposed

---

## Entry 7 -- 2026-04-02: Empirical validation of human postural space construction -- joints+z-score vs PCA+z-score

**Context**: Entry 6 argued that the human observation space should use PCA+z-score over HOGraspNet joint angles. However, after extended analysis, the technical advantage of PCA over joints+z-score is not conclusive. The variability inter-sujeto is captured by the dataset (1.5M frames, 99 subjects), not by the PCA transformation itself. PCA additionally removes redundancy between correlated joints and reduces dimensionality, but whether this produces meaningfully better soft labels is an empirical question, not a theoretical one.

**Decision**: Implement and compare both approaches before committing to either.

- **Approach A**: joints directly + z-score. Centroids and distances computed in full joint space (20D normalized). Simpler, symmetric with robot side, harder for a reviewer to question.
- **Approach B**: PCA + z-score. Centroids and distances computed in PCA space (7-9D). Connected to Santello (1998), Jarque-Bou (2019/2021), Segil (2014). Removes joint correlation redundancy.

**Why both**: the theoretical discussion converges on "both should work similarly because the data is the same." The only way to know which produces better soft labels -- and whether the model learns a better postural locator with one vs the other -- is to train and compare.

**Expected outcomes**:
- If equivalent: use PCA for the journal argument (literature connection). Document equivalence as validation.
- If PCA better: confirmed technical advantage, use PCA.
- If joints better: surprising result, worth reporting. Use joints, document why PCA hurt (possibly overfitting to dominant variance directions).

**References**:
- Entry 6: full argument for asymmetric PCA-joints architecture
- Entry 3: training objective (KL divergence, soft labels from postural distances)

**Status**: Proposed

---

## Entry 8 -- 2026-04-02: Open hand anchor -- origin of the postural space

**Context**: The 28 Feix anchors define grasp configurations -- all of them closed or semi-closed. The postural space they span is biased toward closure. If the operator opens their hand, the model has nowhere to place it within that space and distributes weights arbitrarily among the 28 grasps. The space is incomplete.

The open hand anchor completes the postural space. With it, the model can locate any real hand configuration -- from fully open to any grasp -- and the weighted sum always produces something meaningful. In practice, the open hand is also the resting state: where the robot sits when no grasp is active, and the point from which most grasp trajectories naturally begin and end. But the technical reason to include it is completeness, not convention.

**Decision**: Add a 29th anchor -- open hand (palma abierta) -- as a full participant in the weighted sum, not as an external fallback.

```
q = w_open * anchor_open + sum_i( w_i * anchor_i )    # 29 anchors total
```

This is not a special case or a fallback. The open hand anchor behaves exactly like any other anchor -- the model assigns weight to it, the weighted sum includes it, the robot executes it. No special logic required.

The open hand anchor is the origin of the postural space -- every grasp trajectory starts and ends there:

```
w_open=1.0, w_target=0.0  →  fully open
w_open=0.5, w_target=0.5  →  mid-transition
w_open=0.0, w_target=1.0  →  full grasp
```

This makes the morphing trajectory physically meaningful: it matches the actual hand movement sequence (open → grasp → open) rather than interpolating between two closed configurations.

**Data source**: open hand frames are available from deploy experiments already recorded. These frames can be used as training data for the open hand class.

**Why not a fallback**: a fallback is external logic that intercepts when something fails. An anchor is part of the normal mechanism -- the model decides how much weight to give it continuously, same as any other class. This produces smooth transitions rather than discrete mode switches.

**Expected impact**:
- Natural resting state when model is uncertain or hand is in transition
- Smooth open-to-grasp trajectories captured by the weighted sum
- The model learns the full grasp cycle, not just the closed configurations

**Prerequisites**:
- Collect and label open hand frames from deploy experiments
- Define anchor_open as Shadow Hand fully open configuration (all joints at 0 or anatomical rest)
- Add open hand as class 28 (index) in the model and training pipeline

**Status**: Proposed

---

## Entry 9 -- 2026-04-02: Human hand DOF inventory -- resolved from Chen et al. (2013)

**Context**: Pendiente A -- before building the human postural space, we needed a citable, explicit inventory of the anatomical DOFs of the human hand. This entry documents the resolution of that question from Chen et al. (2013), "Constraint Study for a Hand Exoskeleton: Human Hand Kinematics and Dynamics," Journal of Robotics, Article ID 910961.

**Decision**: Chen et al. (2013) is the primary citation for the human hand DOF inventory. The primary sources within the paper are:

- **Figure 2 caption**: "The thumb is defined by 3 links and 4 degrees of freedom whereas index, middle, ring, and little fingers are defined by 4 links and 5 DoFs." → count total: 4 + 5×4 = 24 DOFs.
- **Table 8** (MDH thumb): lists exactly θ_TMC_ab/ad, θ_TMC_f/e, θ_MCP_f/e, θ_IP_f/e — 4 DOFs, no Thumb MCP abd.
- **Table 7** (MDH fingers 1–4): lists exactly θ_CMC_f/e, θ_MCP_ab/ad, θ_MCP_f/e, θ_PIP_f/e, θ_DIP_f/e — 5 DOFs per finger.
- **Sec 2**: "A single DoF (flexion/extension) characterizes the MCP and IP joints of the thumb as well as the PIP and DIP joints of the fingers." — confirms Thumb MCP has no abd DOF.
- **Table 6**: static ROM constraints (limits on θ values). NOT the DOF definition — a joint can have passive ROM without being an independent active DOF. Thumb MCP shows 5° abd/add in Table 6 but is not modeled as an independent DOF (absent from Table 8).

Wu et al. (2005) provides the measurement framework (JCS) but not the inventory. Chen provides both.

### Complete DOF inventory -- Chen et al. (2013) Tables 7+8 + Figure 2

**Kinematic model**: 19 links, 24 DOFs, Modified Denavit-Hartenberg (MDH) convention.

**Thumb (3 links, 4 DOFs):**
| Joint | DOF type | Flexion ROM | Extension ROM | Abd/Add ROM |
|-------|----------|-------------|---------------|-------------|
| TMC   | 2 DOF (flex + abd) | 50-90° | 15° | 45-60° |
| MCP   | 1 DOF (flex only) | 75-80° | 0° | 5° |
| IP    | 1 DOF (flex only) | 75-80° | 5-10° | 5° |

**Index finger (4 links, 5 DOFs):**
| Joint | DOF type | Flexion ROM | Extension ROM | Abd/Add ROM |
|-------|----------|-------------|---------------|-------------|
| CMC   | 1 DOF (flex, small) | 5° | 0° | 0° |
| MCP   | 2 DOF (flex + abd) | 90° | 30-40° | 60° |
| PIP   | 1 DOF (flex only) | 110° | 0° | 0° |
| DIP   | 1 DOF (flex only) | 80-90° | 5° | 0° |

**Middle finger (4 links, 5 DOFs):**
| Joint | DOF type | Flexion ROM | Extension ROM | Abd/Add ROM |
|-------|----------|-------------|---------------|-------------|
| CMC   | 1 DOF (flex, small) | 5° | 0° | 0° |
| MCP   | 2 DOF (flex + abd) | 90° | 30-40° | 45° |
| PIP   | 1 DOF (flex only) | 110° | 0° | 0° |
| DIP   | 1 DOF (flex only) | 80-90° | 5° | 0° |

**Ring finger (4 links, 5 DOFs):**
| Joint | DOF type | Flexion ROM | Extension ROM | Abd/Add ROM |
|-------|----------|-------------|---------------|-------------|
| CMC   | 1 DOF (flex, small) | 10° | 0° | 0° |
| MCP   | 2 DOF (flex + abd) | 90° | 30-40° | 45° |
| PIP   | 1 DOF (flex only) | 120° | 0° | 0° |
| DIP   | 1 DOF (flex only) | 80-90° | 5° | 0° |

**Little finger (4 links, 5 DOFs):**
| Joint | DOF type | Flexion ROM | Extension ROM | Abd/Add ROM |
|-------|----------|-------------|---------------|-------------|
| CMC   | 1 DOF (flex, small) | 15° | 0° | 0° |
| MCP   | 2 DOF (flex + abd) | 90° | 30-40° | 50° |
| PIP   | 1 DOF (flex only) | 135° | 0° | 0° |
| DIP   | 1 DOF (flex only) | 90° | 5° | 0° |

**Total: 4 + 5 + 5 + 5 + 5 = 24 DOFs**

### Key observations for postural space design

1. **Only MCP joints have meaningful abduction** -- PIP and DIP are pure flexion. Abduction DOFs: MCP of 4 fingers (4 DOF) + TMC of thumb (1 DOF) = 5 abduction DOFs total.

2. **CMC joints of 4 fingers have minimal ROM** (5-15° flexion only) -- commonly approximated as fixed in kinematic models. Chen's model includes them but their contribution to postural discrimination is small.

3. **DIP is coupled to PIP** via intrafinger constraints (Eq. 1 in paper): theta_DIP ≈ (2/3) * theta_PIP. This means DIP is not an independent DOF in practice -- it is determined by PIP.

4. **Thumb TMC is the most complex joint**: 2 DOF (flexion 50-90° + abduction 45-60°). Largest abduction range of any joint in the hand.

5. **MCP abduction range varies by finger**: Index 60°, Middle 45°, Ring 45°, Little 50°. Not uniform -- z-score normalization is essential before computing distances.

### Definición de los tipos de DOF

Cada DOF es un único eje de rotación entre dos segmentos óseos adyacentes. Las articulaciones de los dedos tienen a lo sumo 2 ejes de rotación (Chen 2013 compila esta caracterización anatómica estándar en Table 6):

- **Flexión** (flex): rotación en el eje que cierra el dedo hacia la palma. La tabla de Chen incluye columnas "Flexion" y "Extension" para describir el rango de movimiento (ROM) en cada dirección del eje -- pero ambas direcciones corresponden a **un solo DOF**. La extensión es simplemente el extremo negativo del mismo eje. Chen lo explicita: "A single DoF (flexion/extension) characterizes the MCP and IP joints of the thumb as well as the PIP and DIP joints of the fingers" (Section 2).

- **Abducción** (abd): rotación en el eje lateral, en el plano de la palma. Separa o junta los dedos entre sí. Solo presente en MCP de los 4 dedos y en TMC del pulgar.

No existe rotación axial del dedo sobre su propio eje longitudinal -- por eso el máximo es 2 DOF por articulación, no 3. La mano humana tiene en total **24 DOFs**: 5 articulaciones × 2 DOF (TMC + 4 MCPs) + 14 articulaciones × 1 DOF (CMCs, PIPs, DIPs, thumb MCP, thumb IP) = 10 + 14 = 24.

### Candidate set for postural space (DIP excluded -- coupled to PIP)

La lista de ángulos candidatos se deriva de Chen (2013) Table 6 aplicando tres criterios:

1. **Por cada segmento óseo, se toman los DOFs que presenta Table 6.** Cada articulación tiene a lo sumo flexión y abducción. Si el ROM de un DOF es 0° en ambas direcciones, ese eje no existe funcionalmente y se omite. Por ejemplo, PIP tiene 0° de abducción → solo se incluye PIP flex.

2. **Los DIP se eliminan porque no son DOFs independientes.** Chen (2013) Eq. 1 establece θ_DIP ≈ (2/3) θ_PIP -- el DIP está determinado por el PIP, no se mueve libremente. Incluirlo sería redundante.

3. **Los CMC de los 4 dedos se retienen por ahora.** Su ROM es pequeño (5-15°) y corresponden a la deformación del arco palmar, no al movimiento de los dedos. Su computabilidad desde 21 keypoints XYZ se decide en Pendiente B.

Resultado: la lista completa de ángulos posibles de la mano humana derivada de Chen (2013):

Removing only DIP of 4 fingers (coupled to PIP via Eq. 1, Chen 2013 -- not an independent DOF).
CMC of 4 fingers retained -- their computability from XYZ will be decided in Pendiente B.

| DOF | Joint | Type | ROM | Note |
|-----|-------|------|-----|------|
| 1 | Thumb TMC flex | flexion | 50-90° | |
| 2 | Thumb TMC abd | abduction | 45-60° | |
| 3 | Thumb MCP flex | flexion | 75-80° | |
| 4 | Thumb IP flex | flexion | 75-80° | |
| 5 | Index CMC flex | flexion | 5° | palmar arch deformation, not finger movement |
| 6 | Index MCP flex | flexion | 90° | |
| 7 | Index MCP abd | abduction | 60° | |
| 8 | Index PIP flex | flexion | 110° | |
| 9 | Middle CMC flex | flexion | 5° | palmar arch |
| 10 | Middle MCP flex | flexion | 90° | |
| 11 | Middle MCP abd | abduction | 45° | |
| 12 | Middle PIP flex | flexion | 110° | |
| 13 | Ring CMC flex | flexion | 10° | palmar arch |
| 14 | Ring MCP flex | flexion | 90° | |
| 15 | Ring MCP abd | abduction | 45° | |
| 16 | Ring PIP flex | flexion | 120° | |
| 17 | Little CMC flex | flexion | 15° | palmar arch |
| 18 | Little MCP flex | flexion | 90° | |
| 19 | Little MCP abd | abduction | 50° | |
| 20 | Little PIP flex | flexion | 135° | |

**20 DOFs** -- lista completa de ángulos posibles de la mano humana derivada de Chen (2013). DIP excluido (acoplado a PIP). CMC retenido pendiente de análisis de computabilidad (Pendiente B).

### Implication for Pendiente B

The question now is: which of these 16 DOFs are directly computable from 21 XYZ keypoints? Flexion angles are computable via 3-point angle. Abduction requires projection to the palmar plane to avoid flexion contamination (Pendiente B, subproblem C).

**References**:
- Chen et al. (2013): Figure 2 caption (DOF count), Table 8 (thumb θ_j), Table 7 (finger θ_j), Sec 2 (Thumb MCP = 1 DOF), Eq. 1 (DIP coupling), Table 6 (ROM constraints only)
- Jarque-Bou et al. (2019, 2021): 16 CyberGlove angles -- cross-validation of the reduced set
- Wu et al. (2005): JCS measurement framework -- how to measure, not what exists

**Status**: Resolved -- Pendiente A closed. DOF inventory and candidate set manually reviewed against Chen (2013) Table 6 by Y. Ramírez (2026-04-02).

---

## Entry 10 -- 2026-04-02: Two-stream architecture -- hand configuration vs wrist orientation

**Context**: In teleoperation, the operator controls both the hand configuration (which grasp) and the arm pose (where the hand is pointing). These are two orthogonal problems. The question was whether wrist orientation should be included in the model or treated separately.

**Decision**: Two independent streams.

**Stream 1 -- Hand configuration (this system):**
- Input: 21 XYZ keypoints → finger DOFs (Chen inventory)
- Model: GNN → w (28+1 Feix classes)
- Output: qpos of robot hand

**Stream 2 -- Wrist/arm orientation (separate controller):**
- Input: wrist orientation from MediaPipe Holistic or similar
- Controller: separate module, independent of GNN
- Output: pose of robot arm end-effector (rotation, position)

**Why separate:**
- Wrist orientation does not discriminate between Feix grasp classes -- Power Disk and Lateral Tripod can occur at any wrist angle
- HOGraspNet has no arm/wrist orientation data -- training Stream 1 on wrist features would add noise without signal
- The two problems are orthogonal: what the hand does vs where it points
- Separating them keeps each stream simple and independently trainable/replaceable

**Note on computability:** MediaPipe Hands gives wrist position but not orientation. MediaPipe Holistic gives full body pose including wrist orientation. Stream 2 requires Holistic or equivalent.

**Status**: Proposed

---

## Entry 11 -- 2026-04-02: Postural space as implicit grasp stabilization

**Context**: OAT (Chen et al., ICRA 2026) identifies a "grasp stabilization" problem: when hand configuration changes during grasping, wrist pose estimation drifts, causing unintended robot arm movement. They address this with an external bias term that freezes wrist estimation during large hand pose transitions. This is analogous to our VotingWindow -- a patch external to the model.

**Observation**: Operating in a postural space may implicitly mitigate this problem without external logic. With a hard classifier, transitions produce discrete class jumps that propagate as discontinuities to qpos. With a postural locator, the model outputs a continuous distribution w that shifts gradually during transitions -- the weighted sum absorbs the uncertainty and produces smooth intermediate qpos rather than discontinuous jumps.

**This is a possibility, not a confirmed property.** It has not been validated experimentally. The argument is architectural: continuous w → smooth qpos by construction, regardless of what the hand is doing frame to frame.

**Relationship to OAT**: OAT (Chen et al., ICRA 2026) proposes the same two-stream architecture (hand retargeting + wrist retargeting) and explicitly documents the stabilization problem. Their solution is external to the model. Our postural space approach could address the same problem internally -- if validated, this is a concrete architectural advantage over geometric retargeting approaches like OAT.

**OAT as reference for Stream 2**: OAT's wrist retargeting (Section III-B) is a concrete implementation of our Stream 2 -- estimating wrist position and orientation from MediaPipe 2D/3D keypoints via optimization. It achieves 30 Hz on a laptop. It can be adopted directly as the Stream 2 module in our system.

**References**:
- Entry 1: ED-TCN as temporal solution to erratic behavior
- Entry 10: two-stream architecture
- Chen et al. (ICRA 2026): OAT -- two-stream teleoperation, grasp stabilization problem, wrist retargeting implementation

**Status**: Proposed -- postural stabilization property not yet validated

---

## Entry 12 -- 2026-04-02: Arm control as a pluggable module

**Context**: Teleoperation requires controlling both the hand configuration (Stream 1, this system) and the robot arm pose (Stream 2). The arm control problem is orthogonal to hand grasp intent recognition and should not be coupled to it.

**Decision**: The arm control module is defined as a pluggable slot in the architecture with a fixed interface. It is not implemented as part of this work.

**Interface:**
```
input:  operator wrist pose (6 DOF -- position + orientation)
output: robot arm joint commands
```

**Candidate implementations** (not implemented, documented as future work):
- **AnyTeleop** (Qin et al., RSS 2024): full arm-hand teleoperation system with motion generation and collision avoidance. Modular design -- their hand retargeting module can be replaced by ours while keeping their arm motion generation. Open source, Docker-based, supports Shadow Hand explicitly.
- **OAT** (Chen et al., ICRA 2026): lightweight wrist pose estimation from monocular RGB via optimization. 30 Hz on laptop, no depth sensor required. Simpler than AnyTeleop -- no collision avoidance. Good candidate for quick integration.

**Why pluggable**: the hand grasp intent system (Stream 1) produces qpos for the robot hand independently of how the arm moves. The two streams share only the MediaPipe keypoints as input. Neither depends on the internals of the other.

**The complete system when plugged:**
```
MediaPipe
    |
    ├── Stream 1 (this work): keypoints → GNN → w → qpos hand
    └── Stream 2 (AnyTeleop/OAT): keypoints → wrist pose → motion generation → qpos arm
```

**Status**: Proposed -- Stream 2 not implemented. Interface defined.

---

## Entry 13 -- 2026-04-02 (revised 2026-04-02): Computable angles from 21 XYZ keypoints -- Pendiente B resolved

**Context**: The 20-DOF candidate set from Entry 9 was derived from anatomy (Chen 2013). Before building the human postural space we need to determine which of those 20 DOFs are computable from the 21 XYZ keypoints available in our domain (HOGraspNet training, MediaPipe deploy). This is Pendiente B, which includes subproblem C: how to compute abduction in 3D without contamination from flexion.

**Decision**: 16 of the 20 candidate DOFs are computable from 21 XYZ keypoints. The 4 non-computable DOFs are the CMC flexion angles of the 4 fingers (index, middle, ring, little) -- palmar arch deformation, requires metacarpal base keypoints absent from the 21-point model. The postural space is built from these 16 DOFs.

### Role of each source

Four papers establish the evidence base for this entry, each with a distinct role:

- **Chen et al. (2013)** -- *Anatomical inventory*. Table 6 lists which DOFs exist at each hand joint and their ROM. Tells us *what exists*: MCP has flex + abd (2 DOF), IP/PIP has only flex (1 DOF), DIP coupled to PIP. This is the source for the candidate set in Entry 9. Does not define how to measure the angles mathematically.

- **Wu et al. (2005)** -- *Measurement standard*. ISB recommendation for joint coordinate systems of the hand (Section 4). Defines *how the angles are measured*: flexion = rotation α around the Z-axis of the proximal segment (e1); abduction = rotation β around the floating axis (e2), perpendicular to e1. Because flex and abd are rotations around **different orthogonal axes**, they cannot be captured by a single dot product -- they require separate measurement. Critical limitation for our domain: Wu's JCS requires both the base and head of each metacarpal (Section 4.3.4). The metacarpal base is not available in 21-point models, so Wu cannot be implemented directly -- it motivates the decomposition, but the adaptation is left to the researcher.

- **Gionfrida et al. (2022)** -- *Computational method for 1-DOF joints*. Validates finger angle computation from 21-keypoint OpenPose against gold-standard optical motion capture. For joints with 1 DOF (PIP, IP), the 3-point arccos formula directly gives the single flexion angle -- no decomposition needed. Uses WRIST→MCP as metacarpal proxy for MCP flex (Fig 7, vector α), validated at RMSE ~6.82°. Abduction validated at RMSE <9° using inter-finger angles. Limitation: 2D only.

- **Schlegel et al. (2024)** -- *Computational method for 2-DOF joints*. Formalizes computation of ISB joint angles from 3D keypoints (Section 3.2). For joints with 2 DOF (MCP, TMC), a simple 3D dot product conflates flex and abd into one number. Schlegel's method: project the distal bone onto the sagittal plane of the proximal segment, measure flex as the in-plane angle and abd as the out-of-plane elevation. Builds explicitly on Wu's axis definitions. Validated on body joints (shoulder, hip); the mathematical principle transfers directly to finger MCP joints, which have the same 2-DOF structure.

**Argument chain**: Chen establishes *which DOFs exist* per joint. Wu establishes *why they need separate measurement* (different axes). Gionfrida provides *the computation for 1-DOF joints* (validated, 21 keypoints). Schlegel provides *the computation for 2-DOF joints* (implements Wu from keypoints). The method used for each angle follows directly from the number of DOFs of that joint.

### 21-keypoint index (MediaPipe / HOGraspNet)

```
0: WRIST
1: THUMB_CMC   2: THUMB_MCP   3: THUMB_IP    4: THUMB_TIP
5: INDEX_MCP   6: INDEX_PIP   7: INDEX_DIP   8: INDEX_TIP
9: MIDDLE_MCP  10: MIDDLE_PIP 11: MIDDLE_DIP 12: MIDDLE_TIP
13: RING_MCP   14: RING_PIP   15: RING_DIP   16: RING_TIP
17: LITTLE_MCP 18: LITTLE_PIP 19: LITTLE_DIP 20: LITTLE_TIP
```

### Operaciones base

**Sistema de coordenadas local (Wu 2005 Sec 4.3.4–4.3.5)**

Para metacarpianos y falanges, Wu define un JCS local con tres ejes:
- **Y_m**: eje longitudinal del hueso, de cabeza a base (proximal) — Wu Sec 4.3.4
- **X_m**: dirección volar (hacia la palma) — define el plano sagital del dedo
- **Z_m**: cross(X_m, Y_m) — eje lateral (radial para mano derecha)

Wu Sec 4.3.5: "The 14 coordinate systems for the phalanges can be described in a manner analogous to the metacarpal systems." — el mismo sistema aplica a todos los huesos de los 5 dedos.

**X_m = dirección volar = normal palmar n̂**, estimada desde keypoints disponibles:
```
n̂ = normalize( cross(kp[5]-kp[0], kp[17]-kp[0]) )   # WRIST→INDEX_MCP × WRIST→LITTLE_MCP
```

Para cada joint: ŷ = normalize(J - P) (eje longitudinal local, distal).
Por convención usamos ŷ apuntando distalmente (-Y_m de Wu), la dirección del movimiento.
```
ẑ = normalize( cross(ŷ, n̂) )    # eje lateral local — Wu Sec 4.3.4 Z_m
```

---

**FLEX(P, J, D)** — joint de 1 DOF (Schlegel 2024 Sec 3.2):
```
ŷ    = normalize(J - P)
b    = normalize(D - J)
flex = arccos(ŷ · b)
```
P = keypoint proximal, J = joint center, D = keypoint distal.

**FLEX_ABD(P, J, D)** — joint de 2 DOF (Schlegel 2024 Sec 3.2, ejes de Wu 2005 Sec 4.3.4):
```
ŷ     = normalize(J - P)
ẑ     = normalize(cross(ŷ, n̂))
b     = D - J
b_sag = b - dot(b, ẑ) * ẑ                  # proyección al plano sagital (Wu X_m–Y_m)
flex  = arccos( dot(normalize(b_sag), ŷ) )  # ángulo en plano sagital
abd   = arcsin( dot(normalize(b), ẑ) )      # elevación fuera del plano sagital
```

### Computability table

| DOF | Anatómica | Keypoints (exacto) | Adaptación | Veredicto |
|-----|-----------|-------------------|------------|-----------|
| Thumb TMC flex | `FLEX_ABD(CARPAL, TMC, MCP)` → flex | Falta CARPAL | `FLEX_ABD(kp[0], kp[1], kp[2])` → flex | **Con proxy** — WRIST sustituye CARPAL (trapecio no expuesto en ningún sensor de keypoints) |
| Thumb TMC abd | `FLEX_ABD(CARPAL, TMC, MCP)` → abd | Falta CARPAL | `FLEX_ABD(kp[0], kp[1], kp[2])` → abd | **Con proxy** — WRIST sustituye CARPAL (trapecio no expuesto en ningún sensor de keypoints) |
| Thumb MCP flex | `FLEX(TMC, MCP, IP)` | `FLEX(kp[1], kp[2], kp[3])` | — | **Exacto** |
| Thumb IP flex | `FLEX(MCP, IP, TIP)` | `FLEX(kp[2], kp[3], kp[4])` | — | **Exacto** |
| Index CMC flex | `FLEX_ABD(CARPAL, CMC, MCP)` → flex | Falta CARPAL, CMC | — | **NO computable** — CMC ausente del modelo de 21 puntos; sin joint center no se puede formar la cadena |
| Index MCP flex | `FLEX_ABD(CMC, MCP, PIP)` → flex | Falta CMC | `FLEX_ABD(kp[0], kp[5], kp[6])` → flex | **Con proxy** — WRIST sustituye CMC; validado Gionfrida 2022 Fig 7 (RMSE ~6.82°) |
| Index MCP abd | `FLEX_ABD(CMC, MCP, PIP)` → abd | Falta CMC | `FLEX_ABD(kp[0], kp[5], kp[6])` → abd | **Con proxy** — WRIST sustituye CMC; no validado independientemente para abd en 3D |
| Index PIP flex | `FLEX(MCP, PIP, DIP)` | `FLEX(kp[5], kp[6], kp[7])` | — | **Exacto** |
| Middle CMC flex | `FLEX_ABD(CARPAL, CMC, MCP)` → flex | Falta CARPAL, CMC | — | **NO computable** — CMC ausente del modelo de 21 puntos; sin joint center no se puede formar la cadena |
| Middle MCP flex | `FLEX_ABD(CMC, MCP, PIP)` → flex | Falta CMC | `FLEX_ABD(kp[0], kp[9], kp[10])` → flex | **Con proxy** — WRIST sustituye CMC; validado Gionfrida 2022 Fig 7 |
| Middle MCP abd | `FLEX_ABD(CMC, MCP, PIP)` → abd | Falta CMC | `FLEX_ABD(kp[0], kp[9], kp[10])` → abd | **Con proxy** — WRIST sustituye CMC; no validado independientemente para abd en 3D |
| Middle PIP flex | `FLEX(MCP, PIP, DIP)` | `FLEX(kp[9], kp[10], kp[11])` | — | **Exacto** |
| Ring CMC flex | `FLEX_ABD(CARPAL, CMC, MCP)` → flex | Falta CARPAL, CMC | — | **NO computable** — CMC ausente del modelo de 21 puntos; sin joint center no se puede formar la cadena |
| Ring MCP flex | `FLEX_ABD(CMC, MCP, PIP)` → flex | Falta CMC | `FLEX_ABD(kp[0], kp[13], kp[14])` → flex | **Con proxy** — WRIST sustituye CMC; validado Gionfrida 2022 Fig 7 |
| Ring MCP abd | `FLEX_ABD(CMC, MCP, PIP)` → abd | Falta CMC | `FLEX_ABD(kp[0], kp[13], kp[14])` → abd | **Con proxy** — WRIST sustituye CMC; no validado independientemente para abd en 3D |
| Ring PIP flex | `FLEX(MCP, PIP, DIP)` | `FLEX(kp[13], kp[14], kp[15])` | — | **Exacto** |
| Little CMC flex | `FLEX_ABD(CARPAL, CMC, MCP)` → flex | Falta CARPAL, CMC | — | **NO computable** — CMC ausente del modelo de 21 puntos; sin joint center no se puede formar la cadena |
| Little MCP flex | `FLEX_ABD(CMC, MCP, PIP)` → flex | Falta CMC | `FLEX_ABD(kp[0], kp[17], kp[18])` → flex | **Con proxy** — WRIST sustituye CMC; validado Gionfrida 2022 Fig 7 |
| Little MCP abd | `FLEX_ABD(CMC, MCP, PIP)` → abd | Falta CMC | `FLEX_ABD(kp[0], kp[17], kp[18])` → abd | **Con proxy** — WRIST sustituye CMC; no validado independientemente para abd en 3D |
| Little PIP flex | `FLEX(MCP, PIP, DIP)` | `FLEX(kp[17], kp[18], kp[19])` | — | **Exacto** |

**No computables: 4 DOFs** (CMC flex de 4 dedos — CMC joint y CARPAL ausentes del modelo de 21 puntos, no se puede formar la cadena).
**6 exactos (sin proxy)**: Thumb MCP flex, Thumb IP flex, 4× PIP flex.
**10 con proxy**: Thumb TMC flex/abd (CARPAL ausente → proxy kp[0]), 4× MCP flex/abd (CMC ausente → proxy kp[0]).

### Final 16-DOF set

```
Thumb (4):   TMC_flex, TMC_abd, MCP_flex, IP_flex
Index (3):   MCP_flex, MCP_abd, PIP_flex
Middle (3):  MCP_flex, MCP_abd, PIP_flex
Ring (3):    MCP_flex, MCP_abd, PIP_flex
Little (3):  MCP_flex, MCP_abd, PIP_flex
Total: 16
```

Consistente con la configuración de 16 sensores del CyberGlove usada en Santello et al. (1998) y Jarque-Bou et al. (2019, 2021), que excluían sensores de arco palmar. Esta coincidencia es validación cruzada independiente: esos instrumentos fueron diseñados alrededor de lo que caracteriza biomecánicamente la postura de la mano.

**Referencias**:
- Chen et al. (2013): inventario anatómico de DOFs, Table 6 (Entry 9)
- Wu et al. (2005): estándar ISB de medición, Sec 4.3.4 (coordenadas metacarpianas), Sec 4.4 (JCS articulaciones de la mano)
- Gionfrida et al. (2022): validación empírica de cómputo desde 21 keypoints, Fig 7 (fórmula flexión), resultados RMSE
- Schlegel et al. (2024): descomposición 3D flex/abd desde keypoints, Sec 3.2

**Status**: Resolved -- Pendiente B closed. Next step: implementar `add_anatomical_angles` en `tograph.py` con estos 16 DOFs.

---

## Entry 14 -- 2026-04-04: Dual-VGAE architecture -- cross-domain postural space alignment

**Context**: El GCN clasificador (abl04, 71.07%) es un aproximador de una función analítica computable (XYZ → 16 DOFs → PCA → distancias → w). El control postural via Σ w·anchor es lineal en espacio de joints -- no garantiza coherencia mecánica en poses intermedias. Dexonomy no tiene transiciones entre clases, solo poses estáticas (pregrasp/grasp/squeeze). Un MLP directo z→qpos no garantiza que poses intermedias sean físicamente ejecutables.

**Decision**: Arquitectura dual-VGAE con dos espacios latentes explícitos:

- **VGAE Encoder A** (humano): aprende manifold postural humano desde HOGraspNet. Input: grafo de mano (21 XYZ, T frames, CAM). Output: z_humano.
- **VGAE Encoder B** (robot): aprende manifold postural robot desde Dexonomy (pregrasp/grasp/squeeze). Input: qpos [22 joints]. Output: z_robot.
- **Manifold Mapping**: dado z_humano, obtiene z_robot equivalente. Mecanismo pendiente de definir -- las 29 clases Feix son las anclas de alineación entre espacios.
- **VGAE Decoder B**: z_robot → qpos Shadow Hand.

Pipeline de inferencia:
```
1. Input: Stream de video (webcam) procesado por MediaPipe para extraer 21 landmarks 3D de la mano.
2. Preprocessing (Multiframe): Ventana deslizante de N frames que estabiliza los puntos y aporta contexto temporal contra el ruido.
3. Graph Construction (Multiframe CAM): Construcción del grafo de la mano con topología aprendida.
4. VGAE Encoder A: Ubica la pose en el espacio latente humano Z_humano.
5. Manifold Mapping: Dado un punto en Z_humano, obtiene su equivalente en Z_robot. (Mecanismo pendiente de definir.)
6. VGAE Decoder B: ¿Qué configuración de joints del robot corresponde a ese punto en Z_robot?
7. Movimiento.
```

El Encoder B solo se usa durante entrenamiento. En inferencia solo se ejecutan Encoder A, el mapeo, y Decoder B.

**Alternatives considered**:
- MLP directo z_humano → qpos: funcional pero no garantiza coherencia geométrica en poses intermedias. No tiene espacio robot explícito -- las transiciones entre agarres caen en huecos no cubiertos por Dexonomy.
- Kernel regression (Σ w·anchor): equivalente a Diseño 2 pero en z-space no-lineal. Pierde la geometría del espacio robot.
- Diseño 1 (clasificador discreto): control discreto, sin morphing continuo.

**Expected impact**: Morphing continuo y mecánicamente coherente entre agarres. Agnosticidad al robot -- nuevo robot = nuevo Encoder B + Decoder B entrenados con sus datos de Dexonomy. Extensión no-lineal de Segil & Weir (2014).

**References**:
- Segil & Weir (2014): PCA sobre espacio postural humano, control via anclas ponderadas -- baseline lineal que este diseño extiende.
- Kipf & Welling (2016): Variational Graph Auto-Encoders -- arquitectura base.
- Leng et al. (2021): Stable Hand Pose Estimation under Tremor -- justificación de Multiframe CAM para robustez temporal.
- Dexonomy: pregrasp/grasp/squeeze qpos [11, 29] por agarre -- datos de entrenamiento Encoder B.

**Status**: Proposed -- Manifold Mapping pendiente de definir.

---

## Entry 15 -- 2026-04-11: Angle extraction basis -- adopt Dong kinematics and wrist-frame formulation

**Context**: Se definió como objetivo inmediato extraer variables posturales (ángulos) desde landmarks 3D de MediaPipe de forma biomecánicamente consistente. Hacía falta fijar un marco teórico único para evitar ambigüedad en la definición de ejes, transformaciones y parámetros angulares.

**Decision**:
- Se adopta Dong and Payandeh (2025) como referencia metodológica para el cálculo de ángulos de mano en 3D.
- Razón principal: el paper provee explícitamente:
  - la construcción del frame de muñeca/palma desde landmarks 3D (Eq. 5-8),
  - la transformación de world a frame local de muñeca (Eq. 9 y Eq. 16),
  - y el esquema jerárquico cinemático para resolver parámetros angulares por dedo.
- El pipeline de referencia queda:
  - `world 3D landmarks -> wrist local frame -> angle extraction`
- Se fija este criterio para el desarrollo siguiente del módulo de ángulos y para reportes experimentales.

**Alternatives considered**:
- Definiciones ad-hoc de ejes/ángulos sin formulación publicada: descartado por baja trazabilidad científica.
- Mezclar múltiples convenciones de ejes entre papers: descartado por riesgo de inconsistencia entre DOFs.

**Expected impact**:
- Base matemática única y reproducible para calcular postura.
- Coherencia entre preprocesamiento geométrico y extracción angular.
- Mejor defendibilidad metodológica en manuscrito y revisión por pares.

**References**:
- Dong and Payandeh (2025): Hand Kinematic Model Construction Based on Tracking Landmarks.

**Status**: Implemented

---

## Entry 20 -- 2026-04-14: Step4 MCP First Layer implemented (Dong Eq.19-24) with Eq.23 translation constrained by Eq.15

**Context**: Tras cerrar Palm Base (Eq.13/16/17), el siguiente bloque era MCP de primera capa (`i={1,5,9,13,17}`) para extraer `beta_i` y `gamma_i`. Surgio la duda clave de si la traslacion de Eq.23 debia usar `d_i^0` observado (con `z` medido) o el `d_i^0` parametrizado por Eq.15.

**Decision**:
- Se implemento el paso `4.1` en `dong_step4_to_wrist_local.py` con trazabilidad completa de Eq.19->Eq.24.
- Para **orientacion** MCP:
  - Eq.20/21/22 se calculan con puntos `d_i^0` y `d_{i+1}^0` observados en 3D (post Eq.16), construyendo `X_i^0`, `Y_i^0`, `Z_i^0` y `R_i^0`.
- Para **traslacion** de Eq.23:
  - se usa `d_i^0` de Eq.15, es decir `d_i^0=[l0,i cos(theta_i), l0,i sin(theta_i), 0]`.
  - Por tanto, el componente `z` de la traslacion en `T_i^0` se fija a `0` conforme al supuesto de palma en el plano `xy` del marco de muneca `{0}`.
- Se imprimen ambos vectores por raiz para auditoria:
  - `d_i^0(obs)` (medido por Eq.16) y `d_i^0(Eq.15)` (usado en Eq.23).
- Se agregaron checks MCP:
  - `det(R_i^0)`, `||R_i^{0T}R_i^0-I||`, y diferencia `||d_i^0(Eq.15)-d_i^0(obs)||`.

**Alternatives considered**:
- Usar `d_i^0` observado (con `z`) en Eq.23: descartado por inconsistencia con la parametrizacion de Eq.15 y con el supuesto explicito de primera capa en el plano de la palma.
- Forzar Eq.20 tambien a plano (`z=0`): descartado; el paper define Eq.20 desde landmarks transformados y su ejemplo numerico (Eq.25, dedo indice) muestra componente `z` no nula en `X_5^0`.

**Expected impact**:
- Reproduccion mas fiel del procedimiento de Dong para primera capa MCP.
- Menor ambiguedad metodologica en la construccion de `T_i^0`.
- Extraccion angular alineada al objetivo:
  - `beta_i` y `gamma_i` se obtienen de `R_i^0` (Eq.24), manteniendo consistencia con el modelo.

**References**:
- Dong and Payandeh (2025): Eq.15, Eq.19, Eq.20, Eq.21, Eq.22, Eq.23, Eq.24, Eq.25.
- Implementacion local: `grasp-app/hand_preprocessing/dong_step4_to_wrist_local.py`.

**Status**: Implemented

---

## Entry 19 -- 2026-04-13: Step4 Palm Base implemented (Dong Eq.13/16/17)

**Context**: Con `world -> wrist-local` ya estable, faltaba implementar explícitamente en Step4 la extracción de parámetros base de palma según Dong (sección 4.1), antes de MCP/PIP/DIP.

**Decision**:
- Se implementó el bloque Palm Base en `dong_step4_to_wrist_local.py` con trazabilidad a ecuaciones:
  - Eq.13: cálculo de longitudes base `l0,i` para `i={1,5,9,13,17}`.
  - Eq.16: transformación a local `d_i^0 = (T0w)^-1 d_i^w`.
  - Eq.17: cálculo de ángulos `theta_i = atan2(d0i_y, d0i_x)`.
- Se añadieron validaciones numéricas para esta etapa:
  - conservación de distancia `|||d0i|| - l0,i|`.
  - diagnóstico de planaridad `|d0i_z|` por raíz y máximo.
- Nota operativa: se reordenó la salida de consola a formato `Paso/Entrada/Salida` para facilitar auditoría manual del procedimiento.

**Alternatives considered**:
- Saltar Palm Base e ir directo a MCP: descartado por romper el orden metodológico de Dong.
- Mantener salida compacta sin desglose por ecuación: descartado por baja trazabilidad en depuración.

**Expected impact**:
- Cierre formal de la fase Palm Base en Step4 bajo Dong.
- Entrada consistente para la siguiente implementación de MCP (Eq.19-24).
- Menor ambigüedad al validar capturas frame-a-frame.

**References**:
- Dong and Payandeh (2025): Eq. 13, Eq. 16, Eq. 17 (Palm Base).
- Implementación local: `grasp-app/hand_preprocessing/dong_step4_to_wrist_local.py`.

**Status**: Implemented

---

## Entry 18 -- 2026-04-12: Next phase checkpoint -- continue with Dong angle extraction from Step 4

**Context**: Se cerró la fase de preparación de dataset RAW y estimación de parámetros base. El siguiente bloque de trabajo retoma la extracción angular bajo el marco de Dong desde el punto alcanzado en Step 4.

**Decision**:
- El próximo paso del proyecto será continuar con el cálculo de ángulos de Dong partiendo del estado actual de Step 4.
- Esta línea de trabajo se retoma en la siguiente sesión (mañana).

**Alternatives considered**:
- Ninguna en esta entrada; se mantiene el rumbo ya acordado.

**Expected impact**:
- Reanudar de forma ordenada la fase de variables posturales angulares.

**References**:
- Entry 15 y Entry 16 (base metodológica y estado del pipeline Step 4).

**Status**: In progress

---

## Entry 17 -- 2026-04-12: RAW dataset truth-first schema + robust estimation of thumb base ratio (0-1)

**Context**: Para estimar longitudes anatómicas (en particular el link `0-1`) sin sesgo de normalización previa, se decidió reconstruir el dataset desde anotaciones RAW de HOGraspNet. El objetivo explícito fue preservar la verdad del dataset y separar claramente: (a) ingesta fiel de anotaciones, y (b) remapeos/cálculos derivados para entrenamiento.

**Decision**:
- Se redefinió `build_hograspnet_raw_csv.py` como ingesta `truth-first`:
  - Sin normalización geométrica, sin escalado, sin mirror.
  - Sin `contact_sum` en el CSV final.
  - `grasp_type` se guarda como FEIX original (`grasp_XX`) parseado del path.
  - Se parsean metadatos explícitos desde ruta de anotación:
    - `subject_id, date_id, object_id, grasp_type, trial_id, cam, frame_id`.
  - XYZ se escribe tal como viene en `hand["3D_pose_per_cam"]`.
- Se fijó orden de filas en cascada para legibilidad y trazabilidad temporal por cámara:
  - `subject_id -> date_id -> object_id -> grasp_type -> trial_id -> cam -> frame_id`.
- Se mantuvo el remapeo FEIX→local (`0..27`) en el dataloader de entrenamiento (`grasps.py`), no en la ingesta RAW.
- Se añadió `compute_l01_ratio.py` para estimar `0-1` de forma robusta:
  - Ratio por frame:
    - `r_i = ||p1-p0|| / (||p9-p0|| + ||p10-p9|| + ||p11-p10|| + ||p12-p11||)`.
  - Agregación robusta en 3 niveles:
    - L1: mediana por bloque `(subject,date,obj,grasp,trial,cam)`.
    - L2: mediana por sujeto.
    - L3: mediana global sobre sujetos.

**Alternatives considered**:
- Mantener `sequence_id` compacta como única clave: útil pero menos práctica para filtros complejos y construcción de tripletes humano-robot.
- Ratio naive `||p1-p0|| / ||p12-p0||`: descartado como ratio final por alta variabilidad postural.
- Media global directa sobre 1.5M frames: descartada por sensibilidad a outliers y sesgo por sujetos/secuencias largas.

**Expected impact**:
- Dataset RAW reproducible, auditable y alineado con la verdad de anotación de HOGraspNet.
- Menor riesgo de leakage metodológico entre “datos observados” y “parámetros derivados”.
- Cálculo defendible de `0-1` para mano canónica, robusto entre sujetos y sesiones.

**Results**:
- CSV RAW generado:
  - `1,489,111` filas de datos (`1,489,112` con header).
  - `70` columnas (`7` metadatos + `63` XYZ).
  - Orden global verificado sobre todo el archivo: `order_ok=True`.
- Estimación robusta final:
  - `ratio_01 = 0.212922061679809` (`21.292206%`).
  - Para `hand_length = 182 mm`:
    - `L01 = 38.751815226 mm`
    - `L01 = 0.038751815226 m`.

**References**:
- HOGraspNet official layout and dataloader conventions (`scripts/HOG_dataloader.py`), especially `(subject, trial, cam, frame)` organization.
- Dong and Payandeh (2025): wrist-frame and kinematic extraction context (upstream of canonical scaling).
- Internal validation run on full RAW CSV (2026-04-12).

**Status**: Implemented and validated

---

## Entry 16 -- 2026-04-11: Dong Step4 pipeline reset -- remove pre-angle scaling, keep world->local first

**Context**: Durante la integración de `dong_step4_to_wrist_local.py`, se probó un flujo `world -> local -> scaled` con longitudes canónicas antes de cualquier extracción de ángulos. En gestos con contacto (p. ej. “like”), la reconstrucción escalada separó visualmente el pulgar de otros dedos. Esto generó conflicto metodológico para el objetivo principal: obtener variables posturales (ángulos) a partir de landmarks observados.

**Decision**:
- Se retiró completamente el bloque de escalado del Step4 (cálculo de `points_0_scaled`, flags `--scale-*`, panel 3D “Scaled”, y reporte de longitudes con target).
- Step4 queda como transformación y validación geométrica de Dong-lite:
  - `world -> wrist local` (Eq. 5-9,16)
  - logging de puntos y checks de consistencia (`||d0^0||`, `||T*T^-1-I||`, reconstrucción world).
- Se mantiene visualización comparativa 1:1 con ejes compartidos entre paneles para evitar sesgo por autoescalado de subplots.
- Se fija como flujo de trabajo para postural space:
  - `landmarks world -> wrist-local -> ángulos -> FK con mano canónica -> coordenadas canónicas`.
- El escalado/canonización no se aplicará antes de extraer ángulos; quedará aguas abajo en la reconstrucción FK.

**Alternatives considered**:
- Mantener `world->local->scaled` dentro de Step4: descartado para el flujo base porque mezcla normalización anatómica con observación antes de extraer postura.
- Reforzar escalado con heurísticas de contacto: descartado en esta etapa por complejidad y por no resolver el objetivo inmediato (ángulos).

**Expected impact**:
- Step4 vuelve a ser un bloque confiable de preprocesamiento para extracción de ángulos.
- Se elimina una fuente de confusión visual y metodológica en poses con oposición/contacto.
- Pipeline explícito y estable para implementación: `world -> wrist-local -> ángulos -> FK canónica -> coordenadas canónicas`.

**References**:
- Dong and Payandeh (2025): definición de frame de muñeca y transformaciones jerárquicas (Eq. 5-9, 16).
- Sesiones de validación internas Step4 (capturas 2026-04-11).

**Status**: Implemented

---

## Entry 21 -- 2026-04-14: Step4 second/third layer integrated + 15 finger lengths moved to fixed calibration block

**Context**: Despues de cerrar palma base (Eq.13/16/17) y MCP first layer (Eq.19-24), faltaba alinear la implementacion de second/third layer (Eq.29-36) con la definicion de parametros fijos del paper. Ademas, la salida de consola se volvio ruidosa para uso operativo.

**Decision**:
- Se implemento second/third layer (PIP/DIP) en Step4:
  - calculo de `beta_j` y `beta_{j+1}` (Eq.29 y Eq.31),
  - `R_j^{j-1}`, `R_{j+1}^j` (Eq.33-34),
  - `T_j^{j-1}`, `T_{j+1}^j` (Eq.35-36).
- Se movieron las **15 longitudes de dedos** a bloque de parametros fijos (calibration/freeze) en el mismo bloque 2:
  - links: `(1,2..3,4), (5,6..7,8), (9,10..11,12), (13,14..15,16), (17,18..19,20)`.
  - congelado por mediana sobre `N` capturas (`--palm-freeze-frames`, default `1`).
- En bloque 3, Eq.35 usa esas longitudes congeladas como `model lengths`.
- Se simplifico la salida de consola:
  - se eliminaron checks de consistencia,
  - se elimino `obs/model` en longitudes de second/third layer (solo modelo fijo),
  - `5.1` MCP quedo en formato de resultados (solo angulos), consistente con `5.2`.

**Alternatives considered**:
- Mantener longitudes de dedos dinamicas por frame en bloque 3: descartado por inconsistencia con el bloque de parametros fijos reportado en el paper.
- Mantener salida `obs/model` y checks en flujo principal: descartado por ruido operativo y confusion durante auditoria manual.

**Expected impact**:
- Step4 queda estructurado en 3 bloques claros:
  - transformacion dinamica `world -> wrist local`,
  - parametros fijos anatomicos (palma + 15 links),
  - parametros dinamicos articulares por frame.
- Menor ambiguedad de lectura para validacion manual y para preparar salida a latente.

**References**:
- Dong and Payandeh (2025): Eq.13, Eq.15, Eq.16, Eq.17, Eq.19-24, Eq.29-36.
- Implementacion local: `grasp-app/hand_preprocessing/dong_step4_to_wrist_local.py`.

**Status**: Implemented

---

## Entry 22 -- 2026-04-14: Scope split for Dong completion -- skip Unity handedness transform for latent pipeline

**Context**: Tras completar Step4 hasta Eq.36, quedo la duda sobre si implementar inmediatamente la conversion de marcos right-handed -> left-handed (Eq.46-56), la cual en el paper se usa para integracion con Unity.

**Decision**:
- Para el objetivo actual (features de postura para espacio latente), **no se implementa por ahora Eq.46-56**.
- Se mantiene la pipeline cinematica en sistema right-handed (sensor/modelo) y el siguiente cierre funcional sera:
  - conversion de matrices de rotacion a cuaterniones (Eq.57-64) sobre las rotaciones ya resueltas.
- La conversion a left-handed queda como modulo opcional de exportacion/compatibilidad para Unity.

**Alternatives considered**:
- Implementar ya Eq.46-56 en el flujo principal: descartado por no aportar valor directo al entrenamiento del latente y por introducir complejidad de convencion de ejes antes de cerrar la salida angular/cuaternion.

**Expected impact**:
- Menor complejidad inmediata y menor riesgo de mezclar convenciones de coordenadas en el dataset de features.
- Camino claro:
  - cerrar salida en cuaterniones para latente,
  - dejar adapter Unity como etapa separada cuando se necesite visualizacion/control en left-handed.

**References**:
- Dong and Payandeh (2025): Eq.46-56 (right-handed to left-handed for Unity), Eq.57-64 (quaternions from rotation matrices).
- Entries 20 y 21 (estado de Step4 hasta Eq.36).

**Status**: Implemented

---

## Entry 23 -- 2026-04-14: Latent features in quaternion space -- do not force signed abduction as parallel correction

**Context**: Surgio la duda de si forzar signo en abduccion/aduccion (`gamma` firmado) para mejorar diferenciacion de agarres, aun cuando la representacion final para el autoencoder sera en cuaterniones.

**Decision**:
- Se mantiene Dong en su formulacion actual para angulos escalares (incluyendo `gamma` por `arccos` sin signo).
- La salida para el latente se tomara desde matrices de rotacion hacia cuaterniones (`R -> q`), sin correccion manual de signo en abduccion.
- No se agregara `gamma` firmado como parche de la ruta principal, ya que resultaria redundante frente a la orientacion completa codificada en `q`.
- Se deja como nota tecnica que la consistencia de cuaterniones se controlara por convencion de signo (`q` y `-q` representan la misma rotacion).

**Alternatives considered**:
- Forzar `gamma` firmado en la ruta principal antes de convertir a cuaternion: descartado por redundancia y riesgo de mezclar representaciones no equivalentes entre angulos escalares y rotacion matricial.

**Expected impact**:
- Pipeline mas limpio y coherente para entrenamiento del latente.
- Menor riesgo de introducir sesgos por convenciones de signo ad-hoc.
- Mayor trazabilidad: la orientacion final usada por el modelo proviene directamente de `R`.

**References**:
- Dong and Payandeh (2025): matriz de rotacion por articulacion (Eq.19-36) y conversion a cuaterniones (Eq.57-64).
- Entry 22 (alcance de implementacion posterior a Eq.36).

**Status**: Implemented

---

## Entry 24 -- 2026-04-14: Scope lock before next session -- finish fingertip chain (last layer); handedness handling closed

**Context**: Cierre de sesion para evitar dispersion de trabajo. Se requiere dejar explicitado que el siguiente bloque tecnico es la ultima capa de la cadena de puntas y que el manejo de cambio de mano ya se considera resuelto en el alcance actual.

**Decision**:
- Se congela el alcance inmediato a un unico pendiente de implementacion: **completar la cadena de puntas (ultima layer)** del pipeline Dong.
- El cambio de mano queda marcado como **resuelto** para el alcance actual.

**Expected impact**:
- Prioridad clara para retomar sin reabrir discusiones laterales.
- Menor riesgo de introducir cambios no prioritarios antes de cerrar la ultima layer.

**Status**: Implemented

---

## Entry 25 -- 2026-04-14: Step4 closure for latent pipeline -- last layer + quaternions + camera modes + single-joint trace

**Context**: Tras cerrar Eq.19-36, faltaba completar la cadena cinematica hasta fingertip, convertir rotaciones a cuaterniones para features del espacio latente y estabilizar el comportamiento de captura en camara (mirror/handedness) sin contaminacion por swaps manuales.

**Decision**:
- Se implementa la **last layer** de Dong (Eq.45) para cada cadena de dedo:
  - `R_tip^dip = I`
  - `d_tip^dip = [l_dip,tip, 0, 0]`
  - `T_tip^dip` homogena correspondiente.
- Se implementa la conversion **R -> q** (Eq.57-64) para todas las joints del modelo:
  - MCP (`R_i^0`), PIP (`R_j^{j-1}`), DIP (`R_{j+1}^j`), TIP (`R_{j+2}^{j+1}`).
  - Salida final normalizada por joint en formato `(w,x,y,z)`.
- Se fija politica de captura para evitar ambiguedad:
  - `webcam`: frame espejado (selfie), handedness raw de MediaPipe, canonicalizacion a derecha por reflexion de landmarks solo cuando detecta Left.
  - `egocentric`: mismo pipeline pero sin espejar frame.
  - No se hace swap manual de etiquetas (`Left/Right`).
- Se agregan campos explicitos de trazabilidad de fuente:
  - `hand_detected`, `hand_canonical`, `landmark_reflected`, `frame_mirrored`.
- Se agrega modo de auditoria de una sola joint:
  - `--trace-joint N` para imprimir entrada -> transformacion -> salida de esa joint.
  - `--trace-joint-only` para ocultar el listado completo cuando se quiera depurar una sola joint.
- Se limpia salida regular de `[5.4]`:
  - ahora imprime solo cuaternion final por joint (sin procedimiento intermedio).
  - el procedimiento completo queda reservado a `--trace-joint`.

**Alternatives considered**:
- Aplicar Eq.46-56 (right-handed -> left-handed Unity) dentro del flujo principal: descartado por alcance actual (features latentes), se mantiene como adapter opcional posterior.
- Corregir handedness con swap de labels: descartado por riesgo de contaminar la semantica raw de MediaPipe.

**Expected impact**:
- Pipeline Step4 cerrado para extraccion de orientacion articular en cuaterniones.
- Menor confusion operativa entre modo espejo de camara y semantica de mano detectada.
- Salida de consola mas util:
  - compacta para uso normal,
  - detallada y auditable cuando se depura una joint puntual.
- Este cierre no congela la interfaz operativa actual:
  - la version presente funciona como herramienta de captura bajo demanda,
  - la siguiente mejora prevista es pasar a modo continuo con dos etapas:
    - un tiempo explicito de calibracion inicial para congelar parametros fijos,
    - despues calculo `frame-by-frame` en tiempo real durante el resto de la ejecucion.

**References**:
- Dong and Payandeh (2025): Eq.45, Eq.57-64.
- Entries 20, 21, 22, 23 y 24 (contexto de cierre de alcance en Step4).

**Status**: Implemented and validated

---

## Entry 26 -- 2026-04-14: Replace current graph angle features with Dong quaternions

**Context**: El pipeline GNN actual usa features angulares heuristicas:
- `flex`: un escalar por nodo calculado como el angulo entre el vector padre->joint y el vector joint->child.
- `theta_cmc`: un escalar global del grafo para la orientacion del pulgar respecto al plano de la palma.

Estas magnitudes no representan una cinematica articular completa. En particular:
- `flex` no separa flexion, abduccion ni orientacion local completa.
- `theta_cmc` queda aislado como escalar global y tampoco preserva una orientacion articular completa.

En paralelo, Step4 con Dong ya produce una representacion local por joint basada en matrices de rotacion y sus cuaterniones asociados.

**Decision**:
- La direccion de trabajo pasa a ser **reemplazar los angulos actuales usados como features del grafo por los cuaterniones locales de Dong**.
- La representacion objetivo deja de ser:
  - `flex` por nodo,
  - `theta_cmc` global,
  y pasa a ser:
  - `q_i` local por nodo/joint, derivado de `R_i^parent`.
- Para el grafo actual de 21 nodos:
  - `WRIST (0)` se tratara aparte como raiz.
  - los nodos `1..20` se alinean naturalmente con `q1..q20`.
- Este cambio se trata como una **reparametrizacion cinemática del espacio postural**, no como un ajuste menor de una feature escalar.

**Technical note to investigate**:
- Los cuaterniones presentan ambiguedad de signo:
  - `q` y `-q` representan la misma rotacion.
- En robotica esto no es problema, pero para ML puede introducir discontinuidades artificiales si una misma orientacion aparece a veces como `q` y a veces como `-q`.
- Queda explicitamente pendiente investigar y fijar una convencion numerica estable para entrenamiento/inferencia, por ejemplo una regla canonica tipo `w >= 0`, u otra que preserve continuidad temporal cuando convenga.

**Alternatives considered**:
- Mantener `flex` y solo agregar Dong como complemento: descartado como direccion principal, porque conserva la mezcla de una heuristica angular pobre con una representacion cinemática completa.
- Mantener `theta_cmc` como escalar global aislado: descartado como representacion principal una vez que ya se dispone de orientaciones locales completas por joint.

**Expected impact**:
- Sustitucion de features angulares heuristicas por una representacion articular local completa.
- Mayor coherencia entre la cinematica de la mano y las features que consume el grafo.
- Mejor base para comparar entrenamiento actual vs pipeline cinemático Dong.

**References**:
- `grasp-model/src/grasp_gcn/transforms/tograph.py`: `flex` actual y `theta_cmc` actual.
- Dong and Payandeh (2025): Eq.19-45, Eq.57-64.
- Entry 25 (Step4 ya cerrado hasta cuaterniones).

**Status**: Proposed

---

## Entry 27 -- 2026-04-14: Step4 continuous runtime mode -- timed calibration + per-joint live output

**Context**: Step4 ya estaba cerrado para captura puntual (snapshot), pero hacia falta operarlo como flujo continuo para uso en agarre en vivo: primero calibrar parametros fijos y luego calcular por frame sin romper la logica Dong ya validada.

**Decision**:
- Se agrega modo continuo en `dong_step4_to_wrist_local.py`:
  - `--continuous`
  - `--calib-seconds` para ventana de calibracion inicial
  - `--continuous-print-every` para controlar frecuencia de salida en consola.
- Se mantiene la misma matematica del pipeline; solo se reorganiza ejecucion:
  - durante calibracion: se acumulan muestras para palma y longitudes de dedos,
  - al terminar calibracion: se congelan parametros fijos,
  - despues: se ejecuta calculo completo por frame.
- Se implementa salida compacta en vivo por joint (sin dump completo de procedimiento):
  - MCP roots: `beta`, `gamma`, `q`
  - PIP/DIP: `beta`, `q`
  - TIP: `q`
- Se conserva `SPACE` para snapshot detallado completo cuando se requiera auditoria profunda.

**Expected impact**:
- Step4 deja de ser solo captura puntual y pasa a modo operativo en tiempo real.
- Menor ruido de consola en continuo, pero con visibilidad suficiente por articulacion para depuracion.
- Base lista para conectar features dinamicas de agarre sobre streaming frame-by-frame.

**References**:
- Implementacion local: `grasp-app/hand_preprocessing/dong_step4_to_wrist_local.py`.
- Entry 25 (cierre de last layer + quaternions en snapshot).

**Status**: Implemented and validated

---

## Entry 28 -- 2026-04-15: Coupled latent space for the hand -- decoupling by finger is not justified

**Context**: While designing the latent space architecture for cross-embodiment hand teleoperation, the question arose whether to decouple the latent space by finger (or finger group), following the segment-wise decoupling strategy of Yan and Lee (2026), "Learning a Unified Latent Space for Cross-Embodiment Robot Control" (arXiv:2601.15419). Since the system targets multiple robot hands with different morphologies (Shadow Hand: 5 fingers/24 DOF, Allegro: 4 fingers/16 DOF, future grippers), decoupling seemed like a candidate for handling missing or different fingers across embodiments.

**Decision**: The latent space is **coupled** (single space, hand as a unit). Decoupling by finger is not justified for the hand domain.

### Why Yan and Lee decouple (body segments)

Yan and Lee's decoupling is motivated by three concrete properties of the whole-body domain:

1. **Partial embodiments**: TIAGO has no legs, H1 has limited trunk, ATLAS has everything. A coupled latent space would force the encoder to represent legs for a robot that has none, producing ambiguous mappings. Each robot participates only in the subspaces it has.

2. **Heterogeneous similarity metrics per segment**: Arms need end-effector position accuracy (for manipulation). Legs need rotational fidelity (for visual resemblance while walking). A single space cannot optimize for both criteria simultaneously.

3. **Functional independence**: In whole-body motion, limbs are largely independent -- you can walk (legs) while reaching (right arm) while pointing (left arm). The information in each segment is orthogonal.

### Why these three conditions do not hold for the hand

1. **No partial embodiments in the relevant sense**: The human hand always has 5 fingers. Target robot hands have 4-5 fingers with different DOF counts, but the difference is kinematic (different joint configurations for the same finger), not structural (missing an entire limb). Allegro without a pinky is not analogous to TIAGO without legs -- the Allegro still executes 5-finger grasps by compensating with the remaining 4 fingers. This is handled by robot-specific embedding layers, not by latent space decoupling.

2. **Uniform similarity metric**: All fingers serve the same functional role in grasping -- coordinated closure around an object. There is no analog to the arms-need-EE-position vs legs-need-rotation distinction. A single similarity metric (postural distance) applies uniformly to the entire hand.

3. **Fingers are functionally dependent, not independent**: This is the fundamental difference. Santello et al. (1998) showed that 2 PCs explain >80% of variance in human hand postures. The first synergy is "all fingers open/close together." Feix grasp classes are defined by inter-finger coordination patterns (VF count, opposition type), not by individual finger states. Decoupling by finger would destroy exactly the signal that defines grasping.

### What decoupling would lose

- **Coordination patterns**: "Tip Pinch" is defined by the relationship between thumb and index -- not by thumb state alone and index state alone in separate subspaces.
- **Santello synergies**: The dominant modes of postural variation are whole-hand co-activation patterns. Decoupling by finger fragments these patterns across subspaces.
- **Feix taxonomy structure**: Every Feix class is a global hand configuration. The taxonomy does not decompose by finger.

### How multi-robot support works without decoupling

Following Yan and Lee's robot-specific embedding layer design (their E_r, D_r), each target robot hand has its own lightweight embedding layers that map between the shared latent space and its specific joint space:

```
z_latent (grasp intent) --> Embedding_Shadow (24 DOF) --> qpos Shadow Hand
                        --> Embedding_Allegro (16 DOF) --> qpos Allegro
                        --> Embedding_Gripper (6 DOF)  --> qpos Gripper
```

The latent space encodes postural intent (what type of grasp, with what configuration). Each robot's embedding layer translates that intent into its own kinematics. New robot = new embedding layer, latent space unchanged. This is the same mechanism as Yan and Lee but without subspace decoupling.

This is consistent with Entry 4 (two-space architecture) and Entry 6 (asymmetric design): the semantic bridge between human and robot is the Feix taxonomy, not geometric correspondence. The embedding layer is the robot-specific adapter.

### Only plausible decoupling: thumb vs digits 2-5

If decoupling were ever considered, the only biomechanically motivated split would be thumb vs the other 4 fingers, because the thumb is cinamatically distinct (TMC with 2 DOF, 3-phalanx chain vs 4-phalanx) and its opposition state is the row separator in the Feix taxonomy matrix. However, even this is not justified at this stage because thumb opposition only has meaning relative to the other fingers -- "abducted" and "adducted" are defined by the spatial relationship between thumb and palm/fingers, not by the thumb alone.

This can be revisited as an ablation experiment if the coupled latent space shows poor separation between opposition-driven Feix classes.

**Alternatives considered**:
- Decouple by finger (5 subspaces, one per finger): rejected. Destroys inter-finger coordination, which is the primary signal in grasping. No partial-embodiment justification.
- Decouple thumb vs digits 2-5 (2 subspaces): deferred. Some biomechanical motivation but thumb opposition is relational, not independent. Can be tested as ablation.
- Decouple by phalanx level (proximal/medial/distal): no precedent, no clear justification.

**Expected impact**:
- Simpler architecture (one encoder, one decoder, one latent space per embodiment side).
- Preserved inter-finger coordination in the latent space.
- Multi-robot support via embedding layers without architectural changes to the core model.
- Clean comparison point: if a future decoupled variant is tested, the coupled baseline exists.

**References**:
- Yan and Lee (2026), arXiv:2601.15419: cross-embodiment latent space with segment-wise decoupling for whole-body control; explicit limitation: "hand motion retargeting is not handled in this paper" (Sec III-E).
- Santello, Flanders and Soechting (1998): 2 PCs explain >80% of hand posture variance -- fingers are highly coordinated.
- Feix et al. (2016): GRASP taxonomy -- classes defined by whole-hand coordination (VF count, opposition, contact type).
- Entry 4: two-space architecture -- semantic communication via Feix distribution.
- Entry 6: asymmetric design -- robot-specific adapters, not shared geometry.
- Entry 14: dual-VGAE architecture -- robot-specific encoder/decoder with shared alignment.

**Status**: Proposed

---

## Entry 27 -- 2026-04-15: Quaternion Hemisphere Convention and Architecture Modularity

**Context**: 
The extraction of kinematic features (quaternions) to build a unified latent space (following the Yan & Lee framework) requires clean inputs. The paper relies on quaternions ($n=4$) as input for the human encoder. However, quaternions suffer from a topological "double cover" of SO(3): $q$ and $-q$ reflect exactly the same physical 3D rotation. This creates severe discontinuities (antipodal ambiguity) for neural networks that use euclidean or standard latent distances, hindering optimization and creating false jumps in the latent space.
Additionally, the original Dong implementation was a massive internal monolith coupling math with CV2 tools, making precomputation impossible.

**Decision**: 
1. **Mathematical Convention**: We apply a strict hemisphere projection to the quaternions right as they are generated by Dong, making sure $w \ge 0$. If $w < 0$, the quaternion is inverted ($-q$). This resolves the double cover ambiguity, removing massive jumps in neural network input distributions, while remaining 100% compatible with Yan & Lee's $n=4$ architecture.
2. **Architecture Refactor**: We decoupled the core mathematical blocks of Dong & Payandeh (2025) into a clean, reusable stateful object (`DongKinematics` in `dong_kinematics.py`). The camera debug functionalities were isolated into a separate file (`mediapipe_live.py`).
3. **Precomputation Pipeline**: A script (`precompute_dong_features.py`) was introduced to process the raw HOGraspNet dataset and save a clean feature dataset (`hograspnet_dong.csv`). We chose to separate raw data and precomputed feature data logically to keep versioning clean.

**Alternatives considered**:
- **Continuous 6D Representation (Zhou et al. 2019)**: Rejected. Although 6D coordinates resolve the topological problem intrinsically in ML, Yan & Lee's architecture natively expects quaternions ($n=4$). Moving to 6D would mean rewriting the baseline encoder and decoding framework, invalidating the foundational architecture that proves cross-embodiment retargeting.
- **Putting all features in the original dataset CSV**: Rejected. Harder to version, corrupts the clean raw structure, and unnecessarily widens the data frame when we only need a lightweight join logic based on primary keys.

**Expected impact**:
- Clean gradient flow when feeding the human encoder, protecting the continuous latent space structure against discontinuous rotation spikes.
- Full reusability of the Dong math solver across precomputation and real-time teleoperation endpoints.
- Ready to delete the obsolete `dong_step4_to_wrist_local.py` monolith.

**References**:
- Zhou et al. (2019): "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019. (Proof of topological discontinuity in $S^3$ quaternions and need for mitigation).
- Yan & Lee (2026): "Learning a Unified Latent Space for Cross-Embodiment Robot Control".

**Status**: Implemented


---

## Entry 29 -- 2026-04-16: Offline Precomputation & Structural Calibration of Dong Kinematics

**Context**: 
The GraphGrasp architecture relies on mapping 3D keypoints into Dong Kinematics features (wrist frame translation, quaternions, and beta/gamma angles) to achieve a topologically stable latent space. When ingesting the 1.5M frame HOGraspNet dataset, three critical questions arose: (1) whether to compute features dynamically during training or precompute them offline, (2) how to scale the anatomical bone calibration across thousands of separated trials/cameras, and (3) whether the t=8 temporal stabilization used for live webcams should be applied to the training dataset.

**Decision**:

1. **Strict Offline Precomputation**:
Kinematics are extracted offline resulting in `hograspnet_dong.csv`. The Dong algorithm relies on extensive matrix inversions and cross-products for all 21 joints, pushing 100% CPU load for ~45 minutes over 1.5M frames. Attempting this on-the-fly inside a PyTorch DataLoader would catastrophically bottleneck GPU training loops. Precomputing caches the mathematical translations so training relies purely on fast matrix multiplications.

2. **Subject-Level Rigid Calibration (N=50)**:
The dataset is processed by explicitly grouping on `subject_id` (ignoring `sequence_id`, `trial_id`, and `cam`). For each new subject, the algorithm captures the first 50 valid frames from their subset, computes the statistical median of their skeletal lengths, and freezes an immutable kinematic base. All subsequent frames for that person—across all objects and cameras—are evaluated against this fixed template. This prevents the neural network from learning false correlations caused by hands "magically shrinking" due to intra-trial perspective noise.

3. **Exclusion of t=8 Temporal Smoothing**:
The t=8 stabilizing buffer is strictly designated as an **inference-time mechanism** for real-time robotic teleoperation to combat heavy webcam noise. It is deliberately omitted from the training dataset. Imposing moving averages over the ground-truth training data would blur the sharp boundaries between grasp transitions and create a massive domain gap (training on artificially slow, smoothed physics while deploying against erratic ones). 

**Alternatives considered**:
- **On-the-fly dataloader extraction**: Rejected due to unviable I/O and CPU bottlenecking.
- **Trial-level calibration**: Rejected. Allowing skeletal dimensions to vary per trial for the same physical person destroys intra-subject topological consistency.
- **Applying t=8 smoothing to training**: Rejected as it would over-smooth the target loss landscape and degrade the responsiveness of the learned model.

**Expected impact**:
- Massive acceleration in GPU training iterations via the precomputed CSV.
- The model will explicitly learn to separate static intrinsic shape (handled by the rigid subject skeleton) from dynamic angular intent.
- 100% topological integrity (e.g. Tips fixed at w=1.0) and math-safe execution (Double-Cover w>=0 forced on all coordinates).

**References**:
- Dong & Payandeh (2025): Primary kinematics base.
- Precomputation validation mapping: Output columns rigorously logged to match MediaPipe semantic array blocks (`verify_mapping.py`).

**Status**: Implemented and Validated.
---

## Entry 30 -- 2026-04-19: abl10 result and objective realignment -- Dong quaternions for latent space, not classification

**Context**:

abl10 trained `GCN_CAM_8_8_16_16_32` with Dong quaternions only (F=4, no XYZ, no AHG, no flex) against the 28-class HOGraspNet benchmark. Result: **62.6% test accuracy, Macro F1 = 0.561**, versus abl04 baseline (XYZ+AHG+flex, F=24): 71.07%, F1=0.657.

The original plan stated "validate that Dong features improve over abl04 before investing in Dual-VGAE." That criterion was not met. However, reviewing the plan against the actual thesis objective exposed that the criterion was misaligned from the start.

**Decision**:

Proceed with Dong quaternions as the human encoder input for the Dual-VGAE latent space. The classification benchmark is not the right validation criterion for this architecture.

**Why 62.6% is sufficient validation**:

The thesis objective is not a standalone grasp classifier. It is a cross-embodiment latent space following Yan and Lee (2026): human grasp pose (Dong quaternions) maps to a latent representation z_human, which bridges to robot joint space z_robot via Feix anchors. The 62.6% accuracy with only 4 features per node (vs. random chance at 3.6%) demonstrates that Dong quaternions carry substantial discriminative information about grasp configuration. The latent space does not need to be optimized for argmax classification -- it needs semantic clustering by grasp type, which the quaternion signal is sufficient to support.

**Why XYZ is not added back**:

The lower accuracy of abl10 vs abl04 is expected: F=4 vs F=24. XYZ provides shape information (fingertip positions, inter-joint distances) that quaternions do not. However, adding XYZ to the latent space encoder would mix two fundamentally different representations -- rotation state (quaternions) and position (XYZ). This creates an asymmetry with the robot side, where joints are represented as scalar angles, not positions. Yan and Lee use quaternions (n=4) for human joints and scalar angles (n=1) for robot joints, with no XYZ on either side. The cross-embodiment similarity metric D_R operates on quaternions alone (plus D_ee for arm end-effectors in their system, which has no direct analog in grasping because all 5 fingertips are end-effectors with symmetric roles). XYZ would contaminate the similarity metric with scale/distance information that does not transfer to robot joint space.

**Why AHG is not used**:

AHG features (10 wrist-relative angles per node, 10 distances per node) are heuristic -- designed for classification discriminability, not for kinematic semantics. They have no correspondence on the robot side and would not improve the cross-embodiment alignment. Rejected on the same grounds as XYZ.

**Corrected validation criterion**:

The relevant validation for the Dual-VGAE is not classification accuracy but latent space quality: do semantically similar grasps cluster together? Do Feix-adjacent classes lie near each other in z_human? This is evaluated via reconstruction quality, manifold smoothness, and retargeting error -- not top-1 accuracy on a 28-class benchmark. The 62.6% result is evidence of signal; it is not the primary metric.

**Alternatives considered**:

- Add XYZ back (F=7: xyz+dong_quats): would improve classification accuracy but mix position and rotation in the encoder input. Incompatible with a clean D_R-based similarity metric. Rejected.
- Use XYZ+AHG+flex (stay at abl04): best classification accuracy, but features are heuristic and have no principled robot-side correspondence. Cannot support cross-embodiment latent alignment. Rejected as the latent space input.
- Require abl10 to beat abl04 before proceeding: criterion was wrong from the start. The thesis is about the latent space, not the argmax accuracy of a classifier. Rejected.

**Expected impact**:

- Human encoder input fixed: Dong quaternions only (21 joints × 4 = 84 dim, wrist-local frame, w>=0 hemisphere convention, subject-level calibration).
- WRIST (joint 0) = identity quaternion [1,0,0,0] by definition (reference frame).
- Similarity metric for contrastive learning: D_R(quats) over finger joints, following Yan and Lee Eq. 1.
- Robot side: Shadow Hand qpos (scalar angles), robot-specific embedding layer following Entry 14 / Yan and Lee E_r, D_r design.
- Next step: implement Dual-VGAE Encoder A (human graph, Dong quats) + Encoder B (robot qpos).

**References**:

- Yan and Lee (2026): human joints represented as quaternions (n=4), no XYZ; D_R similarity metric; robot-specific embedding layers.
- Entry 14: Dual-VGAE architecture -- Encoder A (human) + Encoder B (robot) + manifold mapping via Feix anchors.
- Entry 28: coupled (not decoupled) latent space for the hand -- inter-finger coordination must be preserved.
- Entry 29: Dong quaternions precomputed in hograspnet_dong.csv, hemisphere convention w>=0.
- abl04: 71.07% (xyz+AHG+flex, F=24) -- best classifier, wrong representation for latent space.
- abl10: 62.6% (Dong quats, F=4) -- lower classifier accuracy, correct representation for latent space.

**Status**: Decided

---

## Entry 31 -- 2026-04-19: Architecture alignment to Yan & Lee -- unified latent space, dataset selection, and cross-embodiment similarity criterion

**Context**:

A deep review of Yan and Lee (2026, arXiv:2601.15419) and the major hand-object interaction datasets (HOGraspNet, DexYCB, GRAB, FreiHAND) was conducted to inform the encoder architecture, dataset choice, and similarity criterion for the cross-embodiment latent space.

**Decision 1: One unified latent space, not dual**

Entry 14 proposed a Dual-VGAE with two separate latent spaces (z_human, z_robot) bridged via Feix anchors. This is revised. Yan and Lee use one unified latent space where both human and robot encoders map to the same z. The architecture is:

```
Human:  x_H (21 joints, Dong quats) --> E_h (graph encoder) --> z
Robot:  x_R (Shadow Hand qpos)       --> E_r (robot-specific embedding) --> E_X (shared MLP) --> z
```

The unified space is simpler, better supported by prior work, and directly enables robot-agnostic retargeting: new robot = new E_r/D_r, shared latent space unchanged. Entry 14's two-space design is discarded.

**Decision 2: HOGraspNet is the correct dataset for E_h**

Four datasets were reviewed against the objective of learning a latent space organized by grip configuration type:

- **HOGraspNet** (Woo et al., ECCV 2024): 1.5M frames, 99 subjects, 30 YCB objects, 28 Feix classes labeled per frame, dynamic sequences. Organized by **grip configuration type** -- same object grasped 28 different ways. Directly matches the retargeting objective.
- **DexYCB** (Chao et al., CVPR 2021): 582K frames, 10 subjects, 20 YCB objects, 1000 sequences. Protocol: "from hand approaching, opening fingers, contact, to holding the object stably." One grasp per sequence, no grasp taxonomy. Organized by **object shape** -- each cluster = how you grab a specific object. Useful for pose estimation, not for grip-type retargeting.
- **GRAB** (Taheri et al., ECCV 2020): 1.6M frames, 10 subjects, 51 objects, 4 intents (use/pass/lift/off-hand). Protocol: T-pose -> approach -> intent -> T-pose. Includes in-hand manipulation and re-grasping, but not X->Y transitions between Feix classes. Organized by **functional intent x object** -- hammer-use cluster, binoculars-use cluster. Uses SMPL-X (theta_h in R^60, not 21 XYZ landmarks). Wrong organization for retargeting.
- **FreiHAND** (Zimmermann et al., ICCV 2019): 37K frames, single images, no sequences, no grasp taxonomy. Organized by **raw geometric pose variation**. Designed for pose estimation generalization, not functional grasp analysis.

None of the datasets capture X->Y transitions between grasp types within a sequence while holding the same object. This gap is universal -- it is not a limitation of HOGraspNet specifically. The gap is addressed architecturally by the VGAE KL term (see Decision 4), not by dataset selection.

HOGraspNet is the only dataset that organizes data by grip configuration type, which is exactly what the robot needs to know for retargeting. It is also the only dataset with Feix labels at scale, compatible 21-landmark format (same as MediaPipe inference), and sufficient subject diversity (99 subjects) for E_h generalization.

**Decision 3: Feix + D_ee as cross-embodiment similarity criterion**

Two approaches were considered for mining positive/negative pairs across embodiments:

- **D_ee only** (geometric): human fingertip XYZ (direct from landmarks) vs robot FK fingertip positions. Continuous, no taxonomy required. Problem: Power Sphere with small object and Tip Pinch both produce close fingertips -- D_ee cannot distinguish them. Object size confounds grip type.
- **Feix only** (semantic): same Feix class = positive. Problem: destroys intra-class geometric variation. Large-object and small-object Power Sphere have very different joint configurations -- forcing them to identical latent points collapses the continuous structure of the manifold.
- **Feix + D_ee hybrid** (adopted): Feix defines coarse positive/negative (same class = positive cross-embodiment, different class = negative). D_ee provides fine-grained ordering within each class (object size, aperture variation). This preserves both semantic cluster identity and continuous intra-class geometry.

Robot side (Dexonomy): 26/28 HOGraspNet Feix classes have Shadow Hand qpos from Dexonomy (RSS 2025). Missing: Feix 9 (Palmar Pinch) and Feix 19 (Distal). E_h trains on all 28 classes (human data complete). E_r trains on 26 classes (robot data available). The 2 missing classes have no robot-side alignment -- these regions exist in z but Shadow Hand cannot reach them. This is a dataset limitation, not an architectural flaw. Documented as a known limitation.

**Decision 4: VGAE KL term handles intermediate latent zones**

Since no dataset contains X->Y transitions between grip types, the latent space has no explicit training signal for intermediate zones between Feix clusters. This is not resolved by switching datasets (the gap is universal). Instead, the VGAE KL divergence term:

```
KL(q(z|x) || N(0, I))
```

regularizes the latent space toward a smooth Gaussian prior. This forces intermediate zones to be populated with meaningful interpolations rather than garbage. The decoder learns to produce valid hand poses for intermediate latent points even without explicit transition data. This is the architectural justification for using VGAE over a standard VAE or AE -- the prior-matching pressure creates a complete, navigable manifold.

**Decision 5: Yan's explicit limitation validates the thesis gap**

Yan and Lee (2026) state explicitly: "Since the SMPL model does not capture hand movements, it considers the hands as part of the forearms. As a result, due to the missing hand motion in the SMPL, hand motion retargeting is not handled in this paper, limiting applications that require fine-grained teleoperation."

This confirms that the thesis addresses a real, acknowledged gap in the cross-embodiment retargeting literature. The contribution is not incremental -- it extends Yan's framework to the hand domain, which requires a different human representation (Dong quaternions over 21 landmarks vs. SMPL body joints), a different dataset (HOGraspNet vs. motion capture body sequences), and a graph-based encoder to exploit the skeletal topology of the hand.

**Architecture summary (updated from Entry 14)**:

```
E_h:  graph encoder, input = 21 joints x 4 Dong quats = 84-dim per frame
      trained on HOGraspNet (28 Feix classes, 1.5M frames)
E_r:  robot-specific MLP embedding, input = 22 Shadow Hand qpos
      trained on Dexonomy (26/28 Feix classes)
E_X:  shared MLP encoder (following Yan: 8 FC layers, 256 neurons, ELU, Tanh output)
z:    unified latent space, coupled (Entry 28), continuous via VGAE KL
D_r:  robot-specific MLP decoder, output = 22 Shadow Hand qpos
Loss: L_contrastive (Feix + D_ee triplets) + L_rec + L_ltc + L_KL
```

**References**:

- Yan and Lee (2026), arXiv:2601.15419: unified latent space architecture, E_r/D_r design, L_ltc loss, explicit hand limitation quote.
- Taheri et al. (2020), ECCV: GRAB dataset -- protocol, SMPL-X representation, intent taxonomy.
- Chao et al. (2021), CVPR: DexYCB dataset -- protocol quote, 21-joint annotation, single-grasp-per-sequence structure.
- Woo et al. (2024), ECCV: HOGraspNet -- dynamic sequences, 28 Feix classes, 99 subjects.
- Entry 14: original Dual-VGAE proposal (now revised to unified space).
- Entry 28: coupled latent space for hand.
- Entry 30: Dong quaternions as human encoder input.

**Status**: Decided

---

## Entry 32 -- 2026-04-19: Similarity criterion revised to D_R + D_ee (no Feix in triplets), and loss function confirmed from Yan

**Context**:

Entry 31 Decision 3 adopted Feix + D_ee as the triplet similarity criterion. After a closer reading of Yan and Lee (2026) Eqs. 1-4, this decision is revised. Feix labels are not used in triplet mining.

**Decision 1: Similarity criterion is D_R + omega*D_ee, following Yan Eqs. 1-3**

For arm subspaces (LA, RA), Yan defines:

```
D_R(x_A, x_B) = sum_j (1 - <q_A^j, q_B^j>^2)          (Eq. 1)
D_ee(x_A, x_B) = ||p_A^ee - p_B^ee||_2                  (Eq. 2)
S_hand = D_R + omega * D_ee    for arm/hand subspace     (Eq. 3)
```

The hand is the analogue of Yan's arm subspace (manipulation extremity). Therefore, for our hand subspace:

```
D_R: sum over 21 hand joints of (1 - <q_A^j, q_B^j>^2)
     using Dong quaternions (wrist-local, w>=0 convention)
     wrist (joint 0) excluded -- identity quaternion by definition

D_ee: L2 distance between 5 fingertip positions in wrist-local normalized frame
      p_ee = {thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip}

S_hand = D_R + omega * D_ee
```

omega is a tunable weight (Yan does not fix it; to be determined empirically).

D_R captures rotational configuration (how the fingers are oriented). D_ee captures where the fingertips end up in task space. Both are necessary cross-embodiment: D_R is joint-space (embodiment-specific), D_ee is task-space (comparable across human and robot via FK).

**Decision 2: Feix not used in triplet criterion**

Feix labels are NOT used to define positives/negatives in triplet mining. The similarity is purely geometric (D_R + D_ee). Feix cluster structure emerges in z as a consequence of geometry, not as an explicit constraint.

Rationale:
- Feix boundaries are arbitrary divisions of a continuous manifold -- imposing them as hard triplet labels collapses intra-class variation
- D_R + D_ee is continuous, differentiable, and directly comparable cross-embodiment (human Dong quats vs robot FK output)
- Yan uses no grasp taxonomy and achieves cross-embodiment retargeting -- the same approach applies here
- AHG features (angles and distances from Aiman and Ahmad 2024) are equivalent to D_R + D_ee information and are already used as E_h node features (abl04); they are not needed as a separate similarity criterion

**Decision 3: Loss function confirmed from Yan Eqs. 6-9**

Retargeting network total loss (Yan Eq. 9):

```
L_total = lambda_c * L_contrastive + lambda_rec * L_rec + lambda_ltc * L_ltc + lambda_temp * L_temporal
          lambda_c=10,              lambda_rec=5,         lambda_ltc=1,         lambda_temp=0.1
```

Individual losses:

```
L_contrastive = sum over triplets max(||z^a - z^+||_2 - ||z^a - z^-||_2 + alpha, 0)
                alpha = 0.05 (Yan Eq. 5)

L_rec = ||x_A - x_hat_A||_2                                                   (Yan Eq. 6)
        for robot poses only -- human x_H has no paired robot ground truth

L_ltc = ||E_h(x_H) - E_X(D_X(E_h(x_H)))||_2                                  (Yan Eq. 7)
        latent consistency: decode human embedding to robot, re-encode, check round-trip

L_temporal = ||v_H^hand - v_ee^robot||_2                                       (Yan Eq. 8)
             velocity of human fingertips vs robot EE between consecutive frames
             applicable because HOGraspNet contains dynamic sequences
```

c-VAE control policy loss (separate component, Yan Eq. 10):

```
L_cvae = L_reconstruction + lambda_KL * L_KL    lambda_KL = 1e-4
         L_KL = KL(N(mu, sigma) || N(0, I))
```

Note: L_KL is part of the c-VAE control policy (goal-conditioned motion generation), not the retargeting network. The VGAE KL term documented in Entry 31 Decision 4 serves the same regularization role but belongs to our encoder, not Yan's control policy.

**Architecture summary (final)**:

```
E_h:  graph encoder, input = 21 joints x 4 Dong quats = 84-dim per frame
E_r:  robot-specific MLP, input = 22 Shadow Hand qpos --> 1024-dim embedding
E_X:  shared MLP (8 FC, 256 neurons, ELU, Tanh), output = 16-dim z
D_X:  shared MLP decoder, input = z --> pose in shared space
D_r:  robot-specific MLP, input = shared space --> 22 Shadow Hand qpos

Similarity: S_hand = D_R(Dong quats) + omega * D_ee(5 fingertips, wrist-local)
Loss:       L_total = 10*L_contrastive + 5*L_rec + 1*L_ltc + 0.1*L_temporal + L_KL
```

**References**:

- Yan and Lee (2026) Eqs. 1-9: D_R, D_ee, S_k, L_contrastive, L_rec, L_ltc, L_temporal, L_total, L_cvae.
- Entry 31: unified latent space architecture, HOGraspNet justification, VGAE KL rationale.
- Entry 28: coupled (not decoupled) latent space.
- Aiman and Ahmad (2024): AHG features = angles + distances, already used as E_h node features; not needed as similarity criterion.

**Status**: Decided

---
## Entry 33 -- 2026-04-19: Exact component inventory of Yan & Lee (2026) Stage 1

**Context**:

Full paper read from /home/yareeez/AIST-hand/knowledge/yan2026/main.tex. This entry documents every component of Yan Stage 1 exactly as described in the paper.

Yan has two stages. Stage 1 = unified latent space (retargeting). Stage 2 = c-VAE goal-conditioned control policy. These are separate systems.

---

**Component 1: E_h -- Human Encoder**

Maps human pose to latent subspaces z.
Implementation: MLP, 8 fully connected layers, 256 neurons/layer, ELU activations, Tanh output.
Input: human joint quaternions x_H in R^(J_H x 4).
Output: 5 latent vectors, one per body segment (LA, RA, TK, LL, RL), each 16-dim, bounded [-1, 1].

---

**Component 2: E_r -- Robot-Specific Embedding Layer**

Projects robot-specific joint space to fixed-size shared feature space.
Implementation: learnable MLP.
Input: robot qpos (scalar joint angles, J_robot joints), x in R^(J_robot x 1).
Output: 1024-dim feature vector.
Purpose: handles dimensionality mismatch between robots with different numbers of joints.
When adding new robot: only E_r and D_r are trained. E_h, E_X, D_X are frozen.

---

**Component 3: E_X -- Shared Cross-Embodiment Encoder**

Shared MLP that maps the 1024-dim robot embedding to latent subspaces z.
Implementation: MLP, 8 fully connected layers, 256 neurons/layer, ELU activations, Tanh output.
Input: 1024-dim robot embedding from E_r.
Output: 5 latent vectors (LA, RA, TK, LL, RL), each 16-dim, bounded [-1, 1].
Shared across all robots and human. Frozen when adding new robots.

---

**Component 4: D_X -- Shared Cross-Embodiment Decoder**

Shared MLP that reconstructs from latent z back to shared embedding space.
Implementation: MLP, architecture symmetric to E_X.
Input: latent z (16-dim per subspace).
Output: shared embedding (1024-dim), then passed to D_r.
Frozen when adding new robots.

---

**Component 5: D_r -- Robot-Specific Inverse Embedding**

Projects shared 1024-dim embedding back to robot-specific joint space.
Implementation: learnable MLP, inverse of E_r.
Input: 1024-dim shared embedding from D_X.
Output: robot qpos (J_robot scalar angles).
When adding new robot: only E_r and D_r are trained.

---

**Decoupled latent space**

Yan decouples z into 5 subspaces for 5 body segments (LA, RA, TK, LL, RL).
Reason: robots have asymmetric morphologies (TIAGO = arms only, H1 = limited trunk). Decoupling prevents cross-segment interference.
Each subspace is 16-dim, bounded [-1, 1].
Each subspace has its own similarity metric:
  - Arms (LA, RA): S = D_R + omega*D_ee, omega=1.0 (confirmed by ablation)
  - Trunk/legs (TK, LL, RL): S = D_R only

---

**Loss functions (Yan Eq. 9)**

```
L_total = 10*L_contrastive + 5*L_rec + 1*L_ltc + 0.1*L_temporal
```

L_contrastive (Eq. 5): triplet loss, alpha=0.05. Triplets from ANY embodiment (human or robot). Similarity criterion = S_k per subspace.
L_rec (Eq. 6): ||x_robot - x_hat_robot||_2. Applied to ROBOT poses only. Human has no direct reconstruction because no paired human-robot ground truth exists.
L_ltc (Eq. 7): ||E_h(x_H) - E_X(D_X(E_h(x_H)))||_2. Latent round-trip consistency for human. Encodes human, decodes to robot space, re-encodes, checks round-trip fidelity.
L_temporal (Eq. 8): ||v_H^hand - v_robot^ee||_2. L2 distance between human hand velocity and robot EE velocity across consecutive frames.

---

**Training data (Yan)**
Human: HumanML3D dataset, 29,224 sequences, 4M poses, full body, SMPL representation (quaternions, n=4).
Robot: NOT collected. Sampled uniformly at random from joint space via FK at each training step. Over 10^5 new robot poses per step, immediately discarded. Billions of poses per embodiment over full training. FK computed via PyTorch-Kinematics from URDF.
Batch size: 10^5. Optimizer: Adam, lr=1e-3. Hardware: NVIDIA A4000 GPU.

---

**Stage 2: c-VAE Goal-Conditioned Control Policy**

Separate from Stage 1. Trained exclusively on human data after Stage 1 is complete.
Input: current latent z_t + goal EE velocity v_ee.
Output: latent displacement d_t = z_{t+1} - z_t.
Implementation: MLP encoder + MLP decoder, 8 linear layers each, ELU, no output activation.
Encoder output: 32-dim Gaussian distribution (mu, sigma).
Loss: L_cvae = L_reconstruction + lambda_KL * L_KL, lambda_KL = 1e-4.
L_reconstruction = ||d_t - d_hat_t||_2^2.
At inference: z_{t+1} = z_t + d_hat_t, autoregressive.

---

**References**:
- Yan and Lee (2026), main.tex: all equations, hyperparameters, implementation details.

**Status**: Decided

---
## Entry 34 -- 2026-04-19: Human-to-robot inference pipeline clarification

**Context**:

Clarification of which components are active during human-to-robot inference vs training.

---

**Human-to-robot inference pipeline (exact, from Yan main.tex line 168):**

```
x_H (human pose, quaternions)
  --> E_h  [Human Encoder, MLP 8x256]
  --> z    [16-dim latent vector]
  --> D_X  [Shared Cross-Embodiment Decoder, MLP] --> 1024-dim shared embedding
  --> D_r  [Robot-Specific Inverse Embedding, MLP] --> x_hat_A (robot qpos)
```

E_r and E_X are NOT in the human-to-robot inference path.

---

**Why D_X and D_r are two separate MLPs (not one):**

D_X is shared across all robots -- it decodes from z to a 1024-dim shared embedding space.
D_r is robot-specific -- it projects from the 1024-dim shared space to robot joint space.

Separating them enables: freeze D_X when adding a new robot, train only D_r.
If they were one MLP, you could not freeze the shared part independently.

---

**E_r and E_X: when they are used**

During training only -- for the robot side of triplets and for L_rec (robot reconstruction).
Also used in L_ltc: E_X(D_X(E_h(x_H))) -- round-trip consistency check for human.
At inference for human-to-robot: not used.
At inference for robot-to-robot: x_A --> E_r --> E_X --> z --> D_X --> D_r --> x_hat_B.

**References**:
- Yan and Lee (2026), main.tex line 168: exact human-to-robot pipeline.

**Status**: Decided

---

## Entry 35 -- 2026-04-19: GCN/GAT encoder for E_h, MLP decoders for D_X and D_r

**Context**:

Yan & Lee (2026) implement all components (E_h, E_X, D_X, E_r, D_r) as MLPs. We considered whether to replace some components with graph-based networks, given that SAME (Lee et al., SIGGRAPH Asia 2023) uses a GCN/GAT autoencoder for skeleton-agnostic motion embedding.

---

**Decision**:

- **E_h**: Replace MLP with GCN/GAT (SAME-style encoder). The input to E_h is the hand pose -- 21 joints represented as nodes in a graph, with bones as edges and Dong quaternions as node features. GCN layers propagate information between neighboring joints before compression. This exploits the structural topology of the hand: MCP influences PIP, PIP influences DIP, thumb has different connectivity than fingers. A flat MLP treats all node features as an unstructured vector and must learn these relationships implicitly.

- **D_X**: Keep as MLP (Yan). Input is z (flat latent vector), output is a 1024-dim vector. No graph structure in input or output.

- **D_r**: Keep as MLP/linear embedding (Yan). Input is 1024-dim vector, output is robot qpos (flat vector of scalar joint angles). No graph structure.

---

**Why the graph is only needed in the encoder:**

The graph topology captures relationships between joints. In E_h, those relationships are computed and compressed into z -- the structural information is encoded inside the latent vector. Once z exists, D_X and D_r only need to map that vector to another vector. They do not need to reinterpret hand structure -- it is already there inside z.

SAME uses a GCN decoder because its output must be per-node (a pose for each joint of an arbitrary target skeleton, with variable dimensionality). Our D_X and D_r produce flat vectors of fixed dimensionality. The condition that justifies a GCN decoder in SAME does not apply here.

---

**Alternatives considered**:

- GCN decoder for D_X or D_r: rejected. Output dimensionality is fixed. No graph structure to exploit. SAME uses GCN decoder specifically because output topology varies per target skeleton -- that is not our case.
- Pure MLP for E_h (Yan-style): rejected. Loses the inductive bias from hand joint topology. GCN is strictly more expressive given graph-structured input.

**References**:
- Yan and Lee (2026), main.tex line 257: E_h, E_X, D_X implemented as MLPs (our baseline).
- Lee et al. (2023), SAME Section 4.2: GCN encoder + GCN decoder. Decoder uses GCN because output is per-node with variable target skeleton topology.

**Status**: Decided

---

## Entry 36 -- 2026-04-19: CAM included in E_h alongside GAT

**Context**:

E_h uses GAT layers (Entry 35). Question: also include CAM (Constraint Adjacency Matrix, Leng et al., IEEE VR 2021)?

---

**What CAM actually is (from Leng et al.):**

CAM is a learnable N×N matrix of weights, one entry per joint pair. It replaces the fixed skeleton topology entirely. Aggregation becomes:

    f_k = sum_j (CAM_kj * h_j)  for all j in 1..N

Every joint communicates with every other joint. The weights are static parameters learned by gradient descent -- the same for all samples and all frames.

This is different from GAT: GAT computes attention weights dynamically from current node features, but only between neighbors in the fixed graph. CAM learns which joints are globally correlated (static), GAT learns how much each neighbor matters per pose (dynamic). They are complementary -- not redundant.

---

**Decision**: Use GAT + CAM in E_h.

CAM and GAT address different aspects:
- CAM: global topology (any joint to any joint), learned statically. Captures that e.g. index MCP and ring MCP are always correlated across grasps.
- GAT: local dynamic attention (neighbors only), computed per input frame.

CAM adds 21x21 = 441 parameters. Cost is negligible.

abl04 already demonstrated CAM works for this exact hand graph (21 joints). Excluding CAM would be a regression relative to what is already empirically validated, with no theoretical justification for the removal.

Ablation of GAT vs GAT+CAM is not planned -- the system has too many components to ablate each independently, and this choice has both theoretical grounding (Leng) and empirical support (abl04).

**References**:
- Leng et al. (2021), IEEE VR: CAM definition and aggregation equation.
- abl04: validated CAM on 21-joint hand graph for grasp classification.

**Status**: Decided

---

## Entry 37 -- 2026-04-19: Multi-frame CAM inside E_h, no external stabilization

**Context**:

Entry 36 decided GAT + CAM (single-frame, spatial only) for E_h. Question: also add a multi-frame CAM outside the model to stabilize Dong features before they enter E_h?

---

**Decision**: Single multi-frame CAM inside E_h. No external CAM or smoothing.

E_h receives 8-frame sliding windows of Dong features (computed per-frame from raw landmarks). The multi-frame CAM inside E_h handles both spatial (joint-to-joint) and temporal (frame-to-frame) relationships in a single learned matrix. No external preprocessing step needed.

Two-CAM option (external multi-frame CAM + internal single-frame CAM) rejected: redundant, adds complexity, no evidence of benefit. If a frame is lost (no landmarks detected), the sliding window retains previous frames -- handled internally.

w>=0 hemisphere convention (Entry 27) ensures quaternion sign consistency across frames by definition.

**References**:
- Leng et al. (2021), IEEE VR: multi-frame CAM-GNN.
- Entry 27: hemisphere convention for Dong quaternions.
- Entry 36: GAT + CAM decision for E_h.

**Status**: Decided

---

## Entry 38 -- 2026-04-19: Next session plan

Review Yan human-to-robot pipeline (Entry 34) in detail. From there, define exactly what SAME replaces (the encoder E_h), then design the remaining components bottom-up.

**Status**: Completed (2026-04-22)

---

## Entry 39 -- 2026-04-22: Full system architecture for hand retargeting (Yan + SAME)

**Context**:

Complete architectural design of the cross-embodiment hand retargeting system, adapting Yan & Lee (2026) to the hand domain with SAME-style encoder for E_h.

---

**Decision**:

Five components, all implemented in `grasp-model/src/grasp_gcn/network/`:

```
E_h  (human_encoder.py):      GAT 3 layers, hidden=32, heads=4, global_max_pool, Linear, Tanh → z[32]
E_X  (cross_embodiment.py):   MLP 3 layers (256→128→64→32), ELU, Tanh → z[32]
D_X  (cross_embodiment.py):   MLP 3 layers (32→128→256→256), ELU, Tanh → [256]
E_r  (robot_embedding.py):    Linear 24→256  (Shadow Hand, 24 actuable joints)
D_r  (robot_embedding.py):    Linear 256→24  (Shadow Hand)
```

Input to E_h: 21 nodes x 4 features (Dong quaternions, wrist node = identity [1,0,0,0]).
Latent dim z=32, shared embedding dim=256.
All encoder/decoder outputs bounded [-1,1] via Tanh (Yan convention).

**Deviations from Yan**:
- E_h: GAT+CAM (SAME-style) instead of MLP -- exploits hand graph topology
- Fewer layers (3 vs 8) and smaller dims -- hand domain is simpler than full-body
- z_dim=32 instead of 16 -- 28 Feix classes require more latent capacity
- shared_dim=256 instead of 1024 -- Shadow Hand has 24 joints, not 50+
- 1 subspace (hand) instead of 5 (LA, RA, TK, LL, RL) -- hand is a single connected system (Entry 28)
- CAM not yet added to E_h -- deferred until full system is validated end-to-end

**Training data**:
- Human: hograspnet_dong.csv (Dong quaternions) + hograspnet_raw.csv (XYZ for D_ee)
- Robot: qpos sampled uniformly from Shadow Hand joint limits via FK on-the-fly (pytorch-kinematics + URDF)
- No paired human-robot demonstrations needed (Yan approach)

**Losses** (Yan, weights: λ_c=10, λ_rec=5, λ_ltc=1, λ_temp=0.1):
- L_contrastive: triplet loss with similarity metric D_R + D_ee
- L_rec: robot reconstruction (E_r→E_X→z→D_X→D_r→qpos)
- L_ltc: human round-trip consistency ||E_h(x_H) - E_X(D_X(E_h(x_H)))||
- L_temporal: fingertip velocity alignment between consecutive frames

**Similarity metric for triplets** (analog of Yan Eq. 2-4):
- D_R: sum_j(1 - <q_human^j, q_robot^j>^2) using Dong quaternions (human) and FK quaternions (robot)
- D_ee: ||p_human_fingertip_norm - p_robot_fingertip_norm||, where positions are wrist-relative and normalized by hand_length = ||MCP_middle - wrist||
- Both normalization steps computed on-the-fly from raw XYZ -- no CSV recomputation needed

**Inference (human to robot)**:
```
x_H [21,4] → E_h → z[32] → D_X → [256] → D_r → qpos[24]
```
E_r and E_X not used at inference.

**References**:
- Yan & Lee (2026): framework, losses, similarity metric
- Lee et al. (2023) SAME: GATEnc architecture for E_h
- Entry 28: single subspace for hand
- Entry 35-37: E_h design decisions

**Status**: Implemented, pending training validation

---

## Entry 40 -- 2026-04-22: Training data strategy and FK setup for L_contrastive

**Context**: L_contrastive requires robot poses to form triplets with human poses. No robot demonstration dataset exists. Need FK to compute D_R and D_ee for triplet construction.

**Decision**:
- Robot poses sampled **uniformly at random** from Shadow Hand joint limits every training step
- Poses generated on-the-fly, never stored -- matches Yan exactly
- FK via pytorch-kinematics `build_serial_chain_from_urdf` using `/home/yareeez/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf`
- URDF preferred over MJCF: standard format, matches Yan pipeline

**Triplet construction**:
- For each step: batch = B human frames (HOGraspNet) + B random robot qpos
- Encode all → z_h and z_r in same R^32 latent space
- Compute pairwise S = D_R + omega*D_ee in pose-space to rank similarity
- Sample triplets (anchor, positive, negative) by similarity rank
- L_contrastive = triplet loss on z-space vectors (alpha=0.05)

**Triplet loss does NOT need exact matches**: only relative ordering matters. With large batch, statistically some robot qpos will be functionally closer to a given human pose than others. Model learns to cluster by functional similarity across embodiments.

**Alternatives considered**:
- Robot demonstration dataset: not available, not needed per Yan
- MJCF for FK: would work (pytorch-kinematics supports it) but URDF is cleaner and matches Yan

**Expected impact**: L_contrastive aligns human and robot latent spaces by functional pose similarity without paired data.

**References**:
- Yan & Lee (2026), Section IV-B (Datasets): "sample robot joint configurations uniformly at random... compute FK... immediately discarded after updating"
- Entry 39: full system architecture

**Status**: Proposed

---

## Entry 41 -- 2026-04-22: Postural Control implementation (Segil analog)

**Context**: With 3.5 hours before advisor presentation, needed a working end-to-end demo. Dual-VGAE/Yan retargeting training takes 17-33h on Colab T4 (FK on CPU bottleneck). Decision: implement postural control using abl04 + canonical poses v5 as the thesis contribution for today.

**Decision**: Implement postural control analog to Segil et al. (2014) C3 controller, replacing EMG with vision-based GNN.

**Architecture**:
```
MediaPipe XYZ [21,3] → ToGraph (xyz+AHG+flex, F=24) → abl04 GNN → softmax [28]
                                                                         ↓
                                                          top-2: (k1,p1), (k2,p2)
                                                                         ↓
                                   qpos = (p1·C[k1] + p2·C[k2]) / (p1 + p2)   ← JAT equivalent
                                                                         ↓
                                                           Shadow Hand qpos [24]
```

**Canonical poses**: `shadow_hand_canonical_v5_grasp.yaml` — 27/28 Feix classes defined (medoid of Dexonomy grasps). Missing: "Distal" (class 9) → fallback hand-flat.

**Analog to Segil**:
- Segil C3: EMG RMS → 2D coordinate → JAT (linear transform) → joint angles
- This work: XYZ → GNN → 28-dim probability vector → top-2 weighted interpolation → joint angles
- Interpolation mechanism identical; selection criterion differs (geometric distance vs semantic probability)
- Key advantage over Segil: sensor-agnostic (only RGB camera needed), 27 classes vs 6

**Files**:
- `grasp-robot/postural_control.py`: PosturalController class, xyz → qpos pipeline
- Next: integrate into `grasp-app/mujoco_canonical_demo.py` (replace discrete keyframe lookup with PC interpolation)

**MLP baseline**: running in parallel to validate GNN contribution. abl04 GNN target: 71.07%. MLP (504→256→128→28) val_acc ~67% at epoch 23 — GNN leads by ~4 points. Test result pending.

**Status**: postural_control.py implemented and verified. MuJoCo integration pending.

**References**:
- Segil et al. (2014): C3 postural controller, JAT, PC domain
- Entry 14: Dual-VGAE (future work, after training validates latent space)
- abl04: best current classifier, 71.07% test acc, F1=0.657

---

## Entry 42 -- 2026-04-24: Bug fix -- Shadow FK quats were absolute, not parent-relative

**Context**: Retargeting training (Entry 40) showed Lc=0.0000 from epoch 2 onward and Ltemp constant throughout. Root cause investigation via comparison with Yan & Lee (2026).

**Bug**: `shadow_fk.py` computed joint quaternions as absolute rotations in palm frame:
```python
# WRONG (before fix)
R_palm_inv @ ret[link].get_matrix()[:, :3, :3]   # absolute in palm frame
```

Dong kinematics (human quats) uses parent-relative convention: MCP = R_i^0 (rotation relative to wrist/parent), PIP = R_j^{j-1} (relative to MCP parent), DIP = R_{j+1}^j (relative to PIP parent). This is the standard joint-by-joint representation -- the same convention SMPL uses for whole-body, which is exactly why D_R works in Yan.

The mismatch: human PIP quat at 45 deg flex with MCP straight = [cos(22.5), sin(22.5), 0, 0]. Robot PIP quat (old) = absolute in palm = accumulated rotation including MCP and all structural offsets. These produce very different numbers for the same physical pose.

**Consequence**: D_R(human, robot) was always large and noisy regardless of semantic similarity. Cross-embodiment pairs never became positives in triplet mining. Both human and robot anchors found same-embodiment positives trivially. Lc collapsed to 0 after epoch 1 (when L_rec=5 forced z_r into a compact cluster, making human-robot latent gap large).

**Fix**: Added `JOINT_PARENT_LINKS` table to `shadow_fk.py` and changed `forward()` to compute parent-relative quaternions:
```python
# CORRECT (after fix)
R_cache[parent].transpose(-1, -2) @ R_cache[child]   # parent-relative
```

Verified: FFJ3 at 45 deg flex returns [0.924, 0.383, 0, 0] regardless of FFJ4 value -- correct parent-relative behavior. Zero pose returns identity for all finger joints (non-identity for LFJ5/THJ5 which have structural URDF offsets, also correct).

**Why Yan does not have this problem**: Yan uses HumanML3D/SMPL where human joints are local/parent-relative by definition. Robot FK via PyTorch-Kinematics also yields parent-relative rotations when computed correctly. Our implementation had an error in the reference frame for non-palm-parent joints.

**Files changed**: `grasp-model/src/grasp_gcn/network/shadow_fk.py`

**Expected impact**: D_R cross-embodiment now meaningful. Cross-embodiment triplet positives should appear. Lc should be nonzero throughout training.

**Status**: Fixed. Re-training needed to validate.

## Entry 43 -- 2026-04-24: Stage 2 -- Dong-style kinematics applied to robot FK positions

**Context**: After fixing the Shadow FK convention bug (Entry 42), D_R cross-embodiment was still broken because parent-relative URDF joint quaternions are not semantically comparable to Dong human quaternions. URDF joints split what is anatomically one MCP orientation into two separate DOFs (abduction + flexion), using different reference frames per robot.

**Decision**: Implement Stage 2 as a generic Dong Block 1 + Block 3 algorithm applied directly to FK link positions (not FK rotation matrices). The key insight: Dong is a position-to-quaternion algorithm -- it does not care whether positions come from a sensor (MediaPipe) or from FK of a URDF. The URDF link origins are pivot points of joints, directly analogous to MediaPipe keypoints (both are 3D positions of anatomical joint centers).

**What Stage 2 does**:
1. Extract XYZ positions of relevant links from `fk_out[link][:, :3, 3]`
2. Block 1 (Dong Eq. 5-7): build palm reference frame from wrist anchor + 3 MCP positions
3. Block 3 MCP (Eq. 20-22): compute finger orientation in palm frame from consecutive positions
4. Block 3 PIP/DIP (Eq. 29-34): compute flexion angles Ry(beta) in parent frame
5. Output: `[B, N, 4]` quaternions in Dong convention (wxyz, w>=0), `N = num_fingers * layers`

**Why this is correct for D_R**: Dong's MCP quaternion encodes "where does this finger point in the palm frame" -- a single 3D orientation combining abduction and flexion. This is the same representation on both human and robot sides, making D_R meaningful. Parent-relative URDF quats were non-comparable because human has 1 MCP DOF per finger while Shadow has 2 (FFJ4 + FFJ3).

**Generic via YAML config**: Each robot needs a YAML mapping file specifying:
- `wrist_link`: anchor link for Block 1
- `frame_index_mcp`, `frame_middle_mcp`, `frame_ring_mcp`: 3 links to build palm frame
- `fingers`: per-finger chains of link names [MCP, PIP, DIP, TIP]

The algorithm (Block 1 + Block 3) is identical for all robots. Only the YAML changes.

**Shadow Hand link mapping insight**: In Shadow URDF, FFJ4 (abduction) and FFJ3 (flexion) are colocated -- `ffknuckle` and `ffproximal` have identical world positions (FFJ3 has `origin xyz="0 0 0"`). The proximal phalanx starts at `ffmiddle` (45mm above). Therefore the correct chain is `[ffknuckle, ffmiddle, ffdistal, fftip]`, not `[ffknuckle, ffproximal, ...]`.

**Allegro Hand link mapping**: Joint axes in URDF determine anatomical role: `axis Z` = abduction (joint_0.0/4.0/8.0), `axis Y` = flexion (joint_1.0/5.0/9.0 = MCP, joint_2.0/6.0/10.0 = PIP, joint_3.0/7.0/11.0 = DIP). Correct chain: `[link_1.0, link_2.0, link_3.0, link_3.0_tip]`.

**Output sizes**:
- Shadow (5 fingers x 3 layers): `[B, 15, 4]` -- isomorphic to human Dong
- Allegro (4 fingers x 3 layers): `[B, 12, 4]`
- Robots with 2 links/finger: `[B, N_fingers*2, 4]` (MCP + PIP only)

**Block 2 (calibration) not needed for robots**: Human Dong requires per-subject calibration because anatomical link lengths vary between people. Robot geometry is fixed in the URDF -- link lengths are constants known from joint origins. No calibration step.

**Validation**:
- Quaternion norms = 1.000000 across 1000 samples (both Shadow and Allegro)
- w >= 0 always enforced
- Zero pose: index/middle/ring/pinky MCP = identity (fingers aligned with palm frame)
- Zero pose: thumb MCP = [0.924, 0, 0, -0.383] (structural offset from URDF, correct)
- PIP/DIP at zero pose = identity (no flexion)

**Files created**:
- `grasp-model/scripts/dong_stage2.py` -- generic Dong Block 1+3 in batched PyTorch
- `grasp-model/scripts/hand_configs/shadow_hand_right.yaml`
- `grasp-model/scripts/hand_configs/allegro_hand_right.yaml`

**Files modified**:
- `grasp-model/scripts/robot_randomizer.py` -- added `run_dong_stage2()` method and `--hand-config` CLI arg

**Status**: Stage 2 implemented and validated for Shadow Hand and Allegro Hand. Stage 3 (subspace selection for D_R) pending.

## Entry 45 -- 2026-04-25: Stage 2 completion -- robot_randomizer integration, normalization, validation

**Status**: IMPLEMENTED and validated.

**What was done**:

1. **Merged dong_stage2.py into robot_randomizer.py** -- Stage 2 geometry functions (`_dong_block1_wrist_frame`, `_dong_block3_mcp`, `_dong_block3_pip_dip`, etc.) live as module-level functions inside `robot_randomizer.py`. `dong_stage2.py` deleted. Reason: in training, Stage 1 (FK) and Stage 2 (Dong) are always called together as one atomic operation per step (Yan 2026, Section Datasets: "at each training step, over 10^5 new robot poses are sampled").

2. **Added `sample_dong()` method** -- single call for training loop:
   ```python
   q, quats, labels, meta = rnd.sample_dong(num_samples, hand_config_path, seed)
   ```
   Returns sampled joint angles, Dong quaternions, labels, and meta with normalized tips.

3. **Added fingertip normalization** -- `meta["tips"]` is `[B, F, 3]` in Dong palm frame, divided by `hand_length`. `hand_length` = `||palm -> mftip||` at q=0, computed once and cached in `self._hand_length`. Equivalent to Yan Eq. 2 normalization (shoulder-relative, divided by arm length).

4. **Validated correctness**:
   - q=0: all PIP/DIP = (1,0,0,0), middle tip norm = 1.0 exactly.
   - FFJ2=pi/2: index_pip = (0.7071, 0, 0.7071, 0) = Ry(90 deg).
   - Shadow hand_length = 19.53 cm (matches datasheet: ~45+25+26 mm finger + ~95 mm palm).
   - D_R (human open hand vs Shadow q=0) = 0.017 mean per joint -- correctly low for matching poses.

**Stage 2 output** (`run_dong_stage2` and `sample_dong`):
- `quats`: `[B, N, 4]` wxyz quaternions (w>=0). Shadow: N=15, Allegro: N=12.
- `labels`: list of strings (`thumb_mcp`, `index_pip`, ...).
- `meta["tips"]`: `[B, F, 3]` fingertip XYZ / hand_length (dimensionless). Shadow: F=5, Allegro: F=4.
- `meta["tip_labels"]`: finger names.
- `meta["hand_length"]`: float (meters).
- `meta["R_wrist"]`: `[B, 3, 3]` Dong palm frame.

**Why tips normalized by hand_length**: Yan D_ee = `||p_A^ee - p_B^ee||_2` where positions are normalized by limb length to be scale-invariant across embodiments. For hands: hand_length = palm to middle fingertip at q=0.

**Robots with fewer fingers/joints**: YAML defines exactly which links exist. Allegro has no pinky -> 12 quats, 4 tips. Stage 3 (pending) will select the common subspace for cross-robot D_R comparison.

**Abduction captured implicitly**: FFJ4 (abduction) moves the MCP knuckle laterally -> `mcp_l` changes -> `block3_mcp` Xi vector changes -> MCP quat encodes abduction. No explicit abduction term needed.

## Entry 44 -- 2026-04-24: URDF link origin clarification

**Question**: When FK returns the transform of a link, what physical point does it correspond to?

**Answer**: The origin of a child link in URDF is placed exactly at the joint that connects it to its parent. There is no distinction between "joint position" and "child link origin" -- they are the same point.

```xml
<joint name="FFJ4">
  <parent link="palm"/>
  <child link="ffknuckle"/>
  <origin xyz="0.033 0 0.095"/>   <!-- ffknuckle origin placed HERE -->
</joint>
```

`fk_out["ffknuckle"][:, :3, 3]` = world position of the origin of link ffknuckle = world position of joint FFJ4 = position of the MCP knuckle.

This holds for every link in URDF: its origin is at the joint connecting it to its parent.

**Consequence for Stage 2**: FK positions of Shadow links are directly analogous to MediaPipe keypoints -- both are 3D positions of anatomical joint centers. The comparison is geometrically valid.

**Citation**: ROS URDF documentation (https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/Building-a-Visual-Robot-Model-with-URDF-from-Scratch.html): "The origin for the child link will be positioned by the joint origin, regardless of the child link's visual origin tag. The joint origin explicitly determines the child link frame location in the kinematic chain."

---
## Entry 47 -- 2026-04-25: Stage 3 complete -- CrossEmbodimentSampler, loaders, and cross_emb package

**Summary**: Stage 3 (cross-embodiment training data pipeline) is complete and validated.

**Files implemented**:
- `grasp-model/scripts/human_loader.py` -- `HumanLoader`: loads `hograspnet_abl11.csv` into RAM, S1 subject split, per-subject hand_length normalization (median ||middle_tip||), temporal pairs via `next_idx` array. 1,015,342 train frames, 98.8% valid temporal pairs.
- `grasp-model/scripts/robot_loader.py` -- `RobotLoader` (formerly `URDFRandomizer`): URDF FK + Dong-style kinematics, samples random joint configs uniformly. Robot-agnostic via hand YAML config.
- `grasp-model/scripts/cross_embodiment_sampler.py` -- `CrossEmbodimentSampler`: single entry point for training loop. Wraps both loaders and assembles the batch. `get_batch(B)` and `get_batch_temporal(B)`.

**Output contract of `get_batch(B)`**:
```
q_r            [B, J]      raw robot joint angles          -> E_r
quats_h        [B, 20, 4]  human Dong quats frame t        -> E_h
quats_h_sub    [B, K, 4]   common joints only              -> D_R human
quats_r_sub    [B, K, 4]   common joints only              -> D_R robot
tips_h_sub     [B, Fc, 3]  common fingers only             -> D_ee human
tips_r_sub     [B, Fc, 3]  common fingers only             -> D_ee robot
common_labels  list[str]
common_fingers list[str]
```

`get_batch_temporal` adds `quats_h_t1 [B,20,4]` and `tips_h_t1 [B,Fh,3]`.

**v_H^hand for dexterous hands**: Yan uses wrist velocity (SMPL has no fingers). For dexterous hands, analog is velocity per fingertip: `v = tips_t1 - tips_t` giving `[B, Fc, 3]`. One velocity vector per finger EE, not a centroid. Robot poses for L_temporal are retargeted (forward pass), not random samples.

**Package reorganization**:
- `grasp-model/src/cross_emb/` -- new package for cross-embodiment network (separated from `grasp_gcn/` classifier)
- Moved: `human_encoder.py`, `cross_embodiment.py`, `robot_embedding.py`, `losses.py`
- Deleted: `shadow_fk.py` (replaced by `RobotLoader`), `retarget_dataset.py` (replaced by `HumanLoader`)

**Pending**:
1. `cross_emb/graph_utils.py` -- `quats_to_batch(quats [B,20,4]) -> PyG Batch` for E_h input
2. Fix `losses.py` -- remove hardcoded Shadow indices, use sampler subspace contract
3. New `train_retarget.py` -- connect sampler + encoders + corrected losses

## Entry 46 -- 2026-04-25: LEAP Hand YAML validation and chain_positions addition

**Context**: LEAP hand YAML was created for Stage 3 testing (cross-robot with fewer fingers than human). Two issues required investigation: (1) q=0 producing non-identity PIP/DIP quaternions, (2) hand_length underestimation due to bent q=0 pose.

**Findings**:

1. **LEAP q=0 is not a flat hand** -- large rpy rotations in URDF joint frames (e.g., rpy="1.57079 1.57079 0") make the finger geometry bent at zero joint angles. This is a mechanical design property of LEAP, not a URDF error or a Stage 2 bug. Dong correctly encodes whatever geometry is present. Non-identity quaternions at q=0 are expected and correct.

2. **`*_tip_head` links are the physical fingertip** -- they have no visual mesh (the mesh is on `fingertip`/`fingertip_2` links), but they mark the distal endpoint ~4.8 cm from the DIP joint in world space at q=0. The `fingertip.obj` mesh bounding box is 5.7 cm long, consistent with this offset. Using `*_tip_head` as TIP in the YAML is correct.

3. **hand_length fixed with path length** -- `_get_hand_length` was computing straight-line distance from palm frame origin to tip. At LEAP q=0 (bent), this gave 13.7 cm instead of the physical ~18.3 cm. Fixed to use path length (sum of segment lengths along middle finger chain). Path length is pose-invariant because each segment length = URDF joint xyz magnitude, fixed regardless of joint angles. Shadow unaffected (q=0 flat -> straight-line = path length).

4. **`palm_lower` as wrist anchor is valid** -- LEAP has no wrist joint. `palm_lower` is the base link (motor housing). It is 3 cm proximal to the MCPs. For Dong Block 1, wrist_link only needs to be a fixed proximal anchor from which to build the palm frame -- anatomical wrist location is not required.

5. **LEAP hand_length = 18.3 cm** -- less than expected ~23 cm (30% larger than human per LEAP paper). Discrepancy explained by `palm_lower` being close to MCPs rather than at wrist level. The 30% larger comparison includes the full actuator housing from wrist mount to tip.

**Added to Stage 2 output**:
- `meta["chain_positions"]`: `{finger: [B, L, 3]}` -- all chain link positions in palm frame, normalized by hand_length. Equivalent to robot landmarks (analogous to 21 MediaPipe XYZ). Enables future D_ee using all keypoints, not just tips. Tips and chain_positions are now normalized in both `run_dong_stage2` and `sample_dong`.

**LEAP YAML**: uses `*_tip_head` as chain endpoints (physical fingertip), `palm_lower` as wrist anchor, 4 fingers (no pinky). Stage 3 will select {thumb, index, middle, ring} as common subspace for human-LEAP D_R.

**Status**: Implemented. LEAP YAML validated against URDF geometry. Shadow Stage 2 remains validated.

---
## Entry 48 -- 2026-04-26: Stage 4 training -- OOM on T4 GPU, pending architecture decision

**Context**: First run of Stage 4 training loop (train_stage1_colab.ipynb) on Colab T4 (14.56 GB VRAM) with B=100k fails OOM in E_h (GATConv). Splitting the 2B forward pass into two separate B-size passes halved memory but still OOM at B=50k. Root cause: GATConv activations (~3.6 GB for two passes) plus 8-layer MLPs called 5 times per step (E_X x2, D_X x3) accumulate ~12 GB of activations for backward.

**Decision**: PENDING. Two options on the table:

1. **Lower B** (e.g. B=20k) -- keeps architecture identical, sacrifices contrastive pool diversity. Quick fix.
2. **Replace GATConv with GCNConv** -- maintains graph structure, drastically reduces memory (no attention weights), allows B closer to Yan's 100k.

**Alternatives considered**: Gradient checkpointing (complex to implement for PyG), chunked mini-batch accumulation (complex), MLP flat encoder like Yan (loses graph structure entirely).

**Expected impact**: Resolution unblocks Stage 4 training.

**References**: Yan et al. 2026 used A4000 (16 GB) with MLP encoder -- no GATConv. GAT specifically was adopted here based on Lee et al. 2023 (SAME: Skeleton-Agnostic Motion Embedding). Pending: review SAME to determine whether GAT attention is essential to their method or can be replaced by simpler graph convolution (GCNConv) without losing the key insight.

**Status**: Resolved -- see Entry 49.

---

## Entry 49 -- 2026-04-26: Memory diagnostics, GAT -> CAM decision, Yan multi-robot training analysis

**Context**: B=20k first step completed on Colab T4 (15.64 GB total, session reports slightly higher than 14.56 GB). Memory diagnostic after step 0:

- Reserved: 11.56 GB
- Free: 4.08 GB
- K=15 common joints (thumb/index/middle/ring/pinky × MCP/PIP/DIP)

Memory breakdown estimated at B=20k:
- MLPs (E_X×2, D_X×3, E_r, D_r): ~9.11 GB -- dominant cost, scales linearly with B
- E_h (GATConv, heads=4, hidden=32): ~1.44 GB -- scales linearly with B
- L_contrastive matrix [N_TRIPLETS, N_TRIPLETS, K=15, 4]: ~1.01 GB at N=2048

Contrastive pool utilization: N_TRIPLETS=2048 out of B2=40k available -- only 5.1% of the batch contributes to the contrastive gradient per step.

Review of SAME paper (Lee et al. 2023): GAT is NOT the core contribution. SAME's skeleton-agnostic property comes from (1) using GNN instead of MLP and (2) the embedding consistency loss. The paper itself calls the architecture "graph convolution networks (GCN)" in the abstract. GAT is an expressiveness enhancement, not the architectural differentiator. Ablation study (Table 1) does not ablate GAT vs GCN -- if GAT were essential, they would have. Prior ablation (camgat01 vs abl04): CAM alone outperformed CAM+GAT (70.46% vs 71.07% F1=0.643 vs 0.657). Evidence against GAT.

**Decision**: Replace GATConv with CAMLayer in E_h. CAM [21,21] learned static adjacency -- same role as GAT (weight neighbor contributions) but static across samples, no per-sample attention weights. Already validated in this domain (abl04). Memory reduction: E_h from ~1.44 GB to ~0.18 GB at B=20k (8x less, no heads multiplier).

Projected memory with CAM at B=20k: ~10.3 GB reserved, ~5.3 GB free.

Safe operating points with CAM (estimated, must be verified empirically):

| B | N_TRIPLETS | MLPs | E_h | Contrastive | Total |
|---|---|---|---|---|---|
| 20k | 4096 | 9.11 | 0.18 | 4.03 | 13.32 GB |
| 25k | 3072 | 11.39 | 0.23 | 2.26 | 13.88 GB |
| 25k | 2048 | 11.39 | 0.23 | 1.01 | 12.63 GB |

Target: B=20k + N_TRIPLETS=4096 as first test. N_TRIPLETS matters more than B for our use case -- see below.

**Why N_TRIPLETS > B for our use case**: Yan uses B=1e5 primarily for robot kinematic space coverage ("ensures diverse and unbiased exposure... results in billions of robot poses per embodiment"). With B=20k × 5000 steps = 100M Shadow Hand poses, coverage of a single 24-DOF hand space is already adequate. Yan covers 4-6 robots with diverse full-body morphologies -- their coverage requirement is much larger. For us, the contrastive gradient quality per step (N_TRIPLETS) is the binding constraint.

**Yan multi-robot training -- clarification**: Yan trains Phase 1 with 4 robots simultaneously (TIAGo++, JVRC, H1, + 1 more), all in the same batch, B=1e5 shared across all embodiments. Phase 2: freeze E_h/E_X/D_X, train only E_r/D_r for new robots (ATLAS, G1, Kinova) -- takes ~15 minutes. Our architecture (E_r/D_r per robot, shared E_h/E_X/D_X) matches this design exactly.

Current choice: Phase 1 with Shadow Hand only. Risk: latent space anchors to Shadow Hand morphology, Phase 2 generalization to Allegro/LEAP/Inspire may be weaker than if Phase 1 included 2-3 morphologically distinct hands. Mitigating factors: (1) all dexterous hands share 4-5 finger topology with MCP/PIP/DIP -- morphological gap is much smaller than Yan's humanoid diversity; (2) hardware constraint (T4) makes multi-hand Phase 1 impractical. Decision: proceed with Shadow Hand Phase 1, empirically test Phase 2 generalization before committing to multi-hand retraining.

**Alternatives considered**: GCNConv (less expressive than CAM, no learned topology), chunked contrastive computation (avoids O(n²) peak but complex to implement), multi-hand Phase 1 training (better generalization but 2-3× memory cost at current B).

**Expected impact**: CAM unblocks B=20k training and allows N_TRIPLETS=4096, improving contrastive gradient quality from 5.1% to ~10.2% of the batch per step.

**References**: Yan et al. 2026 (batch size justification, multi-robot training protocol, Phase 1/2 architecture); Lee et al. 2023 SAME (GAT ablation analysis); abl04/camgat01 (empirical evidence against GAT in this domain).

**Status**: Implemented -- CAM-GNN live in human_modules.py, first training run complete (see Entry 50).

---

## Entry 50 -- 2026-04-26: Stage 4 first training run -- 5000 steps, B=30k, N_TRIPLETS=4096

**Context**: First completed training run after GAT→CAM-GNN migration (Entry 49). Colab T4, B=30k, N_TRIPLETS=4096, N_STEPS=5000, LR=1e-3 constant, CKPT_EVERY=500.

**Observed behavior**:

- Step 0: total=19.26, cont=0.013, rec=3.43, ltc=0.43, temp=0.24 -- all losses active immediately
- GPU RAM at step 0: 14.1/15.0 GB reserved (94% utilization). Allocated ~1.06 GB active tensors, rest is reserved but fragmented
- rec loss decreased from ~3.43 to best ~0.94 around steps 2500-2700
- ltc converged to near zero rapidly (expected -- robot FK is exact, no noise in latent code)
- temp loss plateaued at ~0.31
- cont loss near zero throughout (could indicate collapse or saturation -- not yet diagnosed)
- Periodic loss spikes observed at multiple steps (approx. steps 2000, 2750, 3150, and others). Spikes correspond to checkpoint save events (CKPT_EVERY=500), but causality not established -- could be checkpoint I/O, stochastic batch variance, or memory pressure at 94% GPU utilization
- Final losses at step 4950: total=6.06, rec=1.20, ltc=0.006, temp=0.31
- Final rec (1.20) is worse than best rec (~0.94 at step ~2700) -- model did not sustain improvement after step 2700

**Decision**: Do not train further with constant LR=1e-3. Best checkpoint is step 2500. Before next run, add LR scheduler (cosine decay or step decay after ~2000 steps). This is the most actionable hypothesis for the degradation after step 2700 -- but no other causes have been ruled out.

**What remains unknown**:
- Whether spike pattern is harmful or merely stochastic
- Whether cont~0 reflects embedding collapse or appropriate saturation given Shadow Hand Phase 1 only
- Whether step 2500 checkpoint produces usable retargeting -- not yet evaluated

**Next steps**:
1. Add cosine LR decay to training notebook
2. Load step 2500 checkpoint, run retargeting on a held-out pose, visually inspect result
3. If retargeting quality is acceptable, proceed to Phase 2 (Allegro/LEAP/Inspire)

**Alternatives considered**: Continue from step 4950 checkpoint (rec is worse than step 2500, not preferred); reduce N_TRIPLETS to lower memory pressure and allow higher B (would require re-running from scratch).

**References**: Yan et al. 2026 (training protocol); Entry 49 (architecture and hyperparameter selection).

**Status**: Complete -- training run documented. Step 2500 checkpoint not yet evaluated.

---

## Entry 51 -- 2026-04-27: Post-training test evaluation plan

**Context**: Yan et al. evaluate retargeting quality post-training on a held-out test set (80/20 split on HumanML3D), reporting NDS and NVS metrics. We have no equivalent evaluation yet -- training loss alone does not measure whether fingertips of the robot actually follow the human hand.

**Decision**: After each training run completes, run a dedicated test evaluation script. Protocol:

1. Hold out 20% of `hograspnet_dong.csv` subjects (subject-level split to avoid data leakage)
2. Load best checkpoint
3. Inference: `q_r_hat = D_r(D_X(E_h(quats_h)))` on test poses
4. FK: run `robot_loader` FK on `q_r_hat` to get robot fingertip positions
5. Metric: normalized fingertip distance similarity (NDS analog) -- fingertip positions in wrist-local frame, normalized by hand span

This is post-training only. No val loop during training.

**Alternatives considered**: Val loop during training (adds overhead, not needed since we are not doing early stopping based on val loss yet); loss-only evaluation (does not measure geometric retargeting quality).

**References**: Yan et al. 2026 Sec. IV-A (NDS/NVS definition); Entry 50 (training run, motivating need for quality metric).

**Status**: Proposed -- implement after next training run completes.

---

## Entry 52 -- 2026-04-27: Embedding collapse diagnosis and training fixes

**Context**: Entry 50 reported `cont~0` throughout training. Loaded the step 2500 checkpoint and ran a retargeting probe -- Shadow Hand output showed no meaningful pose variation across different human inputs (fingers near rest pose regardless of input). Inspected latent variance directly: z.std()~0.015, range ±0.27 vs theoretical ±1.0 from Tanh output. Confirmed embedding collapse.

Root cause: margin=0.05 too small relative to typical latent distances. With z in [-1,1]^16, Euclidean distances between distinct poses are typically 1-3. At margin=0.05 nearly all triplets already satisfy d_pos - d_neg + margin < 0 by chance -- L_cont saturates to 0 immediately, E_h receives no separation gradient.

With L_cont dead, the only gradients reaching E_h are:
- L_ltc: round-trip consistency (z_t ≈ E_X(D_X(z_t))) -- does not pressure pose separation
- L_temp: fingertip velocity signal -- weak and indirect

L_rec does not touch E_h at all (only trains E_r/E_X/D_X/D_r on the robot side). Result: E_h learned to produce z that survive the round-trip but collapsed to a small region of latent space.

**Changes applied**:

1. **margin 0.05 → 0.3**: Triplet margin increased to force meaningful separation. At margin=0.05 with z_dim=16 and Tanh output, nearly all triplets satisfy the constraint by chance. At 0.3 the loss stays active through training.
2. **Cosine LR decay**: Added `CosineAnnealingLR` starting after `lr_warmup=500` steps, decaying 1e-3 → 1e-5. Addresses the degradation observed after step 2700 in Entry 50, where constant LR likely caused overshooting.

**Code locations**: Both fixes live in `grasp-model/src/cross_emb/train_cross_emb.py` (single source of truth). Colab notebook `train_stage1_colab.ipynb` delegates to the script via `subprocess.run` -- no inline training loop remains.

**Expected effect**: L_cont should be >0 at step 0 and decrease gradually. z.std() should reach ~0.3-0.5 range by step 2000. rec loss should sustain improvement past step 2700 due to LR decay.

**What remains unknown**: Whether margin=0.3 is optimal or overshoots (too aggressive separation). If L_cont dominates and rec degrades, lower lambda_c or reduce margin.

**Next steps**:
1. Retrain in Colab with these fixes
2. At step 50, verify L_cont > 0
3. After run, evaluate with NDS analog (Entry 51)

**Status**: Implemented -- not yet validated by a training run.

---

## Entry 53 -- 2026-04-27: Stage 1 run 2 results and failure analysis

**Context**: Second training run with margin=0.3, cosine LR decay, B=20k, N_STEPS=5000. Tested final checkpoint (step 4972, rec=0.70) in live_retarget.py with webcam.

**Observed behavior**:
- Shadow Hand detects open/close motion of human hand — flexes when human flexes
- Never produces a recognizable pose (no fist, no pinch, no tripod)
- Initial pose with human hand open is bent/semi-closed with colliding fingers
- Movement is smooth but non-specific

**Problem 1 — E_h dataset gap**:
HOGraspNet contains only grasping poses — zero open-hand frames. E_h maps an open human hand near the centroid of all grasp embeddings → Shadow Hand defaults to semi-closed. This is a dataset problem, not a training bug. Fix requires either augmenting training data with open-hand poses or a different dataset.

**Problem 2 — rec=0.70 too high**:
Shadow Hand joints have mean range 1.337 rad. rec=0.70 is 52% of the mean joint range — far too large to produce defined poses. Causes finger collisions and inability to reach pose extremes (e.g., full fist). Target for functional retargeting: rec < 0.1 rad (~7% of mean range). The floor of rec=0.70 was reached around step 2650 and did not improve further despite cosine LR decay to step 5000. This is a training convergence problem, independent of the human dataset — the robot autoencoder (E_r → E_X → D_X → D_r) trains on random uniform joint poses.

**What remains unknown**:
- Why rec plateaued at 0.70. Hypotheses: (a) D_r MLP capacity too small, (b) shared_dim=1024 bottleneck, (c) E_X/D_X adding unnecessary distortion for the robot-only path, (d) random uniform sampling produces too many unreachable configurations.
- Whether reducing the robot path (E_r → D_r directly, bypassing E_X/D_X) would produce lower rec.

**Root cause hypothesis for Problem 2**:
Yan uses 5 decoupled subspaces (LA, RA, TK, LL, RL), each with z_dim=16. Each subspace handles ~2-7 DOF — so 16 dims is more than sufficient per subspace. We use a single subspace with z_dim=16 for 24 DOF — the bottleneck forces lossy compression and the decoder cannot reconstruct with sufficient precision.

**Proposal 1 — Increase z_dim**:
Increase z_dim from 16 to 64 (or 32) as a first experiment to validate the bottleneck hypothesis. This is the minimal change: one argument, no architectural refactor. If rec drops significantly, the hypothesis is confirmed.

**Proposal 2 (future) — Decoupled subspaces by finger**:
If z_dim increase is not sufficient, adopt Yan's decoupled design adapted to hands: 5 subspaces (thumb, index, middle, ring, pinky), each with z_dim=16. E_h, E_X, D_X remain single networks but output 5×16=80 dims, partitioned per finger. Contrastive loss applied independently per finger subspace. This mirrors Yan's architecture exactly for the hand domain.

**Proposal 3 — Fix robot sampler: filter colliding poses**:
Empirical measurement (5000 random poses, same `np.random.uniform(lower, upper)` method as current sampler): 83.8% of poses have at least one self-collision in MuJoCo (`data.ncon > 0`). Only 16.2% are physically valid. The decoder is forced to learn to reconstruct physically impossible configurations, which contaminates the latent space and raises the rec floor artificially. Fix: after sampling, run `mj_forward` and reject poses with `ncon > 0`, resample until B valid poses are collected. This adds overhead per step but is the correct fix — training on invalid poses is the likely dominant cause of rec=0.70.

**Proposal 4 — Hybrid contrastive: label + geometry**:
Current S = D_R + D_ee is weakly discriminative for hand grasps — different Feix classes can have similar rotation and fingertip distances. Two extremes are both wrong: (a) geometry only (current) → cont~0, no inter-class separation; (b) label only (SupCon) → 28 discrete clusters, intra-class variability collapses, embedding loses continuity — interpolating between poses would produce discrete jumps.

Correct approach: hybrid. Feix label decides which class is positive vs negative (inter-class separation guaranteed). Within the chosen class, S geométrica selects the hardest positive (intra-class continuity preserved). For human-robot and robot-robot pairs where no label exists, S geométrica applies as now.

This gives: semantic separation across classes + continuous structure within each class + cross-embodiment alignment via geometry. Labels never enter E_h forward pass — the latent space remains continuous.

**Proposal 5 — Labeled robot poses via Dexonomy**:
Dexonomy generates physically valid, collision-free Shadow Hand qpos for each Feix grasp type. Using these instead of random uniform poses would: (a) eliminate colliding poses (Proposal 3 already addresses this but Dexonomy gives semantically labeled poses too), (b) enable SupCon for the robot side as well (same grasp_type = positive across embodiments). Requires downloading Dexonomy, generating Shadow Hand poses per class, and modifying the sampler to load from file instead of random FK. More complex but addresses Problems 1, 2, and 3 simultaneously.

**Priority order**:
1. Proposal 3 (filter collisions) — highest impact, minimal code change, addresses likely dominant cause of rec=0.70
2. Proposal 1 (z_dim=64) — validate bottleneck hypothesis
3. Proposal 4 (richer S) — improve contrastive signal if rec is fixed but cont stays dead
4. Proposal 2 (decoupled subspaces) — architectural refactor, only if 1+3 insufficient
5. Proposal 5 (Dexonomy) — most complex, highest potential

**Next step**: Implement Proposal 3 (collision filter in sampler) + Proposal 1 (z_dim=64) together in next run.

**Status**: Full analysis complete. Next experiment addresses root cause of rec floor.

## Entry 54 -- 2026-04-28: Hypothesis -- anatomy-informed S for hand contrastive loss

**Context**: If Proposals 1+3 (z_dim=64 + collision filtering) do not activate the contrastive loss (cont still ≈0), the likely cause is that S = D_R + D_ee is too weak to discriminate hand grasps. Different Feix classes can score similar S values because D_R weights all joints equally and D_ee uses raw fingertip distance -- neither captures the axes of variation that actually separate grasp types.

**Hypothesis**: S can be made more discriminative while preserving Yan's geometric continuity (no labels, no discrete clusters) by weighting joints according to their anatomical discriminative power for hand grasps.

**Proposed modification**:
```
S = α * D_thumb + β * D_fingers + γ * D_ee
```
where:
- D_thumb = D_R restricted to thumb joints (THJ5, THJ4, THJ3) -- opposition axis
- D_fingers = D_R restricted to remaining joints
- D_ee = fingertip position distance (same as Yan)
- α > β -- thumb weighted higher, justified by Feix taxonomy

This is a geometric metric, not label-based. Latent space remains continuous. The modification adapts Yan's S to the hand domain by recognizing that thumb opposition is the primary axis separating Feix classes -- morphology-agnostic since the principle applies to any hand (human or Shadow Hand).

**Candidate references**:
- **Feix et al. (2016)** -- GRASP taxonomy. Defines the 28 classes via virtual finger opposition (VF1=thumb vs VF2=remaining fingers) and contact type. Directly informs which joints are most discriminative for S.
- **Stival et al.** -- hand pose analysis, potential source for joint importance ranking.
- **Jarque-Bou et al.** -- PCA of hand grasps, identifies principal axes of postural variation. Could inform α/β weight ratios empirically.

**Constraint for any S modification**: To preserve Yan's navigable latent space, any modification to S must satisfy three conditions simultaneously:
1. **Continuous** -- no thresholds, no if/else by class. Small change in pose → small change in S.
2. **Geometric** -- derivable from joint positions/rotations only, not from external labels.
3. **Cross-embodiment compatible** -- computable from both human Dong quaternions and Shadow Hand FK positions.

If all three hold, the latent space maintains Yan's interpolation property. The open question (to be informed by Feix/Jarque-Bou/Stival) is which geometric variables of the hand satisfy these three conditions while also discriminating between grasp types.

**When to pursue**: Only if cont ≈ 0 persists after running with Proposals 1+3. Do not implement speculatively.

**Status**: Hypothesis. Pending experimental validation of Proposals 1+3.

## Entry 55 -- 2026-04-28: Run 3 results -- NPZ + z_dim=64

**Config**: B=20k, N_STEPS=5000, z_dim=64, margin=0.3, cosine LR decay. Robot sampler: VALID_NPZ (10M collision-free poses, 15.7% acceptance rate).

**Results**:
- rec: 3.23 → 0.11 (97% reduction). Previous floor was 0.70 -- NPZ + z_dim=64 resolved reconstruction completely.
- cont: ≈ 0 throughout entire run. Confirmed: S = D_R + D_ee does not discriminate hand grasps.
- ltc: 3.48 → 0.006. Converged.
- temp: ~0.32 stable.

**Live test observations**:
- Shadow Hand poses are now physically valid -- no finger collisions or impossible configurations.
- Motion is smooth. Hand opens and closes in response to human input.
- Problem 1 -- only flexion: no grasp-type differentiation. Fist and pinch produce similar output. cont=0 means the latent space has no semantic structure -- E_h maps all poses to the same region regardless of grasp type.
- Problem 2 -- no rest pose: hand never reaches open/neutral configuration. Always partially flexed. Same dataset gap as Run 2 (HOGraspNet contains only grasping poses, zero open-hand frames).

**Conclusions**:
- Proposals 1+3 (z_dim=64 + collision filtering) fully resolved the reconstruction problem.
- cont=0 is now the sole remaining problem. Root cause confirmed: S is too weak to discriminate hand grasps.
- Next step: Entry 54 -- modify S (anatomy-informed weighting or alternative geometric features).
- Dataset gap (no open-hand poses) is a separate problem, lower priority until cont is resolved.

**Status**: Run complete. Proceeding to S modification (Entry 54).

## Entry 56 -- 2026-04-29: Decoupled latent subspaces -- architecture decision and design

**Context**: cont=0 confirmed in Run 3. Two candidate fixes discussed: (A) modify S with anatomical weights (Entry 54), (B) decouple latent space by body segment following Yan exactly. Decision: implement B first -- it is the approach Yan validates empirically, not a new hypothesis. A is a novel proposal without validation.

**Decision: 3 subspaces (thumb / index / ulnar)**

Justified by Jarque-Bou et al. (2019) kinematic synergy analysis (77 subjects, 20 grasps, NinaPro DB1/DB2/DB5):
- Synergy #1 (most prevalent, most variance): MCP flexion fingers 3-5 + PIP flexion fingers 2-5. Index PIP couples with ulnar fingers -- index is not fully independent.
- Synergy #3: thumb CMC/MCP/IP acts independently of all other fingers.
- Thumb independence is robust. Full index independence is not supported by the dominant synergy.

2 subspaces (thumb + digits) would be most faithful to the dominant synergy. 3 subspaces (thumb + index + ulnar) cuts synergy #1 at the index PIP -- the index PIP will show redundant signal between z_index and z_ulnar. Accepted: 3 subspaces provides higher S granularity, which is the primary goal, and the redundancy is minor.

**Subspace joint assignment:**

Three subspaces: thumb / precision / support. Justified by functional independence analysis:
- Thumb: maximum independence — opposition axis, CMC 2-DOF
- Precision (index+middle): index has high independence; middle is the ambiguous case. Placing middle with index (not ulnar) avoids the contrastive loss penalizing middle for deviating from ring+pinky in grasps like tripod where middle acts with index, not ulnar.
- Support (ring+pinky): ring and pinky never act independently from each other across any Feix grasp. Always co-vary. Placing together is unambiguous.

Human side (21 CAM-GNN nodes: 0=wrist identity, 1-20=Dong joints):
- z_thumb:     nodes [1,2,3,4]   — thumb_mcp/pip/dip/tip
- z_precision: nodes [5..12]     — index_mcp/pip/dip/tip + middle_mcp/pip/dip/tip
- z_support:   nodes [13..20]    — ring_mcp/pip/dip/tip + pinky_mcp/pip/dip/tip

Wrist (node 0): part of CAM graph (propagates info to all nodes), no projection head. Human wrist = identity quaternion (Dong reference frame). Robot WRJ1-2 excluded from S_k, participates in L_rec via E_r/D_r.

Fingertip labels per subspace (for D_ee):
- z_thumb:     ["thumb"]
- z_precision: ["index", "middle"]
- z_support:   ["ring", "pinky"]

Robot side (Shadow Hand, 24 DOF):
- z_thumb:     THJ1-THJ5 (filtered by label prefix "thumb_")
- z_precision: FFJ1-4 + MFJ1-4 (prefixes "index_", "middle_")
- z_support:   RFJ1-4 + LFJ1-5 (prefixes "ring_", "pinky_")

Wrist joints (WRJ1-2): excluded from all S_k. Human has no wrist quaternion (muñeca = Dong reference frame, always identity). WRJ encoded in E_r → 1024-dim and reconstructed by D_r via L_rec. No contrastive signal for wrist -- acceptable because wrist angle encodes approach direction, not grasp type.

**Similarity metric per subspace (all subspaces):**

All three subspaces use:
```
S_k = D_R_k + D_ee_k
```
where D_R_k sums over joints in subspace k only, D_ee_k uses fingertips of subspace k only. Same formula as Yan's arm subspaces -- fingertips are the end-effectors for hands, directly analogous.

**Architecture changes (following Yan exactly):**

Current: E_h → z [B, z_dim], E_X → z_r [B, z_dim], one contrastive loss over all 24 joints.

New:
- E_h: [B, Nh, 4] → [z_thumb, z_index, z_ulnar], each [B, z_dim]. One MLP, 3 output heads (K*z_dim output dim).
- E_X: [B, 1024] → [z_thumb_r, z_index_r, z_ulnar_r], each [B, z_dim]. One MLP, 3 output heads.
- L_cont = L_cont_thumb + L_cont_index + L_cont_ulnar. Each computed independently with its own S_k.
- D_X: concat([z_thumb, z_index, z_ulnar]) → [B, 1024]. Input dim 3*z_dim. Rest unchanged.
- E_r, D_r, L_rec, L_ltc, L_temp: unchanged.

z_dim per subspace = 64 (same as Run 3). Total latent dim = 3 * 64 = 192. Yan uses 16 per subspace (5 body parts) -- our z_dim is larger because hands have more complex intra-subspace structure than a 2-joint arm.

**What changes in code:**
- `human_modules.py`: E_h output dim 64 → 192 (3 heads)
- `shared_modules.py`: E_X output dim 64 → 192; D_X input dim 64 → 192
- `train_cross_emb.py`: split z into 3 subspaces, compute S_k and L_cont_k per subspace, sum
- `cross_embodiment_sampler.py`: return per-subspace joint/tip indices for S_k computation

**Evidence base:**
- Yan et al. (2026): decoupled subspaces validated empirically for cross-embodiment full-body retargeting. Directly applied here.
- Jarque-Bou et al. (2019): kinematic synergy analysis justifies thumb independence (synergy #3) and ulnar grouping (synergy #1). Informs subspace boundaries.
- Yan (Discussion): explicitly states hand retargeting is future work -- this entry is the hand-domain adaptation.

## Entry 57 -- 2026-04-29: Run 4 results -- decoupled subspaces, qualitative evaluation, and Run 5 plan

**Run 4 config**: B=20000, n_steps=5000, z_dim=64, shared_dim=1024, margin=0.3, lambda_c=10, lambda_rec=5, lambda_ltc=1, lambda_tmp=0.1, n_triplets=2048. Checkpoint: stage1_best_run4_subspaces.pt.

**Quantitative**: cont=0.001-0.012 (non-zero from step 0, decoupling fixed collapse). L_rec=3.19 at step 0, ~0.63 at step 1200. Training did not crash.

**Qualitative (live retargeting via live_retarget.py)**:
- Thumb responds better than all prior runs -- subspace isolation working.
- Some fingers always bent regardless of human input -- D_r has mean-pose bias toward grasps.
- Cannot fully close hand to fist -- range of motion not fully covered.
- Wrist (WRJ1/WRJ2) moves -- incorrect, human signal is in wrist frame (Dong), wrist rotation not encoded.

**Root cause analysis:**

1. **WRJ spurious movement**: D_r predicts all 24 DOF including WRJ1/WRJ2. But E_h receives Dong quaternions which are wrist-relative -- no wrist angle info in input. D_r generates noise in WRJ channels. Fix applied in inference: `q[:, 0:2] = 0.0` in `retarget.py`. Permanent fix for training: zero WRJ in valid_robot_poses so D_r learns WRJ=0 is canonical.

2. **Fingers always bent (bias toward closed pose)**: HOGraspNet contains only grasp frames -- all poses are bent configurations. D_r learned that the average output is a semi-closed hand. When the latent embedding is ambiguous, D_r reverts to mean = grasp. No open-hand signal in human training data. This is the primary qualitative failure.

3. **Cannot fully close**: Likely two sub-causes: (a) 5000 steps insufficient -- L_rec was still ~0.63 and descending at step 1200, model did not converge; (b) D_r capacity is not the bottleneck, but the extreme closed-hand poses are underrepresented in valid_robot_poses (random uniform sampling may undersample extreme flexion configs due to collision filtering).

4. **Asymmetric finger response**: Some fingers respond more than others. Likely due to imbalance in S_k triplet mining -- support subspace (ring+pinky) has fewer discriminative grasps in HOGraspNet and fewer fingertip degrees of freedom in the Shadow Hand's LFJ (little finger has 5 joints vs 4).

**Run 5 proposed config:**

Training changes:
- n_steps: 5000 → 20000 (model was still converging at step 1200/5000)
- B: 20000 → 30000 (more triplet diversity per step, if GPU VRAM allows on T4; fallback to 20000)
- n_triplets: 2048 → 4096 (more mined triplets per subspace per step)
- WRJ fix in training: zero WRJ1/WRJ2 columns in q_r batch before L_rec (or use separate mask). Prevents D_r from wasting capacity on wrist prediction.
- lambda_rec: 5.0 → 8.0 (finger bias is the main failure -- increase reconstruction pressure)

Augmentation changes (two-source batch):
- Source A (HOGraspNet): all current losses -- L_cont + L_rec + L_ltc + L_temporal
- Source B (open-hand): L_cont + L_rec only (no temporal pairing, no L_ltc)
- Open-hand human poses: 11K Hands dataset (Afifi 2019) -- 11,076 static images, real pose variety. Pipeline: MediaPipe → Dong precompute → CSV. No sequence metadata available, so temporal pairs duplicated (velocity=0) and excluded from L_temporal.
- Open-hand robot poses: already partially covered by valid_robot_poses random sampling. Augment by adding extended-finger configurations with low flexion angles explicitly to the valid poses pool.
- Closed fist: HOGraspNet power grasps cover semi-closed. Add fully-closed synthetic robot poses (all flexion joints at upper limit) as fixed anchor samples -- these anchor D_r to learn full flexion range.

**Decision on augmentation order**:
- 11K Hands (open-hand) is the highest-priority gap. Fingers always bent = human encoder never saw extension.
- Fully-closed fist is lower priority -- power grasps in HOGraspNet are close enough to closed. Add only if "cannot fully close" persists after open-hand augmentation and more steps.

**WRJ masking in training (proposed)**:
In `train_cross_emb.py`, after sampling q_r batch:
```python
q_r[:, 0:2] = 0.0  # zero WRJ before L_rec so D_r does not learn wrist prediction from human signal
```
Or equivalently, zero WRJ columns in valid_robot_poses.npz at generation time so the dataset never contains non-zero WRJ.

**Next concrete steps**:
1. Precompute 11K Hands Dong features (MediaPipe → `precompute_dong_features.py` → `11k_hands_dong.csv`)
2. Add two-source batch logic to `cross_embodiment_sampler.py` or a new `HumanAugLoader`
3. Zero WRJ in training
4. Run 5 with above config on Colab T4

**Status**: Run 4 evaluated. Issues diagnosed. Run 5 plan above.

**Status**: Design complete. Implementation next (Run 4).

## Entry 58 -- 2026-04-29: Replace all-pairs contrastive mining with explicit random triplets

**Context**: The current Stage 1 contrastive block already uses Yan-style random candidates per anchor, but it still builds an all-pairs physical similarity matrix `S_k [n_triplets, n_triplets]`. This matrix is inherited from the previous hard-mining implementation (`argmin` positive / `argmax` negative). With the current random-candidate logic, only two entries of `S_k` are used per anchor.

**Yan reference**: Yan et al. describe "randomly sampled triplets" from batches containing human and robot poses. The paper defines the physical metric `S_k = D_R + w D_ee` and margin `alpha=0.05`, but does not require all-pairs mining.

**Decision**: Replace all-pairs `S_k` construction with explicit random triplets:
- sample `n_triplets` anchors from the human+robot pool;
- sample two non-self candidates per anchor;
- compute only `S(anchor, cand_a)` and `S(anchor, cand_b)`;
- assign positive = lower `S`, negative = higher `S`;
- keep the same triplet loss in latent space.

**Metrics preserved**:
```
D_R   = mean_j(1 - <q_a^j, q_b^j>^2)
D_ee = mean_tip(||tip_a - tip_b||_2)
S     = D_R + D_ee
```

**What does not change**:
- no changes to GNN/encoders;
- no changes to decoder;
- no changes to FK;
- no changes to HaGRID mixing;
- no changes to `margin=0.05`;
- no changes to `lambda_c`, `lambda_rec`, `lambda_ltc`, `lambda_tmp`.

**Expected effect**: The contrastive miner becomes O(`n_triplets`) instead of O(`n_triplets^2`) for physical similarity computation. This should allow much larger explicit triplet counts, improving per-step coverage of the 40k human+robot pool without allocating all-pairs matrices.

**Logging change**: `metric_stats` should report statistics over the sampled anchor-candidate pairs actually used by the loss, not over all-pairs similarities.

**Implementation status**: Implemented in `train_cross_emb.py`.

## Entry 59 -- 2026-04-29: Treat `n_triplets` as a legacy memory cap, not a method hyperparameter

**Context**: Yan et al. report batch size and randomly sampled triplets, but do not define a separate `n_triplets` hyperparameter. Our `n_triplets` existed because the older all-pairs physical similarity matrix made full-batch triplet coverage too expensive.

**Decision**: The default Stage 1 behavior is now full-pool triplet coverage per subspace:
- `B` still means `B` human poses and `B` robot poses;
- the contrastive pool is therefore `2B`;
- if `--n_triplets` is omitted or `<= 0`, the miner uses all `2B` pool entries as anchors;
- if `--n_triplets` is provided, it is only a legacy/debug memory cap.

**Notebook behavior**: `train_stage1_colab.ipynb` sets `N_TRIPLETS = None`, so the command omits `--n_triplets` and uses the full human+robot pool.

**What does not change**:
- anchors and candidates are still sampled from the mixed human+robot pool;
- positive/negative assignment still uses physical `S_k`;
- triplet loss is still applied in latent space;
- no changes to GNN, FK, losses, margin, or HaGRID mixing.
