# Eigengrasp Generation Procedure (per hand)

Standardized procedure to build a robot hand's eigengrasp (synergy) basis for the
cross-embodiment retargeter. The **method** (steps + acceptance criteria) is the
same for every hand; only the **dataset adapter** (how grasps are loaded) changes,
because different grasp datasets have different structure.

## Goal

Produce `eigengrasp_<hand>.npz` (PCA synergy basis) whose sampling range spans a
fully **open** hand and a **closed** fist. The retargeter samples robot training
poses from this basis (`generate_valid_robot_poses.py`); if the basis cannot reach
open/close, the robot never learns to open/close.

## The contract (same for all hands)

A basis is accepted only if, projecting the open and close anchor poses onto it:
1. Reconstruction error is low at k=9 PCs (extremes representable by the synergies).
2. Anchor coefficients fall **inside `[coeff_p01, coeff_p99]`** — the sampling
   range. Otherwise `Uniform(p01, p99)` never generates the extreme.
3. GATE A passes (joint order verified by per-column span).
4. GATE B passes (PCA round-trip near-exact).

## Steps

1. **Get a grasp dataset for the hand.**
   - Multi-hand ready data: BODex (`huggingface.co/datasets/JiayiChenPKU/BODex`,
     `synthesized_grasps/<hand>.tar.gz`) — covers shadow/allegro/leap. Place in
     `robot/hands/<hand>/datasets/raw/BODex_<hand>.tar.gz`.
   - Shadow uses Dexonomy instead (it has Feix grasp-type labels).
   - Datasets are gitignored (`robot/**/datasets/`); they travel via Drive, not git.

2. **Build the basis with phase balancing + synthetic anchors.**
   - Use ALL grasp phases, not just the final grasp. BODex/Dexonomy store
     `pregrasp_qpos` (hand approaching = more OPEN), `grasp_qpos`, `squeeze_qpos`
     (tighter). Equal rows per phase = automatic 1:1:1 phase balance → the basis
     spans open->close from the data.
   - **Downsample** real data to a modest size (`--max-rows-per-phase`, default
     100k/phase). This keeps the appended synthetic anchors a meaningful fraction
     (~5-15%) so they land inside p01/p99. (Shadow does the same via
     `rows_per_weight_unit=9890`; synthetic ~6.5% of its total.)
   - **Append synthetic open/close anchors.** The data never reaches a flat-open
     hand (pregrasp keeps PIP joints slightly bent) nor an empty fist (grasps are
     around objects), so synthetic anchors are required for those extremes.

3. **Generate the synthetic anchors** (`generate_synthetic_open_close_<hand>.py`):
   - **open** = parametric, all flexion=0 (flat hand). Automatic, no tuning.
   - **close** = a fist target, collision-filtered against the hand MJCF. The fist
     thumb is hand-specific (thumb kinematics differ per hand): tune it ONCE with
     `mujoco_joint_tuner.py --robot <hand> --thumb-only`, press `p`, paste the
     values into `CLOSE_TARGETS`. (~10 min, the only manual step.)
   - Jitter (~6% of joint span) around each target → a cloud (PCA needs variance).

4. **Verify the basis** against the contract above (recon error + p01/p99 coverage
   + gates). Eyeball with `mujoco_eigengrasp_viewer.py --robot <hand>` (PC1 should
   sweep open<->close).

5. **Vendor the hand MJCF** if not present:
   `third_party/mujoco_menagerie/<hand>/` (sparse-clone from
   google-deepmind/mujoco_menagerie). Needed by the collision filter and viewers.

## Tooling

- `build_<hand>_eigengrasps.py` — load phases, downsample, inject synthetic, PCA.
- `generate_synthetic_open_close_<hand>.py` — open/close anchors + collision filter.
- `mujoco_joint_tuner.py` — interactive per-joint pose crafting (tune the fist).
- `mujoco_valid_poses_viewer.py` — browse raw poses (arrows + PageUp/Down).
- `mujoco_eigengrasp_viewer.py` — drive the PCA synergies (validate the basis).

## Per-dataset adapter notes

| hand | dataset | grasp-type balance | notes |
|------|---------|--------------------|-------|
| shadow | Dexonomy | yes (28 Feix classes) | class + phase balanced; basis validated, Run 20 trained on it |
| allegro | BODex | no (object-organized) | **PENDING**: currently built from MultiDex (final grasps only, no phases) -> likely fails the contract; rebuild from BODex 3-phase |
| leap | BODex | no (object-organized) | done (2026-06-13): 3-phase + downsample + 2 tuned fists |

Dexonomy supports allegro/leap too, but regenerating its data needs a GPU
(RTX 3090) + a human-annotated template per hand x grasp type + the full synthesis
pipeline — out of scope. BODex (same authors, Dexonomy's trajectory-synthesis
component) provides ready multi-hand data; grasp-type diversity comes from its
many objects instead of Feix labels.

## Status

- shadow: done (Dexonomy, validated).
- leap: done (BODex 3-phase, validated numerically + visually).
- allegro: **pending** — rebuild from BODex 3-phase, then regenerate valid_poses
  and retrain the allegro retargeter (its current model was trained on the old basis).
