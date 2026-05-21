# Architecture

GraphGrasp is organized by execution pipeline responsibility.

## Pipeline Domains

- `human/` owns observation and human-hand representation: perception backends, Dong kinematics, canonicalization, and human datasets.
- `models/` owns learned prediction: `grasp-intent` imports as `grasp_gcn`, and `retargeting` imports as `cross_emb`.
- `control/` owns runtime policies: smoothing, temporal confirmation, safety clipping, top-k postural decisions, and reset behavior.
- `robot/` owns robot-specific bodies and execution details: Shadow Hand configs, kinematic hand configs, robot datasets, MuJoCo tools, and future ROS/MuJoCo sinks.
- `apps/` owns entrypoints that connect pipeline domains.

## Data Ownership

- Human datasets live under `human/datasets/`.
- Robot datasets live under the specific robot folder, for example `robot/hands/shadow_hand/datasets/`.
- Model checkpoints and experiments live under their model owner in `models/`.
- Non-pipeline history and recovered legacy material lives under `docs/archive/`.
