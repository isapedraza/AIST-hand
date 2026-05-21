# Migration Notes

The repository moved from old repo-name buckets into pipeline domains.

```text
grasp-app/perception/              -> human/perception/
grasp-app/hand_preprocessing/      -> human/kinematics/
grasp-model/                       -> models/grasp-intent-classification/
grasp-model/src/cross_emb          -> models/latent-retargeting/src/cross_emb
grasp-robot/                       -> robot/shadow-hand/
robot/shadow-hand/grasp_configs/   -> robot/hands/shadow_hand/
```

Install both learned-model packages from their new locations:

```bash
pip install -e models/grasp-intent-classification
pip install -e models/latent-retargeting
```

Current data/checkpoint owners:

```text
human/datasets/hograspnet/processed/hograspnet_abl11.csv
robot/hands/shadow_hand/datasets/processed/valid_robot_poses_eigengrasp_dong.npz
robot/hands/shadow_hand/shadow_hand_right.yaml
models/grasp-intent-classification/checkpoints/
models/latent-retargeting/checkpoints/
apps/graphgrasp-live/models/
```

Use importable package names for model code (`grasp_gcn`, `cross_emb`) and
domain packages for repo-local pipeline code (`human.*`, `control.*`).
