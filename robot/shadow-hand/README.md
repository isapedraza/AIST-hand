# shadow-hand

Shadow Hand robot and simulation integration.

This folder owns Shadow Hand-specific configuration, datasets, experiments, and
MuJoCo/pose-map tools. It should contain facts about the robot body and ways to
execute commands, not camera code or learned model definitions.

## Layout

```text
robot/shadow-hand/
  configs/      # canonical grasp YAMLs and Shadow Hand mappings
  datasets/     # robot-side local/generated datasets, ignored when heavy
  experiments/  # robot-side analysis outputs
  tools/        # extraction, visualization, and pose-map scripts
```

Generic hand kinematic configs used by retargeting live in `robot/hand-configs/`.
