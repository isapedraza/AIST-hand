# grasp-robot — GraphGrasp

Robot adapters for **GraphGrasp: A Framework for Grasp Intent-Driven Teleoperation**.

**Status:** pending implementation

## Responsibilities
- Receive `GraspToken` from `grasp-app`
- Load robot YAML configuration (`grasp_configs/<robot>.yaml`)
- Send joint commands via `SrHandCommander` (Shadow Hand) or equivalent
- Pattern: `YAMLRobotAdapter` — new robot = new YAML, no code changes

## Supported robots
- Shadow Dexterous Hand (target)
  - Joint angles sourced from Dexonomy (RSS 2025): https://arxiv.org/abs/2504.18829

## Dependencies
- ROS Noetic (runs inside Shadow Robot Docker container)
- `sr_robot_commander`
- `grasp-model` (for `GraspToken` type)

## Shadow Hand setup
See root README for Aurora one-liner installation.
