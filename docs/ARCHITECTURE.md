# Architecture

GraphGrasp is now a monorepo with separate installable Python packages.

## Packages

- packages/grasp-gcn owns the graph classifier: ToGraph, GCN networks, grasp labels, GraspToken, VotingWindow, training scripts, GCN tests, and GCN experiments.
- packages/cross-emb owns cross-embodiment retargeting: latent-space modules, Retargeter, MediaPipe-to-Dong source, MuJoCo sink, stage1 training, retargeting scripts, robot/human loaders, and cross-emb checkpoints.

## Applications

- apps/graphgrasp-live consumes installed packages and contains camera/perception demos, model variant selection, live inference loops, and app-local deploy checkpoints.

## Robot

- robot/shadow-hand owns Shadow Hand configs, postural control, and MuJoCo/pose-map scripts. It consumes grasp_gcn as an installed package.

## Data and Artifacts

- data/ is for shared local datasets and is ignored by git.
- package-local data folders may exist for model-specific generated files.
- artifacts/ is for shared generated outputs such as runs, reports, and checkpoints.
