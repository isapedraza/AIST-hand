# Docker

Docker files are intentionally not added yet. The new layout supports separate images later:

- GCN training image installs packages/grasp-gcn.
- Cross-emb training image installs packages/cross-emb.
- Live app image installs both packages and runs apps/graphgrasp-live.
- Robot/simulation image installs robot dependencies and mounted MuJoCo assets.
