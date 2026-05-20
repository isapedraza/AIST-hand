# GraphGrasp: AIST-hand

Monorepo for grasp intent recognition and human-to-robot retargeting. The repo is organized as independent Python packages plus applications and robot integration code.

## Repository Layout

```text
AIST-hand/
  packages/
    grasp-gcn/        # Graph-based grasp intent classifier, imports as grasp_gcn
    cross-emb/        # Cross-embodiment retargeting model, imports as cross_emb
  apps/
    graphgrasp-live/  # Live camera/perception demos and app entrypoints
  robot/
    shadow-hand/      # Shadow Hand postural control, configs, MuJoCo scripts
  data/               # Shared local datasets, ignored by git
  artifacts/          # Shared generated outputs, ignored by git
  docs/               # Architecture and migration notes
  docker/             # Docker notes/placeholders
```

## Local Install

```bash
source .venv/bin/activate
pip install -e packages/grasp-gcn
pip install -e packages/cross-emb
```

## Main Workflows

Run the live app:

```bash
python apps/graphgrasp-live/main.py --camera 0
```

Run GCN tests:

```bash
python -m pytest packages/grasp-gcn/tests
```

Run cross-emb retargeting smoke script:

```bash
python packages/cross-emb/scripts/infer_retarget.py --help
```

See docs/ARCHITECTURE.md and docs/MIGRATION.md for package boundaries and moved paths.
