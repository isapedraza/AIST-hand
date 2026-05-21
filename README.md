# GraphGrasp: AIST-hand

Monorepo for grasp intent recognition and human-to-robot retargeting. The repo is organized by execution pipeline responsibilities.

## Repository Layout

```text
AIST-hand/
  human/              # Observation and human-hand representation
  models/
    grasp-intent/     # Graph-based grasp classifier, imports as grasp_gcn
    retargeting/      # Cross-embodiment retargeting model, imports as cross_emb
  control/            # Temporal rules, safety, and stable command policies
  robot/
    hands/
      shadow_hand/    # Shadow Hand configs, datasets, MuJoCo tools
    hand-configs/     # Robot hand kinematic configs used by retargeting
  apps/
    graphgrasp-live/  # Live app and demos that connect pipeline modules
  docs/               # Architecture, decisions, research notes, archived handoffs
```

## Local Install

```bash
source .venv/bin/activate
pip install -e models/grasp-intent-classification
pip install -e models/latent-retargeting
```

## Main Workflows

Run the live app:

```bash
python apps/graphgrasp-live/main.py --camera 0
```

Run GCN tests:

```bash
python -m pytest models/grasp-intent-classification/tests
```

Run cross-emb retargeting smoke script:

```bash
python models/latent-retargeting/scripts/infer_retarget.py --help
```

See `docs/README.md` for architecture notes, migration records, decisions, research notes, and archived handoffs.
