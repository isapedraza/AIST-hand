# Migration Notes

Old top-level directories were moved into domain-specific locations.

```text
grasp-model/              -> packages/grasp-gcn/
grasp-model/src/cross_emb -> packages/cross-emb/src/cross_emb
grasp-app/                -> apps/graphgrasp-live/
grasp-robot/              -> robot/shadow-hand/
```

Install both packages after pulling this migration:

```bash
pip install -e packages/grasp-gcn
pip install -e packages/cross-emb
```

Imports should use package names, not path hacks:

```python
import grasp_gcn
import cross_emb
```

App-local deploy checkpoints remain under apps/graphgrasp-live/models. Cross-emb stage1 checkpoints were moved to packages/cross-emb/checkpoints.
