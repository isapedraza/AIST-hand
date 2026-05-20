# cross-emb

Cross-embodiment retargeting package for GraphGrasp.

This package owns the latent-space retargeting model, inference runtime, and cross-embodiment training helpers. It imports as cross_emb after installation.

Install locally:

    pip install -e packages/cross-emb

Example import:

    from cross_emb.inference import Retargeter

The GCN grasp classifier lives separately in packages/grasp-gcn.
