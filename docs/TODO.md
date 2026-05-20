# TODO

Fecha: 2026-04-19

- [x] Completar la cadena de puntas (ultima layer) del pipeline Dong.
- [x] Precalcular en el dataset las features (agregar nuevas columnas) antes del entrenamiento, para no recomputarlas cada corrida en la nube.
- [x] Agregar flag `add_dong_quats` en tograph.py, grasps.py.
- [x] Correr abl10 (Dong quats only, F=4) -- resultado: 62.6% test acc. Ver Entry 30.
- [ ] Implementar Dual-VGAE Encoder A (grafo humano, Dong quats 84-dim) + Encoder B (robot qpos Shadow Hand).
- [ ] Definir similarity metric D_R para contrastive learning sobre quaterniones Dong (Entry 30 / Yan & Lee Eq.1).
- [ ] Completar 28 canonical Shadow Hand poses en shadow_hand_canonical_v5_grasp.yaml (prerequisito para Encoder B).
- [ ] Implementar Multiframe-CAM GNN para robustez temporal (Entry 1).
