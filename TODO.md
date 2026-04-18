# TODO

Fecha: 2026-04-18

- [x] Completar la cadena de puntas (ultima layer) del pipeline Dong.
- [x] Precalcular en el dataset las features (agregar nuevas columnas) antes del entrenamiento, para no recomputarlas cada corrida en la nube.
- [ ] Agregar flag `add_dong_quats` en tograph.py, grasps.py — mismo patron que `add_mano_pose`.
- [ ] Correr abl10 (clasificador baseline con features Dong) para validar que mejoran sobre abl04 (71.07%) antes de invertir en VGAE.
- [ ] Crear el Multiframe-CAM GNN que alimentara al calculo de Dong (va antes en el flujo).
- [ ] Crear el espacio postural latente / Dual-VGAE — solo despues de validar features Dong.
