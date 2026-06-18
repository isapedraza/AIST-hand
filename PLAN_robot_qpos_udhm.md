# Plan: Robot llena UDHM desde qpos (primitivas), no Dong

## Meta
El robot tiene qpos (verdad). Dong lo adivina desde posiciones y falla en LEAP
(abduccion congelada) y Barrett (distal congelada). Fix: robot llena UDHM desde
qpos via primitivas. Humano usa human_to_udhm (signed, fix arccos). Mismo espacio
UDHM22 para ambos — contrato canonico cross-embodiment.

## STATUS: COMPLETO (2026-06-18)

### Funcion 1 -- `build_primitives` ✓
`robot_primitives.py` — tabla `{dedo: {role: (col_qpos, signo)}}`. Corre 1x por robot.

### Funcion 2 -- `robot_to_udhm` ✓
`robot_primitives.py` — qpos [B,J] → UDHM22 [B,22]. Semantic insertion, /pi normalizado.

### Humano -- `human_to_udhm` ✓
`human_to_udhm.py` — rotaciones Dong → UDHM22 signed (fix arccos unsigned de stage-3).

### Cableado -- sampler.py ✓
- `udhm22_h = human_to_udhm(pose_h, labels)` — siempre
- `udhm22_r = robot_to_udhm(q_r, tabla)` — detrás flag `robot_udhm_from_qpos=True`
- Fallback: `udhm_run_stage3` si flag OFF (default)

### Config/loop ✓
- `--robot_udhm_from_qpos` flag en config.py
- Pasado a CrossEmbodimentSampler en loop.py (train + val)

## Validacion completada
- LEAP: abduccion responde (200/250 slots non-zero) ✓
- Barrett: pip responde ✓
- Shadow/LEAP/Allegro open→close flex direction correcto ✓
- Cross-embodiment: open L1~0.10, close L1~0.20 (esperado por morfologia distinta) ✓
- Script: `scripts/evaluate_udhm_cross_embodiment.py`

## Para entrenar
```
python scripts/train_cross_emb.py \
  --lam_udhm 0.4 \
  --robot_udhm_from_qpos \
  --sk_metric xin \
  ...
```
`lam_udhm` sugerido 0.4-0.5 del total S_k (40-50% rotaciones, resto Xin).

## Archivos generados
- `robot/hands/shadow_hand/datasets/processed/synthetic_open_hand_shadow_qpos.npz`
- `robot/hands/shadow_hand/datasets/processed/synthetic_close_hand_shadow_qpos.npz`
- `robot/hands/allegro_hand/datasets/processed/synthetic_open_allegro.npz`
