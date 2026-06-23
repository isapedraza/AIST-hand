# Allegro degenera en cross-embodiment: abducción ausente en UDHM (qpos-direct)

Fecha: 2026-06-23
Estado: causa raíz identificada; fix candidato sin probar en GPU.

## Síntoma

- Shadow entrena sano (single-robot, base 18B).
- Allegro **degenera totalmente** el latente al entrenar (joint con shadow, o freeze-add).
- No es velocidad (eso es otro problema). Es **correctitud**: el espacio compartido colapsa.

## Causa raíz

Allegro usa `--robot_udhm_from_qpos` → `robot_to_udhm(q_r, primitives_table)`.
La `primitives_table` se construye en `build_primitives()` con un **clasificador
geométrico de ejes**: proyecta el eje de cada joint sobre `[Y, Z, bone]` y elige
el máximo → `FLEX / ABD / ROT`.

Para Allegro ese clasificador **tira las abducciones de los dedos**:

```
joint  finger   projY   projZ   projBone  -> class
joint_0.0  index  0.087  0.000  1.000  -> ROT   (abducción, eje z)  DROPPED
joint_4.0  middle 0.000  0.000  1.000  -> ROT   (abducción)         DROPPED
joint_8.0  ring   0.087  0.000  1.000  -> ROT   (abducción)         DROPPED
joint_13.0 thumb  0.996  0.000  1.000  -> ROT                       DROPPED
joint_14.0 thumb  0.000  1.000  0.000  -> ABD
joint_15.0 thumb  0.000  1.000  0.000  -> ABD   (pisa al anterior)
```

El eje de abducción de Allegro queda **paralelo al hueso** (`projBone=1.0`) en el
palm-frame calculado, y `projZ=0.000` → se clasifica **ROT** (twist) → se descarta
(no hay slot twist para index/middle/ring).

Tabla resultante (abducción de dedos ausente):

```
index:  mcp_flex(1) pip(2) dip(3)      # falta col 0 (abducción)
middle: mcp_flex(5) pip(6) dip(7)      # falta col 4
ring:   mcp_flex(9) pip(10) dip(11)    # falta col 8
thumb:  mcp_abd(14) mcp_flex(12)       # faltan 13, 15
```

`_MANUAL` (overrides en `robot_primitives.py`) tiene entradas para `barrett_hand`
e `inspire_hand` pero **NO para `allegro_hand_right`** → Allegro depende 100% del
clasificador auto, que falla.

## Por qué degenera

`udhm22_r` de Allegro sale con **abducción = 0** en todos los dedos. `udhm22_h`
(humano) SÍ tiene abducción. Con `LAM_UDHM=2.12` (el peso más alto de la métrica
`S_k`), el contrastivo compara humano-con-abducción contra robot-siempre-cero →
no puede distinguir poses abducidas → empareja mal → **colapsa el latente
compartido** (y contamina shadow si se entrena joint).

## Por qué Shadow no sufre

Shadow usa el path **stage3 Dong** (`udhm_run_stage3`), no qpos-direct. Stage3
infiere UDHM desde **posiciones FK de las yemas**, no del eje del joint. Además el
URDF de Shadow ya codifica el espejo anatómico de abducción (FFJ4/MFJ4 axis
`0 -1 0` vs RFJ4/LFJ4 `0 1 0`), así que la geometría trae el signo correcto.

## Origen del flag

`--robot_udhm_from_qpos` se introdujo para **LEAP** (abducción congelada en Dong)
y **Barrett** (pip=0). Allegro **no es ninguno** — su abducción es un joint z
limpio con rango real. Se le aplicó un fix que no necesitaba y que lo rompe.

## Fixes candidatos (sin probar en GPU)

- **C (preferido, menor esfuerzo):** quitar `--robot_udhm_from_qpos` del notebook
  de Allegro → usar stage3 Dong como shadow. Stage3 infiere abducción desde
  posiciones, inmune al bug del clasificador de ejes. Riesgo: el problema de
  abducción-congelada que stage3 tuvo en LEAP; Allegro probablemente no lo hereda.

- **A (targeted):** añadir `allegro_hand_right` a `_MANUAL` forzando
  `joint_0/4/8 -> mcp_abd` (index/middle/ring) y mapeo correcto del pulgar.
  Mantiene la precisión de qpos-direct. Requiere cuidar el signo de abducción.

## Validación pendiente (offline, sin GPU)

Comparar `udhm22_r` de Allegro por ambos paths en poses con abducción real:
si stage3 da abducción != 0 donde qpos-direct da 0, queda probado que stage3 es
el fix. Último paso antes de tocar el notebook.

## Archivos relevantes

- `models/latent-retargeting/src/cross_emb/loaders/robot_primitives.py`
  (`build_primitives`, `robot_to_udhm`, `_MANUAL`).
- `models/latent-retargeting/src/cross_emb/loaders/sampler.py:300` (selección de path).
- `models/latent-retargeting/src/cross_emb/training/config.py:60` (`--robot_udhm_from_qpos`).
- `robot/hand-configs/allegro.yaml`, `robot/hands/allegro_hand/allegro_hand_right.urdf`.
