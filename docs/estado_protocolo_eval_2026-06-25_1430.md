# Estado — Protocolo de experimentación (Eval 1 + Eval 2)

**Fecha:** 2026-06-25 14:30:22 CST

Reporte de estado tras sesión de diseño del protocolo de evaluación y validación
de la integración Shadow en DexJoCo. Continúa `docs/sim_eval_findings.md`.

## Resumen

Quedó definida la estructura de dos evaluaciones, verificado que el attach Shadow
funciona en las tareas single-hand, y acotado lo que falta para bimanual (un
bloqueo del lado modelo). El "PROBADO" del doc previo se confirmó reproducible hoy
(los scripts existen como `dexjoco/shadow_ext/build.py`, no con los nombres que
citaba `sim_eval_findings.md`).

## Las dos evaluaciones

### Eval 1 — Cinemático (métricas Yan)
- Métricas: RS / NDS / NVS / MPJPE / S_k sobre el test split. Sin simulador.
- Cubre ambos robots (Shadow + Allegro). Es el núcleo cuantitativo defendible.
- Mide precisión del retargeting (qué tan cerca queda la mano robot de la humana).

### Eval 2 — Funcional (sim, task success)
- **Qué es:** correr el modelo (teleop) en una combinación de tareas DexJoCo y
  reportar tabla de task success. Es un **reporte de capacidad**, complemento de
  Eval 1, no una ablación controlada.
- **Quién mide el éxito:** DexJoCo. Cada env trae su propio `_compute_success` y
  emite `{"succeed": True/False}`. No se inventa métrica; se lee y se cuenta.
- **Aclaración de alcance (importante):** Eval 2 como número en vivo mezcla la
  pericia del operador con la calidad del modelo y no es perfectamente
  reproducible. Eso lo hace débil SOLO si se vende como comparación controlada.
  Como reporte de capacidad ("el retargeter maneja N tareas en 2 robots a X%")
  es legítimo y estándar. El modelo ya funciona (probado en MuJoCo); Eval 2
  cuantifica/reporta, no demuestra desde cero.

## Suite de tareas propuesta para Eval 2

Combinación curada (no las 11), eligiendo por tipo de agarre:

| Task | Mano | Skill que estresa | Éxito (de DexJoCo) |
|---|---|---|---|
| `pick_bucket` | 1 | power grasp + lift | comida en balde + levantada 15cm |
| `water_plant` | 1 | trigger (índice aislado) | boquilla dentro de región cilíndrica |
| `pinch_tongs` | 1 | pinch + apertura | succeed propio del env |
| `hammer_nail` | 1 | power + muñeca 6D dinámica | por `nail_depth` (clavo hundido) |
| `bimanual_hanoi` | 2 | pick-place simétrico (showcase) | discos asignados a poste + estático |

4 single-hand (cubren power / trigger / pinch / wrist-6D) + 1 bimanual de
exhibición. Si bimanual falla, Eval 2 se para en las 4 single. No se apuesta la
tesis a infra no probada.

## Verificación del attach Shadow (hoy, reproducible)

`shadow_ext.build.build_spec(arena)` compila Panda+Shadow para cada arena.
Resultados de `spec.compile()` verificados:

| Arena | Resultado | nq | nu | nbody |
|---|---|---|---|---|
| `arena_arm_hand_bucket_pick` | OK | 46 | 28 | 45 |
| `arena_arm_hand_plant` (water_plant) | OK | 39 | 28 | 48 |
| `arena_arm_hand_table_tongs` (pinch_tongs) | OK | 39 | 27 | 44 |
| `arena_arm_hand_hammer_nail` | OK | 38 | 27 | 45 |
| `arena_arm_hand_bimanual_hanoi` | OK (parcial) | 75 | 34 | 84 |

**Single-hand: las 4 compilan con Shadow igual que el bucket.** El attach
generaliza por arena; el trabajo por-task es solo el swap de listas + mapeo en su
env. Mecánico, ~chico, pero 5 ports separados, cada uno con su viewer-check.

**Bimanual: nu=34 = 14 Panda + 20 Shadow = UNA sola mano.** `build_spec` ataca un
solo `attachment_site`; el segundo Allegro se borra pero no se reemplaza. Bimanual
requiere extender el attach a 2 sites (lh-/rh-, 2 Shadows).

## Mapeo retargeter qpos (24) → ctrl Shadow (20)

Resuelto y verificado (`shadow_ext/mapping.py`).

- El retargeter devuelve 24 valores en orden Menagerie/URDF; las 2 juntas de
  muñeca (`WRJ1/WRJ2`) ya vienen en 0 (`retarget.py:235-236`,
  comentario: "wrist DOFs — not encoded in wrist-frame human signal").
- Adapter Shadow (`shadow_udhm_adapter.yaml`) declara `ignored_joints: [WRJ2, WRJ1]`.
  UDHM = 22 slots, puros dedos, sin muñeca (confirmado en `UDHM22_SLOTS`).
- Único transform real: 4 tendones acoplados. En el MJCF cada dedo FF/MF/RF/LF
  tiene `<fixed name="rh_*J0"> joint *J2 coef 1 + joint *J1 coef 1`, manejado por
  un actuador de posición `A_*J0`. Por tanto `ctrl[A_*J0] = J2 + J1` (suma),
  fijado por el modelo (coef 1/1), no es decisión nuestra.
- Regla: `A_XXJ0` → `q[XXJ2]+q[XXJ1]`; `A_XXJn` (n>0) → `q[XXJn]`; muñeca → 0.
- Mapeo construido por nombre desde el modelo compilado (robusto a reorden y al
  prefijo `rh-`). Test: 20 actuadores, 4 sumas, 16 directos. Verificado.

## Distinción de "muñeca" (resuelta conceptualmente)

Dos cosas distintas que se llaman muñeca, no mezclar:

| | Qué es | DoF | De dónde sale |
|---|---|---|---|
| WRJ1/WRJ2 Shadow | juntas articuladas dentro de la mano | 2 | nadie — UDHM no las modela → ctrl=0 |
| Muñeca global | pose 6D de la mano en el espacio | 6 | source → `mocap_pos/quat` vía brazo Panda |

El brazo (7 DoF) entrega los 6D en el attachment site; la mano rígida hereda la
orientación. Incluir WRJ1/WRJ2 además = doble control del mismo DoF, peleándose.
La mayoría de manos dextras (Allegro: 0 juntas de muñeca) se montan rígidas al
flange; Shadow es la rara con 2 DoF de muñeca. UDHM, embodiment-agnóstico, los
omite a propósito porque no generalizan.

## Driver funcional stage-1 (hecho)

`shadow_ext/teleop_driver.py`: carga la escena attach, sostiene la muñeca fija vía
OSC del Panda (`opspace`), alimenta dedos vía el mapeo 24→20, `mj_step`, y computa
métricas (grasp-lift ahora; bucket-place listo para cuando llegue la traslación de
muñeca). Corre headless limpio. `grasp_lift=False` esperado: el objeto está en su
posición default de mesa, no en la zona de agarre (colocación se deja al teleop
en vivo; lo importante es la muñeca 6D).

## Decisiones pendientes (del usuario)

1. **Tasks de Eval 2**: confirmar la mezcla de 5 (4 single + hanoi) o ajustar.
2. **Intentos por task**: cuántos para que el % signifique algo (p.ej. 10).
3. **Robots**: Allegro (éxito gratis, nativo, sin attach) / Shadow (+swap por env)
   / ambos (soporta el claim cross-embodiment).

## Bloqueos para bimanual

- **(a) sim:** extender `build_spec` a 2 sites (2 Shadows). Lado nuestro, ~chico.
- **(b) modelo:** doble alimentación (2 instancias del retargeter, izq = espejo).
  Lado usuario, **no enchufado aún**. Bloquea el demo bimanual.

Ambos gatean bimanual; (a) puede dejarse listo de antemano sin esperar (b).

## Plan inmediato propuesto

1. Port Shadow a los 4 envs single-hand + verificar que `succeed` dispara.
2. Extender `build_spec` a 2 manos (deja el sim bimanual listo para el doble feed).

## Archivos relevantes (fork dexjoco, untracked)

- `dexjoco/shadow_ext/build.py` — attach Panda+Shadow vía MjSpec (parametrizado por arena).
- `dexjoco/shadow_ext/mapping.py` — mapeo qpos 24 → ctrl 20 (suma de tendones acoplados).
- `dexjoco/shadow_ext/teleop_driver.py` — driver funcional stage-1 + métricas.
- `dexjoco/shadow_ext/finger_close_test.py` — test aislado ctrl→dedos (de-risk).
- `dexjoco/shadow_ext/TODO.md` — estado fork-interno.
