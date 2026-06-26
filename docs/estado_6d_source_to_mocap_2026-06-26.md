# Estado — Arranque 6D source → mocap (lo GRANDE)

**Fecha:** 2026-06-26

Continúa `docs/estado_eval2_suite_2026-06-25_1614.md`. Captura lo rescatado del
`TODO.md` del fork (público) ANTES de borrarlo. Doc/notas viven SOLO en AIST-hand
(privado); el fork `dexjoco-shadow` es público.

## Por qué este paso primero
Suite Eval2 single (4 tasks) ya corre pero da `succeed=False`: muñeca FIJA, sin
teleop vivo. 6D source → mocap convierte la suite de "corre" a "tiene éxito en
vivo". Se reusa tal cual en bimanual (cada mano = su mocap). Por eso va primero.

## Camino vivo
```
WiLoR pose global muñeca (global_orient + traslación)
  -> data.mocap_pos/quat[target]
  -> opspace (OSC) de DexJoCo
  -> brazo Panda sigue
```
Mocap→brazo YA probado (sweep senoidal del recorder). Hoy el driver pone seno en
el mocap (stage 1). Falta: enchufar fuente 6D viva en vez del seno.

## Hallazgo clave: env nativo `sim_teleop.py` ya tiene el patrón
`dexjoco/dexjoco/dexjoco/tasks/sim_teleop.py` — env teleop nativo de DexJoCo por
UDP. Puertos (verificado en código, líneas 19-32):

| Puerto | Señal |
|---|---|
| `5012` | muñeca (VIVE tracker, 6D) |
| `5014` | mano derecha (qpos) |
| `5016` | mano IZQUIERDA (qpos) — bimanual YA cableado aquí |

Implicación: el camino UDP muñeca→mocap ya existe en DexJoCo (vía VIVE). Nuestro
trabajo = emitir la muñeca 6D de WiLoR a 5012 (en vez de VIVE), o replicar el
patrón de recepción en `shadow_ext/teleop_driver.py`. Bimanual: 5014+5016 ya
contemplados nativamente.

## Muñeca WiLoR: dónde sale la pose global
`apps/graphgrasp-live/sources/wilor_source.py` produce quaternions Dong desde
keypoints WiLoR (orden OpenPose/MediaPipe 21-joint, drop-in para Dong). Maneja
chirality (canonicalize_to_right_hand vía flag `is_right`).

PENDIENTE de surfacear (hoy se descarta al canonicalizar a frame Dong):
- `pos` = keypoint[0] en cámara (traslación muñeca global)
- `rot` = frame Dong / global_orient
- Señal monocular: tiembla + ~253ms lag → filtro one-euro o anclaje.

## Plan por etapas (cada una ya es demo; riesgo creciente)
1. Muñeca fija + dedos en vivo (objeto pre-colocado, cerrar → agarrar). ← stage 1 actual
2. + orientación muñeca en vivo (frame Dong, confiable).
3. + traslación muñeca en vivo (cam_t + filtro). Si tiembla mucho, quedarse en 2.

## Refs
- `dexjoco/shadow_ext/teleop_driver.py` — env mínimo (dónde está el seno del mocap).
- `dexjoco/dexjoco/dexjoco/tasks/sim_teleop.py` — patrón UDP nativo (5012/5014/5016).
- `apps/graphgrasp-live/sources/wilor_source.py` — fuente WiLoR (pose global a surfacear).
- `docs/sim_eval_findings.md` — hallazgos/diseño completos.
