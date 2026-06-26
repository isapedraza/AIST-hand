# Investigación — Muñeca 6D desde WiLoR → mocap (frames y fuente real)

**Fecha:** 2026-06-26
Continúa `docs/estado_6d_source_to_mocap_2026-06-26.md`. Investigación a fondo de
DE DÓNDE sale la muñeca 6D y CÓMO mapearla al frame que espera el env. Implementación
de cuidado: documentar antes de tocar código.

## Cadena de datos actual (qué se produce y qué se tira)
```
WiLoR-mini (Colab)  ->  server /infer  ->  WiLoRBackend  ->  WiLoRSource  ->  retargeter
  pred_keypoints_3d        SOLO manda       sample =          SOLO emite
  pred_cam_t (?)           keypoints[21,3]  {kp, is_right}    quats dedos [1,20,4]
  global_orient (?)        + is_right
```
- `servers/wilor_colab_server.ipynb:100` — server extrae SOLO `wilor_preds["pred_keypoints_3d"][0]` (21,3, orden OpenPose) + `is_right`. Tira el resto de `wilor_preds`.
- `human/perception/wilor_backend.py` — sample = `{JOINT: xyz, is_right}`. Nada global.
- `apps/graphgrasp-live/sources/wilor_source.py:149-153` — emite SOLO quats de dedos. Muñeca global descartada.

## HALLAZGO CLAVE: la orientación 6D ya se computa (y se tira)
`human/kinematics/dong_kinematics.py::_compute_wrist_frame` (`:58-106`) construye
cada frame la rotación de muñeca global desde anatomía (Dong Eq. 5-7):
```
Y0 = normalize(d13 - d5)            # línea de nudillos (pinky_mcp -> index_mcp)
Z0 = normalize((d9 - d0) x Y0)      # normal de palma
X0 = Y0 x Z0                        # sale de la palma
r0w = [X0 | Y0 | Z0]                # 3x3 ROTACIÓN GLOBAL DE MUÑECA  <-- esto buscamos
```
- `_build_t0w(r0w, d0w)` (`:109`) ya arma la 4x4 `[R|t]`. Origen `d0w = points_w[0]` (WRIST).
- Es ortonormal (`det≈1`, `orth_err≈0`, ya chequeado en el dict). Frame anatómico bien definido.
- **Orientación NO necesita cambio de server.** Sale gratis del Dong que ya corre.

## EL HUECO REAL: traslación
Los `pred_keypoints_3d` de WiLoR-mini son **root-relative + escala-normalizada**
(MANO recentrada en muñeca). => `points_w[0]` (WRIST) ≈ origen SIEMPRE => **no hay
señal de traslación global** en lo que el server manda hoy.
- Para traslación real en frame cámara hace falta `pred_cam_t` (o `pred_cam`) de
  `wilor_preds`, que el server NO surfacea. **Stage 3 BLOQUEADO hasta** añadir
  `pred_cam_t` al payload del server (`:101`) + backend + source.
- Stage 2 (solo orientación) NO lo necesita. Por eso el plan por etapas calza:
  orientación primero (gratis), traslación después (toca server).

## Frames en juego (el riesgo de fondo)
Tres convenciones distintas; hay que alinearlas:
1. **Frame cámara WiLoR** — ejes de `pred_keypoints_3d`.
2. **Frame Dong (r0w)** — anatómico: X=fuera de palma, Y=nudillos, Z=normal. Lo que tenemos.
3. **Frame VIVE/flange** que espera `sim_teleop._vive_action` — pose 3x4 delta desde `ee_start`, `pose_scale=1.5`, `target = ee_start @ delta`.

El env asume convención del tracker VIVE físico. Nosotros damos r0w (Dong). =>
falta una **rotación de alineación fija `R_align`** (Dong→flange) calibrada en
viewer. Riesgo real = atinar esa R_align (toda la rotación, no solo signos). Una vez
fija es matemática determinista.

## Chirality (importa para bimanual)
`canonicalize_to_right_hand` (source `:128`) refleja izq→der ANTES de Dong. => r0w
se computa sobre puntos right-canonical. Para mano IZQ (bimanual) hay que des-reflejar
la rotación o manejar el frame izq aparte. Consistente con [[project-shadow-teleop-eval-protocol]]
"izq = espejo + flip abducción".

## Ruido de señal
Monocular: tiembla + ~253ms lag (ver doc anterior). => filtro one-euro sobre la 6D
ANTES de empacar. El esquema delta-relativo del env ya ayuda (ancla a ee_start).

## Plan refinado (orden por costo/riesgo)
| Etapa | Qué | Toca | Bloqueo |
|---|---|---|---|
| A | Dedos en vivo, muñeca fija (stage1 actual ya demo) | fork driver (recv 5014) | ninguno |
| B | + orientación muñeca (r0w → 3x4 → 5012) | source surfacea r0w; driver recv 5012; calibrar R_align | ninguno (r0w gratis) |
| C | + traslación (pred_cam_t) | **server payload** + backend + source + driver | requiere reabrir Colab server |
| D | filtro one-euro + bimanual (izq des-reflejada, 5016) | source + driver 2 manos | tras B/C |

## Decisiones abiertas (a resolver antes de codear)
1. ¿Surfacear r0w como nuevo método en `WiLoRSource` (p.ej. `wrist_pose()`) o
   devolver 6D junto a quats? Mantener flaggeable/desacoplado [[feedback_flaggeable_decoupled]].
2. ¿Calibrar R_align a mano en viewer o derivarla (alinear X0 con flange forward)?
3. Traslación: ¿abrir server para `pred_cam_t`, o quedarse en orientación-sola
   (stage B) si el temblor de traslación no vale la pena? (Dong doc original sugería
   "si tiembla mucho, quedarse en 2".)

## VERIFICACIÓN (2026-06-26, no asumir) — correcciones a supuestos

Se auditaron los supuestos antes de codear. Resultados:

1. **NO existe emisor UDP.** `grep sendto/5012/5014` en apps/human/servers = vacío.
   `sim_teleop` solo RECIBE; el feed era el tracker VIVE FÍSICO. => construir el
   emisor WiLoR→UDP es PIEZA REAL, no "enchufar".
2. **`sim_teleop` está cableado a Allegro, no Shadow** (`_hand_action` usa
   `action[7:23]`=16 dof). => NO se reusa directo; se COPIA su esquema VIVE al
   `teleop_driver` Shadow.
3. **Math VIVE confirmada** (`_vive_action:181-199`): `delta = inv(start)@now` (4x4
   completa rot+trasl), `delta[:3,3]*=pose_scale(1.5)`, `target = ee_start@delta`,
   manda `[pos(3),quat_wxyz(4)]`. Receptor lee 12 float64 reshape(3,4) = `[R|t]`.
   Como es delta, R_align conjuga la delta (no se cancela sola); sigue haciendo falta.
4. **CORRECCIÓN: mocap→brazo con ROTACIÓN nunca se probó.** `teleop_driver:108`
   hace OSC HOLD en mocap FIJO (soldado al inicio, no se actualiza en loop). El
   "seno" es de DEDOS (`:106`), no muñeca. El "sweep del brazo" del doc Eval2 era
   script tirado / exageración. => `opspace` recibe `ori=` pero nadie manejó mocap
   móvil/rotando y confirmó que el flange sigue. **PRIMER experimento (cimiento):**
   manejar mocap con pose conocida variable (trasl+rot) en el driver, ver en viewer
   que el Panda sigue. Local, sin WiLoR. Valida todo antes de tocar arriba.
5. **`canonicalize_to_right_hand` = IDENTIDAD para mano derecha** (`:898`,
   reflected=False). => `r0w` mano der limpio, sin rotación oculta. Izq refleja X
   (manejar en bimanual).
6. **RESUELTO (clonado `third_party/WiLoR-mini`, ignorado):** `wilor_preds` trae
   mucho más de lo que el server manda. Bloqueo de traslación = FALSO, el dato existe.

### WiLoR-mini: qué hay en `wilor_preds` (verificado en código)
- **`pred_keypoints_3d` (21,3) = root-relative** ✓ (se proyectan a 2D CON
  `translation=pred_cam_t_full` aparte, pipeline `:153-154`). Solos no tienen traslación.
- **`pred_cam_t_full` (3,) = traslación global en frame cámara** (`pipeline:148-150`,
  `utils.cam_crop_to_full:138`):
  ```
  tz = 2*focal/bs   (bs = box_size*cam_bbox[0])  -> PROFUNDIDAD por tamaño aparente
  tx = 2*(cx-w/2)/bs + cam_bbox[1]   (plano imagen)
  ty = 2*(cy-h/2)/bs + cam_bbox[2]
  ```
  Pseudo-métrica (weak perspective). **tz = derivada del tamaño aparente => eje débil/
  ruidoso** (la profundidad monocular que se advirtió). tx/ty decentes.
- **`global_orient` (axis-angle 3,) = rotación global de mano, frame cámara**
  (`wilor.py:48` rotmat 3x3 -> `:62` `roma.rotmat_to_rotvec`). Fuente NATIVA de
  orientación, alternativa a `r0w` de Dong.

### Consecuencias firmes
- Orientación: DOS fuentes — `r0w` (local, sin server) o `global_orient` (nativa,
  requiere server). Stage B usa `r0w` (gratis). Si se abre server, mandar ambas y comparar.
- Traslación: `pred_cam_t_full` ya existe. Stage C = añadir 2 campos (`global_orient`,
  `pred_cam_t_full`) al JSON del server Colab (`servers/wilor_colab_server.ipynb:101`)
  + backend + source. Barato, NO bloqueado.
- Reto real único: ruido de tz (profundidad). One-euro fuerte en Z, o escala fija +
  solo XY si tz inusable. (Calza con "si tiembla, quedarse en stage 2" de Dong.)

### Etapa 0 — CIMIENTO VALIDADO (2026-06-26)
Implementado `wrist_sweep_pose()` + flag `--sweep-wrist` en `teleop_driver.py`
(desacoplado; path default sigue HOLD fijo). Sweep = lissajous ±8cm + tilt ±0.5rad.
Medido error de tracking (sensor flange vs mocap comandado, salta transitorio):

| Velocidad | gain | POS err | ANG err |
|---|---|---|---|
| 1.0x | 400 | 60.4mm | 12.7° |
| 0.25x | 400 | 21.4mm | 6.2° |
| 0.25x | 800 | 15.8mm | 6.2° |

**Resultado: mocap→brazo SIGUE posición Y orientación.** El error es lag/gain (no
muro): encoge con movimiento lento (velocidad humana real) + gains altos. El
supuesto del doc Eval2 ("mocap→brazo ya funciona") queda PROBADO cierto. Residual
~16mm = steady-state OSC. Caveat para etapas 2-3: subir gains o suavizar fuente.

### Orden de trabajo corregido
0. **Experimento cimiento:** mocap móvil+rotando → Panda sigue (driver, local). ✅ HECHO
1. Etapa A: dedos vivos vía recv 5014 + construir emisor de qpos retargeter.
2. Etapa B: orientación r0w → emisor 3x4 → recv 5012 en driver + calibrar R_align.
3. Etapa C: traslación → primero verificar WiLoR-mini coords + pred_cam_t; si root-
   relative, reabrir server Colab para surfacear pred_cam_t.
4. Etapa D: one-euro + bimanual izq.

## Refs
- `human/kinematics/dong_kinematics.py:58-127` — `_compute_wrist_frame`, `_build_t0w`, transforms.
- `apps/graphgrasp-live/sources/wilor_source.py:121-153` — extract points + emit quats (dónde inyectar wrist_pose).
- `human/perception/wilor_backend.py:126-164` — payload server→sample.
- `servers/wilor_colab_server.ipynb:100-101` — recorte a keypoints (dónde añadir pred_cam_t).
- `dexjoco/dexjoco/dexjoco/tasks/sim_teleop.py:120-196` — patrón VIVE delta-relativo.
- `dexjoco/shadow_ext/teleop_driver.py:80-113` — weld fijo (dónde meter recv mocap).
