# Estado — Pendientes para hacer en casa (PC personal + webcam)

**Fecha:** 2026-07-01 CST
**Contexto:** escrito desde PC trabajo (sin webcam). Las 3 tareas de la lista
están CONSTRUIDAS; falta probarlas end-to-end vivo con webcam en casa. Este doc
es el checklist de arranque para cuando llegues.

Regla: este doc vive SOLO en AIST-hand main. El código de sim vive en el fork
`dexjoco-shadow` (`dexjoco/shadow_ext/`, branch `shadow-support`).

---

## Resumen de las 3 tareas

| # | Tarea | Estado código | Qué falta en casa |
|---|---|---|---|
| 1 | Nuevo modelo Allegro con sinergias modificadas | Hecho (commit `adfdd5c`) | Bajar el `.pt` de Colab (no está en disco aquí) |
| 2 | Entrenamiento cross-embodiment (multi-robot B=100k) | Corriendo anoche en Colab | Confirmar que terminó + bajar el `.pt` |
| 3 | Setup entorno simulación | Suite 4/4 corre limpio | Enchufar teleop 6D vivo → ver `succeed=True` |

**Eslabón nunca probado:** fuente 6D (webcam→WiLoR→muñeca) enchufada a la suite
de sim. Todo lo demás ya se validó por partes.

---

## Tarea 1 + 2 — traer los pesos de Colab

Entrenaste anoche en Colab. El checkpoint destino (según `adfdd5c`) es:

    stage1_shadow_allegro_bodex_objbal.pt

(Se renombró para NO pisar el checkpoint de Run C. B pasó de 50k→100k para que
cada robot reciba 50k: el loop parte `B // n_robots`. Shadow+Allegro juntos =
esto ES el cross-embodiment de la tarea 2.)

Pasos:
1. Abrir Colab, confirmar que el run de `train_stage1_multi_udhm_B.ipynb` terminó
   sin caerse (revisar métricas finales / eigenvalues).
2. Descargar `stage1_shadow_allegro_bodex_objbal.pt`.
3. Colocarlo en: `models/latent-retargeting/checkpoints/`
4. (opcional) Comparar contra `stage1_best_run20.pt` / Run C para sanity de que
   las sinergias cambiaron.

---

## Tarea 3 — probar sim con teleop 6D vivo

La suite Eval2 single-hand ya corre 4/4 limpio (bucket, water_plant,
pinch_tongs, hammer_nail) con la métrica `succeed` de DexJoCo conectada.
HOY dan `succeed=False` — ESPERADO, no bug: muñeca fija, sin teleop vivo, nada
alcanza el objeto. El éxito lo dispara el operador al enchufar la fuente 6D.

### Puertos UDP (ya cableados)
- Dedos (qpos retargeter): **5014**
- Muñeca (pose 6D): **5012**

### Cadena a levantar (3 procesos)

**A) Servidor WiLoR (Colab)** — backend de percepción
- Notebook: `servers/wilor_colab_server.ipynb`
- Arrancar, copiar la URL del túnel `https://xxxx.trycloudflare.com`.

**B) Emisor: webcam → retargeter → UDP** (PC casa, venv principal)

    python apps/graphgrasp-live/live_retarget.py \
      --ckpt models/latent-retargeting/checkpoints/stage1_shadow_allegro_bodex_objbal.pt \
      --source wilor --url https://xxxx.trycloudflare.com \
      --camera 0 \
      --emit-udp   --emit-port 5014 \
      --emit-wrist --wrist-port 5012

- `--emit-udp` manda qpos dedos a 5014. `--emit-wrist` manda muñeca 6D a 5012.
- `--emit-wrist` EXIGE `--source wilor` (la pose sale de WiLoRSource).
- Si la webcam no es índice 0, ajustar `--camera`.

**C) Receptor: sim** (venv `.venv-dexjoco`, dentro de `dexjoco/`)

    # desde /home/pc_pro/AIST-hand/dexjoco, con .venv-dexjoco activo
    python -m shadow_ext.teleop_driver pick_bucket \
      --recv-fingers --recv-wrist --view

- Positional = nombre de task: `pick_bucket` | `water_plant` | `pinch_tongs` | `hammer_nail`
- `--recv-fingers` escucha 5014, `--recv-wrist` escucha 5012, `--view` abre viewer.
- Sin `--recv-*` corre en modo held-fixed (lo de hoy → succeed=False).

### Criterio de éxito
- Con la cadena viva, mover la mano frente a la webcam debe mover brazo+dedos en
  sim. Buscar `succeed=True` en al menos una task.
- Grabar figuras con el recorder: flags `--record <path> --shot <n> --cam front`
  en `teleop_driver` (ver `shadow_ext/recorder.py`).

---

## Orden sugerido al llegar a casa

1. Confirmar Colab terminó → bajar `stage1_shadow_allegro_bodex_objbal.pt` a
   `models/latent-retargeting/checkpoints/`.
2. Levantar server WiLoR (Colab) → copiar URL túnel.
3. Correr emisor (B) con webcam → verificar en la ventana de `live_retarget` que
   la mano se detecta y el robot sigue.
4. Correr receptor (C) `pick_bucket --recv-fingers --recv-wrist --view`.
5. Verificar que brazo+dedos siguen la mano; cazar `succeed=True`.
6. Repetir las 4 tasks, grabar figuras.

## Riesgos conocidos / dónde puede fallar
- 6D source→mocap NUNCA probado vivo — es el eslabón nuevo. Anchoring en 1er
  paquete (`tracker_start`), revisar frames/escala si la muñeca deriva.
- Chirality WiLoR: knob empírico en `wilor_source.py:55` (identidad por default,
  Dong opcional) si la mano sale espejada.
- Percepción MCP cierra de lado no frontal (dato de Run 38) — vigilar orientación
  de cámara.
