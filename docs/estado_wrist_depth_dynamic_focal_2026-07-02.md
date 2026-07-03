# Estado — focal_length dinámico por bbox (fix intento 2, NO probado en vivo)

**Fecha:** 2026-07-02 (continúa `estado_wrist_depth_wilor_2026-07-02.md`)

## Qué se probó hoy

Se corrió la cadena completa (Colab + `live_retarget.py --emit-wrist` + `teleop_driver.py
--recv-wrist --wait-anchor`) con el fix de `focal_length=179.4` estático (commit anterior).
Resultado: **rotación perfecta, traslación (reach) prácticamente muerta.**

## Causa raíz encontrada (confirmada leyendo código fuente de `wilor_mini`)

`wilor_backend.py` manda a WiLoR un **crop** de la mano (MediaPipe detecta bbox, recorta,
resize a 256x256 fijo) en vez del frame completo -- optimización de latencia de un commit
anterior (`6166368b9`).

WiLoR estima profundidad así (`wilor_mini/utils/utils.py:cam_crop_to_full`):
```
tz = 2 * focal_length / bs
```
`bs` = bbox que WiLoR detecta (su propio YOLO interno) **dentro de la imagen que recibe**.
Como el crop ya viene pre-recortado y normalizado a 256x256 con padding fijo (30%), ese
`bs` sale ~constante en cada frame sin importar la distancia real de la mano -> `tz` no
se mueve -> cero señal de reach. El script `verify_wrist_depth.py` que validó r=0.981 NO
pasa por este path (manda el frame HOGraspNet completo directo a `pipe.predict()`), por
eso no detectó el problema.

## Fix aplicado (código ya escrito y pusheado, `pip` faltante: re-subir notebook a Colab)

En vez de mandar el frame completo (más latencia), se manda el **bbox original en píxeles**
(`bbox_px`, calculado del lado cliente con MediaPipe antes de recortar) junto al crop, y el
servidor reescala `focal_length` por request:
```
focal_length_dinamico = K_CALIB / bbox_px      # K_CALIB = FX_REAL_PX * 256
```
Esto compensa el "zoom" que mete el crop, sin cambiar tamaño de payload ni costo de
inferencia -- cero latencia extra.

Archivos tocados:
- `human/perception/wilor_backend.py`: `_infer_async` ahora manda `bbox_px` como form field.
- `servers/wilor_colab_server.ipynb`: Cell 2 define `FX_REAL_PX=897`, `K_CALIB=FX_REAL_PX*256`
  (mismo estimado por specs de antes, sin medir). Cell 3 `/infer` acepta `bbox_px` (Form,
  opcional) y hace `pipe.FOCAL_LENGTH = K_CALIB / bbox_px` antes de cada `predict()`.
  Sin `bbox_px` (cliente viejo) cae al estático `FOCAL_LENGTH_ARG=179.4` de antes.

## Pendiente para mañana

1. **Re-subir/reiniciar el notebook en Colab** -- el server que corrió hoy tiene el código
   VIEJO (focal estático). Hay que resubir `servers/wilor_colab_server.ipynb` (o reiniciar
   runtime + correr todas las celdas) para que el fix de hoy quede activo.
2. Correr la cadena de nuevo (mismo comando de siempre, ver
   `docs/estado_pendientes_casa_2026-07-01.md` sección B/C) y ver si el reach ahora sí
   mueve el brazo.
3. Si sigue sin moverse: sospechar de `padding=0.3` en `wilor_backend.py` (quizás el bbox
   de MediaPipe también es poco sensible a distancia por el padding fijo) o revisar si
   `bbox_px` llega bien al server (loggear en Cell 3).
4. Si se mueve pero mal calibrado: sigue pendiente la foto con objeto de ancho/distancia
   conocida (doc de ayer) -- ahora esa calibración da directo `K_CALIB` (ya no hace falta
   `FX_REAL_PX` y `_WRIST_POSE_SCALE` por separado, `K_CALIB` los absorbe a ambos del lado
   de profundidad; `_WRIST_POSE_SCALE` en `teleop_driver.py:60` sigue siendo el gain final
   post-metros, separado).

## Caveat

Nada de esto se probó en vivo todavía -- es el fix escrito hoy en base al diagnóstico de
código fuente (`wilor_mini`), no verificado contra datos reales como el fix de ayer. Mañana
el primer paso es simplemente confirmar que ahora sí hay señal de reach, antes de afinar
calibración.
