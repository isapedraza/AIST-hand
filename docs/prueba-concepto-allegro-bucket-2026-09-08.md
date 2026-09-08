# Prueba de concepto Allegro bucket — 2026-09-08

Objetivo: comprobar dedos, control de brazo y registro del bucket con Allegro. Se conserva `trial-kind practice`; no es evaluación formal.

No hay poses/waypoints Allegro guardados actualmente. La primera prueba utiliza posicionamiento manual. Las trayectorias Shadow no se asumen transferibles, porque cambia la geometría de montaje de la mano.

En la primera terminal, sustituir la URL por la del servidor WiLoR activo:

```bash
cd /home/yareeez/AIST-hand
.venv/bin/python apps/graphgrasp-live/live_retarget.py \
  --ckpt models/latent-retargeting/checkpoints/active/stage1_shadow_allegro_bodex_objbal_15k.pt \
  --source wilor --url "YOUR_CURRENT_WILOR_URL" \
  --robot allegro --camera 0 --emit-udp --no-viewer \
  --session /home/yareeez/AIST-hand/recordings/allegro-bucket-practice-001
```

Esperar calibración. En la segunda terminal:

```bash
cd /home/yareeez/dexjoco-shadow
/home/yareeez/AIST-hand/.venv/bin/python -m shadow_ext.teleop_driver pick_bucket \
  --hand allegro --view --key-control --recv-fingers \
  --pose-file shadow_ext/saved_key_pose_allegro.json \
  --session /home/yareeez/AIST-hand/recordings/allegro-bucket-practice-001 \
  --evaluation-start manual --time-limit 600 \
  --trial-kind practice --operator-id yahel \
  --condition-id allegro-bucket-practice-v1
```

Comprobar respuesta de dedos antes de iniciar el intento. Los dedos siguen la webcam. Con foco en el simulador y NumLock activado:

| Teclas de teclado numérico | Movimiento del brazo |
|---|---|
| 4 / 6 | -X / +X |
| 8 / 2 | +Y / -Y |
| 7 / 9 | +Z / -Z |
| 1 / 3 | Pitch |
| / y * | Yaw |
| - y + | Roll |
| Punto decimal | Guardar pose de muñeca Allegro |

Sin `--playback-poses`, las flechas derecha/izquierda no recorren waypoints. Se prepararán poses propias de Allegro con la experiencia de esta prueba.

Pulsar F5 antes de manipular comida/cubo para iniciar el intervalo de diez minutos reales. El tiempo previo se conserva como preparación. Finaliza por éxito, timeout o interrupción manual; después cerrar cámara con Q/Esc o Ctrl+C. Una interrupción también produce evidencia útil de práctica.

Usar un nombre nuevo en ambos comandos para cada intento posterior. El exportador y las comprobaciones son los mismos que en Shadow; la selección de mano y el modelo congelado identifican Allegro. La ruta utiliza implicitfast ya implementado, sin asumir que su dinámica coincide con Shadow.

Las fuerzas de contacto directas son una ampliación opcional para diagnóstico; no bloquean esta prueba. [Justificación bibliográfica](fuerzas-contacto-prueba-concepto-2026-09-08.md).
