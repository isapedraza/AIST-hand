# Grabación básica de una sesión Shadow

Estado: 2026-09-07. Captura y exportación integradas; seis pruebas automatizadas
aprobadas con MuJoCo 3.6.0. Comprobación práctica completada con webcam, WiLoR
y Shadow: tarea exitosa y grabación confirmadas por el operador.
Este documento describe la funcionalidad actual, no el protocolo completo de
evaluación de tesis.

Actualización 2026-09-08: el registro ampliado, inicio F5, límite real, eventos y
resúmenes se describen en la [guía de evaluación funcional](registro-evaluacion-funcional-2026-09-08.md).
Esta página conserva la descripción y evidencia de la integración básica del 7 de septiembre.

## Uso rápido con tus comandos habituales

Selecciona una carpeta nueva por intento, por ejemplo:

`/home/yareeez/AIST-hand/recordings/shadow-take-001`

Añade a tu comando habitual de retargeting WiLoR:

```text
--robot shadow --emit-udp --session /home/yareeez/AIST-hand/recordings/shadow-take-001 --no-viewer
```

Conserva tus opciones actuales de checkpoint, cámara, URL, calibración e interpolación.
`--no-viewer` omite la ventana adicional de previsualización de la mano;
la ventana de webcam y la del simulador siguen siendo las de uso normal.
La captura de fotos del robot desde la ventana adicional no está disponible en ese modo.

Añade al comando habitual del simulador:

```text
--session /home/yareeez/AIST-hand/recordings/shadow-take-001
```

Conserva los controles de brazo y waypoints que ya utilizas.
Durante esta primera integración no se cambiaron las ganancias, el control por
teclado ni las trayectorias. La compensación de deslizamiento conserva su opción
existente; grabar no la activa.

Para evitar renderizado costoso durante la interacción, utiliza la nueva sesión
sin añadir `--record`. Los videos del simulador se generan después.
No es necesario añadir el antiguo `--state-log`: `--session` ya conserva estados
más completos.

## Ejemplo completo

En una terminal de AIST-hand, sustituye checkpoint y URL por los que ya utilizas:

```bash
cd /home/yareeez/AIST-hand
.venv/bin/python apps/graphgrasp-live/live_retarget.py \
  --ckpt /ruta/al/checkpoint.pt \
  --source wilor --url https://TU-SERVIDOR \
  --robot shadow --camera 0 --emit-udp --no-viewer \
  --session /home/yareeez/AIST-hand/recordings/shadow-take-001
```

Inicia la webcam primero y espera a que termine la calibración. En la otra terminal:

```bash
cd /home/yareeez/dexjoco-shadow
/home/yareeez/AIST-hand/.venv/bin/python -m shadow_ext.teleop_driver pick_bucket \
  --view --key-control --recv-fingers --duration 300 \
  --session /home/yareeez/AIST-hand/recordings/shadow-take-001
```

Puedes conservar tus opciones habituales `--playback-poses`, `--pose-file` y
`--grasp-class` en ese segundo comando. El ejemplo no selecciona tus waypoints.

Mueve brazo y dedos como siempre. Para terminar, detén el simulador con Ctrl+C o
cerrando su ventana; luego detén la webcam con Q/ESC en su ventana o Ctrl+C.
Si la tarea alcanza éxito, el simulador conserva su comportamiento actual de
terminar automáticamente. Detén después la webcam para finalizar su archivo.

La captura empieza cuando arranca cada proceso; los timestamps alinean sus
grabaciones aunque no comiencen al mismo instante. Ambos deben ejecutarse en el
mismo equipo y arranque del sistema. La exportación comprueba esa identidad.

El driver actual interpreta `--duration` como presupuesto de tiempo simulado:
lo convierte a pasos físicos. El tiempo real puede ser mayor.
El registro indica esta diferencia explícitamente. No usar ese argumento como
si garantizara un límite de tiempo real.

## Exportar el mismo ensayo desde dos cámaras

Una vez detenidos ambos procesos:

```bash
cd /home/yareeez/dexjoco-shadow
MUJOCO_GL=egl /home/yareeez/AIST-hand/.venv/bin/python -m shadow_ext.replay \
  /home/yareeez/AIST-hand/recordings/shadow-take-001/sim \
  --record /home/yareeez/AIST-hand/recordings/shadow-take-001/render/take.mp4 \
  --cam front,left --width 1280 --height 720 --fps 30 --with-operator
```

`egl` funcionó para la exportación sin ventana en este equipo. Las cámaras
disponibles dependen del modelo guardado; en la escena bucket existen
`front`, `left` y `right`.

La exportación produce:

- `take_front.mp4`, `take_left.mp4`: dos vistas del mismo movimiento.
- `take_operator.mp4`: webcam ajustada a la misma línea temporal.
- `take_paired_front.mp4`, `take_paired_left.mp4`: webcam y simulador lado a lado.
- `take_timeline.csv`: cuadro exportado, tiempo, índice del estado, índice de webcam,
  tiempo simulado y éxito de tarea.
- `take_export.json`: parámetros, archivos y estado de finalización de la exportación.

Omite `--with-operator` para exportar solamente el simulador.
Para otra resolución o selección de cámaras, usa un prefijo de salida nuevo.
Los archivos previos no se sobrescriben.

No se vuelve a avanzar la física ni a ejecutar la tarea durante replay.
Todas las cámaras usan el mismo estado por cuadro.
Se conserva el último estado, incluido el de éxito, aunque para mostrarlo sea
necesario extender la salida hasta un intervalo de cuadro adicional.

## Qué contiene la sesión

```text
shadow-take-001/
  operator/
    camera.avi
    frames.csv
    metadata.json
  sim/
    model.mjb
    metadata.json
    000000.npz
    000001.npz
    ...
    teleop_driver.py
    tasks.py
    build.py
    mapping.py
    state_recording.py
    waypoints.json        # si se proporcionaron para reproducción
    initial_pose.json     # si existía el archivo de pose
  render/
    ...
```

El AVI de webcam es un contenedor de cuadros a 30 fps nominales. Su reproducción
directa no representa necesariamente la duración real. Para video sincronizado
utiliza la exportación, que lee `frames.csv`.

La captura guarda imágenes crudas, sin espejo ni esqueleto superpuesto.
Los timestamps corresponden al retorno de `camera.read()`, no al instante de
exposición del sensor. No eliminan la latencia de cámara, inferencia o control.
Si el codificador se sobrecarga, se cuenta y comunica la pérdida de cuadros.
La exportación mantiene la última imagen disponible entre adquisiciones; muestra
un cuadro vacío identificado fuera de la cobertura temporal de la webcam.

El simulador guarda:

- Estado `mjSTATE_INTEGRATION` completo: incluye posiciones de brazo, dedos y objetos,
  velocidades, controles, activaciones, objetivos mocap y otros campos de integración.
- `body_gravcomp` por paso para conservar cambios de compensación gravitacional.
- Objetivos de dedos antes de compensación y sesgo de deslizamiento aplicado.
- Tiempo real monotónico, tiempo simulado y booleano de éxito por estado.
- Un estado inicial y un estado después de cada paso y actualización de tarea.
- Modelo compilado congelado, versión de MuJoCo, hash del modelo y copias/hashes del
  código principal del simulador y de los archivos de poses usados.
- Condiciones principales y resultado resumido en `sim/metadata.json`.

El resumen incluye `success_ever`, primer instante de éxito,
`time_to_success_wall_s`, `time_to_success_sim_s`, duraciones y `stop_reason`.
El tiempo empieza antes del primer paso físico e incluye preparación dentro del
simulador; todavía no existe una marca separada de “iniciar evaluación”.

Motivos de terminación actuales: `success`, `step_limit`,
`manual_interruption`, `technical_error`.
`complete` describe si el registro se cerró; no significa que la tarea tuvo éxito.
La recuperación de una ejecución sin cierre puede usar los bloques ya escritos;
no garantiza recuperar el bloque que estaba en memoria.

El checkpoint se identifica por ruta y SHA-256 en `operator/metadata.json`.
No se copia automáticamente el checkpoint; conservar ese archivo por separado.

## Verificación realizada

Comando de las pruebas:

```bash
cd /home/yareeez/dexjoco-shadow
MUJOCO_GL=egl /home/yareeez/AIST-hand/.venv/bin/python -m pytest \
  shadow_ext/tests/test_session_recording.py -q
```

Resultado: 6 pruebas aprobadas.

- Igualdad exacta de los vectores de estado de integración después de guardar/restaurar,
  incluidos estados de brazo, dedos, objeto, controles y mocap; cambios de gravedad.
- Lectura entre bloques, acceso hacia atrás y recuperación de bloques completos.
- Captura de webcam sintética, selección temporal y exportación de dos cámaras
  más vistas emparejadas con igual número de cuadros y fps.
- Exportación con `mj_step` prohibido por la prueba: confirma que no reintegra física.
- Cierre del driver ante éxito, interrupción y error, conservando cambios de tarea
  antes de guardar el estado.

También se ejecutó una captura corta de la escena Shadow bucket real y se exportaron
las cámaras front/left. No fue una prueba humana ni una ejecución exitosa de la tarea.
Las pruebas de éxito usan una tarea sintética para verificar el registro del resultado.

## Próximas comprobaciones

1. Completado: toma real con brazo, dedos y objeto; éxito de tarea y grabación
   confirmados por el operador (véase el resultado al final).
2. Comprobar el encuadre para las figuras/videos definitivos.
3. Extender el registro para tesis: magnitudes del criterio de tarea, eventos de
   waypoints/correcciones, marca de inicio de evaluación y resumen consolidado por ensayo.
4. Validar el flujo de la otra mano antes de incluir resultados de Allegro.

El registro básico no implementa todavía todos los campos del
[protocolo metodológico](../../ThesisGCN/protocolo-validacion-funcional-registro-2026-09-07.md).


## Primera comprobación en vivo completada

Sesión local: `recordings/shadow-live-check-20260907-224629`.
El operador completó `pick_bucket` con Shadow; el driver registró éxito y terminó
automáticamente. Compensación de deslizamiento y compensación de gravedad del
objeto estaban desactivadas. Éxito a 63.296 s simulados / 310.351 s reales,
medidos desde el comienzo de captura del simulador.

Se guardaron 31,649 estados y 3,775 cuadros de webcam, sin descartes en la cola
del codificador. Se exportaron y verificaron cinco videos sincronizados,
incluidas front/left y las dos vistas emparejadas, con 3,105 cuadros cada uno a
10 fps. El último cuadro conserva el resultado exitoso; la webcam cubre toda
la línea temporal exportada.

El estado final restaurado también satisface la comprobación derivada de lift
mínimo de 0.152282 m y posición del sensor de comida dentro de los límites de
referencia. Este resultado no reemplaza el booleano original de la tarea.

[Nota local del ensayo y videos](../recordings/shadow-live-check-20260907-224629/README.md).
Los datos del ensayo permanecen locales. El operador confirmó que tanto la
simulación como la grabación fueron exitosas. Sigue pendiente el registro
ampliado de tesis y preparar las repeticiones del protocolo.
[Resumen numérico versionado](resultado-shadow-live-2026-09-07.json).
Una sola toma exitosa documenta factibilidad, no una tasa de éxito general.
