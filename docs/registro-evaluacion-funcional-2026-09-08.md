# Registro ampliado de evaluación funcional

Estado: 2026-09-08. Continuación de la [toma Shadow exitosa](grabacion-sesion-shadow-2026-09-07.md).
La captura y el video existentes se conservan como base. La comparación con/sin
compensación de deslizamiento queda fuera del trabajo actual.

Continuación de física: [auditoría del simulador](auditoria-fisica-shadow-2026-09-08.md),
con inventario compilado, 31 649 estados comprobados y 18 continuaciones locales.
Antes de afirmar correspondencia con hardware, resolver la carga del montaje
Panda–Shadow y justificar masas/materiales. La configuración de trabajo se conserva.

## Qué se implementó

- Inicio explícito de evaluación con **F5**, después de preparar el ensayo.
- Límite de **tiempo real** desde esa marca, independiente del tiempo simulado.
- Progreso numérico de la tarea en cada estado, obtenido de la actualización
  original de la tarea, sin ejecutarla dos veces.
- Eventos de teclado, cambios de waypoint, guardado de waypoints y marcas de fase **F6**.
- Resumen por sesión y consolidación de varias sesiones, conservando prácticas,
  interrupciones y errores con motivos de exclusión explícitos.
- Identificación de operador, condición, mano, checkpoint, modelo y código efectivo.
  La webcam conserva copias de los módulos Python del proyecto cargados al iniciar
  su grabador, con hashes y estado de Git.
- Ruta Allegro de 16 articulaciones para el mismo grabador y exportador.
- Detección de inestabilidad/reinicio de MuJoCo y cierre de recursos con errores
  identificados; un fallo de cierre no impide intentar los demás cierres.

El driver y los tests viven en el repositorio hermano `dexjoco-shadow/shadow_ext`.
El registro de webcam y su procedencia viven en AIST-hand.

## Próxima prueba práctica

Usa una carpeta nueva por intento y conserva tus comandos habituales de cámara,
checkpoint, calibración, poses y waypoints. Arranca primero la webcam y espera
la calibración. Añade al retargeter:

```text
--robot shadow --emit-udp --no-viewer --session /home/yareeez/AIST-hand/recordings/shadow-practice-001
```

Añade al simulador, que debe conservar `--view --key-control --recv-fingers`:

```text
--session /home/yareeez/AIST-hand/recordings/shadow-practice-001
--evaluation-start manual --time-limit 600
--trial-kind practice --operator-id yahel --condition-id shadow-bucket-v1
```

Las líneas anteriores son opciones del mismo comando. **600 s es un ejemplo
configurable**, no un límite metodológico ya acordado.

1. Prepara brazo y dedos sin completar ni adelantar físicamente la tarea.
2. Pulsa **F5 en la ventana del simulador** cuando comience el intento.
3. Realiza la tarea. Opcionalmente pulsa **F6** en fronteras de fase previamente
   definidas; el evento es una marca del operador, no una detección de agarre.
4. El simulador termina por éxito o límite real. Detén después la webcam como antes.
5. Consolida la sesión después de cerrar ambos procesos y exporta los videos
   con el comando habitual de replay.

F5 se acepta una sola vez. Se reinician los contadores y referencias de la tarea
al comenzar evaluación; para bucket, la primera actualización posterior toma
las alturas de referencia. La preparación permanece en el video y en la traza,
pero sus tiempos y éxitos se separan de los resultados de evaluación.
El comienzo inmediato para pruebas automáticas usa `--evaluation-start immediate`.

Sin `--evaluation-start`, se conserva el presupuesto histórico de `--duration`
en tiempo simulado. Con inicio explícito, ese presupuesto implícito deja de
terminar el ensayo; `--n-steps`, si se pasa expresamente, sigue siendo un tope
adicional y su terminación se identifica como `step_limit`.

La comprobación del límite ocurre entre pasos. Puede sobrepasarse por el costo
de un paso o del visor; un éxito observado al alcanzar o superar el límite
se conserva como resultado bruto, pero no cuenta como éxito de evaluación.

## Archivos y medidas

```text
SESSION/
  trial_summary.json
  operator/
    metadata.json
    source/...             # copias del código del retargeter
    camera.avi
    frames.csv
  sim/
    metadata.json
    events.jsonl
    model.mjb
    000000.npz
    ...
    trial.py               # junto con las demás fuentes congeladas
```

La traza pasa a formato 2; el lector conserva compatibilidad con formato 1.
Las columnas nuevas son `task_metrics` (JSON Unicode, sin pickle) y
`evaluation_active`. Los estados de integración y el replay mantienen su formato
de contenido físico; el exportador no reintegra la simulación.

Para bucket se guardan posición del sensor de comida, ocho esquinas de referencia,
alturas base, incrementos de los cuatro sitios inferiores, incremento mínimo,
umbral de 0.15 m y booleanos de contención en AABB/elevación. Las otras tareas
guardan sus cantidades, umbrales, contadores y estados de activación respectivos.

`sim/metadata.json.evaluation` contiene inicio, primer éxito y duraciones real y
simulada del intervalo de evaluación. Los campos históricos de captura siguen
midiendo desde el inicio de grabación. Un tiempo a éxito ausente es `null`, nunca cero.

Los eventos incluyen reloj monotónico, tiempo simulado y estado grabado precedente.
Los eventos de teclado conservan además el instante de recepción y las correcciones
solicitadas. El estado físico posterior conserva el objetivo mocap aplicado.
Los controles de apertura canónica no gobiernan los dedos cuando se usa
`--recv-fingers`; los dedos siguen el retargeter.

El resumen creado al cerrar el simulador es provisional si la webcam sigue
grabando. El comando siguiente vuelve a leer ambos metadatos y actualiza el resumen.

## Consolidar intentos

Desde `/home/yareeez/dexjoco-shadow`, una vez detenidos ambos procesos:

```bash
/home/yareeez/AIST-hand/.venv/bin/python -m shadow_ext.trial \
  /home/yareeez/AIST-hand/recordings/shadow-practice-001 \
  --output /home/yareeez/AIST-hand/recordings/resumen-practice-001.json
```

Se pueden pasar varias carpetas. Una carpeta representa un intento, aunque tenga
varios videos. Una carpeta repetida se rechaza. El archivo de salida debe ser nuevo.

El informe conserva todas las sesiones y agrupa las configuraciones compatibles.
Cuenta como evaluable una sesión marcada `evaluation`, iniciada explícitamente,
con traza completa, terminación `success` o `time_limit`, y webcam completa de
la misma mano/reloj que cubra la evaluación. Las restantes aparecen con sus
motivos de exclusión. No se borran interrupciones, errores ni prácticas.

Cada grupo muestra éxitos/intentos evaluables, excluidos, motivos de terminación,
proporción de éxito y mediana/rango de tiempos reales de los éxitos.
El informe identifica el criterio de inclusión empleado.

Antes de iniciar una serie formal, fija cantidad de intentos, límite temporal,
waypoints, checkpoint y correcciones permitidas. Usa entonces
`--trial-kind evaluation` con operador y condición explícitos. Si cambias la
configuración, usa otro identificador de condición. El software registra la
configuración y evita mezclar diferencias identificadas; no decide cuántas
repeticiones bastan para la tesis ni ejecuta por sí solo los intentos humanos.

## Allegro

Retargeter: `--robot allegro --emit-udp`. Simulador:
`--hand allegro --recv-fingers`, junto con las mismas opciones de sesión/evaluación.
Usa poses y waypoints preparados específicamente para Allegro.

El mapeo respeta el orden ff/mf/rf/th, cuatro articulaciones por dedo. El receptor
rechaza paquetes de otra dimensión y valores no finitos. Allegro usa su escena
nativa y sus ganancias; se selecciona integración `implicitfast` para evitar la
inestabilidad observada con Euler y realimentación de velocidad a 2 ms.
Esta elección y el modelo efectivo quedan registrados. Shadow conserva sus ajustes.

La ruta Allegro requiere dedos recibidos; no reutiliza los agarres canónicos de
Shadow ni sus opciones de asistencia/diagnóstico de agarre.
La prueba automática de dinámica, mapeo y restauración no demuestra éxito humano
en la tarea. Falta esa comprobación práctica y calibrar sus trayectorias.

## Verificación y continuación

Resultado: **26 pruebas aprobadas** en el repositorio del simulador, incluidas
500 iteraciones físicas Allegro y exportación front/left con avance de física
prohibido durante replay. También se restauró la toma histórica de 31,649 estados:
su último estado conserva el éxito. Las advertencias restantes provienen de
`dm_robotics.transformations` y su uso de `where` sin `out`.

Suite enfocada:

```bash
cd /home/yareeez/dexjoco-shadow
MUJOCO_GL=egl /home/yareeez/AIST-hand/.venv/bin/python -m pytest \
  shadow_ext/tests/test_session_recording.py shadow_ext/tests/test_trial.py \
  -q --tb=short -o tmp_path_retention_policy=failed
```

La suite cubre fidelidad de estados, compatibilidad antigua, exportación sincronizada
sin física, tiempos y éxito tardío, inicio manual único, métricas de las cuatro tareas,
resúmenes, UDP de ambas manos, dinámica Allegro, procedencia y fallos de cierre.

La toma Shadow de práctica con F5 ya se comprobó; véase el resultado actualizado
al final. Sigue la comprobación Allegro en vivo antes de fijar la serie formal.
La observación histórica de salida 1 sin traceback no se atribuye a una causa
demostrada; ahora los fallos de cierre Python quedan identificados en el registro.

## Ensayo Shadow bucket comprobado — 2026-09-08

Alcance acordado: **únicamente pick_bucket con Shadow y Allegro**.
La sesión `shadow-bucket-practice-003` completa la comprobación en vivo del
registro ampliado Shadow: un inicio F5, éxito a 399.808 s reales desde F5,
34 565 estados y 4 819 cuadros de cámara sin descartes del codificador.
Ambos procesos cerraron correctamente; la cámara cubre todo el intervalo medido.
Los estados, métricas, eventos y hashes se comprobaron; el estado final restaurado
conserva el criterio de éxito. Las sesiones 001/002 son interrupciones antes de F5.

[Resultado y videos locales](../recordings/shadow-bucket-practice-003/README.md).
El resumen actualizado conserva sólo `not_an_evaluation_trial` como exclusión,
por tratarse de práctica. Siguiente paso: poses/waypoints propios de Allegro y
su ensayo en vivo. La justificación física sigue pendiente antes de la serie formal.

Alcance de las próximas sesiones: pruebas de concepto, no ensayos finales.
[Prueba Allegro bucket: comandos](prueba-concepto-allegro-bucket-2026-09-08.md).
[Fuerzas de contacto: recomendación y fuentes](fuerzas-contacto-prueba-concepto-2026-09-08.md).
El logging directo de fuerzas queda como ampliación opcional; no bloquea Allegro.
