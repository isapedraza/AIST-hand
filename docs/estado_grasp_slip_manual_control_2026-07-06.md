# Estado 2026-07-06 — grasp slip (MuJoCo) + herramienta de control manual

## Contexto

Reporte del usuario: en sim, el brazo llega bien al objetivo (reach ok), pero
los objetos **siempre se resbalan** de la mano. Específicamente: agarra bien
estático, pero al subir el brazo (fase de "place") el agarre se desarma --
los dedos se ven aflojar/abrir, como si el slip deformara la mano. Pasa en
TODAS las tareas del eval suite (bucket/water_plant/pinch_tongs/hammer_nail),
no es específico de un objeto. Confirmado explícitamente por el usuario:
**no es culpa del modelo/checkpoint** ("modelos buenos, no el switching").

## Diagnóstico (confirmado vía código fuente + búsqueda web)

Causa raíz: contacto dedo-objeto sin fricción torsional + cono de fricción
piramidal (default MuJoCo). Todos los fingertip geoms (Allegro
`panda_allegro_copy.xml` y Shadow `right_hand.xml`) heredaban el default
global de MuJoCo (`condim=3`, `friction="1 0.005 0.0001"`) -- sin resistencia
al giro (torsional) ni al rolling. Esto explica el patrón exacto: agarre
estático aguanta (sin torque), pero al mover el brazo el objeto queda libre
de girar dentro de la mano y se escapa por rotación, no por deslizamiento
plano.

Confirmado por búsqueda web que es un problema **conocido y común** en
MuJoCo para manos diestras (no bug propio): contacto "soft" por diseño +
cono piramidal + sin fricción torsional por default es la combinación
clásica reportada en foros/papers de manipulación diestra.

## Fixes aplicados (repo `dexjoco`, fork separado, branch `shadow-support`)

1. `condim="4" friction="1.0 0.02 0.005"` en fingertip collision de:
   - `dexjoco/dexjoco/sim/envs/xmls/panda_allegro_copy.xml` (Allegro)
   - `dexjoco/dexjoco/sim/envs/xmls/shadow_hand/right_hand.xml` (Shadow)
2. `cone="elliptic" impratio="10"` en `<option>` de las 5 arenas del eval
   suite (bucket_pick, glass_v2, plant, hammer_nail, table_tongs).
3. `noslip_iterations="5"` ya estaba presente (no se tocó, ya era correcto).

Hipótesis secundaria (no aplicada, evaluada y descartada por ahora): subir
`kp` de los actuadores de posición para que no "cedan" bajo el torque del
peso del objeto. Investigado en literatura -- resultado mixto: papers RL de
Allegro usan `kp=1` (nuestro Allegro ya tiene `kp=2`, en línea o arriba de lo
típico), así que no es un fix "estándar conocido" como el de fricción.
Riesgo de meter oscilación/inestabilidad si se sube a ciegas. **Pendiente
medir** (qpos target vs. real durante el lift) antes de tocarlo.

## Herramienta nueva: control manual sin teleop (`teleop_driver.py --key-control`)

Motivación: usuario quería mover el brazo con teclado y cerrar con un
agarre canónico pre-hecho, para poder probar el grasp/slip directamente sin
depender del pipeline de teleop humano (WiLoR/retargeter).

Implementado en `dexjoco/shadow_ext/teleop_driver.py`:
- Traslación: `W/S/A/D/Q/E` (ya existía).
- Rotación (nuevo): `I/K`=pitch, `J/L`=yaw, `U/O`=roll -- quaternion acumulado
  sobre la pose home vía `mju_axisAngle2Quat`+`mju_mulQuat`.
- Agarre continuo (nuevo): `UP/DOWN` interpolan aperture 0..1 entre un ancla
  **open fija y universal** (`qpos=0`, extensión completa -- el `pose_open`
  del YAML NO sirve para esto, es medoid de "grasp menos cerrado" del
  dataset, no mano abierta real, ver nota abajo) y el ancla **close variable
  por clase Feix** (`pose_close` de la clase elegida, default ahora
  `"Parallel Extension"`, la más cercana a "parallel grasp" en esta
  taxonomía -- no existe clase "Parallel" exacta). `G` = snap open/close.
- Guardar/cargar pose (nuevo): tecla `P` guarda posición+orientación+aperture
  actual en JSON (`shadow_ext/saved_key_pose.json` default, override con
  `--pose-file`); si el archivo existe, se retoma ahí al arrancar -- evita
  reposicionar desde cero cada vez.
- **Cambio de mecanismo de input**: pynput (hook global vía X Record) no
  funciona bajo WSLg -- el listener arranca sin error pero nunca recibe
  teclas. Reescrito para usar `key_callback` del propio
  `mujoco.viewer.launch_passive` (eventos GLFW de la ventana enfocada), que
  sí funciona en WSLg. Requiere `--view` y foco en la ventana del visor.
  `_PoseRecorder` (`--record-poses`, tecla Space) sigue en pynput sin tocar
  -- fuera de alcance de esta sesión, puede tener el mismo problema.

Comando:
```
cd /home/pc_pro/AIST-hand/dexjoco
/home/pc_pro/AIST-hand/.venv-dexjoco/bin/python -m shadow_ext.teleop_driver \
  pick_bucket --view --key-control --duration 86400
```

## Resultado y plan de contingencia

El usuario probó manualmente con estas modificaciones (fricción/cono +
control manual) y **sigue sin poder agarrar el objeto correctamente**. Va a
reintentar en casa con esta versión ya pusheada.

**Si de plano no funciona**: se abandona esta prueba cualitativa de
grasp-hold-transport en sim, y la evaluación se apoya únicamente en los
resultados numéricos existentes (métricas de éxito de tarea del eval suite,
ver `[[project_shadow_teleop_eval_protocol]]`), sin insistir en arreglar el
slip físico de MuJoCo para la tesis.

## Pendiente

1. Confirmar en casa si el fix de fricción/cono resolvió o no el slip con el
   nuevo control manual.
2. Si sigue fallando: decisión de abandonar esta vía (arriba) -- no seguir
   invirtiendo tiempo en tuning de física de MuJoCo para este síntoma.
3. Si se decide seguir insistiendo: siguiente sospechoso medible es rigidez
   de actuador (`kp`) vs. reaction force real durante el lift (log qpos
   target vs. real), no un fix a ciegas.
