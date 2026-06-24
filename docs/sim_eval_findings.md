# Hallazgos — Validación funcional en simulación del teleop

Fecha: 2026-06-24

Investigación sobre cómo probar el sistema de teleoperación de mano sin robot
físico: una simulación que actúe como sustituto del robot, donde el operador
teleopera en vivo y agarra/sostiene/mueve objetos.

## Conclusión central

- **No existe un drop-in** que encaje con nuestras restricciones (input RGB +
  nuestro propio retargeter cross-embodiment + CPU sin GPU + MuJoCo).
- La ruta realista = **extender nuestro `apps/graphgrasp-live/sinks/mujoco_sink.py`**
  robando piezas probadas de tres proyectos. No es construir de cero.
- No es imposible; el grueso de la integración (retargeter, fuente WiLoR, robots
  en MuJoCo) ya está hecho. Falta objeto + muñeca + física.

## Qué descartamos y por qué

- **DexJoCo extendido a Shadow**: semanas. MJCF Panda+Shadow (tendones acoplados)
  + cirugía en 11 envs hardcodeados a Allegro + re-tuning de éxito por task. No
  evita el problema de muñeca. Sin payoff para la tesis.
- **SAPIEN / AnyTeleop**: SAPIEN usa PhysX 5, **requiere GPU**. La máquina local
  no tiene GPU dedicada. MuJoCo corre en CPU y tiene mejor contacto para
  manipulación dextra. Cambiar de simulador = peor en todo.
- **DexGraspBench como prueba de teleop**: error de categoría. Ejecuta su propio
  script cerrar+levantar; el teleop (seguir al humano en tiempo real) nunca se
  ejecuta. Mide graspabilidad de una pose, no teleoperación.
- **dex-retargeting (ejemplos públicos)**: `show_realtime_retargeting.py` y
  `hand_robot_viewer.py` son display cinemático / replay de dataset, **sin
  física de agarre**. Mismo nivel que nuestro sink. No sirven tal cual.

## Mapa de robo (cada repo, una pieza)

| Pieza que falta | De dónde | Archivo/clase |
|---|---|---|
| Motor física MuJoCo: mano flotante mocap-soldada + objeto + contacto + lift | **DexGraspBench** | `src/util/hand_util.py` → clase `MjHO` |
| Loop teleop en vivo: aplicar pose streameada cada frame → `mj_step` → viewer | **DexJoCo** | `tasks/sim_teleop.py` (patrón UDP muñeca+mano → mocap+ctrl) |
| Escena realista: mesa + objetos (regadera, balde, martillo, etc.) | **DexJoCo** | `dexjoco/sim/envs/xmls/*.xml` + `table_arena`. XMLs de objeto son separables del robot |
| Diseño: desacople muñeca/dedos, suavizado, motion-gen | **AnyTeleop** | Solo concepto (paper). Código no (SAPIEN/GPU). Confirma que nuestro diseño es el correcto y citable |

## Cómo funciona el motor de DexGraspBench (`MjHO`)

- `timestep=0.004`, `mj_step` con contacto real. CPU.
- **Muñeca = mocap body soldado a la mano** (weld constraint). Se escribe
  `data.mocap_pos/mocap_quat`; el weld arrastra la mano hacia ese target pero la
  mano tiene masa e inercia y los contactos la empujan. Así se sostiene una mano
  flotante en pose comandada y se obtienen fuerzas de contacto.
- **Dedos = actuadores de posición** (`data.ctrl = _qpos2ctrl(qpos)`).
- **Objeto = cuerpo libre** con masa/fricción.
- Nota: `MjHO` **desactiva gravedad** y jala el objeto con fuerza externa para
  testear robustez del agarre. Para pick-and-hold natural → **reactivar gravedad**.
- Nota: carga objetos como `obj_path/urdf/meshes/convex_piece_*.obj` (descomp.
  convexa). Formato distinto al XML de escena DexJoCo → mezclar requiere
  reconciliar la carga de objeto. Tenemos `BODex_shadow` (formato convexo) listo.

## Métricas de Yan (eval cinemático — núcleo, separado de lo funcional)

Las métricas de evaluación de Yan son las **mismas funciones de similitud** del
entrenamiento, computadas sobre el test split:

- **RS** (Rotation Similarity): `D_R = sum_j (1 - <q_j^A, q_j^B>^2)`, distancia
  angular cuaternión por junta.
- **NDS** (Normalized Distance Similarity): `||p_ee^A - p_ee^B||`, posición yema
  en frame local, normalizada por escala.
- **NVS** (Normalized Velocity Similarity): `||v_hand - v_ee||`, velocidad yema
  (ata a nuestra `L_ltc`).
- **MPJPE**: error posición por junta, estándar.

Nuestra `S_k` (yema + pinza + distal + UDHM) ya es esto adaptado a mano. El eval
cinemático **cubre ambos robots** (Shadow + Allegro), sin simulador, y mide lo
que un teleop hace: seguir al humano. Es el núcleo defendible.

## El "problema de muñeca" — resuelto conceptualmente

- El retargeter (`models/latent-retargeting/src/cross_emb/inference/retarget.py`)
  produce `qpos [B, J]` (J=24 Shadow / 16 Allegro, orden Menagerie), clipeado a
  límites. **Solo dedos/mano. No produce muñeca 6DoF de posición en el espacio.**
- Esto **no es problema**: la muñeca es global y embodiment-agnóstica. Sale del
  source, no del retargeter. Arquitectura correcta (= AnyTeleop):
  - dedos → retargeter → `data.ctrl`
  - muñeca global → directo del source → `data.mocap_pos/mocap_quat`
- La muñeca global **ya existe** en el source pero hoy se tira. En
  `apps/graphgrasp-live/sources/wilor_source.py`, `next_frame()` devuelve solo
  los quats Dong locales `[1,20,4]`. El paso `_extract_points()` tiene los 21
  keypoints en frame cámara (muñeca = keypoint[0]); `DongKinematics` calcula la
  rotación world→muñeca-local (= orientación de muñeca). Ambos disponibles un
  paso arriba; solo no se reenvían. Surfacearlos = cambio pequeño al source.

## Riesgo real único

La **calidad** de la traslación de muñeca de WiLoR (`pred_cam_t_full` / keypoint[0]):
monocular, ruidosa, escala ambigua, ~253ms de lag por el túnel cloudflared. La
orientación (frame Dong) es más confiable que la posición.

Mitigación: filtro one-euro/EMA sobre la traslación, o fallback de muñeca anclada.

## Plan por etapas (riesgo creciente; cada etapa ya es demo)

La muñeca tiene 6 DoF: 3 posición + 3 orientación. Las etapas los van soltando.

1. **Muñeca fija, orientación fija, dedos en vivo.** Garantizado. Objeto
   pre-colocado en zona de agarre; cierras dedos → agarra → sostiene. Mano no viaja.
2. **+ orientación de muñeca en vivo** (`global_orient`/frame Dong, señal
   confiable). Bajo riesgo. Mano gira para orientarse antes de cerrar.
3. **+ traslación de muñeca en vivo** (`cam_t` + filtro). Mano vuela libre,
   alcanza/agarra/mueve. Teleop completa. Si tiembla mucho, quedarse en etapa 2.

## Interfaces relevantes (para cuando se implemente)

- Sink host: `apps/graphgrasp-live/sinks/mujoco_sink.py` (viewer + carga
  shadow/allegro/leap desde `third_party/mujoco_menagerie`; hoy `mj_forward` +
  `qvel=0` = cinemático congelado).
- Glue: `apps/graphgrasp-live/live_retarget.py`.
- Retargeter: `cross_emb/inference/retarget.py`, `Retargeter(pose) → qpos`.
- Source: `apps/graphgrasp-live/sources/wilor_source.py`, `next_frame() → quats`.
- Motor a portar: DexGraspBench `src/util/hand_util.py::MjHO`.

## PROBADO (2026-06-24) — attach Panda+Shadow funciona

Re-estimación: extender DexJoCo a Shadow para UNA task NO es "semanas". El "semanas"
asumía portar los 11 tasks + MJCF desde cero. La pieza más temida (autorear MJCF
Panda+Shadow) está resuelta y probada.

Método (operación de árbol con `MjSpec`, mujoco 3.8.1, ~30 líneas):
1. Cargar el spec del Panda+Allegro (o la arena completa).
2. Borrar Allegro: `spec.delete(body "allegro_palm")` + actuadores/sensores/excludes
   que referencian juntas/bodies allegro (si no, refs colgantes → no compila).
3. Cargar Shadow (`mujoco_menagerie/shadow_hand/right_hand.xml`).
4. `spec.site("attachment_site").attach_body(shadow.body("rh_forearm"), "rh-", "")`
   — injerta el subárbol Shadow en el site de muñeca del Panda; prefijo `rh-`
   evita choques de nombres.
5. Rutas de mallas/texturas → absolutas (el xml se mueve de carpeta, las relativas
   rompen; este fue el único tropiezo).
6. `spec.compile()`.

Resultados verificados en viewer (`python -m mujoco.viewer`, glfw, WSLg):
- Panda+Shadow solo: nq=31, nu=27, nbody=42. Carga y simula.
- Escena bucket (mesa + balde + comida) con Shadow: nq=46, nu=28, nbody=45. Carga.
- El brazo cae bajo gravedad sin OSC (Franka usa motores de torque; ctrl=0 → sin
  sostén). Esperado, no bug. La mano (actuadores de posición) sí mantiene pose.

Scripts persistidos y reproducibles desde el repo:
- `experiments/shadow_dexjoco/attach_shadow.py` (Panda+Shadow solo)
- `experiments/shadow_dexjoco/attach_scene_bucket.py` (arena bucket con Shadow)
- Generan `panda_shadow.xml` / `scene_shadow_bucket.xml` en el mismo dir.
- Usan `dexjoco/` local + `third_party/mujoco_menagerie/shadow_hand`.

Nombres clave: en `panda_allegro_copy.xml` (el que usa la arena) los nombres van
SIN sufijo `_right` (body `allegro_palm`, site `attachment_site`); en
`panda_allegro_right.xml` van CON `_right`.

## Falta (no probado aún) para DexJoCo+Shadow funcional

1. Editar UN env (`panda_pick_bucket_env.py`): `_ALLEGRO_JOINT_NAMES`→Shadow,
   `_N_ALLEGRO` 16→(20 actuadores / 24 juntas), ctrl ids, fingertip bodies.
2. Mapeo qpos→ctrl con tendones acoplados de Shadow (24 juntas → 20 actuadores).
3. Arrancar OSC para que el brazo se sostenga/obedezca el target de muñeca.
4. Formato UDP del teleop wrapper (`tasks/sim_teleop.py`, puertos 5012 muñeca /
   5014 mano) para enchufar WiLoR + retargeter sin tocar su código.
5. Señal de muñeca WiLoR (riesgo de siempre; ortogonal a todo lo anterior).

## Fork pendiente

Cuando se formalice: fork de `brave-eai/dexjoco` (cuenta del usuario), mover ahí
los scripts de attach + escenas Shadow + el env editado. Por ahora el trabajo
vive en `experiments/shadow_dexjoco/` contra el clone `dexjoco/` local.

## Decisión pendiente

Editar el env (`panda_pick_bucket_env.py`) para Shadow + arrancar OSC, o cerrar
aquí y escribir la sección de evaluación de la tesis.
