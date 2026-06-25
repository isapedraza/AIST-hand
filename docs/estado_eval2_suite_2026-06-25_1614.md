# Estado — Suite Eval 2 single-hand completa + recorder

**Fecha:** 2026-06-25 16:14:36 CST

Continúa `docs/estado_protocolo_eval_2026-06-25_1430.md`. Reporte tras cerrar las
4 tareas single-hand de Eval 2 (incl. hammer_nail) y validar el grabador de
figuras. Todo el código vive en el fork `dexjoco-shadow` (`shadow_ext/`); este doc
y las memorias viven SOLO en AIST-hand main (regla del usuario).

## Suite Eval 2 single-hand: COMPLETA (4/4)

| Task | Skill | Éxito (copia fiel de DexJoCo) | Estado |
|---|---|---|---|
| `pick_bucket` | power + lift | comida en balde + levantada 15cm | corre |
| `water_plant` | trigger (índice) | boquilla en cilindro + gatillo, 30 steps | corre |
| `pinch_tongs` | pinch | pinzas levantadas + 3 pinchazos, 30 steps | corre |
| `hammer_nail` | power + muñeca 6D | nail_depth >= 0.04 (física de impacto) | corre |

Las 4 corren limpio, sin warnings, con la métrica `succeed` de DexJoCo conectada.

**Importante:** las 4 dan `succeed=False` hoy. Es ESPERADO, no bug — la muñeca
está FIJA y no hay teleop vivo, así que nada alcanza/golpea el objeto. El éxito
real lo dispara el operador cuando se enchufe la fuente 6D al mocap (ver abajo).

## Arquitectura (ruta B, "nuestro env mínimo")

No se tocan los envs gym de DexJoCo (cableados a Allegro). En vez:

- `shadow_ext/build.py` — attach Panda+Shadow por arena (MjSpec).
- `shadow_ext/mapping.py` — qpos[24] retargeter → ctrl[20] (suma de 4 tendones).
- `shadow_ext/tasks.py` — REGISTRY: una clase por task con `_compute_success`
  COPIADO fiel de DexJoCo (no se inventa métrica). 3 son lectura pura de estado;
  hammer_nail es stateful y ESCRIBE en data (baja el mocap del clavo al golpear).
- `shadow_ext/teleop_driver.py` — env mínimo: carga escena, sostiene brazo vía
  `opspace` (OSC) de DexJoCo, dedos vía mapping, `mj_step`, reporta succeed.
  Task-parametrizado (`python -m shadow_ext.teleop_driver <task>`).
- `shadow_ext/recorder.py` — grabador offscreen para figuras (ver abajo).

## Bug arreglado: mocap del Panda por nombre

El driver resolvía el mocap del brazo por índice 0. Pero la escena hammer tiene
2 mocaps: `nail`=índice 0, `target`(panda)=índice 1. Con índice 0 el brazo habría
seguido al CLAVO, no a la muñeca. Fix: resolver `model.body("target").mocapid`.
Las 3 escenas single tienen `target` en índice 0; hammer en índice 1. Verificado.

## Recorder (figuras de tesis) — validado

`recorder.py`: wrapper de `mujoco.Renderer` offscreen sobre cualquier cámara de
escena (`front, back, left, right, top0, handcam_rgb`). Foto (PNG) + video (MP4
vía imageio). Headless, sin display, sin cableado RL.

- Flags en el driver: `--shot foto.png`, `--record video.mp4`, `--cam front`.
- Probado: 6 vistas a la vez a 480p (lento en CPU); en práctica 1-2 cámaras a
  720/1080p (rápido). Calidad confirmada buena en `shadow_demo_recordings/`
  (sweep senoidal del brazo, 6 mp4 + 1 png de referencia).
- Uso real previsto: grabar 1-2 cámaras y sincronizar con video del operador
  (figura side-by-side humano+robot).

## Lo grande pendiente: fuente 6D → mocap

El driver mantiene la muñeca FIJA (stage 1). Para que CUALQUIER task se complete
hay que alimentar `data.mocap_pos/quat[target]` con la muñeca global 6D del teleop
(WiLoR global_orient + traslación). El camino mocap→brazo YA funciona (probado en
el sweep del recorder: el brazo siguió). Solo falta enchufar la fuente viva en vez
del seno. Esto es stage 2/3 y vale para las 5 tareas. Hammer (física del clavo) es
scoring encima; la que mueve todo es la muñeca 6D.

## Bimanual: bloqueado (2 frentes)

- (a) sim: extender `build_spec` a 2 sites (2 Shadows). Hoy bimanual_hanoi compila
  con UNA sola mano (nu=34). Lado nuestro, chico.
- (b) modelo: doble alimentación (2 instancias retargeter, izq=espejo). Lado
  usuario, NO enchufado. Bloquea el demo bimanual. (a) no espera a (b).

## Pendientes

1. **6D source → mocap** (lo grande; desbloquea éxito en vivo de las 5 tasks).
2. Bimanual: build_spec 2 manos + doble feed del usuario.
3. Limpieza opcional: `shadow_demo_recordings/` (6 mp4 de prueba en AIST-hand,
   no ignorados por git).

## Decisiones del usuario (suite Eval 2)

Aún por fijar: nº de intentos por task para el %, y robots (Allegro nativo gratis /
Shadow / ambos para el claim cross-embodiment).

## Commits del fork (branch shadow-support, YahelRamirezP/dexjoco-shadow)

- `390e3e6` task registry (bucket, water_plant, pinch_tongs)
- `34c08cc` recorder (--shot/--record/--cam)
- `e51fd71` hammer_nail + fix mocap por nombre
