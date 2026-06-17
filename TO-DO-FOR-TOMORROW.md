# TO-DO FOR TOMORROW (2026-06-12)

Contexto: EIGENGRASP_ONLINE quedo funcional y corriendo en Colab (multi-robot
shadow+allegro), PERO el sampling online es insostenible: ~38s/step (50 steps
en 32 min). Causa = `_sample_eigengrasp_q` corre `mujoco.mj_forward` en un loop
Python POR cada muestra (50k x 2 robots x cada step, en CPU). robot_loader.py:667.
Por eso existen los valid_poses precomputados. Decision: volver a DONG_CACHE
(precomputado) pero con caches del tamano correcto para caber en T4 free.

---

## TAREA 1: Generar caches precomputados (1M poses/robot) y volver a DONG_CACHE

Objetivo: training rapido (como Run 20) con AMBOS robots en T4 free (~12.7GB RAM).

- [ ] Generar valid_poses de cada robot DESDE el eigengrasp basis, 1M poses c/u.
      Script: `models/latent-retargeting/scripts/generate_valid_robot_poses.py`
      (corre el MISMO collision filter de mujoco, pero UNA vez offline aca, no por step).
      - shadow: eigengrasp=robot/hands/shadow_hand/datasets/processed/dexonomy_shadow_eigengrasps_balanced_sample.npz
                mjcf=third_party/mujoco_menagerie/shadow_hand/right_hand.xml
      - allegro: eigengrasp=robot/hands/allegro_hand/datasets/processed/eigengrasp_allegro.npz
                 mjcf=third_party/mujoco_menagerie/wonik_allegro/right_hand.xml
      - OJO: el DEFAULT_EIGEN del script (linea 32) apunta al nombre VIEJO/muerto
        (dexonomy_shadow_eigengrasps_balanced_phase_open_close_coeffstats_sample.npz).
        Pasar --eigengrasps explicito o arreglar el default. Tambien necesita
        --robot/--mjcf por robot y un flag de cantidad (=1_000_000). REVISAR ARGS.
      - El cache debe ser formato DONG (keys: q, quats, chain, tips, joint_labels,
        tip_labels, hand_config, [rot6]). Verificar que genera todo eso.
- [ ] Costo RAM esperado: ~0.6GB/robot (1M x ~600 bytes) -> ambos ~1.2GB. Sobrado.
- [ ] Subir los 2 valid_poses npz a Google Drive (MyDrive/AIST-hand/).
- [ ] Notebook (train_stage1_colab.ipynb): revertir ROBOTS a DONG_CACHE.
      - ROBOTS: poner 'valid_poses' (path Drive) por robot, QUITAR la dependencia
        de eigengrasp/mjcf del config de training.
      - CUIDADO CON EL GUARD: robot_loader hace EIGENGRASP_ONLINE exclusivo
        (commit 5515c1b57): si se pasa eigengrasp_path, DROPEA valid_poses y cae
        en el online LENTO. Para DONG_CACHE: NO pasar eigengrasp/mjcf, SOLO valid_poses.
      - _ROBOT_STATIC (cell-8): quitar eigengrasp/mjcf (o no escribirlos al yaml).
      - cell-12 (yaml writer): escribir valid_poses, NO eigengrasp/mjcf.
      - Verificar en el log que imprime mode=DONG_CACHE (no EIGENGRASP_ONLINE).
- [ ] RAM: ambos caben en free T4 con 1M. Confirmar con el RAM check de cell-12.

## TAREA 2: Batches cross-embodiment (Yan-style) en el training loop

Objetivo: alinear shadow<->allegro DIRECTAMENTE en el latente (no solo
transitivamente via humano), como Yan plantea.

- [ ] Hoy: loop POR robot (loop.py:400), triplets = humano + ESE robot
      (z_all_k = cat([z_h_k, z_r_k]), loop.py:467). NUNCA shadow vs allegro.
- [ ] Yan: un batch con humano + TODOS los robots; triplets entre CUALQUIER
      embodiment (anchor allegro, positive shadow, negative humano, etc).
- [ ] Comparabilidad cross-robot YA RESUELTA por el diseno del metric (NO es blocker):
      - Todo se proyecta a rep canonica comun (Dong stage-2): [5 dedos, MCP/PIP/DIP/TIP, xyz]
        en frame wrist-local NORMALIZADO por hand_length. No se tocan los 16 vs 24 joints crudos.
      - `common_labels` (loop.py:138,157) = interseccion de joints comparables entre
        los dos embodiments. El metric solo compara esos.
      - Filtrado (loop.py:457-458): `if not jidx: continue` salta dedos sin joints
        comunes (allegro no tiene pinky). "Solo compara lo comparable".
      - Los TRES terminos de S_k (D_R rot6 geodesico, D_joints Cartesiano normalizado,
        D_ahg) operan sobre ese espacio comun -> los tres son cross-embodiment.
        (Correccion: NO es que "solo AHG" sea agnostico; los tres lo son.)
- [ ] Cambio para Yan-style: poolear latentes humano + ambos robots por subespacio,
      samplear triplets cruzados. Para allegro<->shadow: common_labels = interseccion
      allegro INTERSECT shadow (mismo mecanismo, otra interseccion). El sampler ya
      devuelve common_labels para human-vs-robot; extender a robot-vs-robot.
- [ ] Decidir si vale la pena para 2 manos (beneficio incierto empiricamente,
      pero mas fiel a Yan + refuerza claim "unified latent"). Ver explicacion abajo.

---

## Estado actual (lo que YA quedo hecho hoy 2026-06-11)
- Sampler EIGENGRASP_ONLINE validado (smoke local + corrio en Colab).
- Basis eigengrasp shadow+allegro en git (commits 2a30460d3/aa2e73e7e).
- Guard exclusividad eigengrasp>valid_poses (5515c1b57).
- --skip_final_eval (5c4c87b41).
- MJCF menagerie vendorizados a git (b1a303a05).
- mujoco agregado a deps + install (376c108b1).
- Notebook cableado a online (074c8b149) <- ESTO SE REVIERTE EN TAREA 1.

## Divergencias vs Yan (para la tesis)
1. Sampling: eigengrasp (sinergias) en vez de uniform random. JUSTIFICADO
   (mano dextra: uniform da poses colisionadas; Yan tolera uniform por escala metros).
2. Batches per-robot vs cross-embodiment. <- TAREA 2 lo cierra si se decide.
