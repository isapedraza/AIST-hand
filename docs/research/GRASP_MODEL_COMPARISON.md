# Experimentos y Evolución del Modelo

## Modelo Actual — Run 006/007

**Arquitectura:** GCN_CAM_8_8_16_16_32
**Parámetros:** 14,565
**Clases:** 28 (taxonomía Feix)
**Dataset:** HOGraspNet, split S1 (S1-S73 train, S74-S99 test)

### Features actuales (numFeatures=4)

| Feature | Dim | Por nodo | Descripción |
|---|---|---|---|
| XYZ | 3 | ✓ | Posición 3D root-relative, escala WRIST-INDEX_MCP=0.1 |
| joint_angle | 1 | ✓ | arccos(dot(v_in, v_out)) — flexión escalar colapsada |
| theta_cmc | 1 | ✗ (global) | Ángulo abducción pulgar respecto al plano de palma |

**Estimador:** MediaPipe Hands (monocular, 2.5D)
**Normalización:** root-relative + WRIST-INDEX_MCP = 10cm

---

## Reporte Run 006 — MediaPipe

### Deploy — Domain Gap Sessions (2026-03-11)
Sessions: `20260311_145054` (exocentric), `20260311_155647` (near-egocentric)

**Clases que funcionan en deploy (acc > 0.7 en alguna condición):**
- Index Finger Extension: ExoAir=0.967, ExoObj=1.000, EgoAir=0.913, EgoObj=1.000
- Parallel Extension: ExoAir=1.000, ExoObj=1.000, EgoAir=0.907
- Palmar Pinch: ExoAir=1.000, ExoObj=1.000
- Tripod: ExoAir=0.487, ExoObj=1.000, EgoAir=1.000, EgoObj=0.894
- Lateral: EgoAir=1.000, EgoObj=1.000
- Adduction Grip: EgoAir=0.873, EgoObj=1.000
- Sphere 3-Finger: ExoAir=0.947, EgoAir=0.868, EgoObj=0.773

**Clases que fallan en deploy (acc ≈ 0 en todas las condiciones):**
- Large Diameter, Small Diameter, Medium Wrap, Light Tool
- Stick, Adducted Thumb, Tip Pinch, Prismatic 3 Finger
- Power Disk, Power Sphere, Quadpod, Sphere 4-Finger
- Lateral Tripod, Precision Sphere, Palmar, Extension Type

**Accuracy global deploy:** ~17-30% dependiendo de condición

**Casos específicos relevantes:**
- Writing Tripod air: acc=0.090 (colapsa a Palmar Pinch + Lateral)
- Writing Tripod object: acc=0.206 (colapsa a Light Tool + Lateral)
- Sphere 3-Finger object: acc=0.387 (colapsa a Tripod)

---

## Perception Gap Experiment (2026-03-12)

- Error MediaPipe vs GT HOGraspNet: **3.92mm mean** (espacio normalizado)
- Correlación error percepción vs deploy accuracy: r=0.063, p=0.750 (n.s.)
- Conclusión estadística: error uniforme, no diferencial por clase
- **Hallazgo cualitativo (2026-03-13):** con objetos MediaPipe alucina keypoints
  por oclusión. HaMeR con prior MANO es más robusto — Sphere 3-Finger vs Tripod
  diferenciables en vivo con HaMeR donde MediaPipe colapsaba (object acc 38.7%).

---

## Propuesta de Nuevas Features

### Motivación
- `joint_angle` escalar colapsa 3 DOF → 1. Pérdida de información garantizada.
- XYZ no captura orientación local de segmentos.
- Sin contexto temporal: transiciones y mano neutra colapsan a Ring (attractor).
- Para agarres finos (Tripod vs Writing Tripod): diferencia en rotación de falange
  media del dedo medio — invisible en XYZ, marginal en bone, capturada en mano_pose.

### Features propuestas

**Por nodo (reemplaza joint_angle escalar):**

| Feature | Dim | Descripción |
|---|---|---|
| XYZ | 3 | Sin cambio |
| Bone (xyz) | 3 | `bone_i = pos_i - pos_parent(i)` — orientación local explícita |
| Velocity (xyz) | 3 | `v_i = pos_i(t) - pos_i(t-1)` — contexto temporal |
| mano_pose | 3 | 3 DOF por articulación MANO — requiere HaMeR + HOGraspNet mano_pose |

**Global:** theta_cmc eliminado — capturado por mano_pose del CMC del pulgar.

| Configuración | numFeatures | Notas |
|---|---|---|
| XYZ + Bone + Velocity | 9 | Sin cambios en pipeline Colab ni dataset |
| XYZ + Bone + Velocity + mano_pose | 12 | Requiere Colab + ingestion HOGraspNet |

---

## Plan de Implementación

### Paso 0 — Baseline HaMeR
Mismo modelo Run 006 (numFeatures=4), solo cambia estimador a HaMeR.
Script: `grasp-app/domain_gap_experiment_hamer.py`
**Mide:** ganancia del estimador solo, sin cambiar el modelo.

### Paso 1 — XYZ + Bone + Velocity (numFeatures=9)

Archivos a modificar:
1. `grasp-model/src/grasp_gcn/transforms/tograph.py` — bone + velocity (recibe `prev_sample` opcional)
2. `grasp-model/src/grasp_gcn/dataset/grasps.py` — lookup temporal `(seq, trial, cam, frame_num) → idx`
3. `grasp-model/train.py` — pasar `prev_sample` a `to_graph`
4. `grasp-model/src/grasp_gcn/network/gcn_network.py` — `numFeatures=9`, eliminar `use_cmc_angle`
5. `grasp-app/perception/hamer_backend.py` — guardar `_prev_pts` para velocity en inferencia
6. `grasp-app/hamer_simple_demo.py` + `domain_gap_experiment_hamer.py` — pasar `prev_sample`

Velocity en training: lookup dict construido al init del dataset, parseando frame_num
del path en columna `object`. Primer frame de cada trial → velocity=0.

### Paso 2 — + mano_pose (numFeatures=12)

Archivos adicionales:
1. `hamer_keypoints/colab_server.ipynb` — retornar `pred_mano_params` además de keypoints
2. `grasp-model/scripts/ingestion/hograspnet_to_csv.py` — extraer `Mesh[0]['mano_pose']` (45 valores)
3. CSV — regenerar con columnas mano_pose por articulación
4. `tograph.py` — distribuir 3 valores mano_pose por nodo (15 articulaciones → 15 nodos)
5. `hamer_backend.py` — retornar mano_pose junto a keypoints

---

## Tabla de Experimentos de Entrenamiento

| Run | Fecha | Features/nodo | Test Acc | Macro F1 | Notas |
|-----|-------|--------------|----------|----------|-------|
| 006 | — | xyz+flex (4) | 62.0% | 0.550 | Baseline 28 clases |
| 007 | — | xyz+flex (4) | 75.6% | 0.670 | 17 clases colapsadas, mejor global |
| 008 | — | xyz+flex+bone (7) | 64.6% | 0.575 | +bone vectors |
| 009 | — | xyz+flex+bone+vel (10) | 64.6% | 0.574 | velocity=ruido en HOGraspNet estatico |
| 010 | — | xyz+flex+bone+vel+pose (13) | 70.5% | 0.638 | +MANO pose, velocity incluida |
| 011 | 2026-03-21 | xyz+flex+bone+pose (10) | 71.1% | 0.647 | MEJOR 28 clases — sin velocity |
| 012 | 2026-03-23 | xyz+flex+bone+swing (10) | — | — | global swing reemplaza MANO pose |

## Run 012 — Global Swing (2026-03-23)

**Motivacion:** R011 es el mejor modelo en training (71.1%) pero esta roto en deploy.

### Por que se descarto MANO pose para deploy

MANO pose funciono muy bien en training (+9pp sobre R006). El problema es que no es recuperable de forma unica desde los 21 XYZ.

**Argumento completo:**

MANO pose tiene 45 DOF (15 joints x 3 axis-angle). Los 21 joints XYZ dan 63 valores, pero los joints estan sobre los ejes de los huesos. El twist de cada hueso (rotacion alrededor de su propio eje) no mueve ningun joint -- todos estan sobre ese eje. Tampoco mueve los joints hijos, que son la continuacion de la cadena. El sistema queda subdeterminado para pose: infinitas soluciones de θ producen los mismos 21 XYZ.

HOGraspNet y HaMeR son ambos estimadores de MANO (fitters que ajustan θ a los datos observados). Ambos ven el twist en sus datos -- HaMeR en los pixeles RGB (orientacion de unas, textura de nudillos), HOGraspNet en la superficie de profundidad. Pero usan distintos optimizadores con distintos priors y lo resuelven diferente. Resultado: r=0.23 entre los 45 params de pose para el mismo frame fisico.

**Por que los vertices si resolverien el problema:**

Del paper de MANO (Romero et al., 2017, Sec. 3.3):

> "The joint locations, J(β), also depend on shape parameters. These are learned as a sparse linear regression matrix J from mesh vertices."

Los joints no son una salida primaria del modelo -- son una proyeccion lineal aprendida de la malla:

```
J_world = J_regressor @ mesh_vertices    # J_regressor: (21, 778)
```

Esta operacion tiene un espacio nulo enorme. El twist de cada hueso desplaza los 778 vertices en direccion perpendicular al eje, pero J_regressor los promedia de vuelta hasta un punto sobre el eje. El twist queda en el espacio nulo de J_regressor y no aparece en los 21 XYZ resultantes.

Con los 778 vertices directamente el sistema esta sobredeterminado (2334 ecuaciones, 45 incognitas) y θ es recuperable de forma unica. HaMeR de hecho genera esa malla internamente (`pred_vertices`). La estamos tirando.

**Por que no usamos vertices -- argumento en tres partes:**

1. **Sistematico:** El contrato del sistema es `21 landmarks XYZ → GraspToken`, disenado para ser agnostico al sensor (MediaPipe, HaMeR, guante con IMUs, mocap). Usar vertices MANO ataria el sistema exclusivamente a estimadores visuales que corran MANO internamente. MediaPipe no genera malla. Un guante IMU no genera malla. Ese cambio esta fuera del scope de esta tesis.

2. **Dataset:** HOGraspNet fue anotado con un fitter distinto (no HaMeR). Para tener vertices consistentes en training habria que reannotar 1.5M frames con HaMeR -- estimado ~7 dias de computo en Colab T4.

3. **Computacional:** Caches reales medidas en disco como referencia:

   | | R011 (21 nodos) | Malla MANO (778 nodos) |
   |---|---|---|
   | Cache train | 1.8 GB | ~74 GB |
   | Cache total (3 splits) | 2.6 GB | ~107 GB |
   | Aristas por grafo | ~40 | ~4,662 |
   | Costo forward GCN (O(E x F)) | baseline | ~117x |
   | Tiempo estimado Colab T4 | ~60 min | ~90-120 horas |

   Colab tiene limite de ~12h por sesion -- serian 8-10 sesiones con checkpointing manual por cada run. Inviable para iteracion de experimentos.

### Alternativa descartada: reannotar HOGraspNet con HaMeR

La solucion mas directa al gap de convencion seria reannotar todos los frames de HOGraspNet usando HaMeR como estimador. El contrato `21 XYZ → GraspToken` se mantiene intacto -- no se usa la malla, no se usa θ, solo los 21 keypoints que HaMeR produce. La agnósticidad al sensor tambien se preserva porque en inferencia cualquier estimador que entregue 21 XYZ funciona.

El resultado seria que training y deploy ven exactamente la misma distribucion de XYZ. El gap de dominio entre HOGraspNet (depth + fitter propio) y HaMeR (RGB) desaparece por construccion.

**Unico downside: costo computacional.**

- 1,489,111 frames x ~0.1-0.3 seg/frame de inferencia HaMeR en T4
- Estimado: ~40-120 horas de computo continuo
- Colab tiene limite de sesion (~12h) → requiere partir en chunks con checkpointing

No hay argumento arquitectural en contra. Es la solucion mas limpia. Global swing (R012) es el intento de obtener una mejora equivalente sin pagar ese costo. Si R012 se acerca a R011 en accuracy, global swing fue suficiente. Si decae significativamente, la reannotacion con HaMeR es la opcion a considerar.

### Global swing como alternativa

Del FK de MANO (Romero et al., 2017):

```
b_world[i] = R_global[parent[i]] @ b_rest[i]
```

Para cada hueso i (1..15): `R_global[parent[i]] = swing_rotation(b_rest[i], b_world[i])`

La swing rotation impone twist=0 explicitamente. Dado el mismo XYZ, ambas fuentes producen el mismo resultado porque la ambiguedad de twist se elimina por definicion. r=0.83 cross-source (vs r=0.23 con MANO pose raw).

| Representacion | r cross-source (HOGraspNet vs HaMeR) |
|---|---|
| MANO pose original | 0.23 |
| MANO local IK (cadena) | 0.56 |
| Global swing (este run) | 0.83 |
| XYZ crudo (techo) | ~0.94 |

**Implementacion:** `_J_REST` hardcodeado en `tograph.py` (16x3, wrist en origen, de MANO_RIGHT.pkl betas=0). Sin dependencias externas. Solo necesita `hograspnet.csv`.

**Incognita:** cuanto del +9pp de MANO pose era informacion de twist (no recuperable desde XYZ) vs informacion de direccion de hueso (que global swing si captura). El accuracy de R012 lo revela.

### MANO no rompe la agnósticidad al sensor

Una aclaracion importante: MANO como modelo es agnóstico al sensor. Es un mapeo estadístico `(β, θ) → malla` aprendido de escaneos de alta resolucion -- no asume ningun sensor especifico. Cualquier sensor con un fitter adecuado puede producir β y θ:

- RGB monocular → HaMeR, otros
- Depth / RGB-D → fitter de HOGraspNet
- Multicamara → fitters con mas restricciones
- Guante IMU → mapeo directo de angulos a θ
- Mocap con marcadores → ajuste al skeleton MANO

El problema de R011 no era usar MANO -- era usar **θ como feature directamente**. θ no esta determinado de forma unica por los 21 XYZ (por la ambigüedad de twist), entonces distintos fitters producen distintos θ para el mismo estado fisico de la mano. Eso genera features OOD en deploy.

| | Agnóstico al sensor |
|---|---|
| MANO como modelo de mano | Si |
| 21 XYZ de cualquier estimador | Si |
| θ de un fitter especifico como feature | No -- depende del fitter |
| Global swing computado desde XYZ | Si -- determinístico desde XYZ |

Global swing es en esencia "extraer orientacion de hueso desde XYZ usando la ecuacion de FK de MANO, sin depender de ningun fitter para θ".

### Por que no usar solo XYZ

La pregunta natural es: si XYZ ya contiene toda la informacion geometrica, por que añadir features derivadas?

XYZ contiene la informacion de forma implicita. El problema es que la GCN tiene que aprender a extraerla con operaciones de message passing sobre un grafo fijo y una arquitectura poco profunda (5 capas, 32 canales max). Para distinguir Tripod de Writing Tripod, la diferencia relevante es la rotacion de la falange media del dedo medio -- invisible directamente en XYZ, pero capturable como la orientacion del hueso respecto a su posicion de reposo.

Bone vectors (R008) añaden la direccion explicita de cada segmento. Global swing (R012) añade la rotacion completa de cada hueso desde rest (flexion + abduccion juntas), que la GCN no tiene que inferir de diferencias de posicion entre nodos vecinos.

Los resultados empiricos confirman la hipotesis: R006 (solo XYZ) 62.0%, R008 (+bone) 64.6%, R011 (+pose) 71.1%. Cada representacion explicita que elimina trabajo de inferencia implicita para la GCN mejora el accuracy.

## Tabla de Experimentos de Deploy

| Experimento | Fecha | Estado | Script | Resultado |
|---|---|---|---|---|
| Domain Gap MediaPipe | 2026-03-11 | Completado | domain_gap_experiment.py | ~17-30% global |
| Perception Gap MediaPipe vs GT | 2026-03-12 | Completado | — | 3.92mm, r=0.063 n.s. |
| Domain Gap HaMeR exocentrico | 2026-03-20 | Completado | domain_gap_experiment_hamer.py | pendiente registrar |
| Exp 3 (HaMeR + bone, R008) | — | Pendiente | — | — |
| Exp 5 (HaMeR + swing, R012) | — | Pendiente | — | — |
