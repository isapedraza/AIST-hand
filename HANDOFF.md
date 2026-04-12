# GraphGrasp — Handoff Document

> Estado al 2026-03-29. Este documento existe para que un agente o colaborador pueda continuar el trabajo sin perder contexto.

---

## 1. Qué es este proyecto

**Tesis de licenciatura:** "Graph Neural Networks for Intention-Oriented Grasp Recognition in Vision-Based Robotic Hand Teleoperation"

El sistema recibe una secuencia continua de frames de landmarks 3D de la mano (21 joints XYZ), producidos por un estimador de pose como MediaPipe o HaMeR. En cada momento, el humano puede estar ejecutando uno de 28 tipos de agarre de la taxonomía Feix, o transitando entre ellos. El objetivo es emitir una etiqueta de clase en tiempo real que el robot utiliza para replicar el agarre.

**Contrato del sistema:** 21 landmarks XYZ → `GraspToken { class_id, class_name, confidence, apertura }`

**Dataset:** HOGraspNet (ECCV 2024). ~1.5M frames, 99 sujetos, 30 objetos YCB, 28 clases Feix. Split S1 por sujeto (sin data leakage). Cada secuencia = un solo tipo de agarre sostenido (no hay frames de transición entre clases). Estructura de secuencias: `sequence_id` (e.g. `230911_S10_obj_06_grasp_16/trial_0`) + `cam` (sub1/sub2/sub3/mas).

---

## 2. Monorepo

```
AIST-hand/
├── grasp-model/    # paquete Python instalable (pip install -e .)
│   ├── src/grasp_gcn/
│   │   ├── network/gcn_network.py   # arquitecturas (CAMLayer, CAMGATLayer, modelos)
│   │   ├── network/utils.py         # registry de modelos
│   │   ├── dataset/grasps.py        # GraspsClass (PyG dataset)
│   │   └── transforms/tograph.py    # ToGraph (features: xyz, flex, AHG, bone, etc.)
│   ├── train.py                     # script de entrenamiento (configurable por env vars)
│   ├── notebooks/                   # notebooks Colab por experimento
│   └── experiments/                 # checkpoints + TensorBoard + results.json
├── grasp-app/      # implementación de referencia (MediaPipe + RGB)
├── grasp-robot/    # adaptadores por robot (Shadow Hand target)
└── .venv/          # venv compartido (Python 3.11.9 via pyenv)
```

---

## 3. Arquitecturas implementadas

### GCN_CAM_8_8_16_16_32 (baseline actual, ~14k params)

5 capas `CAMLayer` (8→8→16→16→32). Cada capa:
```
x_agg = CAM @ x          # CAM [21x21] learnable, signed, compartida entre capas
x_out = ELU(Linear(cat(x_agg, x_in)))  # skip connection
```
La CAM se inicializa en U(-1,1) y se entrena end-to-end. Entradas negativas = inhibitorio, positivas = excitatorio. No hay topología fija — el modelo aprende qué conexiones importan globalmente.

Readout: mean+max → [B, 64] → FC(64→128→numClasses) + log_softmax.

### GCN_CAMGAT_8_8_16_16_32 (en evaluación, ~16k params)

Reemplaza `CAMLayer` con `CAMGATLayer`. Cada capa añade mecanismo GAT sobre CAM:
```
h_i    = W1 * x_i                                     # proyección [GAT Eq. 1]
e_ij   = LeakyReLU(a^T [h_i || h_j])                  # score de atención [GAT Eq. 3]
beta   = softmax(e_ij)                                 # normalización [GAT Eq. 2]
alpha  = CAM_ij * beta_ij                              # modulación por CAM
x_out  = ELU(W2([sum_j alpha_ij * h_j || x_i]))       # skip connection
```
CAM define qué conexiones existen (prior global aprendido). GAT modula cuánto pesa cada una por muestra (dinámico, dependiente del frame actual). La semántica inhibitoria/excitatoria de CAM se preserva porque el producto `CAM_ij * beta_ij` puede ser negativo.

### GCN_CAM_32_32_64_64_128 (ablación de capacidad, ~70k params)

Igual que GCN_CAM_8_8_16_16_32 pero con dims 32→32→64→64→128. Resultado: 70.81% (abl09) vs 71.07% (abl04 con modelo pequeño). Conclusión: la capacidad no es el cuello de botella.

---

## 4. Features por nodo (ToGraph)

| Feature | Dim | Descripción |
|---------|-----|-------------|
| xyz | 3 | Posición 3D normalizada (wrist=origen, escala=wrist→MMCP) |
| flex | 1 | Ángulo entre bone[i] y bone[child[i]] — ángulo de flexión local |
| AHG angles | 5 | Ángulos desde fingertips al wrist (Aiman & Ahmad 2024) |
| AHG distances | 5 | Distancias a joints críticos |
| bone vectors | 3 | bone[i] = pos[i] - pos[parent[i]] |
| cmc angle | 1 | Ángulo pulgar CMC (graph-level, no por nodo) |

**Configuración abl04 (mejor deploy-safe):** xyz + flex + AHG angles + AHG distances = 14 features/nodo. Hay un efecto de interacción: flex solo empeora, AHG solo apenas ayuda, juntos superan todo.

---

## 5. Tabla de experimentos

| Run | Features | Modelo | Test Acc | Macro F1 | Notas |
|-----|----------|--------|----------|----------|-------|
| run006b | xyz | pequeño | 65.17% | 0.585 | Baseline S1 split + plateau |
| run007 | xyz, taxonomy_v1 | pequeño | 75.6% | 0.670 | 17 clases colapsadas — mejor overall |
| run011 | xyz+bone+pose | pequeño | 71.1% | 0.647 | Roto en deploy (MANO pose OOD) |
| abl01 | xyz | pequeño | 67.99% | 0.619 | — |
| abl02 | xyz+flex | pequeño | 63.9% | 0.575 | Flex solo empeora |
| abl03 | xyz+AHG | pequeño | 68.17% | 0.618 | AHG solo apenas ayuda |
| **abl04** | **xyz+AHG+flex** | **pequeño** | **71.07%** | **0.657** | **Mejor deploy-safe** |
| abl05 | xyz+AHG+cmc | pequeño | 69.18% | 0.634 | — |
| abl06 | xyz+AHG+bone | pequeño | 70.80% | 0.654 | — |
| abl07 | xyz+AHG+swing | pequeño | 69.63% | 0.638 | — |
| abl08 | xyz+AHG+flex+bone | pequeño | 69.67% | 0.641 | flex+bone se solapan con AHG presente |
| abl09 | xyz+AHG+flex | grande (x4) | 70.81% | 0.656 | Capacidad no es el cuello de botella |
| camgat01 | xyz+AHG+flex | CAM-GAT | 70.46% | 0.643 | Scheduler disparó 3 veces, best val 66.6% época 47, convergió en época 60 |

**Scheduler en todos los runs:** plateau (factor=0.5, patience=5). En abl04 activó en época 28, best val acc 66.2% en época 53, test acc final 71.07%.

---

## 6. Resultado camgat01 -- COMPLETADO

- **Modelo:** GCN_CAMGAT_8_8_16_16_32 (~16k params)
- **Features:** xyz+AHG+flex (misma cache que abl04)
- **Test Acc:** 70.46% | **Macro F1:** 0.643
- **Best val acc:** 66.6% en época 47 (supera best val de abl04: 66.2%)
- **LR final:** 1.25e-4 (scheduler disparó 3 veces: 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4)
- **Épocas sin mejorar al final:** 13 -- modelo convergido, no vale continuar
- **Checkpoint:** `grasp-model/experiments/best_model_run_camgat01_xyz_ahg_flex.pth`

**Conclusión:** Bajo las mismas condiciones de entrenamiento y features, CAM-GAT no supera a CAM puro en test (70.46% vs 71.07%). La causa es incierta -- podría ser arquitectural, de hiperparámetros, o de dificultad de optimización. Clases donde CAM-GAT mejora: Sphere 4-Finger (+3.6pp), Lateral (+1.8pp). Clases donde pierde más: Lateral Tripod (-9.7pp), Power Sphere (-4.2pp). Resultado válido para tesis, pero no es concluyente sobre la utilidad general de CAM-GAT.

---

## 7. Próximo paso: Temporalidad

### Por qué

En deploy, el humano transiciona entre agarres continuamente (Pinch → Ring → Sphere). La clasificación frame-a-frame + VotingWindow es inestable. VotingWindow promedia predicciones independientes; temporalidad en el modelo usa la trayectoria directamente.

Adicionalmente, MediaPipe produce frames ruidosos ocasionalmente. Una ventana de T frames hace el modelo robusto ante perturbaciones puntuales (inercia cinemática).

### Limitación del dataset

HOGraspNet no tiene frames de transición entre clases — cada secuencia es un solo agarre sostenido. Por tanto, la temporalidad entrenada con HOGraspNet solo aprende robustez dentro de un agarre. Las transiciones en deploy no tienen soporte en training.

### Dilemas abiertos

**Dilema 1: Transiciones entre agarres sin datos reales**

En deploy, el humano cambia continuamente entre agarres (Pinch → Ring → Sphere). Durante esas transiciones, la mano pasa por configuraciones intermedias que el modelo nunca ha visto en training -- HOGraspNet solo contiene agarres sostenidos y completos. El clasificador entra en una región ambigua y salta erráticamente entre clases, produciendo movimientos bruscos en el robot.

No existe ningún dataset público con transiciones etiquetadas entre clases Feix + landmarks 3D (gap confirmado por búsqueda exhaustiva). Las opciones son:

1. **Datos sintéticos via interpolación MANO:** tomar el medoid de cada clase en espacio de parámetros MANO (45 DoF), interpolar linealmente entre pares de clases, pasar por MANO forward para obtener XYZ, calcular features normalmente, etiquetar extremos con clase A y clase B. Limitación: la interpolación lineal en MANO no produce trayectorias biomecánicamente realistas. Pendiente buscar paper sobre este enfoque.
2. **VotingWindow más agresivo:** más simple pero introduce latencia.
3. **Temporalidad en el modelo:** parcialmente -- aprende a mantener clase A dentro de un agarre, pero sin datos de transición no aprende el cambio correcto.

**Dilema 2: Temporalidad solo resuelve robustez, no transiciones**

HOGraspNet tiene estructura de secuencias (sequence_id + cam), pero todas las secuencias son de un solo agarre sostenido. Entrenar con ventanas temporales de T frames enseña al modelo a ser robusto ante frames ruidosos (oclusión, MediaPipe noise), pero no resuelve las transiciones entre clases porque esas transiciones no existen en los datos.

### Diseño propuesto

**Enfoque modular (recomendado):**

```
Frame t-T ... Frame t-1 ... Frame t
     ↓              ↓            ↓
  [CAM-GAT]    [CAM-GAT]    [CAM-GAT]   ← pesos compartidos
     ↓              ↓            ↓
 embed_{t-T}  embed_{t-1}  embed_t      ← [B, 64] por frame
     └──────────────┴────────────┘
              temporal pool (mean/max o GRU pequeño)
                        ↓
                  clasificador
```

El modelo espacial (CAM-GAT) no cambia — se puede partir del checkpoint de camgat01. La capa temporal va encima. Pesos del CAM-GAT compartidos a través del tiempo (eficiencia de params).

**Datos:** agrupar por `(sequence_id, cam)` en hograspnet.csv para construir ventanas de T frames consecutivos. El split S1 es por sujeto, no por secuencia — las ventanas temporales no introducen data leakage.

**T sugerido:** 5-8 frames. Con 30fps ≈ 167-267ms de contexto, equivalente al VotingWindow actual.

**Pooling temporal:** empezar con mean/max (sin params adicionales). Si no mejora, probar GRU pequeño (aprende qué frames son más informativos).

### Posible extensión: mask de oclusión

Con T frames disponibles, detectar joints anómalos desde la trayectoria (outlier estadístico) y generar una máscara continua:

```
alpha_ij = CAM_ij * mask_j * beta_ij
```

No requiere datos etiquetados de oclusión — la máscara emerge de la consistencia temporal. MediaPipe no da visibility scores (siempre 0), así que el único detector viable es temporal o un autoencoder.

---

## 8. Configuración de entrenamiento

### Variables de entorno para train.py

| Variable | Descripción | Default |
|----------|-------------|---------|
| `GG_RUN_NAME` | Nombre del run | `run_unnamed` |
| `GG_NETWORK_TYPE` | Arquitectura | `GCN_CAM_8_8_16_16_32` |
| `GG_COLLAPSE` | Colapso de clases | `none` |
| `GG_CHECKPOINT` | Nombre del .pth | `best_model_{RUN_NAME}.pth` |
| `GG_JOINT_ANGLES` | Flex angles | `true` |
| `GG_CMC_ANGLE` | CMC angle | `true` |
| `GG_BONE_VECTORS` | Bone vectors | `false` |
| `GG_AHG_ANGLES` | AHG angles | `false` |
| `GG_AHG_DISTANCES` | AHG distances | `false` |
| `GG_NUM_EPOCHS` | Épocas | `40` |
| `GG_PATIENCE` | Early stopping | `10` |
| `GG_LR_SCHEDULER` | Scheduler | `none` |
| `GG_LR_PLATEAU_FACTOR` | Factor scheduler | `0.5` |
| `GG_LR_PLATEAU_PATIENCE` | Patience scheduler | `5` |

### Configuración abl04 (referencia)

```python
GG_NETWORK_TYPE    = "GCN_CAM_8_8_16_16_32"
GG_JOINT_ANGLES    = "true"    # flex
GG_CMC_ANGLE       = "false"
GG_BONE_VECTORS    = "false"
GG_AHG_ANGLES      = "true"
GG_AHG_DISTANCES   = "true"
GG_NUM_EPOCHS      = "60"
GG_LR_SCHEDULER    = "plateau"
GG_LR_PLATEAU_FACTOR   = "0.5"
GG_LR_PLATEAU_PATIENCE = "5"
```

---

## 9. Decisiones de diseño importantes

- **No usar velocity como feature:** HOGraspNet es estático (dt fijo). En deploy con MediaPipe, dt es irregular (latencia de red) → valores OOD. Confirmado en run009/run010.
- **No usar MANO pose como feature:** Los 45 parámetros de pose son incompatibles entre HOGraspNet y HaMeR/MediaPipe (misma forma física, distintos números). OOD en deploy. Confirmado en run011.
- **Global swing (run012):** Derivable desde FK de MANO, r=0.83 cross-source. Pero superado por abl04.
- **Sensor-agnóstico:** El contrato empieza en 21 XYZ landmarks normalizados. Cualquier sensor puede alimentar el pipeline implementando `PerceptionBackend`.
- **Commits en inglés, sin Co-Authored-By de Claude.**
- **No usar em dashes en documentos.**

---

## 10. Argumento de tesis

~14k params vs millones de HOGraspFlow (92.5% con RGB+MANO mesh+DINOv2+contacto). Con 0.003% de params se alcanza ~77% de su accuracy (71% vs 92.5%). Trade-off consciente: agnóstico al sensor, deployable en CPU. HOGraspFlow como upper bound — no sensor-agnóstico.
