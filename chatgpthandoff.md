# ChatGPT Codex Handoff -- GraphGrasp Project

> Estado al 2026-04-04. Este documento es para continuar el trabajo sin perder contexto.

---

## 1. Contexto general

Tesis de licenciatura: GNN para reconocimiento de agarre orientado a intención en teleoperación robótica.

- **Dataset**: HOGraspNet (ECCV 2024). ~1.5M frames, 99 sujetos, 30 objetos YCB, 28 clases Feix.
- **Input del sistema**: 21 landmarks XYZ de la mano (MediaPipe o HaMeR).
- **Output**: GraspToken { class_id, class_name, confidence }

**Monorepo**:
```
AIST-hand/
├── grasp-model/    # paquete Python instalable (pip install -e .)
│   └── src/grasp_gcn/
│       ├── transforms/tograph.py        # pipeline actual (NO tocar)
│       ├── transforms/kinematics.py     # BORRADO a proposito; rehacer limpio
│       └── network/gcn_network.py       # arquitecturas GCN/CAM/CAMGAT
├── grasp-app/      # implementacion de referencia (MediaPipe + RGB)
├── grasp-robot/    # adaptadores por robot (Shadow Hand)
├── DECISIONS.md    # log de decisiones arquitectonicas (fuente de verdad)
└── DONG_VALIDATION_EXAMPLE.md  # ejemplo numerico del paper para validar codigo
```

---

## 2. Estado actual del modelo

**Mejor modelo**: `abl04` -- GCN_CAM_8_8_16_16_32, features xyz+AHG+flex, 71.07% test acc, 28 clases Feix.

Este modelo es **frame-wise** (procesa cada frame independientemente) y el control es **lookup-table** (argmax → keyframe lookup → lerp). Estos son los problemas estructurales que se estan resolviendo con la nueva arquitectura.

---

## 3. Direccion arquitectonica (DECISIONS.md Entry 14)

Se esta migrando hacia una arquitectura **Dual-VGAE**:

```
MediaPipe XYZ
    ↓
compute_16_dofs(kp)   ← funcion en kinematics.py
    ↓
[16 angulos anatomicos]
    ↓
VGAE Encoder A (humano)  →  z_humano
    ↓
Manifold Mapping          →  z_robot   ← PENDIENTE DE DEFINIR
    ↓
VGAE Decoder B (robot)    →  qpos Shadow Hand
```

**Por que VGAE y no suma lineal de anchors (Sigma w_i * anchor_i)**:
- La suma lineal opera en espacio de joints crudo -- no garantiza coherencia mecanica en poses intermedias
- El VGAE aprende el manifold de poses validas del robot -- cualquier punto decodificado esta sobre ese manifold
- La interpolacion lineal asume que el espacio de poses validas es convexo -- el VGAE no asume eso

**Por que angulos y no XYZ**:
- Los angulos son la representacion intrinseca de la postura -- no dependen de distancia a camara, tamano de mano, orientacion global
- XYZ tiene esa informacion mezclada aunque se normalice
- El espacio postural (PCA de Santello/Jarque-Bou) se construye sobre angulos, no sobre XYZ

**Experimento de validacion planificado**: GAE vs AE sobre la mano humana.
- **AE**: vector plano [16 angulos] → MLP encoder → z → MLP decoder → [16 angulos]
- **GAE**: grafo cinematico [21 nodos, features de angulos] → GCNConv encoder → z → decoder
- Misma tarea, mismo z_dim. Si GAE reconstruye mejor, justifica usar la estructura del grafo en Encoder A.

---

## 4. Estado actual de kinematics

**`grasp-model/src/grasp_gcn/transforms/kinematics.py`** -- BORRADO.

Se intento una implementacion inspirada en Schlegel/Wu y luego una inspirada en Dong, pero el usuario pidio borrar el archivo por completo y rehacerlo limpio desde cero.

**Decision actual**:
- No reinterpretar el modelo.
- Implementar el modelo cinemático de **Dong & Payandeh (2025)** de forma fiel.
- Lo importante es obtener **parametros cinematicos** del paper, no imponer etiquetas anatomicas externas.

**Criterio actual del usuario**:
- Le interesa poder decir: "este agarre tiene esta configuracion de angulos / parametros articulares".
- No quiere una interpretacion subjetiva del modelo.
- Prefiere hablar en terminos de `beta`, `gamma`, etc. si eso es lo que define el paper.

**Alcance acordado hasta ahora**:
- Empezar con los dedos largos.
- `MediaPipe` puro en indices.
- Ambas manos pueden espejarse si hace falta para una convencion canonica, pero eso es preprocesamiento, no parte del modelo de Dong.

## 5. Ejemplo de validacion de Dong

Se guardo un archivo nuevo:

- `DONG_VALIDATION_EXAMPLE.md`

Contiene el ejemplo numerico correcto extraido del XML local del paper:
- fuente: `/home/yareeez/Downloads/applsci-15-08921.xml`

Valores clave para validar una implementacion del indice:
- `beta5 = 24.22°`
- `gamma5 = 9.07°`
- `beta6 = 22.22°`
- `beta7 = 8.77°`

Tambien incluye:
- landmarks de `Table 2`
- frame de palma `{0}`
- frame local del indice `{5}`
- puntos intermedios `d6^5`, `d7^5`, `d8^5`

**Regla de oro**:
- Si una implementacion no reproduce esos numeros aproximadamente con los landmarks del ejemplo, no es fiel a Dong.

---

## 6. Siguiente paso de implementacion

Primero rehacer `grasp-model/src/grasp_gcn/transforms/kinematics.py` desde cero.

Orden sugerido:
1. implementar solo el ejemplo numerico del indice de Dong
2. validar contra `DONG_VALIDATION_EXAMPLE.md`
3. extender a los otros dedos largos
4. despues decidir si conviene exponer salida como vector o dict

Solo cuando eso exista y este validado, retomar `to_angle_graph.py`.

## 7. Lo que NO hay que tocar

- `tograph.py`: pipeline del clasificador frame-wise (abl04 y toda la ablacion). Funciona, no romper.
- `gcn_network.py`: arquitecturas existentes. El VGAE sera una arquitectura nueva, no una modificacion.
- Cualquier experimento en `grasp-model/experiments/`: resultados guardados, no modificar.

---

## 8. Referencias clave

- **DECISIONS.md** en la raiz del repo: fuente de verdad para todas las decisiones arquitectonicas. Entries relevantes: 1, 3, 4, 6, 7, 8, 13, 14.
- **Dong & Payandeh (2025)**: referencia principal actual
- **DONG_VALIDATION_EXAMPLE.md**: ejemplo numerico de referencia para testear codigo
- **Chen et al. (2013)**: inventario de DOFs de la mano humana
- **Wu et al. (2005)**: sistemas de coordenadas ISB para joints de la mano
- **Gionfrida et al. (2022)**: baseline simple con angulos entre segmentos
- **Schlegel et al. (2024)**: referencia conceptual para JCS, pero no fuente principal actual
- **Jarque-Bou et al. (2019, 2021)**: z-score para espacio postural, 16 angulos CyberGlove
- **Santello et al. (1998)**: PCA de posturas de mano humana, base metodologica
- **Segil & Weir (2014)**: Postural Control via JAT, baseline lineal que el VGAE extiende
- **Kipf & Welling (2016)**: Variational Graph Auto-Encoders, arquitectura base del VGAE
