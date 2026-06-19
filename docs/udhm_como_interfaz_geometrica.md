# UDHM como interfaz geométrica canónica

## La idea central

UDHM es un vector de 22 posiciones con orden y semántica fijos. Cada posición corresponde exactamente a un grado de libertad anatómico de la mano: flexión del MCP del índice, abducción del MCP del medio, flexión del PIP del anular, etc.

Humano y robot depositan sus ángulos en ese vector. Lo que el robot no tiene → 0. Lo que el humano no tiene → 0. El vector es siempre el mismo: 22 posiciones, mismo orden, mismo significado por posición.

```
slot  5: index_mcp_abd
slot  6: index_mcp_flex
slot  7: index_pip_flex
slot  8: index_dip_flex
...
slot 17: pinky_mcp_abd
slot 18: pinky_mcp_flex
slot 19: pinky_pip_flex
slot 20: pinky_dip_flex
```

---

## El signo es parte del contrato

UDHM define no solo qué slot es cada DoF, sino en qué dirección es positivo:

- **Flexión positiva** = más cerrado (acercar dedos a la palma)
- **Abducción positiva** = alejarse del dedo medio

Sin esto, dos robots haciendo el mismo movimiento pueden producir valores opuestos en el mismo slot, porque sus URDFs definen la dirección positiva del joint de forma distinta:

```
Barrett index_mcp_flex: límites [-2.44, 0]  → cerrar = qpos negativo
Inspire index_mcp_flex: límites [0, 1.47]   → cerrar = qpos positivo
```

Sin corrección → slot 6 de Barrett al cerrar = -0.78, Inspire = +0.47. Mismo movimiento, valores opuestos. La métrica L1 dice "son distintos" cuando son idénticos en intención.

Fix: `udhm[slot] = sign * qpos[col] / π`

El signo se deriva automáticamente del URDF: `sign = +1 si |hi| >= |lo|, else -1`. Para abducción se usa excitación geométrica (FK) porque el rango es simétrico y los límites no dan información de dirección.

**El yaml de cada robot debe declarar qué joint llena qué slot. El signo se computa del URDF. Sin signo, UDHM no es un contrato — es una lista de números sin convención compartida.**

---

## Por qué esto importa geométricamente

Una rotación articular en 3D no es un número — es una matriz 3×3 (o cuaternión). La distancia geodésica entre dos rotaciones mezcla todos los ejes de movimiento de esa articulación en un solo escalar.

Para una articulación MCP de dos grados de libertad (flexión + abducción), la geodésica entre dos rotaciones suma ambos movimientos sin distinción:

```
d_geodésica(R_a, R_b) ≈ Δflexión + Δabducción + interacción cruzada
```

Si el humano abre los dedos (abducción alta, flexión baja) y el robot cierra el MCP (flexión alta, abducción baja), la geodésica puede dar una distancia pequeña si los movimientos se cancelan parcialmente en la representación rotacional. La métrica miente.

**UDHM descompone esa rotación en sus componentes anatómicos antes de comparar:**

```
slot 5 (index_mcp_abd):  Δ_abd = |u_h[5] - u_r[5]|
slot 6 (index_mcp_flex): Δ_flex = |u_h[6] - u_r[6]|
```

Ahora cada componente se compara con sí mismo. Abducción vs abducción. Flexión vs flexión. Sin contaminación cruzada. La métrica refleja la geometría del movimiento.

---

## El problema de las dimensiones variables

Sin UDHM, comparar Shadow vs LEAP vs humano requiere encontrar los joints comunes en cada par:

```
humano:  [thumb_mcp, thumb_pip, ..., index_pip, ..., pinky_dip]  → N=15
shadow:  [thumb_mcp, thumb_pip, ..., index_pip, ..., pinky_dip]  → N=20 (más joints)
leap:    [thumb_mcp, ..., index_pip, ..., pinky_dip]             → N=16
barrett: [finger_1_med, finger_2_med, finger_3_med, ...]         → N=6, nombres distintos
```

Para comparar cualquier par necesitas `filter_to_subspace`: construir la intersección de labels en cada batch, extraer los índices correspondientes, operar en ese subespacio variable. El subespacio de Shadow vs humano tiene dimensión K₁. El de LEAP vs humano tiene K₂. No son iguales. No puedes meter todos en un batch.

**Con UDHM, el espacio siempre es 22:**

```
udhm_humano  [B, 22]  — slots con ángulos, resto 0
udhm_shadow  [B, 22]  — slots con ángulos, resto 0
udhm_leap    [B, 22]  — slots con ángulos, resto 0
udhm_barrett [B, 22]  — slots con ángulos (sin pinky → slots 17-20 = 0)
```

Barrett no tiene pinky → slots 17-20 son 0. Eso es la ausencia. No hay código especial para manejarlo. No hay dimensión variable. El 0 es la representación de "este robot no tiene este DoF". Cualquier operación sobre slot 17 con Barrett simplemente opera sobre 0.

---

## Pool multi-robot sin infraestructura

Con UDHM, todos los robots y el humano viven en el mismo espacio. El contrastive mining puede mezclar todos en un solo batch:

```python
udhm_pool = torch.cat([udhm_h, udhm_shadow, udhm_leap, udhm_barrett], dim=0)
# [4B, 22] — todos comparables directamente

loss = (w_udhm * (udhm_pool[anchors] - udhm_pool[cand]).abs()).sum(-1)
```

Sin `filter_to_subspace`. Sin `common_labels`. Sin reconstruir alineamiento por par. El índice del slot es el alineamiento.

---

## Pesos por slot reflejan geometría del movimiento

No todos los DoFs contribuyen igual a la similaridad de pose. Abducción tiene rango pequeño (±15° típico) vs flexión (0-90°). En espacio de ángulos normalizados por π, la abducción genera valores pequeños que L2 absorbe casi por completo.

Con slots separados, cada DoF tiene su propio peso `w_udhm[slot]` derivado de la varianza real de ese movimiento en datos:

```
w_udhm[index_mcp_flex] = 1/σ_flex   ← más varianza → peso menor
w_udhm[index_mcp_abd]  = 1/σ_abd    ← menos varianza → peso mayor
```

Esto da a abducción visibilidad real en la métrica. Con geodésica de cuaternión, la abducción queda enterrada bajo la magnitud dominante de la flexión.

---

## Añadir un robot nuevo

**Sin UDHM:** URDF + yaml con topología de links + build_primitives clasifica joints por FK geométrico + _MANUAL si falla + synthetic poses + validación de subespacio común.

**Con UDHM:** URDF + yaml que declara qué joint llena qué slot:

```yaml
udhm:
  index_mcp_flex: index_proximal_joint
  index_pip_flex: index_intermediate_joint
  thumb_mcp_abd:  thumb_proximal_yaw_joint
```

El signo se deriva del URDF automáticamente. Lo que no está declarado → slot = 0. `robot_to_udhm` funciona inmediatamente. El robot entra al pool multi-robot sin código adicional.

---

## Dónde empieza UDHM en el sistema

UDHM es la frontera entre percepción y cómputo. Todo lo anterior es "cómo llenar UDHM". Todo lo posterior es "operar sobre UDHM".

```
╔══════════════════════════════╗
║      PERCEPCIÓN / CARGA      ║
║                              ║
║  cámara → MediaPipe/HaMeR    ║
║  qpos   → robot URDF         ║
║  yaml   → mapeo joint→slot   ║
╚══════════╦═══════════════════╝
           ║  adapter por fuente
           ▼
╔══════════════════════════════╗
║         UDHM [B, 22]         ║  ← FRONTERA
║                              ║
║  slot 6 = index_mcp_flex     ║
║  slot 7 = index_pip_flex     ║
║  ausente = 0                 ║
╚══════════╦═══════════════════╝
           ║
     ┌─────┴──────┐
     ▼            ▼
  ángulos        FK(udhm, bone_lengths)
  por slot            ↓
     ↓         XYZ chain, tips
  losses           ↓
  L1 por slot   losses posición
  contrastive
```

Cada fuente de datos tiene su propio adapter que produce `udhm22 [B, 22]`:

| Fuente | Adapter | Estado |
|---|---|---|
| MediaPipe/Dong keypoints | `human_to_udhm(pose_dong, labels)` | Existe |
| HaMeR θ MANO | `hamer_to_udhm(theta_mano)` | A escribir si se cambia backend |
| Robot qpos | `robot_to_udhm(qpos, tabla)` | Existe |

Después del adapter, ningún código sabe de dónde vino el dato. Solo opera sobre `[B, 22]`.

**Consecuencia directa:** cambiar backend de percepción (ej. de MediaPipe a HaMeR) = escribir un adapter nuevo. El pipeline de training, losses, y contrastive no cambia. La percepción está desacoplada del entrenamiento.

HaMeR da `θ ∈ R^48` (parámetros MANO = ángulos por joint directamente). Con UDHM como frontera, `hamer_to_udhm()` mapea esos ángulos a slots canónicos sin tocar nada downstream. Sin UDHM, cambiar de MediaPipe a HaMeR requeriría reescribir `pose_h`, `chain_h`, `tips_h` y toda la infraestructura de subespacio.

---

## Resumen

| Problema | Sin UDHM | Con UDHM |
|---|---|---|
| Comparar abd vs flex | Mezclados en geodésica | Slots separados, pesos distintos |
| Robots con distinta morfología | Subespacio variable por par | Vector fijo 22, missing=0 |
| Pool multi-robot | Imposible (dim distinta) | Cat directo `[nB, 22]` |
| Añadir robot nuevo | 5+ puntos de intervención | 1 yaml con mapeo explícito |
| Robot sin pinky | Código especial de exclusión | slots 17-20 = 0 automático |

UDHM no es solo una representación conveniente. Es la estructura que hace que la geometría del problema (DoFs anatómicos, morfología variable, comparación cross-embodiment) se exprese directamente en el tensor, sin infraestructura de traducción.

---

## UDHM como contenedor: el orden fijo elimina el alineamiento

La idea central no es "UDHM guarda ángulos". Es que **UDHM es un contenedor de orden fijo que puede llevar múltiples representaciones del mismo estado de mano**, todas indexadas por el mismo slot anatómico.

Un slot no es un número — es una posición semántica fija. El slot 6 es `index_mcp_flex` siempre: venga de qpos de un robot, de la inferencia Dong de un humano, de una cámara, o derivado por FK. La fuente, la representación y el robot dejan de importar una vez que el dato está en su slot.

### Múltiples representaciones por slot

El contenedor lleva varias reps del mismo estado, cada consumidor pide la que necesita:

```
UDHM:
  angles [22]     radianes/pi por slot      -> L1 por slot, lam_udhm
  quats  [22, 4]  rotación por slot         -> D_R
  xyz    [22, 3]  posición por slot         -> tips / chain, losses posición
  mask   [22]     presencia (0/1) por slot  -> qué comparar
```

### Llenado: consumir lo que ya viene, derivar lo que falta, guardar

No todo se precalcula ni todo se deriva on-demand. El contenedor **consume lo que la fuente ya trae y deriva (y guarda) lo que falta**:

- Robot llega con qpos → llena `angles` directo (sabe sus ángulos).
- Humano llega con landmarks → `angles` se infieren por Dong.
- Alguien pide `xyz` y no existe → se deriva (FK desde `angles` + bone lengths) y **se guarda** en el slot. El siguiente que lo pida no recomputa.

Lo derivado se cachea. Lo directo se inserta. El orden fijo del slot garantiza que cualquier rep, de cualquier origen, quede alineada con todas las demás sin trabajo extra.

### La presencia se maneja con máscara, no con intersección

Robot con menos DoFs (Barrett sin pinky) → sus slots ausentes = 0, y `mask = 0` ahí. La máscara es lo que hace real el "comparar solo lo que ambos contienen":

```
S = (w_udhm * mask_a * mask_b * (u_a - u_c).abs()).sum(-1)
```

Slot ausente en cualquiera de los dos → su término se anula. **La máscara por slot reemplaza toda la maquinaria de intersección de hoy** (`filter_to_subspace`, `common_labels`, `common_fingers`, `shared_sorted`, `_mini`): esa infraestructura existe solo porque hoy no hay orden fijo y cada métrica reconstruye por su cuenta "qué tienen en común". Con orden fijo + máscara, el índice del slot ES el alineamiento y la máscara ES la intersección.

Nota: esto NO es el conditioning AdaLN por hand-type del paper UniDexTok (Fang et al., Eq. 4). Ese resuelve reconstrucción en un autoencoder VQ condicionado por embodiment. Aquí la tarea es contrastiva entre embodiments (triplets humano↔robot, robot↔robot), y la máscara de presencia es lo que corresponde — no recreamos el paper, tomamos solo el concepto de interfaz de orden fijo con zero-pad semántico.

### Consecuencia: una sola fuente alimenta encoders Y métrica

Hoy `sampler._assemble` ya produce un único dict que alimenta ambos caminos (encoders y métrica), pero con tensores fragmentados (`pose_h`, `q_r`, `pose_*_sub`, `tips_*_sub`, `udhm22_*`). Centralizar en UDHM colapsa eso en una salida canónica:

- **Encoders** (`E_h`, `E_r`) comen UDHM en vez de quats/qpos crudos. Como el input es fijo `[22]` con missing=0, `E_r`/`D_r` dejan de ser **uno por robot** (`Linear(n_joints → ...)`) y pasan a ser **universales** (`Linear(22 → ...)`): un robot nuevo entra sin módulo nuevo, llenando su UDHM. Esto borra la lógica de `robots[name]["E_r"]` y `--freeze_shared`.
- **Métrica** lee de la misma UDHM la rep que cada término necesita (`angles` para L1, `quats` para D_R, `xyz` para tips), ya alineada por slot.

Es un cambio que toca todo el proyecto y obliga re-entrenar desde cero. El payoff: un solo camino fuente → adapter → UDHM → {encoders, métrica}, sin código de alineamiento en ningún punto.

---

## Del encoder por-embodiment al encoder único (y por qué UDHM lo habilita)

### El linaje: Yan & Lee vs UniDexTok

El diseño actual hereda de **Yan & Lee (2026)**, no de UniDexTok. De UniDexTok solo se toma el concepto de UDHM (interfaz de orden fijo + semantic insertion + zero-pad). La arquitectura es Yan & Lee:

```
humano:  x_H [J_h × 4] (quats) ─────────────→ E_h ──→ z (subespacios)
robot:   x_r [J_r × 1] (escalares) → E_r[robot] → 1024 → E_X → z
                                                          ↓
                                                  D_X → shared → D_r[robot] → qpos
```

Punto clave (Yan & Lee, §"Learning a Unified Latent Space", textual): *"Since robots may have differing numbers of joints, their raw pose vectors vary in dimensionality. To handle this variation, each robot pose is first passed through a learnable robot-specific embedding layer, which transforms it into a fixed-size high-dimensional vector."*

Es decir: **`E_r` existe por UNA razón — unificar dimensión variable (J distinto por robot) a tamaño fijo.** No es inteligencia, es un parche de shape aprendido. Y "añadir robot" en Yan & Lee (§II-D) es **few-shot**: congelar E_h/E_X/D_X y entrenar SOLO el `E_r`/`D_r` nuevo (~15 min). No es zero-shot.

Correcciones de atribución importantes:
- **Yan & Lee usan MLPs planas** (8 FC layers, 256 neuronas), NO grafos. (§III-A-1, textual.)
- El **CAM-GNN es aporte propio de este proyecto** (linaje grasp-intent/Dong), no de Yan & Lee.
- El **zero-shot real** es de UniDexTok (Fang, Tabla 2: Inspire hand sin entrenar, 4-8°), no de Yan & Lee.

### UDHM ataca justo la razón de existir de E_r

`E_r` resuelve "J variable → tamaño fijo" con una capa **aprendida**. UDHM resuelve eso **río arriba, determinista**: semantic insertion → `[22]` fijo para todos, signo del URDF, ÷π. La razón #1 de `E_r` desaparece.

Además la unificación NO es solo de dimensión, también de **representación**: `human_to_udhm` (6D/quats → atan2 → ángulos escalares con signo `[22]`) y `robot_to_udhm` (qpos → signo·qpos/π → `[22]`) **ya producen la misma rep escalar `[22]`**. No es lossy: UDHM descompone cada joint en sus DoFs anatómicos de 1-grado (por eso 22 DoFs, no 16 joints); un hinge ES un escalar, el 6D estaba sobre-parametrizado.

Con dimensión Y representación unificadas antes de la red, **mantener `E_h` ≠ `E_r` ya no tiene justificación técnica** — es herencia del diseño viejo. Colapsa a un encoder único `E(UDHM[22])`.

### Consecuencia: todo entra como grafo + zero-shot

Si el encoder único es CAM-GNN sobre los 22 slots:

- El **grafo es la topología anatómica de UDHM** (22 nodos + adyacencia), FIJA y compartida por todos los embodiments — no es del robot.
- **El robot hereda el grafo anatómico que nunca tuvo** (hoy E_r es Linear plano, sin estructura). Su qpos entra a nodos anatómicos; el GNN propaga sobre anatomía.
- El CAM (adyacencia aprendida) se comparte → **una sola matriz de correlación, alimentada por TODOS los embodiments** (más datos por arista).
- Nodos ausentes (Barrett sin pinky) = nodos enmascarados; el GNN sobre grafo parcial es justo para lo que sirven los GNN.

Y habilita **zero-shot real** (no el few-shot de Yan & Lee):

```
robot nuevo → llena UDHM[22] vía su yaml   (adapter determinista, 0 training)
            → MISMO GNN compartido come [22] → latente   (0 módulo nuevo)
            → D → UDHM[22] → udhm_to_robot(yaml) → qpos   (determinista)
```

Cero módulos que entrenar. Yan & Lee es few-shot porque su unificador de dimensión es **aprendido** (`E_r`); aquí es **determinista** (adapter + yaml). **Esa es la contribución que separa este proyecto de ambos papers base: zero-shot cross-embodiment RETARGETING contrastivo** — Yan & Lee hace retargeting pero few-shot; Fang hace zero-shot pero es autoencoder de reconstrucción con AdaLN.

Caveat: zero-shot **corre** siempre (determinista), pero **generaliza** bien solo si el robot nuevo es subset anatómico de los 22 slots (antropomórfico) y hubo diversidad de embodiments en training. Manos no-antropomórficas / acopladas / subactuadas degradan (Fang, §Limitations) → caen a few-shot opcional.

### Qué simplifica "todo grafo"

- **Una sola topología** que mantener, no una por robot.
- **Un solo CAM**, más datos por arista.
- Mueren: `E_r`/`D_r` por robot, `E_X`/`D_X` posiblemente fundidos en el encoder/decoder único, `filter_to_subspace`, `_mini`, `shared_sorted`, `_OUTPUT_SPECS`/límites hardcodeados (→ del URDF).
- Robot nuevo = un yaml. Sin pesos, sin training.
- El zoo de módulos (E_h GNN + E_r Linear×N + E_X + D_X + D_r×N) colapsa hacia: **un encoder GNN + un decoder GNN sobre el grafo de 22 nodos.**

### El escalar `signo·qpos/π` sirve para encoder Y métrica (los 22 DoF son acotados)

Un razonamiento que conviene fijar para no equivocarse:

**El desacople abd/flex viene del SLOT, no de la representación.** UDHM ya separa cada DoF en su slot (22 = por-DoF, no por-joint). Una vez separados, abd y flex están en slots distintos pase lo que pase. La geodésica de rotación los mezclaba en el sistema viejo NO por usar rotación, sino por el **detour lossy** `qpos → FK → puntos → Dong → quat`: el qpos ya tenía abd y flex como columnas separadas, y el round-trip por posiciones los re-empaquetaba en un quaternión por joint. `robot_to_udhm` (qpos directo) nunca los mezcla.

**El escalar acotado aprende bien.** El resultado de Zhou et al. (2019) —reps de SO(3) de baja dim aprenden mal— aplica a **rotaciones 3D completas** (el mapa SO(3)→Rⁿ es discontinuo para n<5). Pero un DoF acotado (hinge en rango `[0, 1.57]`) vive en un arco 1-D, no en SO(3); el ángulo escalar es monótono y continuo sobre el rango, sin wraparound (las manos no hacen 360°). Aprende bien.

Ventajas del escalar `signo·qpos/π`, que lo hacen la rep correcta para ángulos tanto en encoder como en métrica:
1. **Nativo del robot.** El qpos ya es escalar → cero derivación, cero detour lossy (justo lo que mezclaba).
2. **Mínimo y lossless para 1 DoF.** Un hinge ES un escalar; 6D serían 6 números para 1 grado (redundante).
3. **Signo + ÷π interpretables.** El signo codifica dirección (abd+/−); `w_udhm` pesa directo; ×π da grados (MPJAE interpretable, como el paper).

```
UDHM (mismo orden de slots)
  ├─ signo·qpos/π escalar [22]  → ENCODER (GNN) y MÉTRICA  (misma rep de ángulo)
  └─ xyz por slot [22, 3]       → losses de tip / EE  (derivado vía FK + bone lengths)
```

El contenedor multi-rep sigue siendo necesario, pero **por las posiciones XYZ** (losses de fingertip/end-effector), no por duplicar la rep de ángulo. El robot deriva XYZ de su qpos vía FK y se cachea; el humano de sus landmarks. El message passing del GNN debe ser **enmascarado** (no agregar desde slots ausentes).

### La rep es una perilla, no la fundación

Para el latente COMPARTIDO el escalar puede ser mejor: el robot solo tiene escalar (qpos es su única realidad), así que un sustrato escalar común pone humano y robot en el mismo idioma → mejor alineación. Riesgo honesto: el escalar humano es **derivado** (atan2 de Dong) y la abducción se degrada bajo flexión fuerte (~5× ruido, ver `_udhm_w`), mientras el 6D humano era crudo. Por eso la rep se deja como **flag ablatible (escalar | 6D), no decisión fija** — el backbone UDHM es agnóstico a ella; cambiar la perilla solo toca `in_dim` del encoder y qué tensor lee del contenedor, no alineamiento/máscara/adapters/zero-shot.

---

## Diagrama TARGET: backbone UDHM

La rep aparece como flag en el único punto donde importa (lectura del nodo). Todo lo demás es invariante.

### Estructura de nodos del grafo

22 nodos (slot = nodo). Adyacencia derivada del árbol cinemático del URDF + anatomía UDHM:
- cadena intra-dedo: `mcp_abd ─ mcp_flex ─ pip_flex ─ dip_flex`
- abd↔flex del mismo joint: hermanos que comparten frame MCP (en URDF = dos revolutos 1-DoF en cascada, padre-hijo → arista)
- nodo palma/raíz conecta los MCP de los cinco dedos
- CAM (adyacencia aprendida) refina sobre esa base

### Camino de TRAINING

```
FUENTES
  humano: landmarks ─► human_to_udhm ─┐   robot: qpos ─► robot_to_udhm ─┐
  (MediaPipe/HaMeR)                   │   (URDF + yaml slot→joint)       │
                                      ▼                                  ▼
              ╔═══════════════════════════════════════════════════════════╗
              ║   CONTENEDOR UDHM   [B, 22]  + mask[22]                    ║  ← FRONTERA
              ║   slots: angles(signo·qpos/π) | 6D(derivado) | xyz(FK)    ║
              ║   orden fijo · cache deriva-y-guarda · zero-pad ausentes  ║
              ╚════════════╦══════════════════════════════════╦═══════════╝
                           │ [FLAG rep: escalar | 6D]          │ angles escalar [22]
                           ▼ lee rep por nodo                  ▼ + xyz[22,3]
              ╔════════════════════════════════╗   ╔══════════════════════════════╗
              ║  ENCODER ÚNICO (CAM-GNN)       ║   ║   MÉTRICA CONTRASTIVA          ║
              ║  22 nodos · msg passing         ║   ║   pool = cat(humano, robots)  ║
              ║  ENMASCARADO · compartido       ║   ║   S = Σ w·mask_a·mask_b·       ║
              ║  por TODOS los embodiments      ║   ║       |u_a − u_c|  + D_ee(xyz) ║
              ╚════════════╦═══════════════════╝   ║   triplets cross-embodiment    ║
                           ▼                        ╚══════════════╦═══════════════╝
                    z  (latente compartido)                        │ define pos/neg
                           │◄──────────────────────────────────────┘
                           ▼
              ╔════════════════════════════════╗
              ║  DECODER ÚNICO ─► UDHM[22]      ║   losses: L_contrastive (triplet),
              ║  (reconstrucción + consistencia)║           L_rec, L_ltc, L_temporal
              ╚════════════════════════════════╝
```

Sin `filter_to_subspace`, sin `common_labels`, sin `shared_sorted`, sin `_mini`, sin `E_r`/`D_r` por robot. El slot ES el alineamiento; la máscara ES la intersección.

### Camino de INFERENCIA (teleop)

```
cámara ─► human_to_udhm ─► UDHM[22] ─► ENCODER ─► z ─► DECODER ─► UDHM[22]
                                                                      │
                                              udhm_to_robot(yaml_robot)│  determinista
                                              signo·col + ×π + clip    ▼  (límites del URDF)
                                                                    qpos[J]  ─► robot
```

`udhm_to_robot` es aritmética sin pesos (slot→joint del yaml, signo del URDF, ×π, clip a límites del URDF). Los `_OUTPUT_SPECS`/`_LOWER`/`_UPPER` hardcodeados de `retarget.py` mueren → salen del URDF.

### Robot nuevo = zero-shot

```
robot nuevo ─► escribe SOLO su yaml (slot→joint) ─► robot_to_udhm funciona
            ─► MISMO encoder/decoder come [22]  (0 módulo, 0 training)
            ─► udhm_to_robot(yaml_nuevo) ─► qpos
```

Degrada si la mano no es subset anatómico de los 22 (no-antropomórfica/acoplada) → few-shot opcional (Fang §Limitations).

### Lo invariante vs la perilla

| Capa | ¿Cambia con flag rep? |
|---|---|
| Contenedor [22] + mask, adapters, zero-pad | No |
| Grafo 22 nodos, adyacencia URDF, msg passing enmascarado | No |
| Encoder/decoder único, latente compartido | No (solo `in_dim`) |
| Métrica por-slot mask-gated, triplets | No |
| `udhm_to_robot`, zero-shot | No |
| Qué tensor lee el nodo (escalar \| 6D) | **Sí — único punto** |

### El contrato como JSON Schema + yamls autorados por humano

El contrato UDHM se expresa como un **JSON Schema** (definición formal validable): qué es un UDHM válido — 22 slots, cada uno `name/finger/joint/dof/axis`, normalizado por π, orden fijo. Es `udhm_canonical_22dof.yaml` (tiene `schema_version`, `n_dof: 22`, `slots`).

El JSON Schema NO es un stage de transformación — es la **especificación** a la que los adapters se conforman. Cada embodiment tiene un **yaml escrito por un humano** que es su instancia del contrato:
- **robot yaml:** `FFJ3 → index_mcp_flex` (joint qpos → slot)
- **humano yaml:** `dong_index_mcp → index_mcp_flex` (label Dong/MediaPipe → slot)

```
JSON Schema (contrato, udhm_canonical_22dof.yaml)
   ▲ valida
   │
yaml robot   ── instancia humana (joint→slot)
yaml humano  ── instancia humana (label Dong→slot)
```

Por qué importa:
1. **Es el único paso manual.** Todo lo demás (adapters, FK, encoder, decoder) es automático. El humano solo escribe el yaml de mapeo.
2. **El JSON Schema hace seguro ese trabajo humano:** valida nombre de slot, campos, signo → atrapa typos antes de entrenar.
3. **Es la cara humana del zero-shot:** robot nuevo = un humano escribe su yaml (validado por schema), 0 training.

### Estructura del diagrama (drawio)

```
FUENTES                ADAPTERS (yaml humano)         CONTRATO
  Fuente humana ──► human_to_udhm (Dong, yaml labels→slot) ─┐
  Fuente robot  ──► robot_to_udhm (yaml joint→slot)         ─┤   JSON Schema (udhm_canonical)
                                                             │      ▲ valida los yaml
                                                             ▼      │
                              ╔════════════════════════════════════╗
                              ║  CONTENEDOR UDHM[22] + mask         ║  ← frontera
                              ║  reps cacheadas: angles|6D|xyz      ║
                              ╚═══════╦═══════════════════╦═════════╝
                                grafo │                   │ xyz
                                      ▼                   ▼
                              GNN Encoder            ┌─ Losses ─┐
                                      │              │ contrastive (latente)
                                      ▼              │ D_ee (xyz)
                                Latent Space ────────┘ rec, ltc, temporal
                                      │
                          ┌───────────┴───────────┐
                          ▼                        ▼
                    MLP Decoder (default)    GNN Decoder (flag, espejo)
                          └───────────┬───────────┘
                                      ▼
                                 UDHM[22]
                                      │ udhm_to_robot (MISMO yaml del robot, al revés)
                                      ▼
                                 Hand Pose (qpos[J])
```

Notas de corrección sobre el drawio dibujado:
- El humano usa `human_to_udhm` (Dong), no el mismo "YAML" que el robot — ambos tienen yaml pero distinto dominio (labels Dong vs joints qpos).
- El yaml de salida (`UDHM → YAML → Hand Pose`) es el **mismo yaml del robot**, usado al revés (`udhm_to_robot`). No es uno nuevo.
- `JSON Schema` = contrato/validador lateral, no stage en serie del flujo de datos.
- Falta caja explícita del **contenedor UDHM[22]+mask** (hoy el drawio tiene "UDHM Interface" + "Grafo UDHM" por separado; el contenedor es la frontera central de la que salen grafo→encoder y xyz→losses).
- Marcar MLP Decoder como default, GNN Decoder como flag (cross-embodiment/zero-shot).
- Losses conecta a Latent (triplet) Y a UDHM xyz (D_ee), no solo al grafo.
