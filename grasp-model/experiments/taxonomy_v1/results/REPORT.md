# Synergy-Taxonomy Analysis v1 -- Reporte de Metodologia y Resultados

**Fecha:** 2026-03-10
**Objetivo:** Determinar cuales de las 28 clases de agarre de HOGraspNet (Feix et al., 2016)
son posturalmente distinguibles a partir de 21 keypoints XYZ de la mano, y proponer candidatos
de colapso para una taxonomia reducida en el modelo GCN de clasificacion de intento de agarre.

---

## 1. Datos de Entrada

**Dataset:** HOGraspNet (ECCV 2024). 1,489,111 frames, 99 sujetos, 28 clases Feix, 30 objetos YCB.

**CSV:** `data/raw/hograspnet.csv`, generado por `scripts/ingestion/build_hograspnet_csv.py`.
- Columnas: `subject_id`, `sequence_id`, `cam`, `grasp_type`, `contact_sum`, 63 XYZ (21 keypoints x 3 coordenadas).
- Cada fila es un frame individual de una secuencia de agarre.
- `sequence_id` identifica univocamente una secuencia de agarre (un sujeto, un objeto, un agarre, una camara).

**Normalizacion geometrica aplicada en la ingesta (frame a frame):**

1. Traslacion: restar la posicion del WRIST (kp0) a todos los keypoints.
   - Resultado: WRIST = [0, 0, 0] para todo frame.
   - Formula: `kp_i' = kp_i - kp_0` para i = 0..20.

2. Escala: normalizar la distancia WRIST-INDEX_MCP (kp0-kp5) a 10 cm.
   - Formula: `kp_i'' = kp_i' * (0.1 / ||kp_5'||)`.
   - Justificacion: Santos et al. (2025) validaron esta distancia de anclaje en teleoperacion 1:1.
   - Nota: dado que kp_0 = [0,0,0] despues de la traslacion, `||kp_5'|| = ||kp_5 - kp_0||`.

**Propiedad critica:** Esta normalizacion es una traslacion + escala uniforme. Preserva angulos
y proporciones entre segmentos. No afecta ningun calculo angular posterior.

**Filtro de Existencia (solo para el analisis de sinergias, Secciones 2-6):** Se descartan
frames con `contact_sum = 0` (mano no toca el objeto). Solo se analizan frames donde el
agarre esta ejecutandose activamente.
- Frames descartados: 30,623 (2.1% del total).

**Nota importante:** Este filtro se aplica **unicamente al pipeline de sinergias** (extraccion
de angulos, PCA, MLP). El GCN de Run 006 (Seccion 7) se entrena **sin este filtro**, sobre
todos los frames incluyendo aproximacion y liberacion. Las dos pipelines operan sobre
distribuciones ligeramente distintas; las implicaciones se discuten en la Seccion 9.

**Split S1 (por sujeto, no por frame):**

Frames del pipeline de sinergias (con filtro contact_sum > 0):

| Conjunto | Sujetos | Frames | % |
|----------|---------|--------|---|
| Train | S11-S73 (63 sujetos) | 995,886 | 68.3% |
| Val | S01-S10 (10 sujetos) | 138,952 | 9.5% |
| Test | S74-S99 (26 sujetos) | 323,650 | 22.2% |

Frames del GCN Run 006 (sin filtro contact_sum):

| Conjunto | Sujetos | Frames | % |
|----------|---------|--------|---|
| Train | S11-S73 (63 sujetos) | 1,015,342 | 68.2% |
| Val | S01-S10 (10 sujetos) | 146,509 | 9.8% |
| Test | S74-S99 (26 sujetos) | 327,260 | 22.0% |

Justificacion: el split por sujeto evita data leakage temporal. Si separamos frames aleatoriamente,
frames consecutivos del mismo video (misma secuencia, mismo sujeto) aparecen en train y test,
inflando las metricas artificialmente. Con split por sujeto, el modelo se evalua en personas
que nunca vio durante el entrenamiento.

---

## 2. Extraccion de Angulos Articulares (20 DOFs)

Se extraen 20 angulos articulares por frame, siguiendo el modelo de 20 revolute joints de
DexPilot (Handa et al., ICRA 2020): 1 abduccion + 3 flexion por dedo, 5 dedos = 20.

### 2.1 Angulos de Flexion (15)

**Definicion matematica:** Para cada articulacion con un triplete (parent, joint, child), se
calculan dos vectores oseos convergentes y su angulo:

```
v_in  = xyz[joint] - xyz[parent]    # bone vector parent -> joint
v_out = xyz[child] - xyz[joint]     # bone vector joint -> child

theta_flex = arccos( dot(v_in / ||v_in||, v_out / ||v_out||) )
```

Donde `||v|| = max(||v||_2, 1e-8)` para evitar division por cero, y el coseno se clampea
a [-1, 1] antes del arccos para estabilidad numerica.

**Referencia:** Chen et al. (Robotics and Autonomous Systems, 2025), Ecuacion (10). Este es el
metodo estandar de joint space mapping para teleoperacion de manos diestras.

**Tripletes concretos (parent, joint, child -> keypoints):**

| Angulo | parent | joint | child | Articulacion |
|--------|--------|-------|-------|-------------|
| THUMB_CMC_flex | kp0 (WRIST) | kp1 (THUMB_CMC) | kp2 (THUMB_MCP) | CMC del pulgar |
| THUMB_MCP_flex | kp1 | kp2 | kp3 | MCP del pulgar |
| THUMB_IP_flex | kp2 | kp3 | kp4 | IP del pulgar |
| INDEX_MCP_flex | kp0 | kp5 | kp6 | MCP del indice |
| INDEX_PIP_flex | kp5 | kp6 | kp7 | PIP del indice |
| INDEX_DIP_flex | kp6 | kp7 | kp8 | DIP del indice |
| MIDDLE_MCP_flex | kp0 | kp9 | kp10 | MCP del medio |
| MIDDLE_PIP_flex | kp9 | kp10 | kp11 | PIP del medio |
| MIDDLE_DIP_flex | kp10 | kp11 | kp12 | DIP del medio |
| RING_MCP_flex | kp0 | kp13 | kp14 | MCP del anular |
| RING_PIP_flex | kp13 | kp14 | kp15 | PIP del anular |
| RING_DIP_flex | kp14 | kp15 | kp16 | DIP del anular |
| PINKY_MCP_flex | kp0 | kp17 | kp18 | MCP del menique |
| PINKY_PIP_flex | kp17 | kp18 | kp19 | PIP del menique |
| PINKY_DIP_flex | kp18 | kp19 | kp20 | DIP del menique |

**Nota sobre WRIST como parent:** Para los MCP de cada dedo, el parent es WRIST (kp0).
Dado que la normalizacion situa WRIST en [0,0,0], el vector `v_in = xyz[MCP] - xyz[WRIST] = xyz[MCP]`.
Esto es geometricamente correcto: es el vector metacarpal desde la muneca al MCP.

**Nota sobre la semantica del angulo:** `theta_flex` mide el angulo entre los dos segmentos oseos
que convergen en la articulacion. NO es el angulo de flexion anatomico puro (que se mide desde
la posicion neutra extendida). Sin embargo, para el proposito de este experimento (comparar
posturas entre clases), lo que importa es que la metrica sea consistente entre frames y clases,
no que sea anatomicamente absoluta. El mismo calculo aplicado a todas las clases produce una
representacion comparable.

**Rango observado (sanity check, 5 frames de muestra):** [0.026, 1.409] rad ([1.5, 80.7] grados).
Fisiologicamente plausible.

### 2.2 Angulos de Abduccion (5)

**Definicion del plano de la palma:**

```
va = xyz[kp5] - xyz[kp0]     # WRIST -> INDEX_MCP
vb = xyz[kp17] - xyz[kp0]    # WRIST -> PINKY_MCP

palm_normal = normalize( cross(va, vb) )
```

Dado que kp0 = [0,0,0], esto simplifica a `cross(kp5, kp17)`. El plano queda definido
por los dos vectores metacarpales extremos (indice y menique), lo cual captura bien la
orientacion de la palma.

**Potencial problema:** Si kp5 y kp17 son colineales (mano con todos los dedos cerrados
hacia el mismo punto), el cross product sera cercano a cero y la normal sera inestable.
En la practica, estos keypoints estan separados angularmente por ~30-50 grados en la
mayoria de las posturas, y el clip a 1e-8 en la normalizacion previene division por cero.

**Abduccion de dedos largos (4 angulos -- INDEX, MIDDLE, RING, PINKY):**

Para cada dedo i, se definen:
```
r = xyz[MCP_i] - xyz[WRIST]    # vector metacarpal (referencia neutra)
v = xyz[PIP_i] - xyz[MCP_i]    # vector falange proximal (direccion real)
```

Se proyectan ambos vectores al plano de la palma eliminando su componente normal:
```
r_proj = r - dot(r, palm_normal) * palm_normal
v_proj = v - dot(v, palm_normal) * palm_normal
```

El angulo de abduccion es:
```
theta_abd = arccos( dot(normalize(r_proj), normalize(v_proj)) )
```

| Angulo | r (metacarpal) | v (proximal) | Keypoints |
|--------|---------------|--------------|-----------|
| INDEX_MCP_abd | kp5 - kp0 | kp6 - kp5 | 0, 5, 6 |
| MIDDLE_MCP_abd | kp9 - kp0 | kp10 - kp9 | 0, 9, 10 |
| RING_MCP_abd | kp13 - kp0 | kp14 - kp13 | 0, 13, 14 |
| PINKY_MCP_abd | kp17 - kp0 | kp18 - kp17 | 0, 17, 18 |

**Limitacion:** Este angulo es siempre positivo (arccos retorna [0, pi]). No distingue
abduccion (dedos separandose) de adduccion (dedos acercandose). Para este experimento
es aceptable porque buscamos magnitud de desviacion lateral, no su direccion.

**Diferencia con Jarque-Bou et al. (2019):** Ellos miden abduccion **relativa** entre dedos
adyacentes (MCP 3-4 y MCP 4-5, 2 angulos) usando un CyberGlove. Nosotros medimos abduccion
**absoluta** por dedo (4 angulos) derivada de keypoints 3D. Ambas capturan la separacion lateral,
pero con diferente referencia.

**Abduccion del pulgar (1 angulo -- THUMB_CMC_abd):**

El pulgar tiene un mecanismo diferente. Su abduccion es la componente fuera del plano
palmar del metacarpo del pulgar:
```
thumb_dir = xyz[kp2] - xyz[kp1]    # THUMB_MCP - THUMB_CMC

theta_abd_thumb = arcsin( |dot(normalize(thumb_dir), palm_normal)| )
```

Se usa arcsin (no arccos) porque se mide la componente perpendicular al plano, no la
componente en el plano. El valor absoluto asegura que el angulo sea positivo independiente
de la orientacion del metacarpo respecto a la normal.

**Rango observado (sanity check):** [0.079, 1.049] rad ([4.5, 60.1] grados). Plausible.

### 2.3 Sintesis por Mediana -- El Punto de Trial

Cada `sequence_id` contiene multiples frames de una secuencia de agarre completa (aproximacion,
contacto, manipulacion, liberacion). Despues de aplicar el filtro de existencia (contact_sum > 0),
los frames restantes corresponden mayoritariamente a las fases de contacto y manipulacion.

Para cada sequence_id, se calcula la **mediana** de cada uno de los 20 angulos sobre todos
los frames de ese trial:

```
median_angle_j = median({angle_j[frame] : frame in sequence_id})
```

**Justificacion de la mediana (vs media):**
- La mediana es robusta a outliers (frames al inicio/final del contacto donde la postura
  aun esta transitando).
- No asume distribucion normal de los angulos dentro de un trial.

**Resultado:** Un vector de 20 angulos por trial. Esto reduce el dataset de ~1M frames a
~13,500 trials (8,361 train, 1,514 val, 3,671 test).

**Nota critica:** El `grasp_type` asignado a cada trial es el de su primer frame (todos los
frames de un sequence_id comparten la misma clase).

### 2.4 Comparacion con la literatura

Nuestros 20 DOFs se comparan con los 17 angulos anatomicos de Jarque-Bou et al. (2019),
quienes usaron un CyberGlove II de 22 sensores calibrado sobre 77 sujetos:

| DOF | Nosotros (21 keypoints) | Jarque-Bou (CyberGlove) |
|-----|------------------------|------------------------|
| MCP flexion (5 dedos) | Si (5 angulos) | Si (MCP1_F a MCP5_F) |
| PIP flexion (4 dedos largos) | Si (4 angulos) | Si (PIP2 a PIP5) |
| DIP flexion (4 dedos largos) | Si (4 angulos) | No (CyberGlove no mide DIP) |
| IP flexion pulgar | Si (1 angulo) | Si (IP1_F) |
| CMC flexion pulgar | Si (1 angulo) | Si (CMC1_F) |
| CMC abduccion pulgar | Si (1 angulo) | Si (CMC1_A) |
| Abduccion MCP dedos | Si, absoluta (4 angulos) | Si, relativa (2 angulos: MCP 3-4, 4-5) |
| Palmar arch (CMC5) | **No** (no observable) | Si |
| Wrist flexion/deviation | **No** (WRIST = origen) | Si (WRIST_F, WRIST_A) |

**DOFs que no podemos medir:** La flexion/desviacion de la muneca y el arco palmar (CMC5)
no son observables desde 21 keypoints porque WRIST es nuestro punto de referencia (origen)
y no tenemos sensor en la palma. La Sinergia 2 de Jarque-Bou (coordinacion muneca + arco
palmar) no aparecera en nuestros resultados por esta limitacion. Esto es inherente a la
representacion de 21 keypoints, no un error del metodo.

**DOFs adicionales nuestros:** Tenemos 4 angulos DIP que el CyberGlove no mide.

---

## 3. Analisis de Componentes Principales (PCA) con Varimax Rotation

### 3.1 Normalizacion estadistica (z-score)

Los 20 angulos de los trials de entrenamiento se normalizan con z-score:
```
angle_j_normalized = (angle_j - mean_j) / std_j
```

Donde `mean_j` y `std_j` se calculan **exclusivamente sobre el conjunto de entrenamiento**.
La misma transformacion (con los mismos parametros) se aplica a validacion y test.

**Justificacion:** Los angulos de flexion tienen rangos de movimiento (ROM) mas amplios
(~0-80 grados) que los de abduccion (~0-30 grados). Sin z-score, el PCA estaria dominado
por la varianza de las flexiones, ignorando movimientos de abduccion que son criticos para
distinguir agarres como Lateral vs Adducted_Thumb. El z-score permite comparar articulaciones
con diferentes ROM dandoles igual peso en el analisis. Jarque-Bou et al. (2019) aplican
el mismo procedimiento: *"The data standardization procedure followed in this study allowed
us to compare joints with different range of motion"*.

**Nota:** Esta normalizacion estadistica es diferente de la normalizacion geometrica de la Seccion 1.
La geometrica (traslacion + escala) se aplica a los XYZ crudos y preserva angulos.
La estadistica (z-score) se aplica a los angulos derivados y es necesaria para que PCA
opere sobre la matriz de correlacion en vez de la matriz de covarianza, como es estandar
cuando las variables tienen diferentes escalas fisicas.

### 3.2 Ejecucion del PCA

Se aplica PCA completo (20 componentes) sobre los 8,361 vectores medianos normalizados del
conjunto de entrenamiento. Se retienen los componentes que acumulen al menos 85% de la varianza.

**Resultado: 9 componentes para 87.9% de varianza.**

| PC | Varianza explicada | Acumulada |
|----|-------------------|-----------|
| PC1 | 40.65% | 40.65% |
| PC2 | 11.71% | 52.36% |
| PC3 | 7.03% | 59.39% |
| PC4 | 6.42% | 65.82% |
| PC5 | 5.59% | 71.40% |
| PC6 | 5.45% | 76.85% |
| PC7 | 4.22% | 81.07% |
| PC8 | 3.56% | 84.63% |
| PC9 | 3.23% | 87.86% |

**Comparacion:** Jarque-Bou et al. (2019) encontraron 5-6 PCs por sujeto explicando ~83%
de varianza con 17 DOFs. Nosotros encontramos 9 PCs para ~88% con 20 DOFs. La diferencia
se explica por tener mas variables (20 vs 17) y por usar PCA global (vs per-subject).

### 3.3 Varimax Rotation

**Motivacion:** Las PCs originales del PCA son matematicamente optimas para capturar varianza,
pero cada componente puede cargar en muchas variables simultaneamente, dificultando la
interpretacion fisiologica. La rotacion varimax (Kaiser, 1958) rota los ejes en el subespacio
retenido para maximizar la "estructura simple": que cada componente rotado (RC) cargue
fuertemente en pocas variables y tenga carga cercana a cero en las demas.

**Definicion matematica:** Sea `L` la matriz de loadings (20 x 9), varimax busca la
matriz de rotacion ortogonal `R` (9 x 9) que maximiza:

```
V(R) = sum_j [ (1/p) * sum_i (l'_ij^2)^2 - ((1/p) * sum_i l'_ij^2)^2 ]
```

donde `L' = L * R` son los loadings rotados, `p` = 20 variables, y la maximizacion se
resuelve iterativamente via SVD (convergencia tipica en < 20 iteraciones, tolerancia 1e-6).

**Propiedad fundamental:** La rotacion es ortogonal. Esto significa que:
- Las distancias entre puntos en el espacio PCA **no cambian**
- La varianza total explicada **no cambia** (87.9%)
- Solo cambia la orientacion de los ejes (la interpretacion)

Analogia: es como girar un mapa para que el norte quede arriba. Las ciudades no se mueven,
pero el mapa se lee mejor.

**Referencia:** Jarque-Bou et al. (2019) aplican varimax rotation con el mismo proposito:
extraer sinergias fisiologicamente interpretables.

### 3.4 Componentes Rotados (RC) y su interpretacion fisiologica

| RC | Top loadings | Interpretacion |
|----|-------------|---------------|
| RC1 | PINKY_DIP_flex(+0.522), RING_DIP_flex(+0.451), PINKY_PIP_flex(+0.377) | Flexion distal dedos 4-5 |
| RC2 | INDEX_MCP_flex(-0.665), INDEX_MCP_abd(-0.586), MIDDLE_MCP_flex(-0.285) | Extension + abduccion del indice |
| RC3 | THUMB_CMC_abd(+0.921) | Abduccion del pulgar (casi puro) |
| RC4 | PINKY_MCP_abd(+0.575), RING_MCP_abd(+0.567), PINKY_MCP_flex(+0.351) | Abduccion lateral dedos 4-5 |
| RC5 | THUMB_MCP_flex(+0.976) | Flexion MCP del pulgar (casi puro) |
| RC6 | THUMB_IP_flex(+0.984) | Flexion IP del pulgar (casi puro) |
| RC7 | MIDDLE_MCP_abd(+0.736), MIDDLE_MCP_flex(+0.266) | Abduccion del medio |
| RC8 | THUMB_CMC_flex(+0.963) | Flexion CMC del pulgar (casi puro) |
| RC9 | INDEX_DIP_flex(+0.706), INDEX_PIP_flex(+0.532) | Flexion distal del indice |

**Observaciones:**
- **RC3, RC5, RC6, RC8** son componentes casi puros (loading > 0.92 en una sola variable).
  Esto es el efecto deseado de varimax: cada RC captura un DOF interpretable.
- **RC1** (flexion distal dedos 4-5) corresponde parcialmente a la Sinergia 1 de Jarque-Bou
  (flexion MCP/PIP dedos 3-5), aunque nuestra version captura mas la componente DIP.
- **RC3** (abduccion del pulgar, loading 0.921) corresponde a la Sinergia 3 de Jarque-Bou
  (oposicion del pulgar: CMC abd + MCP ext + IP flex).
- Los 5 angulos de abduccion aparecen prominentemente en RC2, RC3, RC4 y RC7. Sin ellos,
  varimax no podria aislar estos componentes.

**Nota:** La Sinergia 2 de Jarque-Bou (coordinacion muneca + arco palmar) no aparece en
nuestros resultados, como se anticipo en la Seccion 2.4.

---

## 4. Analisis de Separabilidad entre Clases

### 4.1 PCA global como espacio comun

El PCA se ajusto sobre todos los trials de entrenamiento conjuntamente (PCA global),
no por sujeto ni por clase. Esto produce un espacio de 9 dimensiones comun donde todos
los trials (de todas las clases y todos los sujetos) se proyectan y son directamente
comparables.

**Justificacion:** Nuestro objetivo es comparar clases entre si. Esto requiere un espacio
comun. Un PCA per-subject (como Jarque-Bou) responderia "que modulos motores comparten los
sujetos", pero no permite comparar directamente los centroides de clase. PCA global es la
opcion natural para nuestra pregunta.

**Posible debilidad:** PCA global maximiza la varianza total, que puede estar dominada por
varianza intra-clase (diferencias entre sujetos haciendo el mismo agarre) en vez de varianza
inter-clase (diferencias entre agarres). Si esto ocurriera, los PCs no serian los mas
discriminantes. Sin embargo, el hecho de que PC1 (cierre global, 40.65%) sea consistente
con la clasificacion Feix de alto nivel (power vs precision) sugiere que la varianza
inter-clase domina.

### 4.2 Centroides de clase

Para cada una de las 28 clases, se calcula el centroide (media aritmetica) de todos sus
trials en el espacio PCA rotado de 9 dimensiones:

```
centroid_c = mean({z_i : grasp_type(trial_i) == c}),   z_i in R^9
```

Los centroides completos se presentan en `class_separability.json`. Aqui los primeros 3 RCs:

| Clase | n trials | RC1 | RC2 | RC3 |
|-------|----------|-----|-----|-----|
| [0] Large_Diameter | 310 | -2.57 | +0.50 | +0.25 |
| [1] Small_Diameter | 162 | +1.61 | -1.10 | -0.36 |
| [2] Index_Finger_Extension | 389 | +1.94 | +1.39 | -0.65 |
| [3] Extension_Type | 489 | -1.11 | -0.13 | +0.82 |
| [4] Parallel_Extension | 309 | -2.37 | -0.42 | -0.64 |
| [5] Palmar | 210 | +1.01 | -0.96 | -0.24 |
| [6] Medium_Wrap | 96 | +0.41 | -0.22 | -0.12 |
| [7] Adducted_Thumb | 450 | +1.22 | -0.71 | -1.09 |
| [8] Light_Tool | 452 | +2.33 | -1.21 | -1.10 |
| [9] Distal | 613 | -1.38 | +1.37 | +0.20 |
| [10] Ring | 308 | -3.06 | +0.82 | -0.02 |
| [11] Power_Disk | 195 | -0.17 | +0.63 | +0.53 |
| [12] Power_Sphere | 181 | -0.87 | +1.13 | +1.21 |
| [13] Sphere_4_Finger | 93 | -2.03 | +0.35 | -0.35 |
| [14] Sphere_3_Finger | 279 | +1.72 | +1.08 | +0.13 |
| [15] Lateral | 673 | +2.31 | -2.11 | -0.76 |
| [16] Stick | 88 | +2.59 | -1.23 | -0.92 |
| [17] Adduction_Grip | 77 | +0.38 | -0.34 | -0.48 |
| [18] Writing_Tripod | 360 | +1.23 | -0.92 | +0.18 |
| [19] Lateral_Tripod | 80 | +0.96 | -1.59 | -0.06 |
| [20] Palmar_Pinch | 265 | -0.95 | -1.07 | +0.56 |
| [21] Tip_Pinch | 188 | -1.21 | -0.89 | +0.12 |
| [22] Inferior_Pincer | 263 | -1.84 | -0.10 | +0.39 |
| [23] Prismatic_3_Finger | 193 | +0.24 | -1.44 | +0.07 |
| [24] Precision_Disk | 890 | -1.04 | +1.66 | +0.68 |
| [25] Precision_Sphere | 203 | -1.02 | +1.06 | +0.57 |
| [26] Quadpod | 84 | -0.47 | +0.18 | +0.79 |
| [27] Tripod | 461 | +1.42 | +0.15 | +0.28 |

### 4.3 Distancias entre centroides

Se calcula la distancia euclidiana entre todos los pares de centroides en R^9:

```
d(c_i, c_j) = ||centroid_i - centroid_j||_2
```

La distancia euclidiana en el espacio PCA rotado es identica a la distancia en el espacio
PCA sin rotar, porque varimax es una rotacion ortogonal. Esta distancia es razonable aqui
porque PCA ya decorrelaciono las variables.

**Top 15 pares mas cercanos:**

| # | Clase A | Clase B | Distancia |
|---|---------|---------|-----------|
| 1 | [24] Precision_Disk | [25] Precision_Sphere | 0.962 |
| 2 | [12] Power_Sphere | [25] Precision_Sphere | 0.979 |
| 3 | [20] Palmar_Pinch | [21] Tip_Pinch | 1.197 |
| 4 | [5] Palmar | [6] Medium_Wrap | 1.337 |
| 5 | [18] Writing_Tripod | [19] Lateral_Tripod | 1.352 |
| 6 | [25] Precision_Sphere | [26] Quadpod | 1.371 |
| 7 | [12] Power_Sphere | [24] Precision_Disk | 1.406 |
| 8 | [0] Large_Diameter | [22] Inferior_Pincer | 1.451 |
| 9 | [5] Palmar | [18] Writing_Tripod | 1.477 |
| 10 | [9] Distal | [25] Precision_Sphere | 1.480 |
| 11 | [11] Power_Disk | [12] Power_Sphere | 1.480 |
| 12 | [12] Power_Sphere | [26] Quadpod | 1.485 |
| 13 | [6] Medium_Wrap | [18] Writing_Tripod | 1.498 |
| 14 | [0] Large_Diameter | [10] Ring | 1.557 |
| 15 | [1] Small_Diameter | [5] Palmar | 1.567 |

### 4.4 Clustering jerarquico de centroides (dendrograma)

Se aplica clustering jerarquico con linkage Ward sobre los 28 centroides en R^9.
El dendrograma resultante se guarda en `results/dendrogram_classes_varimax.png`.

**Linkage Ward:** Minimiza la varianza intra-cluster al fusionar. Es apropiado aqui porque
buscamos grupos compactos de clases. La metrica subyacente es distancia euclidiana.

**Interpretacion del dendrograma:** Se lee de abajo hacia arriba. Las clases que se fusionan
a menor distancia son las mas parecidas posturalmente segun las sinergias. El eje vertical
indica la distancia Ward a la que se produce cada fusion.

**Pares que se fusionan primero (distancia < 1.5):**
- {Precision_Disk, Precision_Sphere} (~0.96) -- practicamente la misma postura
- {Power_Sphere, Precision_Sphere} (~0.98) -- cluster esferico
- {Palmar_Pinch, Tip_Pinch} (~1.20)
- {Palmar, Medium_Wrap} (~1.34)
- {Writing_Tripod, Lateral_Tripod} (~1.35)

**Grupos que emergen a distancia ~2.0:**
- {Precision_Disk, Precision_Sphere, Power_Sphere} -- esferico/disco
- {Palmar_Pinch, Tip_Pinch, Extension_Type} -- pinch/extension
- {Adducted_Thumb, Light_Tool, Small_Diameter} -- pulgar lateral
- {Large_Diameter, Inferior_Pincer, Ring} -- power envolvente
- {Index_Finger_Extension, Sphere_3_Finger, Tripod} -- tripod/3 dedos
- {Writing_Tripod, Lateral_Tripod, Medium_Wrap, Palmar} -- tripod/palmar

**Gran division (~14.3):** El dendrograma se parte en dos macro-ramas correspondientes
a la clasificacion de alto nivel de Feix: agarres con dedos cerrados (power/pinch, RC1 negativo)
vs agarres con pulgar lateral/dedos abiertos (lateral/tripod, RC1 positivo).

---

## 5. Validacion Empirica -- MLP con 28 Clases Originales

### 5.1 Proposito

Las secciones anteriores analizan la separabilidad en el espacio de sinergias (angulos
medianos por trial, proyectados por PCA). Este paso responde una pregunta complementaria:
"dado un frame individual XYZ de un sujeto nunca visto, puede un clasificador distinguir
las 28 clases originales?" La respuesta es la **matriz de confusion**, que muestra
empiricamente que pares se confunden desde la representacion nativa del sensor.

### 5.2 Arquitectura y entrenamiento

**Entrada:** 63 valores XYZ aplanados (21 keypoints x 3 coordenadas) normalizados geometricamente.
Esta es la representacion nativa del sensor, la misma que recibe el GCN.

**Salida:** 28 clases (las etiquetas originales de HOGraspNet).

**Arquitectura:** MLP de 3 capas:
- Linear(63, 256) -> ReLU -> Dropout(0.3)
- Linear(256, 128) -> ReLU -> Dropout(0.3)
- Linear(128, 28) -> LogSoftmax

**Hiperparametros:**
- Optimizador: Adam, lr = 1e-3
- Batch size: 256
- Epochs maximos: 50
- Early stopping: patience = 10 (sobre val accuracy)
- Loss: NLL (Negative Log Likelihood)
- Seed: 42

**Nota sobre la entrada:** Se entrena sobre FRAMES individuales (995,886 train), no sobre
trials medianos. Esto es deliberado: el MLP simula las condiciones del sensor en tiempo real,
donde cada frame se clasifica independientemente.

### 5.3 Resultados

| Metrica | Valor |
|---------|-------|
| Test Accuracy | 69.7% |
| Macro F1 | 0.625 |
| Early stopping | Epoch 48 |
| Best Val Accuracy | ~64.8% |

**Comparacion con el GCN (Run 005, 16 clases colapsadas):**
- GCN Run 005: 71.6% accuracy, 0.672 macro F1 (16 clases)
- MLP 28 clases: 69.7% accuracy, 0.625 macro F1 (28 clases)

El MLP con 28 clases, sin informacion de grafo ni features adicionales, alcanza un rendimiento
comparable. El valor absoluto de accuracy no es el punto central de este experimento --
lo relevante es la **estructura de confusiones**: que pares se confunden sistematicamente.

### 5.4 Confusiones significativas (>10% en alguna direccion)

| Clase verdadera | Predicha como | Tasa | Reversa | Tipo |
|-----------------|---------------|------|---------|------|
| Stick | Light_Tool | 42.4% | 1.3% | Asimetrica: Stick se absorbe |
| Precision_Sphere | Precision_Disk | 36.9% | 2.6% | Asimetrica: P.Sphere se absorbe |
| Lateral_Tripod | Writing_Tripod | 32.8% | 0.6% | Asimetrica: Lat.Tripod se absorbe |
| Medium_Wrap | Adducted_Thumb | 30.3% | 0.3% | Asimetrica: Med.Wrap se absorbe |
| Small_Diameter | Adducted_Thumb | 25.2% | 2.9% | Asimetrica: Small.Diam se absorbe |
| Tip_Pinch | Palmar_Pinch | 23.8% | 1.7% | Asimetrica: Tip.Pinch se absorbe |
| Power_Disk | Large_Diameter | 23.5% | 0.2% | Asimetrica: Power.Disk se dispersa |
| Ring | Inferior_Pincer | 22.4% | 9.7% | Bidireccional parcial |
| Sphere_3_Finger | Tripod | 21.8% | 8.1% | Bidireccional parcial |
| Tip_Pinch | Lateral | 18.6% | 1.1% | Asimetrica |
| Stick | Lateral | 17.5% | 0.3% | Asimetrica |
| Power_Sphere | Precision_Disk | 15.1% | 2.4% | Asimetrica |
| Medium_Wrap | Writing_Tripod | 14.1% | 0.0% | Asimetrica |
| Prismatic_3_Finger | Tripod | 14.0% | 3.1% | Asimetrica |
| Prismatic_3_Finger | Palmar_Pinch | 13.4% | 0.7% | Asimetrica |
| Lateral_Tripod | Light_Tool | 13.2% | 0.1% | Asimetrica |
| Adducted_Thumb | Light_Tool | 12.7% | 4.4% | Bidireccional parcial |
| Lateral | Light_Tool | 12.5% | 11.5% | **Bidireccional** |
| Prismatic_3_Finger | Writing_Tripod | 12.1% | 2.7% | Asimetrica |
| Lateral_Tripod | Prismatic_3_Finger | 11.5% | 2.2% | Asimetrica |
| Writing_Tripod | Tripod | 11.1% | 4.8% | Asimetrica |
| Small_Diameter | Light_Tool | 10.6% | 1.2% | Asimetrica |
| Palmar | Parallel_Extension | 10.4% | 2.1% | Asimetrica |
| Precision_Sphere | Power_Sphere | 10.1% | 5.6% | Bidireccional parcial |

### 5.5 Interpretacion de las confusiones

**Confusiones asimetricas (A -> B alto, B -> A bajo):** La clase A es un subconjunto
postural de B desde la perspectiva del sensor. El MLP clasifica A como B porque sus
coordenadas XYZ son indistinguibles. Colapsar A en B no pierde informacion discriminante.

**Confusiones bidireccionales (ambas > 10%):** Solapamiento genuino. Ambas clases
ocupan la misma region del espacio postural. Solo se encontro un par claramente bidireccional:
Lateral <-> Light_Tool (12.5% / 11.5%).

---

## 6. Concordancia entre Sinergias y Confusiones del MLP

El cruce entre distancias en el espacio PCA (Seccion 4.3) y confusiones del MLP (Seccion 5.4)
ilustra dos patrones:

| Par | Dist. PCA | Confusion MLP | Patron |
|-----|-----------|---------------|--------|
| Precision_Disk / Precision_Sphere | 0.962 | 36.9% | Cercanos en sinergias Y MLP confunde |
| Palmar_Pinch / Tip_Pinch | 1.197 | 23.8% | Cercanos en sinergias Y MLP confunde |
| Writing_Tripod / Lateral_Tripod | 1.352 | 32.8% | Cercanos en sinergias Y MLP confunde |
| Stick / Light_Tool | >2.0 | 42.4% | Lejanos en sinergias, MLP confunde |
| Medium_Wrap / Adducted_Thumb | >2.0 | 30.3% | Lejanos en sinergias, MLP confunde |

**Patron 1 (cercanos + MLP confunde):** Evidencia convergente de que las clases son
posturalmente similares. La distancia de centroides baja y la confusion del clasificador
coinciden.

**Patron 2 (lejanos + MLP confunde):** Los centroides estan separados pero las
distribuciones se solapan en las colas. La distancia de centroides captura la postura
"tipica" de cada clase, pero no su variabilidad. Un clasificador que opera frame a frame
experimenta el solapamiento de las colas aunque los promedios sean distintos.

**Limitacion del MLP como criterio de colapso:** El MLP es un clasificador debil
(XYZ aplanado, sin estructura de grafo ni features angulares). El hecho de que confunda
un par no implica que un modelo mas fuerte tambien lo haga. Medium_Wrap/Adducted_Thumb
es un ejemplo: el MLP los confunde (30.3%) pero el GCN entrenado con features angulares
los separa (c_gcn=0.097, por debajo del umbral de colapso). La confusion del MLP en
este par es un artefacto de su representacion limitada, no una propiedad del espacio
postural.

**El criterio de colapso definitivo es la confusion del GCN** (Run 006, Seccion 7),
no la confusion del MLP. El MLP sirve como referencia exploratoria para identificar
candidatos, pero la decision final requiere el modelo arquitecturalmente correcto
entrenado con la representacion completa.

---

## 7. Decision de Colapso

### 7.1 Inputs

| Input | Origen | Contenido |
|-------|--------|-----------|
| `class_separability.json` | Seccion 4b de este experimento | `class_centroids_pca`: 28 centroides en R^9 (espacio PCA-varimax) |
| `confusion_matrix_norm_gcn.csv` | Run 006 (GCN_CAM_8_8_16_16_32, 28 clases, S1 split) | Matriz de confusion normalizada por fila (recall en la diagonal) |

El Run 006 entrenó el GCN en las 28 clases originales con el mismo split S1 y la misma normalizacion geometrica. Configuracion: 1,015,342 frames de entrenamiento, 40 epochs maximos, early stopping patience=10, batch=256, Adam lr=1e-3, NLL loss. Mejor epoch: 15 (val acc=60.2%). Test: 62.0% accuracy, macro F1=0.550 (327,260 frames, sujetos S74-S99).

### 7.2 Criterio de Colapso

Para cada par (i, j) de las 378 combinaciones posibles se calcula:

```
c_gcn(i,j) = max( CM[i,j], CM[j,i] )
```

Se usa el maximo (no la media) porque en teleoperacion la pregunta relevante no es "son mutuamente indistinguibles?" sino "es alguna de las dos clases no confiable para el operador?". Si CM[A,B] = 21%, el operador intentando producir el agarre A recibira el token B 1 de cada 5 veces, independientemente de si B tambien se confunde como A. La media ocultaria confusiones asimetricas donde una clase pequena se absorbe en una grande.

**Impacto concreto de usar max() vs mean():**

| Par | max() | mean() | Decision con max | Decision con mean |
|-----|-------|--------|-----------------|------------------|
| Light_Tool + Stick | 0.239 | 0.122 | colapsa | no colapsa |
| Lateral_Tripod + Tripod | 0.211 | 0.106 | colapsa | no colapsa |

Con mean() el resultado seria 19 clases en vez de 17, conservando Stick (recall=5.4%) y Lateral_Tripod (recall=22.7%) como categorias separadas no operacionales.

**Umbral:** `c_gcn > 0.15`.

**Justificacion del umbral:** Ordenando los 378 pares por c_gcn descendente, existe un gap entre el par 13 (c=0.177, Writing_Tripod+Tripod) y el par 14 (c=0.135, Medium_Wrap+Precision_Disk). Ningun par tiene c_gcn en el intervalo [0.135, 0.177]. El umbral de 0.15 cae en el centro de este gap -- cualquier valor en [0.135, 0.177] produce exactamente los mismos 13 pares colapsados. Adicionalmente, 0.15 = 4.2 * (1/28), es decir 4.2 veces el nivel de azar con 28 clases, lo que indica confusion sistematica y no error aleatorio. Los 13 pares colapsan estan en el percentil 97 de la distribucion completa (p97=0.178).

El par 14 (Medium_Wrap->Precision_Disk, c=0.135) es una confusion fuertemente asimetrica: CM[Medium_Wrap, Precision_Disk]=0.135 pero CM[Precision_Disk, Medium_Wrap]=0.001. Es una clase pequena absorbida parcialmente por una grande, no una indistinguibilidad genuina.

**Rol de d_synergy:** La distancia sinergica entre centroides se incluye como campo descriptivo en el output -- caracteriza *por que* el GCN confunde cada par, pero no determina si colapsar. No se aplica ningun umbral a d_synergy.

### 7.3 Resultados

**13 pares colapsan** (c_gcn > 0.15, ordenados por c_gcn):

| Par | d_synergy | c_gcn | same_feix | Nota |
|-----|-----------|-------|-----------|------|
| Lateral + Stick | 1.729 | 0.370 | Si | Centroides cercanos |
| Precision_Disk + Precision_Sphere | 0.962 | 0.313 | Si | Par mas cercano en sinergias |
| Prismatic_3F + Tripod | 2.263 | 0.289 | No | Centroides distantes, cross-cell |
| Light_Tool + Lateral | 2.677 | 0.281 | No | Centroides distantes, cross-cell |
| Light_Tool + Stick | 1.615 | 0.239 | No | Asimetrico: Stick->Light_Tool 23.9% |
| Ring + Inferior_Pincer | 1.884 | 0.221 | No | Cross-cell |
| Writing_Tripod + Lateral_Tripod | 1.352 | 0.214 | No | Centroides cercanos |
| Lateral_Tripod + Tripod | 2.980 | 0.211 | No | Asimetrico: Lat_Tripod->Tripod 21.1% |
| Palmar_Pinch + Tip_Pinch | 1.197 | 0.196 | Si | Centroides cercanos |
| Large_Diameter + Power_Disk | 2.921 | 0.189 | Si | Centroides distantes |
| Sphere_3_Finger + Tripod | 1.739 | 0.182 | No | Cross-cell |
| Palmar_Pinch + Inferior_Pincer | 1.762 | 0.179 | Si | |
| Writing_Tripod + Tripod | 1.707 | 0.177 | No | Par mas debil (c justo sobre umbral) |

**Four-case framework (cruce con celdas Feix):**

| Caso | Celda Feix | GCN | Pares | Interpretacion |
|------|-----------|-----|-------|----------------|
| A | Misma | confunde | 5 | Feix y datos coinciden |
| B | Misma | separa | 15 | GCN supera a Feix: e.g. Large_Diameter vs Small_Diameter (c=0.0) |
| C | Distinta | confunde | 8 | Colapso data-driven que Feix no predijo |
| D | Distinta | separa | 350 | Ambos coinciden en separar |

El Caso B es la validacion clave: 15 pares comparten celda Feix pero el GCN los distingue perfectamente. Esto demuestra que la celda Feix no es un criterio suficiente ni necesario para el colapso -- el criterio empirico del GCN puede diverger del criterio taxonomico experto en ambas direcciones.

### 7.4 Taxonomia Propuesta

Los 13 pares se agrupan via union-find (la transitividad es intencional: si el GCN confunde A+B y B+C, el cluster {A,B,C} es la agrupacion natural de confusion). Resultado: **28 -> 17 clases**.

| Nuevo ID | Miembros | Tamano |
|----------|---------|--------|
| 0 | Sphere_3_Finger, Writing_Tripod, Lateral_Tripod, Prismatic_3F, Tripod | 5 |
| 1 | Ring, Palmar_Pinch, Tip_Pinch, Inferior_Pincer | 4 |
| 2 | Light_Tool, Lateral, Stick | 3 |
| 3 | Large_Diameter, Power_Disk | 2 |
| 4 | Precision_Disk, Precision_Sphere | 2 |
| 5-16 | Small_Diameter, Index_Finger_Ext, Extension_Type, Parallel_Ext, Palmar, Medium_Wrap, Adducted_Thumb, Distal, Power_Sphere, Sphere_4_Finger, Adduction_Grip, Quadpod | 1 cada uno |

**Observacion notable:** 17 clases coincide con el numero de categorias posturalmente equivalentes que Feix et al. (2016) identifican al agrupar sus 33 agarres por configuracion de mano (columnas de la Fig. 4, ignorando tipo de oposicion). La coincidencia es independiente: nosotros llegamos desde confusion empirica del GCN entrenado en keypoints 3D; Feix desde analisis biomecanico experto. Los grupos difieren pero la cardinalidad converge, lo que sugiere que 17 es aproximadamente el techo informatico para reconocimiento de agarres sin contacto desde landmarks.

### 7.5 Outputs

- `collapse_decisions.csv` -- 378 pares con d_synergy, c_gcn, same_feix, collapse
- `proposed_taxonomy.json` -- taxonomia de 17 clases
- `scatter_synergy_vs_gcn.png` -- plot 2D (eje x: d_synergy descriptivo, eje y: c_gcn criterio)
- `four_case_summary.txt` -- detalle de los 4 casos con todos los pares

**Script:** `decide_collapses.py --gcn_thresh 0.15 --results_dir experiments/taxonomy_v1/results`

---

## 8. Artefactos y Datos

**Archivos generados:**
- `results/class_separability.json` -- Centroides PCA rotados, distancias, loadings varimax, confusion matrix, F1 por clase.
- `results/dendrogram_classes_varimax.png` -- Dendrograma Ward de los 28 centroides en espacio PCA rotado.
- `results/scatter_rc1_rc2_varimax.png` -- Distribucion de trials en RC1 vs RC2, coloreado por clase.
- `results/confusion_28classes.png` -- Matriz de confusion normalizada por fila del MLP 28 clases.

**Codigo fuente:**
- `config.py` -- Todos los parametros y definiciones (DOFs, split, hiperparametros).
- `data.py` -- Carga CSV, filtro de existencia, split S1.
- `features.py` -- Calculo de 20 angulos (flexion + abduccion) y sintesis por mediana.
- `analyze_class_separability.py` -- Pipeline completo: PCA + varimax + centroides + dendrograma + MLP.

---

## 9. Limitaciones y Puntos a Revisar

1. **DOFs faltantes vs CyberGlove:** No tenemos flexion/desviacion de muneca ni arco palmar.
   La Sinergia 2 de Jarque-Bou (muneca + palmar arch) no es capturada. Esto es una limitacion
   inherente a 21 keypoints, no del metodo. Podria afectar la separabilidad de clases que
   difieren principalmente en la posicion de la muneca.

2. **Abduccion absoluta vs relativa:** Nuestros angulos de abduccion son absolutos (referencia:
   metacarpal), los de Jarque-Bou son relativos (referencia: dedo adyacente). Ambos capturan
   la separacion lateral pero con diferente semantica. No sabemos si una es superior a la otra
   para nuestro proposito.

3. **Angulos de abduccion sin signo:** arccos retorna [0, pi]. Si dos clases difieren en la
   direccion pero no en la magnitud de la abduccion, este calculo no las distinguira.

4. **Centroides con media vs mediana:** Los centroides de clase usan la media de los trials
   medianos. Se podria usar la mediana de los trials medianos para mayor robustez a outliers.
   Impacto esperado: bajo, dado el numero alto de trials por clase (minimo 77).

5. **Distancia euclidiana en PCA:** Asume que las 9 dimensiones son igualmente relevantes.
   Alternativa: ponderar por varianza explicada, o usar Mahalanobis intra-clase. No se
   implemento por simplicidad y porque PCA ya decorrelaciona.

6. **PCA global vs per-subject:** PCA global puede capturar varianza intra-clase (diferencias
   entre sujetos) ademas de inter-clase (diferencias entre agarres). No obstante, la
   consistencia entre nuestro PC1 y la clasificacion Feix top-level sugiere que la varianza
   inter-clase domina.

7. **MLP como proxy del sensor:** El MLP valida si XYZ aplanado puede distinguir las clases,
   pero el GCN tiene informacion adicional (topologia del grafo, features por nodo). Es
   posible que el GCN distinga pares que el MLP confunde. Sin embargo, si el MLP confunde
   un par, es evidencia fuerte de que la informacion postural disponible es insuficiente.

8. **Tamanio de muestra desigual:** Algunas clases tienen ~80 trials (Adduction_Grip,
   Lateral_Tripod, Quadpod) mientras otras tienen ~890 (Precision_Disk). Los centroides
   de clases pequenas son menos estables. El MLP tambien puede sesgar hacia clases grandes.

9. **Filtro asimetrico entre sinergias y GCN:** El pipeline de sinergias (Secciones 2-6)
   aplica filtro contact_sum > 0 (solo frames con contacto activo), mientras que Run 006
   se entreno sobre todos los frames. Esto tiene dos efectos opuestos: (a) la matriz de
   confusion del GCN es mas conservadora porque incluye frames de transicion donde la
   postura es ambigua -- si el GCN confunde dos clases incluso bajo estas condiciones mas
   dificiles, el colapso esta mas justificado; (b) los centroides del espacio de sinergias
   representan posturas "en estado estable" y pueden no capturar la confusion que ocurre
   durante las transiciones. Para un analisis completamente consistente, ambos pipelines
   deberian usar el mismo filtro.

10. **Single-view depth:** HOGraspNet usa una sola camara de profundidad. Las confusiones
   pueden estar amplificadas por auto-oclusiones que un sistema multi-camara resolveria.
   Esto no afecta la validez del analisis para nuestro caso de uso (sensor monocular).

---

## Referencias

- Chen, M. et al. (2025). Robust dexterous hand control strategy cascading bare hand pose estimation and joint jitter suppression. *Robotics and Autonomous Systems*, 194, 105189.
- Feix, T. et al. (2016). The GRASP Taxonomy of Human Grasp Types. *IEEE Transactions on Human-Machine Systems*, 46(1), 66-77.
- Handa, A. et al. (2020). DexPilot: Vision-Based Teleoperation of Dexterous Robotic Hand-Arm System. *ICRA 2020*.
- Jarque-Bou, N.J. et al. (2019). Kinematic synergies of hand grasps: a comprehensive study on a large publicly available dataset. *Journal of NeuroEngineering and Rehabilitation*, 16, 63.
- Kaiser, H.F. (1958). The varimax criterion for analytic rotation in factor analysis. *Psychometrika*, 23(3), 187-200.
- Santello, M. et al. (1998). Postural hand synergies for tool use. *Journal of Neuroscience*, 18(23), 10105-10115.
- Santos, D. et al. (2025). [Teleoperacion 1:1 con escala INDEX_MCP a 10cm].
