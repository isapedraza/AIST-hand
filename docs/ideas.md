# Ideas — próximos experimentos

Surgidas del análisis comparativo contra Yan 2025/2026 y el diagnóstico de runs 20–33.

---

## Idea A — Oracle global + gradiente per-finger

**Problema que ataca:** runs per-finger (29, 30, 31, 33) fallan en MCPs porque el oracle per-finger (3 joints) permite que el modelo satisfaga S_k vía PIP/DIP ignorando MCP. Run 27B resuelve MCPs pero pierde independencia porque usa single latent.

**Propuesta:** mantener arquitectura per-finger (5 subespacios z_k independientes) pero usar oracle GLOBAL — D_R sobre 15 joints + D_joints global — para seleccionar pos/neg. Mismo par pos/neg para todos los subespacios en cada tripleta.

```
S_global = D_R_global(15 joints) + w_joints * D_joints_global
selector: pos_idx, neg_idx  ← mismo para z_thumb, z_index, ..., z_pinky
gradiente: fluye por cada z_k independientemente
```

**Diferencia vs runs anteriores:**
- vs Run 29: oracle era per-finger (3 joints). Aquí es global (15 joints).
- vs Run 33: Run 33 añadió loss global ADICIONAL sobre z_total. Aquí se REEMPLAZA el oracle per-finger con uno global.
- vs Run 27B: mismo oracle pero latente per-finger en lugar de single.

**Riesgo:** todos los z_k reciben gradiente de la misma señal global → podrían aprender representaciones redundantes. Mitigación: L_rec fuerza especialización implícita (si z_index y z_thumb aprenden lo mismo, el decoder no puede reconstruir q_r correctamente).

**Costo:** ~10 líneas en loop.py. Branch nueva desde run33.

---

## Idea B — 5 decoders per-finger

**Problema que ataca:** decoder lineal global `D_r: Linear(1024→24)` tiene pesos cruzados entre dedos. Con per-finger latents, z_index puede afectar joints del pulgar vía el decoder. Independencia no está garantizada estructuralmente.

**Propuesta:** reemplazar D_r global con 5 cabezas independientes:

```
z_thumb  [64] → D_thumb:  Linear(64 → 5)  → THJ1-5
z_index  [64] → D_index:  Linear(64 → 4)  → FFJ1-4
z_middle [64] → D_middle: Linear(64 → 4)  → MFJ1-4
z_ring   [64] → D_ring:   Linear(64 → 4)  → RFJ1-4
z_pinky  [64] → D_pinky:  Linear(64 → 5)  → LFJ1-5
```

Independencia forzada por estructura, no emergente.

**Diferencia vs Idea A:** hipótesis diferente. Idea A asume el oracle es el bloqueador. Idea B asume el decoder es el bloqueador.

**Costo:** cambio en robot_modules.py + loop.py + losses.py (L_rec redefinida per-dedo).

**Orden sugerido:** probar Idea A primero (más barato). Si falla, Idea B con evidencia de que el decoder es el problema.

---

## Idea C — [DESCARTADA] malinterpretación de "6D"

Esta idea nació de leer "6D" como el vector de features 6D de Yan 2025 (5 ratios
de flexión + aducción del pulgar). Fue una interpretación errónea: lo que se
quería decir con "6D" era la representación **6D continua de SO(3)** (Zhou et al.
2019), que ya está capturada correctamente en la **Idea F**. No hay propuesta
separada aquí — ver Idea F.

---

## Idea D — Xin como pérdida directa (no como similitud)

**Problema que ataca:** Xin S_k actualmente es oracle `no_grad` — selecciona pos/neg pero no fluye gradiente. El modelo puede satisfacer el triplet en z sin que las poses robot decodificadas realmente parezcan a las humanas en espacio Cartesiano.

**Propuesta:** añadir Xin como pérdida directa sobre las poses decodificadas:

```python
q_r_hat = D_r(D_X(z_h))          # pose robot decodificada desde humano vía latent consistency
tips_r_hat = FK(q_r_hat)          # fingertips del robot decodificado
L_xin = xin_sk_full(tips_h, tips_r_hat, chain_h, chain_r_hat)
L_total += lambda_xin * L_xin
```

Gradiente fluye: `L_xin → q_r_hat → D_r → D_X → z_h → E_h`.

**Diferencia vs L_ltc:** L_ltc garantiza que el robot decodificado re-encodea cerca del latente humano. L_xin garantiza que el robot decodificado SE PARECE al humano en espacio Cartesiano.

**Riesgo:** requiere pares implícitos (humano → robot vía latent consistency). Sin pares reales, L_xin mide similitud entre pose humana y pose robot generada — que puede ser ruidosa al inicio del training cuando D_r aún no ha convergido.

**Relación con literatura:** Xin et al. usan esto como objetivo IK directo (NLopt). DexMV también lo usa como objetivo IK. La diferencia: ellos tienen correspondencias explícitas; aquí se usa vía la cadena de latent consistency.

---

## Idea E — Sampler basado en primitivas FLEX/ABD/ROT

**Problema que ataca:** eigengrasp PCA samplea en subespacio de agarres Dexonomy/Shadow. MCPs pueden estar subrepresentados en ese subespacio. Además rompe zero-shot (requiere dataset por robot).

**Propuesta:** reemplazar eigengrasp con sampling random uniforme en espacio de primitivas derivadas del URDF:

```
α_flex_thumb  ∈ [0, 1]   → THJ1-3 escalados
α_flex_index  ∈ [0, 1]   → FFJ1-3 escalados
α_flex_middle ∈ [0, 1]   → MFJ1-3 escalados
α_flex_ring   ∈ [0, 1]   → RFJ1-3 escalados
α_flex_pinky  ∈ [0, 1]   → LFJ1-3 escalados
α_abd_index   ∈ [-1, 1]  → FFJ4
α_abd_middle  ∈ [-1, 1]  → MFJ4
α_abd_ring    ∈ [-1, 1]  → RFJ4
α_rot_thumb   ∈ [0, 1]   → THJ4/THJ5
```

Coeficientes random uniform → mapeo a joints via URDF kinematics → poses diversas.

**Ventajas vs eigengrasp:**
- Cubre rango completo de MCPs (no limitado al hull PCA de Dexonomy)
- Zero-shot: clasificación de joints via excitación unitaria en MuJoCo (sysID), sin dataset
- Poses coordinadas por construcción — no dedos sueltos

**Ventajas vs random puro:**
- Sin auto-colisión (flexión coordinada per-dedo)
- Poses funcionales — alineadas a la morfología del robot

**Costo:** requiere sysID (excitación unitaria en MuJoCo para clasificar joints como FLEX/ABD/ROT por dedo). Implementado en DexGrasp-Zero. Reemplaza `generate_valid_robot_poses.py`.

**Relación con literatura:** DexGrasp-Zero usa exactamente este esquema. Diferencia: ellos RL autónomo, tú teleop.

**Impacto en hipótesis MCPs:** si el problema es que eigengrasp no cubre MCPs extremos, este sampler lo resolvería. Pero la evidencia actual (27B funciona con mismo pool) sugiere que el sampler no es el bloqueador principal — es complementario a Ideas A/B.

---

## Idea F — Representación 6D continua de SO(3) en lugar de quaterniones

**Problema que ataca:** quaterniones tienen discontinuidad en SO(3) — `q` y `-q` representan la misma rotación. El encoder CAM-GNN recibe quaterniones como features de nodos. Si dos poses idénticas producen `q` y `-q` (mismo hemisferio pero signo opuesto), el encoder ve inputs distintos para la misma rotación. Dong ya impone la convención `w≥0` para mitigar esto, pero no elimina la discontinuidad.

**Propuesta:** reemplazar quaterniones [4D] con representación 6D continua de SO(3) (Zhou et al. 2019). Las primeras dos columnas de la matriz de rotación 3×3, aplanadas → 6 dims por joint. Continua en SO(3), sin discontinuidades, gradientes más limpios.

```
R ∈ SO(3) → [r1 | r2] ∈ R^6   (columnas 1 y 2 de R)
```

**Impacto en arquitectura:** E_h recibe [B, 20, 6] en lugar de [B, 20, 4]. CAMLayer input dim cambia de 4 a 6. D_R también debería actualizarse — en lugar de dot² entre quaterniones, distancia geodésica entre rotaciones 6D.

**Por qué es valioso aunque no sea la solución:** descarta discontinuidades como fuente de ruido. Si el problema persiste con 6D, se confirma que la representación no es el bloqueador.

**Costo:** cambio en human_modules.py (in_dim 4→6), dong_math.py (output 6D), losses.py (d_r_yan con 6D). Moderado.

**Referencia:** Zhou et al. 2019 "On the Continuity of Rotation Representations in Neural Networks". Entry 95 de DECISIONS lo menciona como alternativa futura.

---

## Idea G — [PENDIENTE] variante en la formulación del triplet/contrastive

**Nota de marcador (sin confirmar):** existe la posibilidad de modificar ligeramente
cómo se lleva a cabo la pérdida contrastive / triplet, no solo a qué se aplica. La
formulación actual es un triplet margin clásico (anchor / pos / neg con
`relu(d_pos - d_neg + margin)`). Recuerdo haber leído un paper con una variante
relevante — falta identificarlo.

**Posibles direcciones a investigar (placeholder, no decididas):**
- InfoNCE / NT-Xent en lugar de triplet margin (multi-negativo, softmax sobre el batch).
- Pesado o minería de pares duros (hard-negative mining) en vez de pos/neg aleatorios.
- Margen adaptativo o por-subespacio.

**Pendiente:** localizar el paper de referencia antes de proponer cambio concreto.
Por ahora solo se anota la posibilidad para no perderla.

---

## Idea H — D_R ponderado por tipo de joint (boost MCP)

**Problema que ataca:** `d_r_yan` actualmente es una suma UNIFORME sobre los joints
comunes: `D_R = sum_j (1 - dot_j^2)`, con peso 1 para todos. El MCP pesa lo mismo
que PIP y DIP, así que el oracle puede rankear pares satisfaciendo PIP/DIP sin
priorizar el cierre del MCP.

**Propuesta:** pesar D_R por tipo de joint, igual que `lam_thumb_tip` da 10x a la
punta del pulgar — pero aquí para el MCP:

```
D_R = sum_j w_j (1 - dot_j^2)     # w_mcp alto, w_pip / w_dip bajo
```

**Relación con lo existente:** `losses.py` ya tiene `_sk_w` (pesos D_R por joint,
orden `[mcp, pip, dip, tip]`), pero esos pesos vienen de `1/sigma` (normalización
de varianza sobre pares HOGraspNet). La idea aquí es distinta en motivación: pesar
por IMPORTANCIA para el agarre (el MCP define el cierre de la mano), no por varianza.
Misma maquinaria, criterio de peso distinto.

**Diferencia vs Idea A:** Idea A ataca el bug del MCP por el lado del GRADIENTE
(triplets per-finger separados). Idea H lo ataca por el lado del ORACLE (pesar la
señal MCP en la selección de pos/neg). Son complementarias — se pueden combinar.

**Costo:** bajo. Reemplazar la suma uniforme en `d_r_yan` por una ponderada, con
pesos por tipo de joint configurables. ~5-10 líneas.

---

## Idea I (PROPUESTA PRINCIPAL) -- Latente híbrido grueso+fino + oracle compuesto

Esta es la propuesta consolidada a partir del análisis comparativo de los runs
20, 21sk, 27B y 28-34 (junio 2026). Es la idea con más evidencia detrás.

### El problema en una frase

Ningún run logra a la vez: (1) cerrar los MCP de forma coordinada, (2) control
independiente por dedo, (3) oposición de pulgar. Cada run consiguió uno o dos,
nunca los tres.

### Evidencia (hechos observados por run)

| run | latente | oracle | comportamiento observado |
|-----|---------|--------|--------------------------|
| run20 | per-finger | quat (D_R + D_joints) | mimic, dedos independientes / MCP NO cierra |
| 21sk | holístico (1 cabeza) | paper-Xin (switching + maquinaria pulgar) + D_R pesado 1/sigma | PULGAR PERFECTO / dedos desastre / MCP no cierra |
| 27B | holístico (1 cabeza) | Xin simple + D_R UNIFORME (16.5) | MCP CIERRA / tieso / pulgar mediocre |
| 28-34 | per-finger | Xin simple + D_R | MCP invertido / busca pinch / dedos colapsados |

Nota: 27B y 21sk tienen la MISMA estructura holística pero resultados opuestos.
Eso prueba que la estructura no lo explica todo: hay un segundo eje (el oracle).

### Los dos ejes (marco derivado de la evidencia)

**Eje 1 -- estructura del latente, decide la CAPACIDAD DE COORDINACIÓN:**
- holístico (1 cabeza, pool global de los 20 joints): puede cerrar coordinado.
- per-finger (5 cabezas, pool por dedo): control independiente, NO cierra.

**Eje 2 -- contenido del oracle, decide QUÉ comportamiento se captura:**
- D_R uniforme: enfatiza rotación de todos los joints igual, captura el MCP, cierra (27B).
- D_R pesado 1/sigma: baja el peso del MCP, no cierra (21sk).
- maquinaria pulgar (paper-Xin): oposición de pulgar perfecta (21sk).
- señales per-finger: control fino (run20).

Los dos ejes son independientes: la estructura habilita, el oracle dirige.

### Ingredientes separables (cada uno probado por UN run)

1. Cierre coordinado: latente holístico + D_R uniforme. (27B)
2. Pulgar perfecto: maquinaria pulgar del paper-Xin (switching-gated pinch +
   `D_thumb_pos` + peso del tip por min-distancia a cualquier dedo). (21sk)
   IMPORTANTE: esta maquinaria se tiró en Run 25 y NO está en 27B/33/34.
3. Control independiente por dedo: cabezas per-finger + triplets per-chunk. (run20)

### La propuesta

Un SOLO latente, dividido en 6 bloques (no 5 como run33): un bloque GRUESO
(global) + 5 bloques FINOS (uno por dedo).

```
ENCODER (forward):
  x = CAM-GNN(21 nodos)                              # backbone, igual que hoy
  z_global = proj_global( maxpool(x[LOS 20 joints]) )   # GRUESO -- NUEVO (cabeza de 27B)
  z_thumb  = proj_thumb(  maxpool(x[nodos pulgar]) )    # FINO  ┐
  z_index  = proj_index(  maxpool(x[nodos índice]) )    # FINO  │ (igual que run33)
  z_middle = ...                                        # FINO  │
  z_ring   = ...                                        # FINO  │
  z_pinky  = ...                                        # FINO  ┘
  z = [ z_global | z_thumb | z_index | z_middle | z_ring | z_pinky ]

CONTRASTIVE:
  1 triplet sobre z_global   -> coordinación / cierre   (oracle whole-hand)
  5 triplets sobre z_dedo    -> control independiente   (oracle per-finger)

ORACLE (selección pos/neg, compuesto de los 3 ingredientes):
  + D_R uniforme / MCP-pesado     -> cierre MCP        (de 27B)
  + maquinaria pulgar (restaurar) -> oposición pulgar  (de 21sk paper-Xin)
  + señales per-finger            -> control           (de run20)

DECODER: recibe el vector z completo (los 6 bloques) -> ángulos de joints.
```

### Intuición grueso/fino (fácil)

Es UN latente, como una mochila con compartimentos. run33 tiene 5 compartimentos
(uno por dedo). El híbrido agrega 1 compartimento más, grande, que mira la mano
entera.

- Bloque GRUESO (`z_global`): "la mano cierra en conjunto" = coordinación, cierre.
- Bloques FINOS (`z_dedo`): "este dedo además así" = ajuste local por dedo.

El decoder combina los dos: el grueso pone el gesto base (cerrar coordinado), el
fino lo modula dedo por dedo. Ejemplo trípode: grueso = "cierre parcial"; fino =
"pulgar+índice+medio se tocan, anular y meñique recogidos distinto".

### Por qué NO es lo mismo que run33

run33 era per-finger (5 bloques) + una LOSS global sobre la suma de los 5 chunks.
Esa suma es info fina apilada: NO contiene coordinación inter-dedo, porque ninguna
cabeza fina la calculó nunca. No se puede recuperar "la mano cierra junta" sumando
5 descripciones de dedos aislados.

El híbrido arregla a nivel ENCODER (una cabeza nueva con pool global de los 20
joints), no a nivel loss. Ese es el órgano de coordinación que a run33 le faltaba.
run33 = arreglo en la loss. Híbrido = arreglo en el encoder. Distinto nivel.

### Por qué Idea A no bastó

Idea A asumió que el cuello era el ALCANCE DE LA SELECCIÓN (per-finger vs global).
Pero run34 (selección global) falló igual que 28-33. La evidencia dice que el
bloqueador es la DESCOMPOSICIÓN per-finger del latente en sí: sin un bloque que
mire la mano entera, el cierre coordinado no tiene dónde vivir. Idea I cambia la
ESTRUCTURA (añade el bloque grueso), no solo la selección.

### Riesgos / incógnitas (honesto)

- ¿El decoder aprende a usar z_global y z_dedo sin que uno domine al otro? run33
  sugiere que mezclar mal no basta; la apuesta es la cabeza global DEDICADA.
- ¿La maquinaria pulgar (switching) funciona como oracle de ranking sin romper los
  demás dedos? 21sk dice que da pulgar perfecto pero rompía dedos: hay que aislar
  su efecto, quizá aplicarla solo al triplet del bloque fino del pulgar.
- Combinar 3 ingredientes de oracle puede crear interacciones no vistas. Probar
  incremental: primero híbrido + D_R uniforme (cierre + control), luego añadir
  maquinaria pulgar.

### Relación con otras ideas

- Idea H (D_R pesado boost MCP) es el ajuste fino del ingrediente 1 del oracle.
- Idea B (5 decoders) es ortogonal: podría combinarse si el decoder único no separa bien.
- Idea F (6D SO(3)) es ortogonal: mejora la representación de rotación en cualquier oracle.

---

## Prioridad sugerida

1. **Idea I** — Latente híbrido grueso+fino + oracle compuesto. PROPUESTA PRINCIPAL: la más fundamentada (evidencia de runs 20/21sk/27B/28-34). Ataca cierre + control + pulgar a la vez.
2. **Idea A** — oracle global + gradiente per-finger. DESCARTADA en la práctica: run34 falló igual que 28-33. La selección global no rescata el latente per-finger.
3. **Idea H** — D_R pesado boost MCP. Ajuste fino del ingrediente de cierre del oracle de Idea I.
4. **Idea B** — 5 decoders. Ortogonal a Idea I; combinable si el decoder único no separa bien.
5. **Idea F** — Representación 6D SO(3). Ortogonal: mejora la rotación en cualquier oracle. Dataset abl14_6d ya generado.
6. **Idea D** — Xin como pérdida. Complementaria a cualquier arquitectura.
7. **Idea E** — Sampler primitivas. Ataca zero-shot y posible bias de MCPs.

(Idea C descartada — era una malinterpretación de F, ver su sección.)
