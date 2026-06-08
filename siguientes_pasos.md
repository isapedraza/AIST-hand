# Siguientes pasos

Lista de ideas posteriores a Run 38. Solo las ideas 2 y 3 están desarrolladas
(las discutidas a fondo). El resto queda como título pendiente de desarrollar.

Contexto que originó la lista (observaciones live, madrugada 2026-06-08):
- El penúltimo modelo (Run 37, con alineación) es **igual o ligeramente mejor**
  que Run 20 y superior al último (Run 38 uniformes). Probablemente por la
  alineación, no por los uniformes.
- Probando la mano **de lado** (cámara ve el cierre MCP), los MCP **sí bajan**
  en ambos modelos. Sugiere que el problema del MCP es en parte de **percepción**:
  desde el frente la cámara no informa con estabilidad la flexión del MCP.
- Run 37 "falla" solo en cerrar MCP por completo, no como modelo total.

---

## 1. Combinar Run 38 (reweighting oracle) + Run 37 (anchor loss)

**Idea.** Una sola corrida que combine los dos mecanismos que atacan el MCP por
vías distintas:
- **Run 38** — reweighting del oracle S_k (interno, no-supervisado): D_R con
  MCP=5/resto=1 (~71% a MCP), D_joints con TIP=3/resto=1 (50% a TIP),
  W_R=2.5 / W_JOINTS=1.0.
- **Run 37** — anchor loss supervisado (externo): alinea anclas humanas
  (`hagrid_dong.csv`) con qpos robot sintético en el espacio compartido.

**Por qué combinar.** Son mecanismos ortogonales (uno cambia la métrica del
oracle, el otro agrega supervisión externa). La esperanza es apilar: la
**suavidad** de 37 + el **empuje de cierre** de 38.

**Observaciones live (madrugada 2026-06-08):**
- Ambos MCP "bajan" en la prueba de mano lateral; no muy diferentes entre sí.
- **Run 37 se siente más suave.**
- **Run 38 más inestable** — los dedos se doblan.
- Ninguno es la panacea.

**Pendiente.** Repetir la misma prueba lateral con **Run 20 original** como
baseline (aún no hecha).

---

## 2. Backends de percepción

**Diagnóstico.** El MCP no cierra principalmente porque la percepción frontal
(MediaPipe) no ve con estabilidad la flexión del MCP — confirmado con la prueba
de mano de lado (el MCP sí cierra cuando la cámara lo puede ver). No es fallo
estructural del modelo. Objetivo: backend que provea el movimiento del MCP de
forma más realista.

**Restricción dura.** Dong es la representación que permite comparar humano y
robot (es el espacio común). Los robots no usan MANO θ. Por lo tanto solo se
puede:
- mejorar la calidad de los XYZ que entran a Dong FK, o
- obtener mejores rotaciones ya en convención Dong.
No se puede meter MANO θ directo al pipeline sin re-entrenar todo / romper el
puente humano-robot.

**Punto clave sobre HaMeR XYZ vs MediaPipe XYZ.** No son equivalentes aunque
ambos den 21 puntos. MediaPipe regresa landmarks directos desde imagen (MCP
frontal ambiguo). HaMeR estima θ con un ViT y luego `XYZ = FK(θ)`: los puntos
salen del modelo paramétrico, internamente coherentes, mejor profundidad. Por eso
HaMeR XYZ → Dong FK **sí mejora** sobre MediaPipe XYZ → Dong FK.

**Opciones evaluadas:**

| Backend | Framework | IK / rotación | MCP | Velocidad | Notas |
|---|---|---|---|---|---|
| HaMeR (Pavlakos 2023) | PyTorch/ViT-H | θ regresado desde imagen | bueno (θ desde imagen completa) | lento (~0.1-0.3 s/frame en T4, solo Colab) | backend ya existe en `human/perception/hamer_backend.py` (Stage 1, async Colab) |
| minimal-hand CalciferZh | TF 1.14 | IKNet aprendido con prior MoCap | bueno (prior MoCap), pero opera sobre XYZ igualmente ambiguos | ~100 fps (1080Ti) | setup TF 1.14 aislado (sin conda); output quats [21,4] |
| minimal-hand MengHao666 | PyTorch | IK analítica (sin IKNet) | marginal vs MediaPipe (misma ambigüedad) | rápido | descartado para MCP: no ataca la raíz |

Ambos repos clonados en `~/minimal-hand` y `~/Minimal-Hand-pytorch`.

**Dirección acordada.** Probar **ambos** (HaMeR y CalciferZh). Empezar por HaMeR
XYZ → mismo Dong FK (backend ya integrado, días no semanas). Si la mejora de MCP
no basta, ir a CalciferZh (venv Python 3.7 + TF 1.14). Falta también repetir la
prueba de mano de lado con **Run 20 puro**.

**Pendiente Stage 2.** `live_retarget.py` usa `MediaPipeSource`. HaMeR solo está
integrado en Stage 1, no en Stage 2 (retargeting). Habría que crear el source
equivalente.

---

## 3. CAM temporal

**Idea.** El encoder humano `E_h` ya usa CAM (Constraint Adjacency Matrix,
Leng et al. IEEE VR 2021), single-frame. Extenderlo a **multi-frame** para dar
consistencia temporal al latente y absorber la inestabilidad de percepción
(detección intermitente del cierre MCP → temblores y movimientos erráticos).

**Por qué dentro del modelo (no filtro externo).** No es OneEuro. OneEuro promedia
señal de forma ciega a estructura — mataría el poco movimiento MCP real junto con
el ruido. La CAM temporal **re-estima** con prior estructural espacio-temporal:
distingue movimiento coherente (intención) de incoherente (ruido). Razona en
espacio de features/latente, no en espacio crudo.

**Diseño acordado (Path B, integrado):**
- E_h temporal: input `[B, 20, 4, T]` → node features concat `4*T` → CAM `[21,21]`
  espacial (Leng Eq. 4). Output z_h temporalmente consistente por construcción.
- **From scratch, sin warm-start.** Warm-start desde Run 20/37 arrastraría el
  sesgo de ceguera-MCP del latente viejo; el punto es un latente mejor.
- Sampler: ya trae `t, t+1`; extender a ventana de T frames (la infra `next_idx`
  ya existe en `human_loader.py`).

**Hallazgos sobre los datos (HOGraspNet):**
- El dataset anotado (1.49M frames) está **downsampleado de 30 FPS a 10 FPS**
  (Cho et al. 2024, sec. preprocessing). `frame_id` step=1 = **100 ms**.
- Nyquist = 5 Hz. El tremor fisiológico (4-12 Hz) **aliasa** a 10 FPS — inyectar
  tremor sintético de alta frecuencia no serviría. Pero el problema real no es
  tremor 12 Hz: es **inestabilidad de detección** (baja frecuencia), que la CAM
  temporal a 10 FPS sí cubre vía coherencia de manifold.
- 20.591 trials, mediana 56 frames/trial. Secuencias densas (step=1 domina).
- **T tentativo = 5** (~500 ms): cubre ráfagas de inestabilidad sin promediar
  el movimiento de agarre real (~0.5-1 s). No copiar T=8 de Leng (era otro fps).

**Requisito de inferencia.** Live webcam ~30 FPS; training a 10 FPS. Subsamplear
el stream vivo a 10 FPS (cada 3er frame) para igualar el spacing temporal, o la
CAM ve escala temporal equivocada.

**Supervisión.** Denoising **implícito**: entrenar sobre secuencias limpias →
E_h aprende coherencia temporal del manifold → en inferencia proyecta ruido al
manifold. No hay datos de tremor real para supervisión explícita.

**Orden / dependencia (importante).** La CAM temporal **estabiliza** la señal pero
no hace que la cámara vea mejor el MCP. Si la señal MCP frontal es débil, da un
MCP estable pero igual de cerrado-a-medias. Es **complemento** del cambio de
percepción, no sustituto. Orden correcto: **percepción primero**, CAM temporal
después (estabilizar una señal que ya vale la pena).

**Nota / bug latente.** ~460k pares con step=0 (`frame_id` duplicado dentro de
trial+cam). El `next_idx` actual podría emparejar un frame con su duplicado
(velocidad cero falsa). Revisar al extender a ventana T.

---

## 4. Xin como loss del latente (no como similitud)

**Marco.** Xin et al. (2025, "Analyzing Key Objectives in Human-to-Robot
Retargeting") planteó sus términos como **losses directas** en una optimización
online (NLopt SLSQP, 20 Hz): cada término mide la discrepancia humano-robot y se
minimiza para resolver el qpos robot — `L_fingertip_pos`, `L_fingertip_rot`,
`L_pinch`, `L_thumb_pos`, `L_joint`, `L_vel`.

**Lo que se hizo hasta ahora (invención propia, no de Xin).** Esos términos se
repurposaron como **oráculo de similitud** `S_k = w_r·D_R + w_joints·D_joints +
w_ahg·D_ahg` para **minar tripletas contrastive** en el latente aprendido. Xin no
lo planteó como medida contrastive para diferenciar agarres.

**Resultados de ese enfoque (observación propia).** No han dado los mejores
resultados, salvo:
- **Run 21sk** — single latent, buen pulgar, resto basura.
- **Run 27** — single latent, buen cierre de MCP, resto basura.

**Idea 4.** Volver al planteo original: usar los términos Xin como **losses
directas del latente / pose decodificada** (aplicar la discrepancia entre input
humano y robot decodificado y hacer backprop), en vez de como oráculo de
tripletas.

**Por qué puede funcionar.** Da **gradiente directo** al path de inferencia
`E_h → E_X → D_X → D_r`. Conecta con un problema ya registrado en DECISIONS: D_r
nunca recibe gradiente del path humano durante el training; el path de inferencia
nunca se optimiza directamente y D_r extrapola al prior semi-flexionado al recibir
z fuera de distribución. El contrastive solo impone orden relativo en el latente y
su gradiente se desvanece cuando la mayoría de tripletas ya cumplen el margen.
Una loss directa ataca eso de frente.

### Contraste con implementaciones pasadas de Xin (3 familias)

- **Familia A — Xin como oráculo de similitud (S_k → contrastive).** Runs 25, 26,
  27B, 28, 29, 30, 34, 21sk, 27. Los términos Xin arman `S_k = w_r·D_R +
  w_joints·D_joints + w_ahg·D_ahg` que **rankea tripletas** para L_cont. Indirecto.
  Resultados pobres salvo 21sk (pulgar) y 27 (MCP), single latents.
- **Familia B — Xin como losses directas sobre FK(q_r_hat). YA EXISTE.** Commit
  `81f1219`, branch `feat/run20-regression`. `grasp-model/src/cross_emb/xin_losses.py`:
  5 términos cartesianos (tip_pos, pinch, tip_rot, pip_pos, L_joint) sobre la pose
  robot decodificada del path humano. Gradiente directo a E_h/D_X/D_r. Aditivo a
  L_cont/L_rec/L_ltc/L_temp. `lam_xin_global=0.6` (~15-17% de L_total). Flag
  `--enable_xin_losses` (default off). **Idea 4 = Familia B.**
- **Pivote a normalize_z (Run 24).** La branch B no llegó a evaluar las losses
  directas: pivoteó a Run 24. Entry 87 diagnosticó que la causa raíz era
  **desalineación de escala del latente** (norma z_H=4.28 vs z_R=1.62; z_H 3.18x
  más lejos de z_R de lo natural; L2-norm post-hoc bajó d_pos 4.68→1.43). Fueron a
  `normalize_z` y dejaron Familia B **sin correr** (o sin recordar haberla corrido).

### Lo esencial de idea 4

La idea (Familia B) es correcta, pero el código existente vive en una branch
**que se desvió y nunca convergió a un modelo bueno**. Idea 4 = **portar la loss
directa sobre una base que YA funciona** (mejor actual: Run 37, o el combinado
37+38 de idea 1) y correrla ahí. No sobre la branch vieja de regresión.

Posiblemente **combinar con `normalize_z`**: la loss directa da gradiente a D_r;
normalize_z alinea la escala del latente donde aterriza ese gradiente. Entry 87
sugiere que sin alinear la escala, la señal no aterriza bien.

**Rol: refinamiento, no reemplazo.** Xin va **siempre aditivo**, nunca en lugar
del contrastive. El contrastive construye la geometría del latente; Xin **refina**
la pose de salida y da gradiente cartesiano directo a D_r. Peso **subordinado**
(`lam_xin_global=0.6` en `81f1219` = ~15-17% de L_total) — tunear. Con peso bajo
refina sin sobreescribir la estructura que arma el contrastive.

### Justificación conceptual (Yan + Xin)

- **Yan = cómo construir el latente.** Contrastive + L_rec (solo robot) + L_ltc
  (consistencia latente, path humano) + L_temporal (velocidad EE). El S_k de Yan
  solo mina tripletas; no es loss directa. El **path humano se supervisa solo
  indirectamente** — no hay objetivo en espacio de pose sobre la salida humana.
- **Yan se lo permite** porque (1) entrena con 6 robots → latente anclado desde
  muchas vistas, y (2) cuerpo humanoide = escala de metros, donde EE-pos +
  contrastive bastan.
- **Tus dos condiciones rompen ambos supuestos:** un solo robot (Shadow) → path
  humano sub-restringido, D_r extrapola al prior semi-flexionado, MCP no cierra;
  y escala milimétrica → ambigüedad geométrica de S_k mucho peor.
- **Xin = cómo medir retargeting en espacio de pose.** Objetivos cartesianos
  morfología-aware (pos relativa a muñeca normalizada por hand_length, pinch con
  rescaling). Esquiva el "no hay datos pareados" de Yan: no compara contra robot
  ground-truth, compara la config cartesiana del **propio humano** vs la del robot
  decodificado (pseudo-target del input humano).
- **Conclusión:** Xin parcha justo el hueco de Yan (supervisión directa del path
  humano en espacio de pose). En el caso 1-robot/mm la falta es más grave → Xin
  vale más aquí que en el paper original de Yan.

---

## 5. Pesos de loss automáticos (Kendall uncertainty weighting)

**Motivación (el driver real).** Nunca se sabe qué peso poner a cada loss, y las
losses tienen **unidades distintas** (Xin cartesiano en mm, L_ltc/contrastive en
latente unitless [-1,1], márgenes de triplet). Es **textualmente** el problema que
ataca Kendall et al. (2017): el peso óptimo depende de la escala (m/cm/mm) y del
ruido de cada tarea. No es over-engineering — es la herramienta exacta para ese
dolor, que se agrava al apilar ideas 1+4.

**Kendall vs GradNorm — Kendall gana aquí.**

| | GradNorm (Chen 2018) | Kendall (uncertainty, 2017) |
|---|---|---|
| Balancea por | ritmo de entrenamiento (normas grad) | ruido/escala por tarea |
| Necesita capa compartida W | sí (ambigua aquí) | **no** |
| Necesita Lᵢ(0) baseline | sí (Xin init inestable) | **no** |
| Costo | ~5% + grad norms | trivial (un `log σ²` por loss) |

Razones de la elección:
1. **Unidades heterogéneas = motivación literal de Kendall.** GradNorm balancea
   ritmos, no escalas → menos alineado con el problema real.
2. **Esquiva los caveats de GradNorm.** No necesita "última capa compartida W"
   (los losses se enganchan en puntos distintos del grafo: contrastive en latente,
   rec/temporal/Xin en pose decodificada, ltc circular). No necesita Lᵢ(0).
3. **Más simple y barato**, robusto a init (converge ~100 iters).

**Caveat compartido — el contrastive es el loss raro para ambos métodos.**
- **No es likelihood:** el triplet `max(||z_a−z_p|| − ||z_a−z_n|| + α, 0)` no tiene
  target y; es restricción relacional (ranking). Kendall sale de `p(y|f(x))` con σ
  = ruido de observación sobre y. Sin y, "σ² del contrastive" no tiene sentido
  limpio — aplicarle `1/2σ²` es heurístico.
- **Es hinge (clamp en 0):** al cumplir margen, loss → ~0. Kendall lo leería como
  "alta confianza → subir peso", pero loss ~0 es **saturación**, no confianza.
  Dinámica rara (misma patología que GradNorm por otra vía: GradNorm leería el
  gradiente saturado como "entrena rápido → bajar peso").

**Recomendación: híbrido.** Auto-pesar con Kendall la **familia de regresión**
(L_rec, L_ltc, L_temporal, los 5 sub-términos Xin) y mantener el **contrastive con
peso fijo** (hand-set). Aprovecha el balanceo de escala donde es principiado, sin
meter el triplet (que ninguno maneja bien) al esquema automático.

**No tocar S_k.** Igual que GradNorm, Kendall opera a nivel de términos top-level
de L_total, NO dentro del oracle S_k (los pesos W_R/W_JOINTS/W_AHG + boost MCP de
Run 38 son tuneados a mano a propósito).

**Orden.** Meta-herramienta → va **al final**, después de que ideas 1+4 metan
suficientes losses que valga la pena balancear.

---

## 6. Cambiar el contrastive: InfoNCE + S_k como peso adaptativo (SiMHand)

**Referencia.** Lin et al. (2025), "SiMHand: Mining Similar Hands for Large-Scale
3D Hand Pose Pre-training". Es un método de pre-training self-supervised, pero lo
transferible es su **formulación de loss**, no el pipeline de 2M imágenes.

**Lo relevante de SiMHand:**
- **Non-self-positives:** positivos de imágenes distintas con pose similar (no
  augmentation de la misma imagen). Similitud vía embedding PCA de 21 keypoints +
  nearest neighbor.
- **Adaptive weighting (la joya):** pesa cada par en el loss por **cuán similar**
  es — `w` = escalado lineal de la distancia entre `d_min` y `d_max` del mini-batch.
  Parameter-free. `w_pos` y `w_neg` separados. Loss = **NT-Xent (InfoNCE) ponderado**
  por similitud, no triplet.

**El problema que resuelve.** Setup actual = **triplet + S_k como oráculo binario**
(positivo/negativo por ranking). Problema recurrente (DECISIONS Entry 87/4111): el
triplet es hinge → al cumplir margen el **gradiente → 0**, satura, el contrastive
deja de empujar.

**Idea 6 = cambiar el contrastive de "triplet + S_k oráculo binario" a "InfoNCE +
S_k como peso adaptativo continuo".** Dos upgrades:
- **Triplet → InfoNCE/NT-Xent:** usa todos los negativos del batch vía softmax,
  gradiente más denso, menos saturación que un margen único.
- **S_k binario → S_k como peso continuo:** ya se computa S_k. En vez de usarlo
  para elegir tripletas, usarlo para **ponderar** el InfoNCE por cuán similar es el
  par. Más informativo, sin saturación dura.

**Qué NO copiar de SiMHand:**
- **PCA de keypoints 2D** (su métrica de similitud) — downgrade. SiMHand la usa
  porque solo tiene imágenes; S_k (quats Dong + chain 3D, cross-modal) es más rica.
- **Augmentation de imagen + inverse-transform (equivariancia PeCLR)** —
  inaplicable. Es para encoders de imagen; aquí el input ya es pose Dong, no píxeles.
- Se adopta solo la **formulación de loss** (InfoNCE ponderado), no la maquinaria
  hand-specific de estimación.

**Posibilidad a echarle un ojo — mining de positivos.** SiMHand mina positivos de
un pool para **garantizar que existe un positivo realmente similar**. Aquí el batch
es muestreado (humano random + robot eigengrasp); si el par más similar intra-batch
**igual es lejano**, el soft weighting no tiene buena señal que pesar (esquiva el
positivo duro, pero no fabrica similitud ausente del batch). Si los positivos
intra-batch resultan débiles, valdría adaptar el **concepto** de mining —usando
**S_k**, no keypoint-PCA— sobre un pool más grande (HOGraspNet/robot completo). No
es prioridad: **medir primero** la calidad de positivos intra-batch.

**Caveat.** El weighting de SiMHand normaliza dentro del mini-batch (`d_min`,
`d_max`). En cross-embodiment el batch mezcla humano+robot y S_k es cross-modal →
cuidado con escalado consistente (otra vez el problema de escala de Entry 87).

**Conexión con idea 4.** Complementarias, mismo problema raíz (supervisión débil
del path humano): idea 6 mejora el **contrastive mismo** (continuo, no-saturante);
idea 4 agrega **gradiente directo de pose** a D_r.

### Comparación de los 3 contrastives contra los problemas reales

Implementación actual (`grasp-model/src/cross_emb/train_cross_emb.py:459-577`):
por subespacio, pool humano+robot; por anchor muestrea **2 candidatos random**,
S_k (en `no_grad`) decide cuál es positivo (`a_closer = S_a <= S_b`); loss =
`relu(||z_a−z_pos|| − ||z_a−z_neg|| + margin).mean()`, margin=0.05, **Euclidean
sobre z crudo** (no normalizado).

Los tres casos:
- **1. Tú actual** — 2-candidatos random, S_k selector binario, Euclidean sobre z
  crudo, hinge.
- **2. Yan paper** — triplet con positivo/negativo **semántico** (S_k mina),
  Euclidean, hinge, α=0.05, **pero entrenado con 6 robots**.
- **3. SiMHand** — NT-Xent/InfoNCE ponderado, todos los negativos, S_k peso
  continuo, cosine, sin saturación.

Problemas relevantes al contrastive: **P1** MCP no cierra (saturación mata
gradiente antes de extremos); **P2** escala latente (Entry 87: z_H norm 4.28 vs
z_R 1.62); **P3** un solo robot (anclaje débil); **P4** escala mm (S_k ambiguo).

| | 1. Tú actual | 2. Yan paper | 3. SiMHand |
|---|---|---|---|
| P1 saturación | ❌ hinge + 2 randoms → muere temprano | ❌ hinge; positivo real jala mejor, igual satura | ✅ softmax nunca cae a 0 |
| P2 escala | ❌ Euclidean sobre z crudo | 🟡 Euclidean, pero 6 robots anclan y compensan | ✅ cosine = normalize gratis |
| P3 1-robot | ❌ positivos random + 1 robot | ❌ el triplet de Yan funciona POR los 6 robots; con 1 se degrada | 🟡 negativos densos compensan parcial; límite persiste |
| P4 mm ambiguo | ❌ binario → empates voltean etiqueta | ❌ binario; Yan opera en metros, a mm se rompe | ✅ peso suave degrada con gracia |

**Lectura clave — estás peor que Yan por DOS razones:**
1. **Tu implementación es un triplet más débil que el de Yan.** Yan usa positivo
   **semántico** (de verdad similar); tú usas "el menos disímil de 2 draws
   aleatorios" → tu positivo a menudo no es similar. Penalización extra.
2. **Régimen equivocado.** El triplet de Yan funciona **porque** tiene 6 robots
   (anclan el latente → P2/P3 compensados) y opera en **metros** (S_k
   discriminativo → P4 no aplica). Trasplantado a **1 robot + mm**, sus dos muletas
   desaparecen. No es que copiaste mal a Yan: el contrastive de Yan depende de
   condiciones que no tienes, y tu variante lo debilita más.

**Por qué SiMHand encaja con TU régimen.** No depende de multi-robot ni de escala
métrica: cosine arregla P2 sin los 6 robots; softmax arregla P1; peso continuo
arregla P4 (la ambigüedad mm se vuelve diferencia de peso, no ruido de etiqueta —
tu talón). P3 solo compensa parcial (límite estructural). SiMHand fue diseñado para
datos in-the-wild, una modalidad, sin anclas múltiples → más cercano a tu situación
pobre-en-anclas que el setup multi-robot de Yan.

**Veredicto:** tu actual < Yan < SiMHand **para tu régimen** (1 robot, mm, S_k
ambiguo). SiMHand gana porque no depende de las muletas de Yan que no tienes.

### Decisión de diseño para idea 6

Para InfoNCE hay que definir el **positivo**. Opciones:
- **Soft/SupCon-style (recomendado):** ponderar todos los pares por afinidad
  continua f(S_k), sin positivo duro. Encaja natural con S_k cross-modal + el
  adaptive weighting de SiMHand. Domina en P4 (ambigüedad mm) porque no toma
  decisiones duras donde S_k es débil.
- **Top-1:** positivo = el más similar del batch (min S_k), resto negativos
  ponderados. Fiel a SiMHand, pero reintroduce un pick duro frágil bajo ambigüedad
  mm (SiMHand tenía similitud limpia de keypoints 2D en una sola modalidad; tu S_k
  cross-modal a mm es más ruidoso).

---

## 7. 6D rotation en lugar de cuaternión

**Qué busca.** Mejor representación de rotación para que el modelo **aprenda
mejor**. No ataca un síntoma específico (como el MCP) — es mejora de fondo de la
calidad de aprendizaje.

**Por qué.** El cuaternión tiene discontinuidad (doble cobertura antipodal: q y −q
= misma rotación) → saltos en el espacio que estorban a la red, gradientes feos.
Zhou et al. (2019) probó que toda representación ≤4D de SO(3) es discontinua; la
**6D es continua** → la red aprende rotaciones más limpio. (Ref también: Geist et
al., "Learning with 3D rotations, a hitchhiker's guide to SO(3)", en lit.)

**Dónde aplica.** (1) input de E_h (quats → 6D como node features); (2) opcional, el
término D_R de S_k (hoy `1 − dot²` de quats → geodésica sobre matrices rot).

**Esencialmente lo mismo, solo recablear.** Mismo dato, forma más amigable para la
red: input dim 4→6 en E_h, conversión quat→6D en el loader, D_R/decoder ajustados a
6D. Plomería de representación, no concepto nuevo.

**Impacta inferencia también.** El modelo entrenado con 6D espera 6D en su input →
el path de deploy debe entregar 6D, no quats. Hoy:
`MediaPipeSource → Dong → quats [1,20,4] → E_h`. Hay que meter la misma conversión
quat→6D en la fuente de inferencia (`mediapipe_source.py` / `retarget.py`) antes de
E_h, o el modelo recibe formato equivocado. Train e inferencia comparten
representación → ambos lados se recablean en espejo (misma lección que el subsampleo
10fps de idea 3).

---

## 8. Sampler de robot por primitivas (agregar robots sin dataset)

**El problema real.** Generar el dataset de poses por robot es **difícil**. Hoy el
robot se samplea de `valid_robot_poses_eigengrasp_dong.npz` — eigengrasps
precomputados **solo para Shadow**. Por eso el proyecto está **atascado en 1
robot**: hacer el NPZ de otro robot cuesta, y si alguien quiere meter su propio
robot, también → nadie lo hace. La barrera para multi-robot **es** la generación de
dataset por robot.

**La idea.** Bajar esa barrera a cero: **URDF → primitivas → samplear poses
válidas, sin NPZ ni recolección de datos.** Cualquiera mete su robot con solo el
URDF.

**Por qué no sirve el sampleo crudo.** Yan samplea sin dataset (joint sampling),
pero en una mano el sampleo random del qpos crudo ignora la **coordinación** entre
dedos → **poses chuecas** (anatómicamente conectadas pero semánticamente vacías —
el problema high-dim ya registrado). Hay que samplear el espacio **coordinado**, no
el joint crudo.

**Dos mecanismos (papers, ambos en lit):**
- **Liu et al. (2021), hand primitives / synergies (= eigengrasps).** PCA sobre
  datos de agarre → subespacio low-dim → sampleo coordinado, poses funcionales.
  **Pero la base necesita datos.** Es lo que ya se hace para Shadow (el NPZ). No
  resuelve "robot nuevo sin dataset", solo explica por qué el sampleo coordinado
  evita lo chueco.
- **Wu et al. (2026), DexGrasp-Zero — el que habilita "sin dataset".** Espacio de
  motion-primitives hand-agnostic: por nodo anatómico, 3 primitivas ortogonales
  (FLEX, ABD, ROT) ancladas en biomecánica. Un mapeo fijo `M_h` por robot convierte
  primitiva-universal → comandos de joint. **`M_h` se construye SOLO del URDF**
  (excitaciones unitarias en sim → regla de indexado). Robot nuevo = URDF → M_h →
  samplear. Cero dataset.

**El plan.** Reemplazar el NPZ eigengrasp por un sampler de primitivas: espacio
hand-agnostic (FLEX/ABD/ROT por nodo) + `M_h` desde URDF por robot + sampleo
**coordinado** (ej. flexión coordinada = cierre) → mapear vía `M_h` → qpos válido.
Combina: base primitiva URDF-derived (DexGrasp-Zero) + sampleo coordinado funcional
(synergy-style de Liu) = poses meaningful para cualquier robot, sin datos.

**Beneficio compuesto (clave).** La razón de **P3** (1 robot, anclaje débil del
latente, la muleta multi-robot de Yan que el proyecto NO tiene — ver ideas 3 y 6)
**es exactamente esta barrera de dataset**. Idea 8 la rompe → más robots → mejor
anclaje del latente → P2/P3 mejoran. No es solo conveniencia: **desbloquea la
condición multi-robot que hace funcionar el contrastive de Yan.**

**Caveat honesto.**
- `M_h` necesita probing de excitación unitaria en sim por robot (barato,
  automatizable del URDF). Ese es el costo "sin dataset": URDF + sondeo en sim, NO
  recolección de agarres.
- Asume que los DOF del robot mapean limpio a semántica FLEX/ABD/ROT. Morfologías
  exóticas pueden no encajar.

**Dataset: NO hace falta regenerar.** 6D se deriva de los quats existentes (quat →
matriz rot → primeras 2 columnas) en el loader.

**Estado.** Implementación previa en commit `4679fbc` (branch
`feat/abl14-r6-rot-repr`, 25 may): `rotations.py` (math reusable) +
`build_abl14_csv.py` + cambios en loaders/encoder, **pero sobre layout viejo**
(`models/latent-retargeting/...`, pre-Run-20). Re-portar a la base actual
(`grasp-model/src/cross_emb/`); no correr sobre la branch vieja (misma lección que
idea 4). No existe CSV R6/abl14 en local (solo `hograspnet_abl11.csv` en quats); si
se generó alguna vez, estaría en Drive/Colab (los CSV son ~3GB, gitignored).
