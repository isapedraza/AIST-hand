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

## Idea C — Dataset humano 6D (Yan 2025 style)

**Problema que ataca:** representación actual = quaterniones Dong (20 joints × 4) = 80 dims de entrada a E_h. Compleja, requiere Dong kinematics pipeline. Yan 2025 usa representación 6D: 5 ratios de flexión (tip_dist/finger_length) + adducción del pulgar.

**Propuesta:** reemplazar input de E_h con la representación 6D de Yan:

```
b_i = d_root-tip / l_i   (i = thumb, index, middle, ring, pinky)
θ   = adducción del pulgar
h   = [b_thumb, b_index, b_middle, b_ring, b_pinky, θ]  ∈ R^6
```

**Ventajas:**
- Elimina dependencia de Dong pipeline
- Representación morfology-agnostic (normalizada por longitud del dedo)
- E_h se simplifica — no necesita CAM-GNN, un MLP basta
- S_k natural: ||phi(x_h) - phi(x_r)||_2 sobre 6D

**Desventajas:**
- Pierde información rotacional (no sabe si el dedo está abducido vs. rotado)
- Yan lo usó con RH8D (robot simple, 1 DOF/dedo). Shadow tiene 4 DOF/dedo — la 6D puede ser insuficiente.
- Abducción inter-dedo (FFJ4) y oposición del pulgar (THJ5) no están en la 6D de Yan.

**Cuándo tiene sentido:** si se quiere simplificar el sistema drásticamente para tener algo funcional rápido. No para el aporte completo de la tesis.

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

## Prioridad sugerida

1. **Idea A** — oracle global + gradiente per-finger. Bajo costo, hipótesis directa, un run.
2. **Idea B** — 5 decoders. Si Idea A falla y hay tiempo.
3. **Idea D** — Xin como pérdida. Complementaria a cualquier arquitectura. Se puede añadir sobre el mejor run.
4. **Idea E** — Sampler primitivas. Ataca zero-shot y posible bias de MCPs. Más costo que A/B.
5. **Idea F** — Representación 6D SO(3). Bajo costo, descarta discontinuidades como fuente de ruido.
6. **Idea C** — 6D features Yan-style. Solo si se decide simplificar el sistema drásticamente.
