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

## Resumen

| Problema | Sin UDHM | Con UDHM |
|---|---|---|
| Comparar abd vs flex | Mezclados en geodésica | Slots separados, pesos distintos |
| Robots con distinta morfología | Subespacio variable por par | Vector fijo 22, missing=0 |
| Pool multi-robot | Imposible (dim distinta) | Cat directo `[nB, 22]` |
| Añadir robot nuevo | 5+ puntos de intervención | 1 yaml con mapeo explícito |
| Robot sin pinky | Código especial de exclusión | slots 17-20 = 0 automático |

UDHM no es solo una representación conveniente. Es la estructura que hace que la geometría del problema (DoFs anatómicos, morfología variable, comparación cross-embodiment) se exprese directamente en el tensor, sin infraestructura de traducción.
