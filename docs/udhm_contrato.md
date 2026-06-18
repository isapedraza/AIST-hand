# UDHM como contrato canónico cross-embodiment

## Qué es UDHM

UDHM (Unified Dexterous Hand Model, Fang et al. 2026) es un vector de 22 valores con orden y semántica fijos:

```
slot 0:  thumb_cmc_flex
slot 1:  thumb_cmc_spread
slot 2:  thumb_mcp_flex
...
slot 12: index_mcp_flex
slot 13: index_pip_flex
...
```

Cada slot tiene un nombre anatómico. El valor es el ángulo en ese DoF, normalizado por π → rango [-1, 1].

Humano y robot depositan en el mismo vector. Lo que el robot no tiene → 0 (sparse / semantic insertion).

---

## Por qué necesitamos signo

El nombre del slot dice QUÉ movimiento es. El signo dice en QUÉ DIRECCIÓN es positivo.

**Problema:** dos robots pueden tener el mismo movimiento anatómico pero con convención de dirección opuesta en su URDF.

Ejemplo concreto:

- Inspire `index_proximal_joint`: límites [0, 1.47]. Cerrar = ir hacia +1.47. Sin corrección → UDHM = +0.47 al cerrar.
- Barrett `finger_1_med_joint`: límites [-2.44, 0]. Cerrar = ir hacia -2.44. Sin corrección → UDHM = -0.78 al cerrar.

Si comparamos humano cerrando (+0.47) vs Barrett cerrando (-0.78): L1 = 1.25. La métrica dice "son opuestos" cuando ambos están cerrando.

**Fix:** multiplicar por signo al llenar el slot:

```python
udhm[slot] = qpos[col] * sign / π
```

- Inspire: sign = +1 (límite positivo mayor) → cerrar = +0.47 ✓
- Barrett: sign = -1 (límite negativo mayor) → cerrar = -2.44 × -1 / π = +0.78 ✓

Ahora ambos son positivos al cerrar. La métrica compara pose, no convención de URDF.

**Convención UDHM:** positivo = más cerrado (flex), positivo = alejado del medio (abd).

El signo se deriva automáticamente de los límites del URDF: `sign = +1 si |hi| >= |lo|, else -1`.

---

## Pipeline completo

```
HUMANO                              ROBOT
-------                             -----
keypoints 3D (Dong/HaMeR)          qpos [B, J]
    ↓                                   ↓
human_to_udhm()                    yaml declara:
  - palm plane                       index_mcp_flex → joint col 4
  - Rodrigues FK inverso             barrett thumb → col 6, sign=-1
  - signed angles                        ↓
    ↓                              robot_to_udhm(qpos, tabla)
udhm22_h [B, 22]                       ↓
                                   udhm22_r [B, 22]
                ↓                       ↓
         métrica contrastiva: L1(udhm22_h, udhm22_r) por slot
         solo slots compartidos (sparse = 0 ignorado)
```

---

## Por qué yaml y no solo UDHM

UDHM define el contrato de destino. Yaml declara cómo cada robot llena ese contrato:

```yaml
udhm:
  index_mcp_flex: index_proximal_joint   # qué joint
  index_pip_flex: index_intermediate_joint
  thumb_mcp_abd:  thumb_proximal_yaw_joint
```

El signo se deriva del URDF automáticamente. Lo que no está declarado → slot = 0.

Sin yaml no hay forma de saber que `qpos[col=4]` de Inspire es `index_mcp_flex` — los nombres de joints varían por fabricante y no tienen semántica estandarizada.

---

## Qué resuelve build_primitives (hoy)

Construye la tabla yaml → slots automáticamente via clasificación geométrica de ejes (FK + frame de palma). Para robots con geometría estándar (Shadow, LEAP, Allegro) funciona sin intervención manual. Para robots con geometría no estándar (Barrett, Inspire thumb) requiere override manual en `_MANUAL`.

Podría reemplazarse por declaración explícita en yaml (más simple, menos magia).
