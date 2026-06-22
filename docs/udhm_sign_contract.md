# Contrato de signos UDHM (resuelto)

Fecha: 2026-06-22. Cierra el dilema de `docs/udhm_sign_dilemma.md`.

## Arquitectura elegida: híbrido

Un encoder compartido. El UDHM[22] del robot se llena por **dos caminos**:

- **20 slots → Dong** (`FK → dong_run_stage2 → human_to_udhm`). El `atan2` da el
  ángulo **ya firmado** por la geometría. **Sin signos manuales.**
- **2 slots → qpos×signo** (los que Dong no modela): `thumb_mcp_abd`,
  `pinky_twist`. `udhm = signo × qpos / π`.

## Marco de referencia (Dong, Eq. 5-7)

- **X** = distal (a lo largo de los dedos).
- **Y** = `ring_mcp − index_mcp` = **ulnar** (hacia el meñique).
- **Z** = normal de palma = **volar** (hacia la palma).
- Pose cero: mano recta, aducta (Fang §3.1).

## Convención por slot (qué significa `+1`)

### Auto-firmados por Dong (20)

| Slot(s) | `+1` significa | Fuente |
|---|---|---|
| Todos los `*_flex` (15) | hacia palma (volar, +Z) | Dong `atan2(xi_z,…)`; Fang/DexGrasp |
| `index/ring/pinky_mcp_abd` | lejos del medio | Dong `atan2(xi_y,xi_x)` + flip radiales |
| `middle_mcp_abd` | ulnar (hacia meñique) | Dong sin flip (medio no es radial); **no ambiguo** porque sale de Dong, no de signo manual |
| `thumb_cmc_spread` | lejos del medio | Dong (pulgar radial, flipeado) |

Nota flip (`human_to_udhm.py` líneas 75-76): índice y pulgar se niegan para que
"lejos del medio = +" en todos. Dong crudo (sin flip) = ulnar+ uniforme = ISB.
El flip es elección DexGrasp (abrir la mano = +). Ambas válidas; usamos con flip.

### Manuales por qpos×signo (2) — ambos solo en Shadow

| Slot | Joint Shadow | `+1` significa | Estado |
|---|---|---|---|
| `thumb_mcp_abd` | THJ3 | lejos del medio (≈ "forward" en render) | **Shadow-libre**, sign `+1` |
| `pinky_twist` | LFJ5 | "forward" (lo que produce LFJ5+) | **Shadow-libre**, sign `+1` |

`thumb_mcp_abd` y `pinky_twist`: **solo Shadow los mapea** (allegro/leap no).
→ Shadow es la **referencia** para estos 2. Su signo es correcto **por
definición** (no hay otro embodiment con qué chocar). Convención decretada =
"lo que produce el `+qpos` de Shadow". Verificado por render (poses
`thumb_abd`, `pinky_twist` en `verify_canonical.py`).

**Si en el futuro entra otro robot con twist / thumb mcp abd:** alinearlo a
esta referencia (o, equivalente, a ISB: rotación axial = pronación+, §4.4.1
de Wu et al. 2005).

## Slots perdidos si se elige DROP en vez de híbrido

Alternativa más simple (0 signos): full-Dong, dejar `thumb_mcp_abd` y
`pinky_twist` en 0 (`mask=False`). Pierdes esos 2 DoF de Shadow (menores). El
lateral del pulgar igual lo capta `thumb_cmc_spread`.

## Manos no-antropomórficas

Cualquier frame (Dong/ISB) rompe sin landmarks (Barrett). Fallback:
semantic-insertion + zero-pad (Fang). Fuera del alcance de este contrato.
