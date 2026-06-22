# UDHM — el dilema de los signos (pendiente de decidir)

Fecha: 2026-06-22. Estado: **sin resolver**, a decidir por Yahel.

## El dilema en una frase

Con **un solo encoder compartido** (objetivo del backbone UDHM), o aceptas
**signos** (para meter twist/abd del robot por qpos) o **sueltas** esos DoF
(full-Dong). No hay las tres a la vez.

## Por qué aparece el problema de signos

El encoder **único** es lo que crea la necesidad de signos:
- **Encoders separados** (camino viejo): cada embodiment aprende su propio
  espacio. No hay que acordar convención. El twist del robot "solo funciona"
  porque su qpos entra a su propio encoder.
- **Encoder compartido** (nuevo): todos los embodiments deben significar lo
  mismo en cada slot → requiere **convención de signo consistente**. El infierno
  de signos = el precio de unificar a un encoder.

## Por qué Dong no da twist ni thumb_mcp_abd

- `pinky_twist`: rotación sobre el eje del propio hueso. Los **keypoints
  (posiciones) no codifican rotación axial** → imposible sacarlo de Dong.
  Humano siempre = 0. Solo se obtiene del **qpos** del robot.
- `thumb_mcp_abd`: Dong modela el pulgar con 4 DoF (cmc flex+spread, mcp flex,
  ip flex), **no incluye** la abd del MCP. Hueco de modelado. El lateral del
  pulgar lo capta `thumb_cmc_spread` (Dong sí lo da); el MCP abd es menor y su
  efecto neto se dobla dentro de cmc_spread al medir de keypoints.

## Las tres opciones

| Quieres | Resultado |
|---|---|
| 1 encoder **+ twist** | necesitas **signos** (mínimo para twist/thumb_mcp_abd) |
| 1 encoder **+ sin signos** | **full-Dong**, pierdes twist + thumb_mcp_abd (siempre 0) |
| twist + sin signos | **encoders separados** (camino viejo) |

## El híbrido (recomendado a evaluar)

- **20 slots → Dong**: `FK → dong_run_stage2(fk, config) → human_to_udhm` → UDHM.
  **Cero signos** (la geometría los computa, igual humano y robot).
- **2 slots → qpos×sign**: `pinky_twist`, `thumb_mcp_abd`, convención **ISB**
  (rotación axial = pronación/supinación; Wu et al. 2005, J Biomech 38:981-992).
- **Un encoder**, máscara (`UDHMContainer.mask`) + condicionamiento por tipo de
  mano (AdaLN, estilo Fang/UniDexTok).
- Resultado: un encoder, conservas el twist, signos bajan de 22 a **2**, y esos
  2 los define ISB (no inventados).

## Maquinaria que YA existe (no es código nuevo grande)

- `dong_run_stage2(fk_out, config)` — docstring: toma `RobotLoader.run_fk()`.
  **Ya acepta FK de robot.** Necesita un `config` por robot (links: wrist,
  frame_*_mcp, chains de dedos) — el adapter ya tiene parte (`frame`, `fingers`).
- `human_to_udhm(quats, labels)` — "rotaciones Dong → UDHM". El nombre dice
  "human" pero solo toma rotaciones + labels; sirve para cualquier fuente.
- Pipeline robot sin signos: `robot FK → dong_run_stage2 → human_to_udhm → UDHM`.

## Estado de la verificación de signos (camino qpos×sign, Shadow)

Solo relevante si se elige conservar signos. Verificado por render
(`verify_canonical.py`, poses canónicas):
- **19/22 confirmados**: 15 flex (`fist` ✓), 4 abd (`spread` ✓:
  index/ring/pinky/thumb_mcp_abd).
- **Pendientes**: `thumb_cmc_spread` (física, "lejos del medio" — falta render),
  `middle_mcp_abd` (libre, decreto ulnar = lo que da Dong), `pinky_twist`
  (libre, ISB).

## Convenciones de referencia

- flex/abd: Dong & Payandeh + DexGrasp ("flex hacia palma", "abd lejos del medio").
  Marco Dong: X distal, Y=(ring−index)=ulnar, Z=normal palma volar.
- twist/rotación axial + posturas neutras: **ISB** (Wu et al. 2005).
- Frame es agnóstico: cualquier frame (Dong/ISB) rompe en manos
  **no-antropomórficas** (sin landmarks) → fallback semantic-insertion + zero-pad
  (lo dice Fang). Problema aparte.

## Decisión pendiente

¿Híbrido (1 encoder + twist + 2 signos ISB) o full-Dong (1 encoder + 0 signos,
sin twist)? Depende de si el twist del meñique / abd del MCP del pulgar importan
para el uso (tokenizer de estado → probablemente menores).
