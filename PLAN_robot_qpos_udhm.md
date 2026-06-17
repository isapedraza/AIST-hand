# Plan: Robot llena UDHM desde qpos (primitivas), no Dong

## Meta
El robot tiene qpos (verdad). Dong lo adivina desde posiciones y falla en LEAP
(abduccion congelada) y Barrett (distal congelada). Fix: robot llena UDHM desde
qpos via primitivas. Humano sigue Dong. Mismo `udhm_run_stage3` para ambos.

## Probado ya (no re-discutir)
- Receta `R_mcp = Ry(flex)@Rz(abd)`, pip/dip `Ry(sign*ang)` -> `udhm_run_stage3`.
- LEAP: abduccion se llena (antes congelada), cero fuga.
- Shadow control: qpos-path vs Dong-path = diff 0.0000 en flex/pip/dip. Mecanismo OK.

## 3 modulos diferenciados (1 responsabilidad c/u)

### Funcion 1 -- `build_primitives(loader, config) -> tabla`  [= M_h del paper]
Corre 1 vez por robot. Tabla: `{dedo: {role: (col_qpos, signo)}}`,
role in {mcp_abd, mcp_flex, pip, dip}.
Por cada joint:
- dedo  <- `_build_joint_to_finger` (reuso)
- tipo  <- eje del joint vs triada del dedo: ||lateral=FLEX, ||normal=ABD, ||bone=ROT
- signo <- REGLA UNICA: `+1 si |hi| >= |lo| else -1` (flexion = lado de mayor rango).
           Determinista. Solo importa en pip/dip (atan2); mcp arccos pliega signo.
- ordenar FLEX por profundidad de cadena -> mcp_flex/pip/dip; ABD -> mcp_abd
- Manos raras: tabla MANUAL que pisa el auto (`if joint in manual`):
  - Barrett (anotado): finger_1/2_prox = spread -> EXCLUIDO; med->mcp_flex; dist->pip;
    finger_3 = thumb. (palm frame radial invalida el auto)
  - Pulgares Shadow/LEAP: auto imperfecto (ejes ~45deg) -> manual despues si hace falta.

### Funcion 2 -- `robot_qpos_to_udhm(qpos, tabla) -> udhm22 [B,22]`
Corre cada batch. ~10 lineas:
```
R_mcp = Ry(qpos[flex]) @ Rz(qpos[abd])      # label {dedo}_mcp
R_pip = Ry(signo * qpos[pip])               # label {dedo}_pip
R_dip = Ry(signo * qpos[dip])               # label {dedo}_dip
return udhm_run_stage3(stack(rot6), labels)  # REUSO, no tocar
```

### Modulo 3 -- Cableado (aparte)
`sampler.py:277`: cambiar `udhm22_r = udhm_run_stage3(pose_r, labels_r)` por
`udhm22_r = robot_qpos_to_udhm(q_r, tabla)`, detras de flag `--robot_udhm_from_qpos`
(default OFF). `q_r` ya esta en `:237`. Humano (`:276`) intacto.

## Reuso sin tocar
`udhm_run_stage3`, `_FILL`, `UDHM22_SLOTS`, `_build_joint_to_finger`, `run_fk`,
`_resolve_limits`, loss en `loop._cross_robot_contrastive`, `losses._udhm_w`.

## Validacion (offline, sin training)
1. Shadow: qpos-path vs Dong-path -> flex/pip/dip diff ~0 (control).
2. LEAP: abduccion responde, cero fuga a flex.
3. Barrett: pip se llena (antes 0), flexion positiva.

## Orden de ejecucion
Funcion 1 sola -> validar -> Funcion 2 sola -> validar -> cableado. Una a la vez.
NO tocar nada existente hasta el cableado (paso final, 1 linea, con flag).
