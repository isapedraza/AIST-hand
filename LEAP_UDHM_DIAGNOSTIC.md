# LEAP UDHM Diagnostic — Problema + Solución

## Qué pasó

Tests #1-3 de UDHM en LEAP **fallan**. Abducción no se lee.

Test #2 (aislar abducción excitando joint "0"): 
- `mcp_abd` NO se mueve (debería). 
- `pip_flex` SÍ se mueve (no debería).
- Conclusión: el joint de abducción de LEAP no mueve la posición de `pip` → Dong no lo ve.

## Por qué pasa

**Dong's method asume:** MCP humano = articulación universal en 1 punto (los 2 DOF ocurren en el mismo lugar, sin torsión). Infiere flex+abd desde un vector: `pip_pos - mcp_pos`.

**Shadow cumple esa premisa:** FFJ4 (abd) y FFJ3 (flex) comparten origin (traslación cero, comentario en shadow.yaml lo confirma). Un joint "acomoda" el otro en el mismo espacio → vector `pip - mcp` captura ambos.

**LEAP NO cumple:** joint "1" (flex) y joint "0" (abd) están **separados en el espacio** (`origin xyz="-0.0122 0.0381 0.0145"` en URDF). El joint "0" rota sobre SU propio eje (torsión axial), NO mueve `pip`. Vector `pip - mcp` ve cero movimiento de ese joint → desaparece de `R_mcp`.

**Root cause:** LEAP hardware no imita anatomía humana (MCP = 1 punto). Tiene 2 joints serie, separados. Dong no fue diseñado para eso.

## Soluciones

### Opción A: Parchear Dong para LEAP (débil)
Usar `dip - mcp` en vez de `pip - mcp` (saltar un eslabón). Captura ambos joints con offset. 
- Pro: cero código nuevo.
- Contra: hack específico por robot. Si future-robot tiene 3 joints, se rompe. Reinterpreta mal PIP.

### Opción B: Robot lee qpos directo (fix completo) ✓
**Novo:** para robots, no inferir ángulos desde posiciones. Leer directo del qpos.
- Paso 1: `_build_motion_primitives` identifica qué joint es FLEX, cuál es ABD, con signo.
- Paso 2: Tomar `qpos[joint_idx]` crudo, normalizar (`/pi` o por límite), escribir directo a slot `index_mcp_flex`/`index_mcp_abd`.
- Paso 3: PIP/DIP quedan igual (1 DOF hinges, Dong ya funciona).

**Por qué es fix, no patch:**
- Dong infiere porque no tiene qpos (mano humana, solo landmarks). Robot TIENE qpos → dato real > inferencia.
- Generaliza: cualquier robot futuro "raro", si tienes qpos, úsalo. Si no (humano), Dong.
- Fuentes de verdad distintas → métodos distintos. No es asimetría, es corrección.

## Dudas / TBD

1. **¿Está confirmado que joint "0" = ABD?** Técnicamente, medición solo vio que `pip` no se mueve (eje axial sobre sí mismo). `_build_motion_primitives` debería confirmar "es ROT o ABD, con signo X". Hay que correr eso.

2. **¿Cuál es el límite exacto de joint "0" en LEAP para normalizar?** Rango en `_ALL_LIMITS` de `build_leap_eigengrasps.py` debería tenerlo. Verificar contra URDF.

3. **¿Shadow también se beneficiaría de Opción B?** Técnicamente sí, porque qpos es más directo. Pero Shadow ya funciona con Dong → no es urgente tocar.

4. **¿Posible que Allegro tenga el mismo problema?** Posible, si sus joints index están separados como LEAP. No se ha testado.

## Plan de implementación (Opción B)

```
1. Run _build_motion_primitives(leap) → confirma joint mapping
2. Crear función new: robot_udhm_from_qpos(qpos, robot_config, primitives)
   - input: [B, J] qpos
   - output: [B, 22] UDHM vector
   - logic: 
     for cada dedo, para cada DOF (flex/abd/pip/dip):
       joint_idx = primitives[dedo][DOF].joint_index
       angle_rad = qpos[:, joint_idx]
       normalized = angle_rad * scale_factor  # /pi o por límite
       slot_idx = UDHM22_SLOTS.index(f"{dedo}_{DOF}")
       udhm[:, slot_idx] = normalized * sign

3. Cambiar sampler/loader: 
   - Humano: usa dong_run_stage2 (posición → ángulos)
   - Robot LEAP: usa robot_udhm_from_qpos (qpos directo)
   - Robot Shadow: usa robot_udhm_from_qpos (opcional, si no hay regresión)

4. Re-correr Tests #1-3 en LEAP
```

## Código actual afectado

- `models/latent-retargeting/src/cross_emb/loaders/udhm_stage3.py` — Dong-based, agnóstico. Sin cambios.
- `models/latent-retargeting/src/cross_emb/loaders/sampler.py` — llama a `udhm_run_stage3`. Agregar rama `if robot == "leap": robot_udhm_from_qpos` else `udhm_run_stage3`.
- `models/latent-retargeting/src/cross_emb/loaders/robot_loader.py` — ya tiene `_build_motion_primitives`. Reusable.

## Validación

Post-fix: Tests #1-3 deberían pasar en LEAP con ábsorción clara (abd monotónico con delta_q, cero leak a flexión).
