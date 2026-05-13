# HANDOFF — Stage 1 Retargeter

Last updated: 2026-05-12
Author: Yahel + Claude session
Current branch: `run19-yan-pure` (pushed to origin)
Para Modelo 2 (retargeter). Para Modelo 1 (clasificador) ver `HANDOFF.md`.

---

## Problema original (no resuelto)

**MCP no cierra correctamente** ni siquiera en el mejor modelo (`Run 20`).
Los dedos no logran flexion completa en la articulacion metacarpofalangica.
Persiste desde Run 14. Documented en DECISIONS.md Entry 79.

Causa hipotetica (no confirmada): D_R formulation usa quaternion completo
sin separar eje de flexion vs eje de abduccion. Decoder satisface D_R via
abduccion (rango chico, facil) en vez de flexion (rango grande, dificil).

Fix propuesto pendiente: **MCP axis decomposition** -- descomponer quaternion
MCP en `q_flex` + `q_abd` antes de calcular D_R. Cada eje recibe escalar
y peso propio. Ver DECISIONS.md Entry 79 (final del entry).

---

## Que se intento (Runs 17 -> 20)

### Punto inicial
- Run 15b: mejor modelo cualitativo. Coordinacion entre dedos suave, pero MCP no cierra.
- Run 15b entreno con random seed (no logged), config 1/sigma D_R + speed opts off + CosineWarmRestarts.

### Secuencia ejecutada

| Run | Cambios | Resultado |
|-----|---------|-----------|
| 17 | Revert D_R a 1/sigma + speed opts on, seed=none | Decente |
| 18 | + seed=42 fijo | Dedos separados (indice+medio vs anular+meñique) |
| 18b | Speed opts revertidos, seed=42 | Identico a Run 18 (speed opts safe) |
| 19 | Yan-pure constant LR + speed opts, seed=42 | Pulgar mejor, dedos siguen mal |
| **20** | Yan-pure + seed=-1 (random) | **MEJOR MODELO. Seed=21266, step=11k. Mano abierta, puno, seguimiento, abduccion.** |

### Conclusiones consolidadas (DECISIONS.md Entry 82-83)

1. **Speed opts A-E confirmados math-safe**. Cache Dong NPZ bit-exact vs runtime (0.0 diff sobre 500 poses). AMP fp16 + torch.compile no afectan basin.
2. **Indexing fix (84c820050 + 7981d9c4f) en todos Run 15+**. No es variable.
3. **CosineWarmRestarts no era unica causa**. Run 19 sin restarts sigue malo con seed=42.
4. **Seed=42 cae en mal basin**. Tres configuraciones diferentes, mismo failure.
5. **Run 15b fue init aleatorio afortunado**, no merito del codigo. No reproducible (seed nunca loggeada).
6. **MCP problem es ortogonal**. Persiste en TODOS los runs, buen o mal basin. Es problema estructural de D_R formulation, no de seed/scheduler/speed.
7. **Seed variance CONFIRMADA (Entry 83)**. Run 20 seed=21266 produce mejor modelo cualitativo de todos los runs: mano abierta, puno cerrado, seguimiento open/close, abduccion de dedos.

---

## Estado actual del codigo

### Branch `run19-yan-pure` en origin

Pipeline consolidado:
- **Adam constant lr=1e-3** (Yan-pure, sin scheduler ni warmup)
- **Speed opts activos** (AMP fp16 + torch.compile + Dong NPZ cache)
- **Seed flag** `--seed -1` (random, default) | `--seed N` (fijo)
- **Seed loggeado** en `ckpt['seed']`
- **Config completo** en `ckpt['config']` (todos hyperparams)
- **Val mid-training** cada 500 steps (subjects 1-10, S1 split). Best por val score.
- **Test al final** (subjects 74-99). Anotado en ambos best_total y best_val.
- **Metrics**: RS (rotation similarity), NDS (normalized distance), NVS (normalized velocity). Adaptadas Yan eq 1/2/loss_v para dominio mano.

Files clave:
- `grasp-model/src/cross_emb/train_cross_emb.py`: training script
- `grasp-model/scripts/cross_embodiment_sampler.py`: sampler con split support
- `grasp-model/scripts/human_loader.py`: HOGraspNet loader con S1 split
- `grasp-model/scripts/robot_loader.py`: RobotLoader con DONG_CACHE / RANDOM_UNIFORM modes
- `grasp-model/notebooks/train_stage1_colab.ipynb`: notebook Colab, auto-checkout branch

### Commits relevantes en `run19-yan-pure`

```
da4275f6d Add DECISIONS Entry 82: Stage 1 pipeline consolidation
b88625925 Add --seed -1 (random) default; notebook SEED=-1
af4722207 Fix OOM: val/test samplers skip valid_poses NPZ
681abc9a9 Notebook: pass eval flags
9351d1978 Add val/test eval with RS/NDS/NVS metrics
56826db24 Save full args config in checkpoint payload
847459f28 Run 19: remove scheduler (Yan-pure constant LR)
```

### Modelos disponibles en `~/Downloads/`

| Ckpt | Step | Seed | Calidad |
|------|------|------|---------|
| **`stage1_best_run20.pt`** | **11000** | **21266** | **MEJOR. Mano abierta, puno, seguimiento, abduccion. REPRODUCIBLE.** RS=0.320 test. |
| `stage1_best_run15b.pt` | ~10k | aleatorio (no logged) | Segundo mejor. Tip pinch/OK aproximado (solo con gesto exagerado). NO reproducible. |
| `stage1_best_run19_seed42.pt` | ~15k | 42 | Constant LR, pulgar mejor, coordinacion mala. |
| `stage1_best_run18b_15k.pt` | ~15k | 42 | Mas responsivo pero dedos separados. |
| `stage1_best_run18b_10k.pt` | 10463 | 42 | Identico a Run 18 sin speed opts. |
| `stage1_best_run18.pt` | 14462 | 42 | Dedos separados. |

---

## Que falta hacer

### Estado actual (2026-05-12)

**Seed variance CONFIRMADA. Baseline reproducible establecido.**

- Run 20 seed=21266: mejor modelo cualitativo de todos los runs.
- Mano abierta + puno cerrado + seguimiento + abduccion.
- MCP problem persiste (estructural, ortogonal a seed).

### Inmediato

**Run 21 con seed=21266 + MCP axis decomposition**:

1. Modificar `_dong_run_stage2` en sampler / S_k: separar quaternion MCP en componente flex y componente abd.
2. Aplicar peso distinto a cada eje en D_R (flex pesa mas que abd).
3. Correr Colab con `SEED=21266` (fijo) y cambio MCP.
4. Live retarget. Comparar coordinacion vs Run 20.
5. Si mejor -> baseline final. Si peor o igual -> MCP fix no era el bottleneck.

Ver DECISIONS.md Entry 79 para diagnostico completo y propuesta de implementacion.

### Mediano plazo

1. **Multi-seed final** (para tesis):
   - 3-5 seeds con config final elegida.
   - Reportar mean +- std de RS/NDS/NVS en test.

2. **Si MCP fix no mejora**: opciones de ataque a investigar:
   - Permutation invariance en init: 5 finger encoders empiezan con misma weight.
   - Auxiliary loss enforzando coordinacion entre subspaces.
   - z_dim reducido (16 -> 8) -> menos basins.

### Largo plazo (tesis)

- Documentar pipeline final en seccion methodology.
- Reportar metrics RS/NDS/NVS en test split como Yan, justificando adaptaciones de mano (chain 4-points, 5 fingertips).
- Speed opts como contribucion (validados safe, ~6x speedup en T4).
- Bug indexing encontrado/corregido (transparencia).
- Seed variance como limitacion conocida + future work.

---

## Comandos clave

### Correr Run en Colab

URL notebook:
```
github.com/isapedraza/AIST-hand/blob/run19-yan-pure/grasp-model/notebooks/train_stage1_colab.ipynb
```

Open in Colab -> Restart runtime -> Run all.

Cell 12 default: `SEED = -1` (random). Cambia a numero fijo para reproducir.

### Live retarget

```bash
cd /home/yareeez/AIST-hand/grasp-model && \
/home/yareeez/AIST-hand/.venv/bin/python scripts/live_retarget.py \
  --ckpt /home/yareeez/Downloads/<ckpt_name>.pt
```

### Inspect ckpt

```bash
/home/yareeez/AIST-hand/.venv/bin/python -c "
import torch
c = torch.load('/home/yareeez/Downloads/<ckpt>.pt', map_location='cpu', weights_only=False)
print('step:', c.get('step'), 'seed:', c.get('seed'))
print('losses:', c.get('losses'))
print('val_metrics:', c.get('val_metrics'))
print('test_metrics:', c.get('test_metrics'))
print('config sample:', {k: c['config'][k] for k in ['lr','b','margin','w_r','w_joints','w_ahg']} if c.get('config') else 'no config')
"
```

### Volver a main

```bash
git checkout main
# Run 19 branch sigue viva en origin, no se pierde
```

### Drive issues en Colab

Si error 107 (transport endpoint not connected) -- causa: internet inestable rompe FUSE mount de Drive:

1. Restart runtime: Runtime -> Restart session
2. Re-run cell 4 (drive.mount)
3. Verificar: `!ls /content/drive/MyDrive/AIST-hand/hograspnet_abl11.csv`
4. Re-run training cells

Mitigacion permanente si red mala -- copiar archivos a `/content/` antes de training para que lecturas sean SSD local (sin red):

```python
!cp /content/drive/MyDrive/AIST-hand/hograspnet_abl11.csv /content/
!cp /content/drive/MyDrive/AIST-hand/valid_robot_poses_eigengrasp_dong.npz /content/
!cp /content/drive/MyDrive/AIST-hand/shadow_hand_right.yaml /content/
!cp /content/drive/MyDrive/AIST-hand/data/processed/hagrid_dong.csv /content/
```

Y apuntar paths en cell 12 a `/content/...` en vez de `/content/drive/MyDrive/AIST-hand/...`.

---

## Contexto para otro LLM

Si retomas esta tesis con otro LLM, lee en orden:

1. `~/.claude/projects/-home-yareeez-AIST-hand/memory/MEMORY.md`: contexto general. Stage 1 retargeter, dominio mano, framework Yan.
2. `DECISIONS.md` Entry 82: estado consolidado del pipeline.
3. `DECISIONS.md` Entry 79: diagnostico MCP (problema original).
4. Este archivo (`HANDOFF_STAGE1.md`): donde quedamos, que sigue.
5. `knowledge/yan2026/main.tex`: paper de referencia. Yan reporta constant LR + Adam + lr=1e-3, MLPs 8 capas 256d, z_dim=16. Eqs 1/2/loss_v para metrics.

Estado mental: pipeline consolidado, MCP problem unresolved, seed variance hipotetica pendiente de testear.
Proximo paso operacional: Run 20 con random seed.

---

## Decisiones pendientes que Yahel debe tomar

1. **¿Implementar MCP axis decomp en Run 21 o primero multi-seed con config actual?** Sugerido: MCP en Run 21 con seed=21266 fija.
2. **¿Pagar Colab Pro temporal o copiar archivos local en cada session?** Decision tuya.

---

## Resumen TL;DR

- Pipeline en `run19-yan-pure` listo y testeado.
- **Run 20 (seed=21266) = mejor modelo. Mano abierta, puno, seguimiento, abduccion. REPRODUCIBLE.**
- MCP problem sigue abierto (Entry 79). Fix propuesto: axis decomposition.
- Proximo paso: Run 21 con seed=21266 + MCP axis decomp. Comparar vs Run 20.
