# HANDOFF -- Stage 1 Retargeter

Last updated: 2026-05-13
Author: Yahel + Claude session
Current branch: `experimental` (pushed to origin, ahead of main)
Para Modelo 2 (retargeter). Para Modelo 1 (clasificador) ver `HANDOFF.md`.

---

## Problema original (no resuelto)

**MCP no cierra correctamente** en ningun run (1-20).
Persiste como problema estructural de D_R formulation.
Ver DECISIONS.md Entry 79.

Causa hipotetica (en investigacion): D_R con quaternion completo mezcla
eje de flexion + eje de abduccion. Decoder satisface D_R via abduccion
(rango chico) en vez de flexion (rango grande).

---

## Mejor modelo actual: Run 20

- Seed: 21266, step: 11000
- Branch: `run19-yan-pure` (mergeado a main, a su vez a experimental)
- Metricas test: RS=0.320, NDS=2.261, NVS=0.088
- Cualitativo: mano abierta, puno cerrado, seguimiento open/close, abduccion dedos
- Ckpt: `~/Downloads/stage1_best_total(4).pt` (o el nombre que descargaste de Colab)
- REPRODUCIBLE: seed=21266 fija

---

## Lo que se hizo en esta sesion (2026-05-13)

### Branch `experimental`

`experimental` = `main` + todo lo siguiente. Es donde trabajamos ahora.

### Run 21 (listo para correr)

Config en `train_stage1_colab.ipynb` con `RUN = 21`:
- CSV: `hograspnet_abl11.csv`
- NPZ: `valid_robot_poses_eigengrasp_dong.npz` (el viejo, ya en Drive)
- MCP D_R weight = 0.5 (boost manual vs 1/sigma baseline)
- Seed: 21266 (fija)
- 15k steps, B=50k

### Run 22 (infraestructura lista, esperando NPZ)

**Idea**: reemplazar D_R quaternion por D_R basado en angulos Dong Eq.24.
En vez de `1 - dot^2`, usar `|beta_a - beta_b|` y `|gamma_a - gamma_b|`
ponderados por 1/sigma calculado desde abl13.

**Implementado** (todos commits en `experimental`, pusheados):

| Commit | Que hace |
|--------|----------|
| `3e7850b8a` | Agrega extraccion de angulos MCP/PIP/DIP en `robot_loader._dong_run_stage2` y `precompute_robot_dong.py` |
| `73f6fa938` | `compute_sk_weights_euler.py`: calcula pesos 1/sigma desde abl13 |
| `fb8f5d984` | DECISIONS Entry 84: plan Run 21 + Run 22 |
| `5a74bdf84` | Training code completo: `human_loader`, `robot_loader`, `cross_embodiment_sampler`, `train_cross_emb` |
| `3a311aee3` | Fix notebook precompute: clona `experimental` no `main` |
| `8402e6baa` | Validacion en `precompute_robot_dong.py`: crashea + borra NPZ si faltan angle keys |
| `350350308` | Validacion en notebook cell-16: idem |

**Archivos modificados para Run 22**:

- `grasp-model/scripts/human_loader.py`: carga columnas `beta*_deg/gamma*_deg` de abl13. Si no existen (abl11), `_has_euler=False` (backward compat). Expone `mcp_angles [B,5,2]`, `pip_angles [B,5]`, `dip_angles [B,5]` en batch.
- `grasp-model/scripts/robot_loader.py`: carga angle arrays de `_dong_euler.npz` si existen. Expone en meta de `sample_dong`.
- `grasp-model/scripts/cross_embodiment_sampler.py`: pasa angle arrays en batch output si ambos lados (human + robot) los tienen. Extra-human (HaGRID) recibe zeros.
- `grasp-model/src/cross_emb/train_cross_emb.py`: auto-detecta euler mode (`"mcp_angles_h" in batch`). Si True: D_R = suma de `|angle_a - angle_b|` ponderada. Si False: fallback a D_R quaternion (Run 21 / abl11).
- `grasp-model/notebooks/train_stage1_colab.ipynb`: selector `RUN = 21` o `RUN = 22` en cell-12.
- `grasp-model/notebooks/precompute_dong_colab.ipynb`: clona `experimental`, verifica angle keys en output.

**Pesos _sk_w_euler** (precomputados desde abl13 train split, ya hardcodeados en `train_cross_emb.py`):
```python
_sk_w_euler = {
    "thumb":  dict(mcp_flex=0.2940, mcp_abd=0.2355, pip=0.2953, dip=0.1753),
    "index":  dict(mcp_flex=0.2159, mcp_abd=0.4193, pip=0.1774, dip=0.1874),
    "middle": dict(mcp_flex=0.2510, mcp_abd=0.2209, pip=0.2453, dip=0.2827),
    "ring":   dict(mcp_flex=0.2798, mcp_abd=0.2344, pip=0.2328, dip=0.2531),
    "pinky":  dict(mcp_flex=0.3130, mcp_abd=0.1853, pip=0.2477, dip=0.2540),
}
```

**RIESGO DOCUMENTADO (Entry 84)**: D_R_euler magnitudes ~5x mayores que D_R_quat
(angulos en [0,pi] vs 1-dot^2 en [0,1]). Con w_r=1.0, D_R podria dominar S_k y
hacer D_joints irrelevante para seleccion de triplets. Primera corrida con w_r=1.0.
Si D_R domina, probar w_r=0.2 en follow-up.

### Problema encontrado con NPZ existente

`valid_robot_poses_eigengrasp_dong_euler.npz` en `/media/yareeez/.../processed/`
NO tiene angle arrays. Fue generado con script viejo (antes de agregar angulos).
El nombre `_euler` fue puesto por el notebook pero el codigo en ese momento no
guardaba angulos todavia.

**Fix aplicado**: notebook ahora clona `experimental` explicitamente. Verificacion
double: script crashea + notebook crashea si faltan angle keys.

---

## Que falta hacer ahora

### Inmediato

1. **Regenerar `_dong_euler.npz`** en Colab:
   - Abrir `precompute_dong_colab.ipynb` desde branch `experimental`
   - `INPUT_NPZ = valid_robot_poses_eigengrasp.npz` (el q-only, 746 MB, ya en `/media/...`)
   - Subirlo a Drive si no esta ya
   - Correr notebook. Output: `_dong_euler.npz` con 11 arrays incluyendo mcp/pip/dip_angles
   - Cell-16 verifica automaticamente. Si falla, es un error real.

2. **Correr Run 21** (no necesita nuevo NPZ, usa abl11 + dong viejo):
   - `train_stage1_colab.ipynb`, `RUN = 21`, `SEED = 21266`
   - Comparar con Run 20 (mismo seed, misma config excepto MCP weight=0.5)
   - Live retarget y comparar MCP flexion

3. **Correr Run 22** (despues de tener NPZ regenerado):
   - `train_stage1_colab.ipynb`, `RUN = 22`, `SEED = 21266`
   - Mismo SEED que Run 20/21 para comparacion limpia
   - Observar si D_R_euler mejora MCP

### Mediano plazo

- Si Run 21 o Run 22 mejoran MCP: multi-seed final (3-5 seeds) para tesis.
- Reportar RS/NDS/NVS mean +- std en test split.
- Si ninguno mejora MCP: replantear hipotesis (ver opciones en DECISIONS.md Entry 79).

---

## Archivos de datos relevantes (local)

```
/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed/
  hograspnet_abl11.csv                            # Run 21 (y todos los runs anteriores)
  hograspnet_abl13.csv                            # Run 22+ (abl11 + columnas beta/gamma deg)
  valid_robot_poses_eigengrasp.npz                # q-only, 746 MB. Input del precompute.
  valid_robot_poses_eigengrasp_dong_euler.npz     # MALO (sin angle arrays). Regenerar.
```

---

## Estado del codigo (branch `experimental`)

Pipeline completo y testeado en human side. Robot side esperando NPZ regenerado.

- Euler mode se activa automaticamente cuando el batch tiene `mcp_angles_h`.
- Si el NPZ no tiene angulos (formato viejo), el training cae silenciosamente a quat D_R.
- No hay flag que forzar: auto-deteccion por contenido del batch.

### Smoke test ejecutado (2026-05-13)

- Human loader: OK. `has_euler=True`. `mcp_angles=(1015342, 5, 2)`.
- Robot loader: NPZ sin angulos -> `mcp_angles_all` ausente en meta.
- Sampler: batch sin `mcp_angles_h/r` -> `_use_euler_dr=False` -> quat fallback.
- Conclusion: Run 21 funciona ahora mismo. Run 22 bloquea en NPZ.

---

## Comandos clave

### Live retarget
```bash
cd /home/yareeez/AIST-hand/grasp-model && \
/home/yareeez/AIST-hand/.venv/bin/python scripts/live_retarget.py \
  --ckpt '/home/yareeez/Downloads/stage1_best_total(4).pt'
```

### Smoke test sampler (verifica euler mode)
```bash
cd /home/yareeez/AIST-hand && \
/home/yareeez/AIST-hand/.venv/bin/python - <<'EOF'
import sys
sys.path.insert(0, "grasp-model/scripts")
DATA = "/media/yareeez/94649A33649A1856/DATOS DE TESIS/AIST-hand_grasp-model_data_cache/processed"
from cross_embodiment_sampler import CrossEmbodimentSampler
cs = CrossEmbodimentSampler(
    f"{DATA}/hograspnet_abl13.csv",
    "/home/yareeez/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf",
    "grasp-model/data/hand_configs/shadow_hand_right.yaml",
    split="train", device="cpu",
    valid_poses_path=f"{DATA}/valid_robot_poses_eigengrasp_dong_euler.npz"
)
batch = cs.get_batch_temporal(32)
print("euler mode:", "mcp_angles_h" in batch)
EOF
```
Debe imprimir `euler mode: True` despues de regenerar el NPZ.

### Inspect ckpt
```bash
/home/yareeez/AIST-hand/.venv/bin/python -c "
import torch
c = torch.load('/home/yareeez/Downloads/stage1_best_total(4).pt', map_location='cpu', weights_only=False)
print('step:', c.get('step'), 'seed:', c.get('seed'))
print('test_metrics:', c.get('test_metrics'))
print('config:', {k: c['config'][k] for k in ['lr','b','margin','w_r','w_joints','w_ahg']} if c.get('config') else 'no config')
"
```

---

## Contexto para otro LLM

Lee en orden:

1. `~/.claude/projects/-home-yareeez-AIST-hand/memory/MEMORY.md`: contexto general.
2. `DECISIONS.md` Entry 84: plan Run 21 + Run 22. Documenta scale risk.
3. `DECISIONS.md` Entry 82-83: estado pipeline + seed variance.
4. `DECISIONS.md` Entry 79: diagnostico MCP original.
5. Este archivo: donde quedamos, que sigue.

**Estado mental actual**: Run 22 infraestructura 100% implementada. Bloqueado
en regenerar NPZ con angulos euler. Run 21 puede correr YA con la config actual.
MCP problem sigue abierto. Seed=21266 es el buen basin.
