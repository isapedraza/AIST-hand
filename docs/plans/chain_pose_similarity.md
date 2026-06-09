# Implementar `D_chain` como `D_ee` de Yan, pero sobre toda la cadena

## Summary

- Esto es analogo a `D_ee` de Yan en el sentido de usar una sola distancia geometrica normalizada para minar triplets.
- La diferencia clave es el objeto comparado: Yan compara solo puntas/end-effectors; aqui se compara toda la cadena `MCP/PIP/DIP/TIP`.
- La metrica debe quedar separada de Xin: no se mezcla con `lam_fp`, `lam_pinch`, `lam_fr` ni `lam_mid`.
- Se activa con `--similarity_metric chain_pose`; el default queda `xin` para no romper runs actuales.

## Key Changes

- En `losses.py`, agregar una funcion nueva:
  - `chain_pose_distance(chain_a, chain_b, finger_idx=None)`
  - Si `finger_idx` es `None`, compara toda la mano comun `[N, Fc, 4, 3]`.
  - Si `finger_idx` viene definido, compara solo ese dedo `[MCP/PIP/DIP/TIP]`.
- La comparacion es punto a punto, no por centroide:
  - `MCP_a` vs `MCP_b`
  - `PIP_a` vs `PIP_b`
  - `DIP_a` vs `DIP_b`
  - `TIP_a` vs `TIP_b`
- La reduccion ocurre despues de comparar puntos, por L2/RMS sobre los puntos.

## Training Wiring

- En `config.py`, agregar:
  - `--similarity_metric`
  - choices: `xin`, `chain_pose`
  - default: `xin`
- En `loop.py`, durante mining contrastive:
  - `similarity_metric == "xin"`: usar comportamiento actual.
  - `similarity_metric == "chain_pose"`:
    - con `--single_latent`: usar `chain_pose_distance(chain_a, chain_b)` sobre todos los dedos comunes.
    - sin `--single_latent`: usar `chain_pose_distance(chain_a, chain_b, finger_idx=...)` para el subespacio de ese dedo.
- La distancia solo decide positivo/negativo en `torch.no_grad()`, igual que ahora.
- El gradiente sigue pasando por la distancia latente del triplet, no por la metrica geometrica.

## Exact Behavior

- No se promedian posiciones antes de comparar.
- No se hace `distance(mean(chain_a), mean(chain_b))`.
- Se hace distancia punto a punto y luego se agregan errores.
- Esto preserva detalles del agarre que `D_ee` literal de Yan perderia al mirar solo fingertips.

## Test Plan

- Actualizar `test_xin_sk.py` o crear test pequeno para verificar:
  - `chain_pose_distance(a, a) == 0`
  - mover solo `MCP` aumenta distancia
  - mover solo `TIP` aumenta distancia
  - salida shape `[N]`
- Verificar que `train_cross_emb.py --help` muestra `--similarity_metric`.
- Smoke test corto con:
  - `--similarity_metric chain_pose`
  - opcionalmente `--single_latent`

## Assumptions

- `finger_pose` significa cadena completa `MCP/PIP/DIP/TIP`, no solo puntas.
- Queremos un port minimo y limpio desde la idea de run33, sin traer el resto de run33.
- La metrica nueva debe quedar aislada para poder comparar runs sin mezclar knobs.
