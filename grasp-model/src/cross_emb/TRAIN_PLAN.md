# train_cross_emb.py — Plan de implementación

Stage 1 de Yan (2026): aprender espacio latente compartido humano-robot.
Adaptación para manos dextrosas (1 subespacio, encoder GNN en vez de MLP).

## Arquitectura

| Red | Tipo | Entrada | Salida |
|-----|------|---------|--------|
| `E_h` | solo humano (GNN) | quats `[B, 20, 4]` → PyG Batch | z `[B, z_dim]` |
| `E_r` | robot-específico (linear) | q_r `[B, J]` | `[B, 1024]` |
| `E_X` | compartida (MLP) | `[B, 1024]` | z `[B, z_dim]` |
| `D_X` | compartida (MLP) | z `[B, z_dim]` | `[B, 1024]` |
| `D_r` | robot-específico (linear) | `[B, 1024]` | q_r `[B, J]` |

## Flujo de datos

```
HUMAN:   quats_h → QuatToGraph → E_h → z_h
ROBOT:   q_r → E_r → E_X → z_r
DECODE:  z → D_X → D_r → q_r_hat
```

## 4 Losses (pesos de Yan)

| Loss | Fórmula | λ |
|------|---------|---|
| `L_cont` | triplet(z_h, z_r) usando D_R + D_ee | 10 |
| `L_rec` | `‖q_r - D_r(D_X(z_r))‖` | 5 |
| `L_ltc` | `‖z_h - E_X(D_X(z_h))‖` | 1 |
| `L_temp` | `‖v_H^hand - v_A^ee‖` | 0.1 |

`v_A^ee` = velocidad robot retargeteada = `D_r(D_X(z_t1)).tips - D_r(D_X(z_t)).tips` vía FK + Stage2.

## Orden de implementación

1. **SAMPLER** — done. Shapes verificadas. ✓
2. **HUMAN** — `QuatToGraph` (ya existe) + `E_h` → `z_t`, `z_t1`
   - cat([quats_h, quats_h_t1]) → [2B, 20, 4] → QuatToGraph → E_h → z_both
   - z_t = z_both[:B], z_t1 = z_both[B:]
3. **ROBOT** — `E_r` + `E_X` → `z_r`
4. **DECODE** — `D_X` + `D_r` → `q_r_hat`, `q_r_ret`, para losses
5. **LOSSES** — una por una: L_rec, L_ltc, L_cont, L_temp
6. **BACKWARD** — optimizer step

## Hiperparámetros (Yan defaults)

- z_dim = 16
- shared_dim = 1024
- lr = 1e-3 (Adam)
- B = 1e5 (objetivo; empezar con 1000 para pruebas)
- α (triplet margin) = 0.05
- ω (peso D_ee en contrastive) = 1.0

## Notas

- Yan usa 5 subspacios (LA, RA, TK, LL, RL). Nosotros: 1 (mano).
- Yan usa MLP para E_h. Nosotros: GNN (GAT x3) — aportación propia.
- FK es diferenciable (pytorch_kinematics) — gradientes fluyen por L_temp.
- Rutas locales:
  - CSV: `/home/yareeez/AIST-hand/grasp-model/data/processed/hograspnet_abl11.csv`
  - URDF: `/home/yareeez/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf`
  - YAML: `/home/yareeez/AIST-hand/grasp-model/data/hand_configs/shadow_hand_right.yaml`
