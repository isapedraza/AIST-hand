# Resultados del barrido de eval (2026-09-01)

Corrido de `run_eval_sweep_2026-09-01.sh` sobre los 18 checkpoints que
pasaron el probe de compatibilidad de arquitectura (ver
`checkpoint_inventory_2026-09-01.md`). Mismos parametros que la tabla de
tesis: `--n_batches 10 --b_eval 3000` (30k muestras), CSV
`~/6d_datasets/hograspnet_abl14_6d.csv`. Log crudo completo en
`eval_sweep_2026-09-01.log`.

RS/NDS/NVS: mas bajo = mas fiel. `rec` = error de reconstruccion del
autoencoder robot.

## shadow

| checkpoint | step | RS | NDS | NVS | rec |
|---|---|---|---|---|---|
| `stage1_shadow_solo_ckpt17_run18b_13k7.pt` (**en tesis**) | 13768 | 0.2030 | 1.9331 | 0.0210 | 0.9309 |
| `stage1_shadow_allegro_bodex_objbal_15k.pt` (**en tesis, multi-robot**) | 14999 | 0.3400 | 2.3781 | 0.0227 | 0.9254 |
| `stage1_best_total(18):b.pt` = `stage1_shadow_udhm_B_15k.pt` (mismo archivo, dos nombres) | 14412 | 0.3222 | 2.0092 | 0.0219 | 0.9370 |
| `stage1_best_total(18):A.pt` | 14496 | 0.3440 | 1.9709 | 0.0217 | 0.9495 |
| `stage1_best_total(25).pt` (multi shadow+allegro) | 10323 | 0.3193 | 2.3542 | 0.0228 | 0.9858 |
| `stage1_best_total(21).pt` (multi shadow+allegro) | 6790 | 0.3193 | 2.3677 | 0.0239 | 1.0638 |
| `stage1_best_total(16).pt` | 14505 | 0.3659 | 2.1749 | 0.0221 | 0.9778 |
| `stage1_best_total(12).pt` | 5985 | 0.3430 | 2.2820 | 0.0225 | 1.3555 |
| `stage1_best_total(22).pt` (multi shadow+allegro) | 9185 | 0.3382 | 2.4361 | 0.0239 | 0.9736 |
| `stage1_best_total(23).pt` (multi shadow+allegro) | 14260 | 0.3414 | 2.4365 | 0.0229 | 0.9036 |
| `stage1_best_total(20).pt` (3-robot, ver leap/barrett abajo) | 13698 | 0.4316 | 2.6341 | 0.0247 | 0.9212 |
| `stage1_best_total(19).pt` (3-robot) | 9885 | 0.4291 | 2.6430 | 0.0247 | 0.9784 |
| `stage1_best_total(15).pt` (3-robot) | 5935 | 0.4555 | 2.5557 | 0.0232 | 1.1722 |
| `stage1_best_total(13).pt` | 5630 | 0.7964 | 3.2845 | 0.0275 | 1.5497 |

**Nota:** ninguno de estos supera al shadow-solo actual de la tesis (RS=0.2030) en RS.
`stage1_best_total(18):b.pt` tiene mejor NDS (2.0092) que el multi-robot de la
tesis (2.3781) con peor RS -- mismo tipo de trade-off que ya se documenta
para Allegro. No se investigo mas a fondo (que hiperparametro distingue este
checkpoint), queda pendiente si se quiere perseguir.

## allegro

| checkpoint | step | RS | NDS | NVS | rec |
|---|---|---|---|---|---|
| **`stage1_best_total(14).pt` -- step 5907, NO esta en la tesis** | 5907 | **0.6104** | **3.5379** | 0.0247 | 0.8896 |
| `stage1_allegro_solo_bodex_noobjbal_ckpt24_14k5.pt` (**en tesis**) | 14553 | 0.6833 | 3.8394 | 0.0235 | 1.2462 |
| `stage1_shadow_allegro_bodex_objbal_15k.pt` (**en tesis, multi-robot**) | 14999 | 0.7104 | 3.7289 | 0.0262 | 1.0954 |
| `stage1_best_total(25).pt` (multi) | 10323 | 0.7026 | 3.7135 | 0.0260 | 1.1444 |
| `stage1_allegro_solo_bodex_objbal_15k.pt` (**en tesis**) | 14999 | 0.7430 | 3.7568 | 0.0241 | 1.2540 |
| `stage1_best_total(21).pt` (multi) | 6790 | 0.7806 | 3.7251 | 0.0291 | 0.6745 |
| `stage1_best_total(23).pt` (multi) | 14260 | 0.7801 | 3.7720 | 0.0282 | 0.6047 |
| `stage1_best_total(22).pt` (multi) | 9185 | 0.7830 | 3.7475 | 0.0288 | 0.6179 |
| `stage1_allegro_freeze.pt` | 2500 | 1.4827 | 6.1069 | 0.0513 | 0.3998 |

**HALLAZGO A REVISAR, no confirmado:** `stage1_best_total(14).pt` (step
5907, robots=['allegro'] solo) domina en RS *y* NDS a los dos checkpoints
Allegro-solo que estan en la tesis. Step mucho mas bajo que los demas
(5907 vs 14-15k) -- no se identifico que corrida/config lo genero ni por
que un checkpoint tan temprano supera a los de step alto; podria ser una
corrida real mejor, o una anomalia del checkpoint temprano (menos overfit,
o config distinta no registrada). **No se cambio la tabla de la tesis por
esto** -- requiere investigar la config antes de confiar en el numero.

## shadow+leap+barrett

`leap` corre y da metricas (peores que shadow, familia nunca antes evaluada).
`barrett` **falla siempre**, en los 3 checkpoints de esta familia, mismo error:

```
RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 2
```

Es sistematico (no depende del checkpoint) -- probable incompatibilidad
entre `robot/hand-configs/barrett.yaml` / `RobotEncoder_E_r` actual y como
se genero ese checkpoint, o un bug en `eval_retarget.py` para ese robot
especifico. No se investigo la causa raiz, queda marcado como error real,
sin inventar un numero.

| checkpoint | step | shadow RS/NDS | leap RS/NDS | barrett |
|---|---|---|---|---|
| `stage1_best_total(20).pt` | 13698 | 0.4316 / 2.6341 | 3.1783 / 9.0438 | ERROR |
| `stage1_best_total(19).pt` | 9885 | 0.4291 / 2.6430 | 3.1780 / 9.0294 | ERROR |
| `stage1_best_total(15).pt` | 5935 | 0.4555 / 2.5557 | 3.1546 / 9.1126 | ERROR |
