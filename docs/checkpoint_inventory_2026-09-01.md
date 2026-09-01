# Inventario de checkpoints de retargeting (2026-09-01)

Barrido de TODOS los `.pt` en `~/Downloads`, `models/latent-retargeting/checkpoints/` y `grasp-model/checkpoints/` (83 archivos, 2 no son checkpoints -- `pre_filter.pt`/`pre_transform.pt` son artefactos de sklearn, se excluyen). Generado leyendo metadata embebida (`step`, `config`, `robots`) con `torch.load`, sin correr eval todavia.

**No estan trackeados en git** (`.gitignore`: `models/*/checkpoints/`) -- para llevarlos a otra maquina hay que copiarlos por fuera de git (USB, rclone, etc.), este documento si viaja.

**Caveat pendiente antes de fiarse de metricas viejas**: varios runs de step bajo (<10k) o resultados que en su momento parecieron "mal modelo" en realidad eran fallas de la camara/percepcion upstream (resuelto despues) -- no descartar un checkpoint solo por su historial de esa epoca sin re-evaluarlo.


## robots = `shadow+allegro`  (n=6)

| archivo | step | z_dim | shared_dim | human_encoder | human_rot_repr | temporal_window | lam_udhm | xin_switching | extra_human_ratio | primitive_sample | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage1_shadow_allegro_bodex_objbal.pt` | 14999 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_shadow_allegro_bodex_objbal_15k.pt` | 14999 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(23).pt` | 14260 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(25).pt` | 10323 | 16 | 1024 | spatial | r6 | 8 | 2.12 | True | 0.1 | False | 21266 |
| `stage1_best_total(22).pt` | 9185 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(21).pt` | 6790 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |

## robots = `shadow`  (n=7)

| archivo | step | z_dim | shared_dim | human_encoder | human_rot_repr | temporal_window | lam_udhm | xin_switching | extra_human_ratio | primitive_sample | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage1_best_total(16).pt` | 14505 | 16 | 1024 |  | r6 |  |  | False | 0.1 | False | 276 |
| `stage1_best_total(18):A.pt` | 14496 | 16 | 1024 |  | r6 |  | 1.2 | False | 0.1 | False | 21266 |
| `stage1_best_total(18):b.pt` | 14412 | 16 | 1024 |  | r6 |  | 2.12 | False | 0.1 | False | 21266 |
| `stage1_shadow_udhm_B_15k.pt` | 14412 | 16 | 1024 |  | r6 |  | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(17).pt` | 13768 | 16 | 1024 |  | r6 |  |  | False | 0.1 | False | 97973 |
| `stage1_shadow_solo_ckpt17_run18b_13k7.pt` | 13768 | 16 | 1024 |  | r6 |  |  | False | 0.1 | False | 97973 |
| `stage1_best_total(18).pt` | 5882 | 16 | 1024 | temporal_cam | r6 | 8 | 0.0 | False | 0.1 | False | 26101 |

## robots = `allegro`  (n=6)

| archivo | step | z_dim | shared_dim | human_encoder | human_rot_repr | temporal_window | lam_udhm | xin_switching | extra_human_ratio | primitive_sample | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage1_allegro_bodex_objbal_udhm.pt` | 14999 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_allegro_solo_bodex_objbal_15k.pt` | 14999 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(24).pt` | 14553 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_allegro_solo_bodex_noobjbal_ckpt24_14k5.pt` | 14553 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(14).pt` | 5907 | 16 | 1024 |  | r6 |  |  | False | 0.1 | False | 9663 |
| `stage1_allegro_freeze.pt` | 2500 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |

## robots = `shadow+leap+barrett`  (n=3)

| archivo | step | z_dim | shared_dim | human_encoder | human_rot_repr | temporal_window | lam_udhm | xin_switching | extra_human_ratio | primitive_sample | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage1_best_total(20).pt` | 13698 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(19).pt` | 9885 | 16 | 1024 | spatial | r6 | 8 | 2.12 | False | 0.1 | False | 21266 |
| `stage1_best_total(15).pt` | 5935 | 16 | 1024 |  | r6 |  |  | False | 0.1 | False | 16074 |

## robots = `shadow-legacy`  (n=59)

Formato viejo (pre multi-robot, `E_r`/`D_r` sueltos, sin dict `robots`). Para evaluarlos con `eval_retarget.py` hay que pasar `--robots shadow` explicito. Solo se listan los 15 de step mas alto -- el resto son intentos tempranos (step < 10k) de la misma familia de runs.

| archivo | step | z_dim | shared_dim | human_encoder | human_rot_repr | temporal_window | lam_udhm | xin_switching | extra_human_ratio | primitive_sample | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage1_best_total(6).pt` | 14957 | 16 | 1024 |  |  |  |  |  | 0.1 |  | 15825 |
| `stage1_best_total(10).pt` | 14874 | 16 | 1024 |  |  |  |  |  | 0.1 |  | 21266 |
| `stage1_latest_run29fix.pt` | 14500 |  |  |  |  |  |  |  |  |  |  |
| `stage1_best_run17.pt` | 14472 |  |  |  |  |  |  |  |  |  |  |
| `stage1_best_run18.pt` | 14462 |  |  |  |  |  |  |  |  |  |  |
| `stage1_best_run18b_15k.pt` | 14462 |  |  |  |  |  |  |  |  |  |  |
| `stage1_best_total(9).pt` | 14237 | 64 | 1024 |  |  |  |  |  | 0.1 |  | 21266 |
| `stage1_best_val(2).pt` | 14000 | 64 | 1024 |  |  |  |  |  | 0.1 |  | 21266 |
| `stage1_best_val(27b).pt` | 14000 | 64 | 1024 |  |  |  |  |  | 0.1 |  | 21266 |
| `stage1_best_total(11).pt` | 13792 | 16 | 1024 |  |  |  |  |  | 0.1 |  | 93901 |
| `stage1_best_total_run29fix.pt` | 13607 | 64 | 1024 |  |  |  |  |  | 0.1 |  | 21266 |
| `stage1_best_run20.pt` | 11000 | 16 | 1024 |  |  |  |  |  | 0.1 |  | 21266 |
| `stage1_run34_global-oracle_step10811.pt` | 10811 | 16 | 1024 |  |  |  |  |  | 0.1 |  | 21266 |
| `stage1_best_run18b_10k.pt` | 10463 |  |  |  |  |  |  |  |  |  |  |
| `stage1_best_run15.pt` | 10396 |  |  |  |  |  |  |  |  |  |  |

## Ya evaluados y en la tesis (Resultados.tex, tab:fidelidad-retargeting)

- `stage1_shadow_solo_ckpt17_run18b_13k7.pt` (step 13768, shadow) -- Solo Shadow
- `stage1_allegro_solo_bodex_objbal_15k.pt` (step 14999, allegro) -- Solo Allegro, BODex balanceado
- `stage1_allegro_solo_bodex_noobjbal_ckpt24_14k5.pt` (step 14553, allegro) -- Solo Allegro, BODex sin balancear
- `stage1_shadow_allegro_bodex_objbal_15k.pt` (step 14999, shadow+allegro) -- Multi-robot


## Barrido de eval -- YA CORRIDO

Ver `eval_sweep_2026-09-01_resultados.md` (log crudo en
`eval_sweep_2026-09-01.log`). Resumen: `barrett` falla siempre (bug real,
no del checkpoint); `stage1_best_total(14).pt` (allegro, step 5907) domina
en RS y NDS a los dos Allegro-solo de la tesis, sin investigar todavia por
que -- no se toco la tabla de tesis por esto.

## Candidatos para el barrido de eval (siguiente paso) -- OBSOLETO, ver arriba

Prioridad: step alto + config distinta a lo ya evaluado. Se corre con `eval_retarget.py --ckpt <f> --csv <r6.csv> --n_batches 10 --b_eval 3000` (mismos parametros que generaron la tabla actual).
- `shadow+allegro`: `stage1_best_total(23).pt` (step 14260) -- multi-robot alterno, no evaluado aun.
- `shadow+leap+barrett`: `stage1_best_total(20).pt` (step 13698) -- 3 robots, familia nunca evaluada en la tesis.
- `shadow-legacy`: `stage1_best_total(6).pt` (step 14957), `stage1_best_total(10).pt` (step 14874) -- podrian superar al shadow-solo actual (13.7k) por tener mas steps, pero son runs viejos, confirmar que el codigo de entonces sigue siendo compatible antes de fiarse del numero.
