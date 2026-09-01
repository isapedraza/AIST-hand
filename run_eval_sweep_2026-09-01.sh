#!/usr/bin/env bash
set -uo pipefail
cd /home/yareeez/AIST-hand
CSV=/home/yareeez/6d_datasets/hograspnet_abl14_6d.csv
OUT=docs/eval_sweep_2026-09-01.log
: > "$OUT"

run() {
  local ckpt="$1"; shift
  local robots="$1"; shift
  echo "=========================================" | tee -a "$OUT"
  echo "CKPT=$ckpt ROBOTS=$robots  $(date -Iseconds)" | tee -a "$OUT"
  .venv/bin/python3 models/latent-retargeting/scripts/eval_retarget.py \
    --ckpt "$ckpt" --csv "$CSV" --robots $robots \
    --n_batches 10 --b_eval 3000 >> "$OUT" 2>&1
}

D=~/Downloads
run "models/latent-retargeting/checkpoints/active/stage1_shadow_solo_ckpt17_run18b_13k7.pt" "shadow"
run "models/latent-retargeting/checkpoints/active/stage1_allegro_solo_bodex_objbal_15k.pt" "allegro"
run "models/latent-retargeting/checkpoints/active/stage1_allegro_solo_bodex_noobjbal_ckpt24_14k5.pt" "allegro"
run "models/latent-retargeting/checkpoints/active/stage1_shadow_allegro_bodex_objbal_15k.pt" "shadow allegro"
run "$D/stage1_best_total(23).pt" "shadow allegro"
run "$D/stage1_best_total(25).pt" "shadow allegro"
run "$D/stage1_best_total(22).pt" "shadow allegro"
run "$D/stage1_best_total(21).pt" "shadow allegro"
run "$D/stage1_best_total(20).pt" "shadow leap barrett"
run "$D/stage1_best_total(19).pt" "shadow leap barrett"
run "$D/stage1_best_total(15).pt" "shadow leap barrett"
run "$D/stage1_best_total(16).pt" "shadow"
run "$D/stage1_best_total(18):A.pt" "shadow"
run "$D/stage1_best_total(18):b.pt" "shadow"
run "$D/stage1_best_total(12).pt" "shadow"
run "$D/stage1_best_total(13).pt" "shadow"
run "$D/stage1_best_total(14).pt" "allegro"
run "$D/stage1_allegro_freeze.pt" "allegro"

echo "=========================================" | tee -a "$OUT"
echo "SWEEP DONE $(date -Iseconds)" | tee -a "$OUT"
