#!/bin/bash
source /home/yareeez/AIST-hand/.venv/bin/activate
python3 /home/yareeez/AIST-hand/grasp-model/scripts/viz_retarget.py \
    --ckpt  /home/yareeez/AIST-hand/grasp-model/checkpoints/stage1_best_run4_subspaces.pt \
    --csv   /home/yareeez/AIST-hand/grasp-model/data/processed/hograspnet_dong.csv \
    --grasp 14 \
    --n     16 \
    --fps   1
