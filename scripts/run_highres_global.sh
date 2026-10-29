#!/usr/bin/env bash
# 1024 像素全局图 + 病灶覆盖裁块（variant hg）的完整流程：训练（scripts/run_queue.sh，config/grading_runs_hg.txt）完成后，
# 全精度对数概率（07g，五个评价队列）→ 去除相机复现斑点后的预测（07h，含 1024 全局图）→ 比较（08q）。GPUS 默认 1 2 3。
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a GPUS <<< "${GPUS:-1 2 3}"
SEEDS=(11 22 33)
trained() { for s in "${SEEDS[@]}"; do [ -f "results/07_classify/C_k4_s${s}_auxhg/PROVENANCE.json" ] || return 1; done; }
until trained; do sleep 120; done
for i in 0 1 2; do
  s=${SEEDS[$i]}
  ( CUDA_VISIBLE_DEVICES=${GPUS[$i]} python scripts/stages/07g_referable_scores.py --tags C_k4_s${s}_auxhg > logs/07g_hg_s$s.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python scripts/stages/07h_predict_camera.py --tags C_k4_s${s}_auxhg G1024_s$s > logs/07h_hg_s$s.log 2>&1 ) &
done
wait
python scripts/stages/08q_highres_global.py > logs/08q_highres_global.log 2>&1
