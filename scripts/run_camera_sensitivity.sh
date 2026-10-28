#!/usr/bin/env bash
# 相机复现斑点的敏感性分析（补充表）：阈值 3%/10%（半径 4 格）与半径 2/8 格（阈值 5%），只重算 Messidor-2 与 IDRiD，
# 只用全局 512 与病灶覆盖的三个模型。每个设置一张 GPU（GPUS 环境变量，默认 1 2 3 4）。
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a GPUS <<< "${GPUS:-1 2 3 4}"
SETTINGS=("t03 --threshold 0.03" "t10 --threshold 0.10" "r2 --radius 2" "r8 --radius 8")
TAGS="G512_s11 G512_s22 G512_s33 C_k4_s11_aux C_k4_s22_aux C_k4_s33_aux"
i=0
for s in "${SETTINGS[@]}"; do
  name=${s%% *}; opt=${s#* }
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  ( python scripts/stages/05d_camera_prior.py $opt --roles messidor2 idrid --out 05d_camera_prior_$name > logs/05d_camera_prior_$name.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$gpu python scripts/stages/07h_predict_camera.py --tags $TAGS --prior 05d_camera_prior_$name \
      --out 07h_predict_camera_$name --roles messidor2 idrid > logs/07h_predict_camera_$name.log 2>&1 ) &
done
wait
