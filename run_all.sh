#!/usr/bin/env bash
# 总控：按依赖顺序运行全部分析 stage 与出图脚本。
# 用法：bash run_all.sh [--resume|--force] [--skip=stageA,stageB]
#   --resume（默认）：results/<stage>/PROVENANCE.json 已存在则跳过该 stage
# 环境：conda 环境 lesioncover（environment.yml）；GPU 由 CUDA_VISIBLE_DEVICES 指定（默认 0）。
# 07_classify 的全部配置（方法 × 种子 × K）列在 config/grading_runs_all.txt（方法 种子 K [训练方式]），可用 scripts/run_queue.sh 多 lane 并行。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
MODE=resume; SKIP=""
for a in "$@"; do
  case "$a" in
    --force) MODE=force ;;
    --resume) MODE=resume ;;
    --skip=*) SKIP="${a#--skip=}" ;;
  esac
done
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
export CONDA_BASE
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate lesioncover
mkdir -p logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

STAGES=(
  "00_download|00_download|bash scripts/stages/00_download.sh && python scripts/stages/00_fetch_ddr_mirror.py && python scripts/stages/00_fetch_messidor2.py && python scripts/stages/00_fetch_messidor2_pairs.py && python scripts/stages/00_fetch_aptos.py && python scripts/stages/00_fetch_eyepacs.py"
  "01_preprocess|01_preprocess|python scripts/stages/01_preprocess.py"
  "02_manifest|02_manifest|python scripts/stages/02_manifest.py"
  "03_segmenter|03_segmenter|python scripts/stages/03_segmenter.py"
  "03b_seg_eval|03b_seg_eval|python scripts/stages/03b_seg_eval.py"
  "04_probmaps|04_probmaps|python scripts/stages/04_probmaps.py"
  "04b_calibration|04b_calibration|python scripts/stages/04b_calibration.py"
  "05_selection|05_selection|python scripts/stages/05_selection.py"
  "06_coverage|06_coverage|python scripts/stages/06_coverage.py"
  "06b_detection_limit|06b_detection_limit|python scripts/stages/06b_detection_limit.py"
  "07_classify|07_classify|cp config/grading_runs_all.txt logs/queue_all.txt && bash scripts/run_queue.sh logs/queue_all.txt ${CUDA_VISIBLE_DEVICES%%,*}"
  "07c_predict_idrid|07c_predict_idrid|python scripts/stages/07c_predict_idrid.py && python scripts/stages/07c_predict_idrid.py --role messidor2 --croponly && python scripts/stages/07c_predict_idrid.py --role eyepacs --croponly --tags $(cat config/eyepacs_tags.txt)"
  "07d_crop_only|07d_crop_only|python scripts/stages/07d_crop_only.py"
  "08_stats|08_stats|python scripts/stages/08_stats.py"
  "08b_attention|08b_attention|python scripts/stages/08b_attention.py"
  "08c_variants|08c_variants|python scripts/stages/08c_variants.py"
  "08d_crop_only|08d_crop_only|python scripts/stages/08d_crop_only.py"
  "08e_dose_response|08e_dose_response|python scripts/stages/08e_dose_response.py"
  "08f_cases|08f_cases|python scripts/stages/08f_cases.py"
  "08g_healthy_spots|08g_healthy_spots|python scripts/stages/08g_healthy_spots.py"
  "04c_adabn_probmaps|04c_adabn_probmaps|python scripts/stages/04c_adabn_probmaps.py --stats-from ddr_train --roles seg_dev ddr_test && for r in deepdrid idrid messidor2; do python scripts/stages/04c_adabn_probmaps.py --stats-from \$r --roles \$r; done && python scripts/stages/04b_calibration.py --subdir probmaps_adabn --out 04b_calibration_adabn"
  "05b_adabn_crops|05b_adabn_crops|python scripts/stages/05b_adabn_crops.py"
  "07e_predict_adabn|07e_predict_adabn|python scripts/stages/07e_predict_adabn.py"
  "08h_adabn|08h_adabn|python scripts/stages/08h_adabn.py"
  "08c_variants_mc|08c_variants_mc|python scripts/stages/08c_variants.py --extra"
  "08g_settings|08g_healthy_spots_auxmchn|for v in auxsmall auxbase auxmc auxmcsmall auxmcbase auxmcart auxmchn; do python scripts/stages/08g_healthy_spots.py --var \$v; done"
  "07f_tta|07f_tta|python scripts/stages/07f_tta.py --tags G512_s{11,22,33} G512_s{11,22,33}_mc C_k4_s{11,22,33}_aux C_k4_s{11,22,33}_auxmc N_k4_s{11,22,33}_aux N_k4_s{11,22,33}_auxmc"
  "08i_dev_screen|08i_dev_screen|python scripts/stages/08i_dev_screen.py"
  "08j_capacity|08j_capacity|python scripts/stages/08j_capacity.py"
  "08k_occlusion|08k_occlusion|python scripts/stages/08k_occlusion.py"
  "08l_screening|08l_screening|python scripts/stages/08l_screening.py"
  "07g_referable_scores|07g_referable_scores|python scripts/stages/07g_referable_scores.py --tags G512_s{11,22,33} G1024_s{11,22,33} N_k4_s{11,22,33}_aux D_k4_s{11,22,33}_aux C_k4_s{11,22,33}_aux"
  "08m_referable_roc|08m_referable_roc|python scripts/stages/08m_referable_roc.py"
  "09_cost|09_cost|python scripts/stages/09_cost.py"
)

for entry in "${STAGES[@]}"; do
  IFS='|' read -r name dir cmd <<< "$entry"
  if [[ ",$SKIP," == *",$name,"* ]]; then echo "[skip] $name"; continue; fi
  if [[ "$MODE" == resume && -f "results/$dir/PROVENANCE.json" ]]; then echo "[done] $name"; continue; fi
  echo "[run ] $name"
  eval "$cmd" 2>&1 | tee "logs/$name.log"
done

echo "[figs] figures"
for f in scripts/figures/Fig[0-9]*.py scripts/figures/FigS*.py; do
  [ -f "$f" ] && python "$f"
done
