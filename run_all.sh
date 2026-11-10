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
  "00_download|00_download|bash scripts/stages/00_download.sh && python scripts/stages/00_fetch_ddr_mirror.py && python scripts/stages/00_fetch_messidor2.py && python scripts/stages/00_fetch_messidor2_pairs.py && python scripts/stages/00_fetch_eyepacs.py && python scripts/stages/00_fetch_mmrdr.py"
  "01_preprocess|01_preprocess|python scripts/stages/01_preprocess.py"
  "01b_mmrdr|01b_mmrdr|python scripts/stages/01b_mmrdr.py"
  "02_manifest|02_manifest|python scripts/stages/02_manifest.py"
  "03_segmenter|03_segmenter|python scripts/stages/03_segmenter.py"
  "03b_seg_eval|03b_seg_eval|python scripts/stages/03b_seg_eval.py"
  "04_probmaps|04_probmaps|python scripts/stages/04_probmaps.py"
  "04_probmaps_mmrdr|04_probmaps_mmrdr|python scripts/stages/04_probmaps.py --manifest results/01b_mmrdr/images.tsv --stage 04_probmaps_mmrdr"
  "04b_calibration|04b_calibration|python scripts/stages/04b_calibration.py"
  "05e_lesion_tokens|05e_lesion_tokens|python scripts/stages/05e_lesion_tokens.py"
  "05e_lesion_tokens_mmrdr|05e_lesion_tokens_mmrdr|python scripts/stages/05e_lesion_tokens.py --manifest results/01b_mmrdr/images.tsv --stage 05e_lesion_tokens_mmrdr"
  "07_models|07n_lcnet/LC_fmrglearn_s33|cp config/lesioncover_runs.txt logs/queue_lc.txt && bash scripts/run_cmd_queue.sh logs/queue_lc.txt ${CUDA_VISIBLE_DEVICES%%,*}"
  "08s_lcnet_eval|08s_lcnet_eval|python scripts/stages/08s_lcnet_eval.py"
  "08t_dev_compare|08t_dev_compare|python scripts/stages/08t_dev_compare.py"
  "08u_grading_standard|08u_grading_standard|python scripts/stages/08u_grading_standard.py"
  "08v_local_calibration|08v_local_calibration|python scripts/stages/08v_local_calibration.py"
  "08w_ruler_check|08w_ruler_check|python scripts/stages/08w_ruler_check.py"
  "08x_learned_rule|08x_learned_rule|python scripts/stages/08x_learned_rule.py"
  "08y_error_direction|08y_error_direction|python scripts/stages/08y_error_direction.py"
  "08z_shift_robustness|08z_shift_robustness|python scripts/stages/08z_shift_robustness.py"
  "08za_interpretability|08za_interpretability|for s in 11 22 33; do python scripts/stages/08za_interpretability.py --seed $s; done; python scripts/stages/08za_interpretability.py --merge"
  "10_tables|10_tables|python scripts/stages/10_tables.py"
)

for entry in "${STAGES[@]}"; do
  IFS='|' read -r name dir cmd <<< "$entry"
  if [[ ",$SKIP," == *",$name,"* ]]; then echo "[skip] $name"; continue; fi
  if [[ "$MODE" == resume && -f "results/$dir/PROVENANCE.json" ]]; then echo "[done] $name"; continue; fi
  echo "[run ] $name"
  eval "$cmd" 2>&1 | tee "logs/$name.log"
done

echo "[figs] figures"
for f in scripts/figures/Fig[0-9]*_*.py scripts/figures/FigS_supplement.py; do
  [ -f "$f" ] && python "$f"
done
