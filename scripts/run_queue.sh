#!/usr/bin/env bash
# 简单的 GPU 作业队列：queue 文件每行一个 "方法 种子 K"，每条 lane 绑定一张 GPU，循环取下一条未领取的作业。
# 启动前等待该卡空闲显存 ≥ MIN_FREE_MB（共享服务器上避免显存溢出）；失败的作业重新排到队尾（最多 3 次）。
# 用法：bash scripts/run_queue.sh <queue_file> <gpu_id>   （同一 queue 可起多条 lane）
# 作业行："方法 种子 K [variant] [recipe]"，variant 为空而需要 recipe 时写 "-"
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
Q="$1"; GPU="$2"; MIN_FREE_MB="${MIN_FREE_MB:-23000}"
source "${CONDA_BASE:-$(conda info --base)}/etc/profile.d/conda.sh"; conda activate lesion_coverage
mkdir -p logs/classify
while true; do
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")" -ge "$MIN_FREE_MB" ]; do sleep 60; done
  job=$(flock "$Q.lock" bash -c "line=\$(grep -v '^#' '$Q' | grep -v '^taken ' | head -1); [ -n \"\$line\" ] && sed -i \"0,/^\$line\$/s//taken \$line/\" '$Q'; echo \"\$line\"")
  [ -z "$job" ] && break
  read -r m s k v r <<< "$job"; v="${v:-}"; r="${r:-}"; [ "$v" = "-" ] && v=""
  tag="${m}_k${k}_s${s}"; [[ $m == G* ]] && tag="${m}_s${s}"; [ -n "$v" ] && tag="${tag}_$v"; [ -n "$r" ] && tag="${tag}_$r"
  if [ -f "results/07_classify/$tag/PROVENANCE.json" ]; then continue; fi
  if pgrep -f "^python scripts/stages/07_classify.py --method $m --seed $s --k $k${v:+ --variant $v}${r:+ --recipe $r}\$" > /dev/null; then continue; fi   # 已在别的 lane 上运行
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
       python scripts/stages/07_classify.py --method "$m" --seed "$s" --k "$k" ${v:+--variant "$v"} ${r:+--recipe "$r"} > "logs/classify/$tag.log" 2>&1; then
    echo "done $tag"
  else
    n=$(grep -c "^FAILED $tag" logs/queue_failures.log 2>/dev/null || true)
    echo "FAILED $tag" | tee -a logs/queue_failures.log
    [ "${n:-0}" -lt 3 ] && flock "$Q.lock" bash -c "echo '$m $s $k ${v:--} $r' >> '$Q'"
  fi
done
