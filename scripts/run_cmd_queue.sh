#!/usr/bin/env bash
# 通用 GPU 作业队列：queue 文件每行 "<日志名> <命令>"；每条 lane 绑定一张 GPU，空闲显存 ≥ MIN_FREE_MB 时领取下一条。
# 用法：bash scripts/run_cmd_queue.sh <queue_file> <gpu_id>   （同一 queue 可起多条 lane）
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
Q="$1"; GPU="$2"; MIN_FREE_MB="${MIN_FREE_MB:-24000}"
source "${CONDA_BASE:-$(conda info --base)}/etc/profile.d/conda.sh"; conda activate lesioncover
while true; do
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")" -ge "$MIN_FREE_MB" ]; do sleep 60; done
  job=$(flock "$Q.lock" bash -c "line=\$(grep -v '^#' '$Q' | grep -v '^taken ' | head -1); [ -n \"\$line\" ] && { n=\$(grep -n -F -x -- \"\$line\" '$Q' | head -1 | cut -d: -f1); sed -i \"\${n}s/^/taken /\" '$Q'; }; echo \"\$line\"")
  [ -z "$job" ] && break
  name="${job%% *}"; cmd="${job#* }"
  if CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash -c "$cmd" > "logs/$name.log" 2>&1; then
    echo "done $name"
  else
    echo "FAILED $name" | tee -a logs/queue_failures.log
  fi
  sleep 30
done
