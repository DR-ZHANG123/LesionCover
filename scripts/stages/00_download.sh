#!/usr/bin/env bash
# 下载 DDR（Google Drive 官方分卷）与 DeepDRiD（官方 GitHub 仓库），合并解压并记录 MD5。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW/DDR" "$ROOT/results/00_download"
cd "$RAW/DDR"
if [[ ! -d DDR_dataset/DDR-dataset ]]; then
  python -m gdown --folder --continue https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC
  cd DDR_dataset && cat DDR-dataset.zip.0* > DDR-dataset.zip && unzip -q DDR-dataset.zip && rm DDR-dataset.zip && cd ..
fi
cd "$RAW"
[[ -d DeepDRiD ]] || git clone --depth 1 https://github.com/deepdrdoc/DeepDRiD.git
git -C DeepDRiD rev-parse HEAD > "$ROOT/results/00_download/deepdrid_commit.txt"
md5sum DDR/DDR_dataset/DDR-dataset.zip.0* > "$ROOT/results/00_download/ddr_archive_md5.txt"
echo '{"stage": "00_download"}' > "$ROOT/results/00_download/PROVENANCE.json"
