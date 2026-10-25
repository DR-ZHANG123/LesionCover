"""多相机训练补充集 APTOS 2019：从 HuggingFace 镜像（sngsfydy/aptos，Kaggle APTOS 2019 训练集 3,662 张照片及其五级分级）
下载 parquet，按原文件名把照片原字节写到 data/raw/APTOS/images/，分级写入 labels.csv；
每个 parquet 与每张照片的 SHA-256 记录在 results/00_download/aptos_files.tsv。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config  # noqa: E402

REPO = "sngsfydy/aptos"
REVISION = "main"


def main():
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw_aptos"]
    hf = raw.parent / "APTOS_hf"
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=str(hf), max_workers=8)
    (raw / "images").mkdir(parents=True, exist_ok=True)
    rows, files = [], []
    for f in sorted((hf / "data").glob("*.parquet")):
        files.append({"file": f"parquet/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
        t = pq.read_table(f)
        for im, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            name = Path(im["path"]).name
            (raw / "images" / name).write_bytes(im["bytes"])
            rows.append({"image": name, "grade": int(lab)})
            files.append({"file": f"images/{name}", "sha256": hashlib.sha256(im["bytes"]).hexdigest()})
    lab = pd.DataFrame(rows).sort_values("image")
    assert lab.image.is_unique
    lab.to_csv(raw / "labels.csv", index=False)
    out = ROOT / "results/00_download"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(files).to_csv(out / "aptos_files.tsv", sep="\t", index=False)
    print(len(lab), lab.grade.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
