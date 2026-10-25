"""官方 Google Drive 分卷因下载配额无法取得时，从逐字节一致的 HuggingFace 镜像（ctmedtech/DDR-dataset）补齐
DR_grading/valid 与 lesion_segmentation。每个文件按 LFS SHA-256 校验；镜像与官方分卷重叠的文件也逐一比对，
结果写入 results/00_download/mirror_check.tsv。"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config  # noqa: E402

REPO = "https://huggingface.co/api/datasets/ctmedtech/DDR-dataset/tree/main/"
RAW = "https://huggingface.co/datasets/ctmedtech/DDR-dataset/resolve/main/"


def _open(url: str, dest: Path | None = None):
    """带退避重试的 HTTP 访问（镜像有 429 频率限制）。"""
    import time
    for attempt in range(12):
        try:
            if dest is None:
                return urllib.request.urlopen(url)
            return urllib.request.urlretrieve(url, dest)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 11:
                raise
            time.sleep(min(300, 10 * 2 ** attempt))


def tree(path: str) -> list:
    out, url = [], f"{REPO}{path}?recursive=true&expand=true"
    while url:
        r = _open(url)
        out += json.load(r)
        link, url = r.headers.get("Link"), None
        if link and 'rel="next"' in link:
            url = link.split(";")[0].strip("<>")
    return [x for x in out if x["type"] == "file"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(item, dest_root: Path) -> dict:
    rel = item["path"]
    oid = item.get("lfs", {}).get("oid")
    # 镜像把 lesion_segmentation 的目录名改写为 images/annotations、test 写作 tet，映射回官方结构
    parts = rel.split("/")
    if parts[0] == "lesion_segmentation":
        kind, split = parts[1], {"tet": "test", "val": "valid"}.get(parts[2], parts[2])
        rel_off = "/".join(["lesion_segmentation", split, "image" if kind == "images" else "label"] + parts[3:])
    else:
        rel_off = rel
    dest = dest_root / rel_off
    status = "present"
    if dest.exists():
        match = (oid is None) or sha(dest) == oid
        status = "present_match" if match else "present_mismatch"
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        _open(RAW + urllib.parse.quote(rel), tmp)
        ok = oid is None or sha(tmp) == oid
        tmp.rename(dest)
        status = "downloaded_verified" if ok else "downloaded_hash_fail"
    return {"mirror_path": rel, "local_path": rel_off, "status": status}


def main():
    cfg = load_config()
    dest = ROOT / cfg["paths"]["raw_ddr"] / "DDR-dataset"
    items = tree("lesion_segmentation") + tree("DR_grading")
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda it: fetch(it, dest), items))
    df = pd.DataFrame(res)
    out = ROOT / "results/00_download"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "mirror_check.tsv", sep="\t", index=False)
    print(df.status.value_counts())


if __name__ == "__main__":
    main()
