"""下载 MMRDR（figshare 29423747，CC BY 4.0）：按 figshare API 列出的分卷逐个下载并核对 MD5，拼接后只解出彩色眼底子集
（MMRDR-CFP）到 data/raw/MMRDR/。另按 config published_methods 的固定 commit 取得 GDRBench 官方代码（07p 使用）。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config  # noqa: E402

ARTICLE = 29423747


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    cfg = load_config()
    out = ROOT / "data/raw/MMRDR"; out.mkdir(parents=True, exist_ok=True)
    files = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{ARTICLE}/files"))
    parts = sorted([f for f in files if f["name"].startswith("MMRDR.zip.")], key=lambda f: f["name"])
    if not (out / "MMRDR-CFP").exists():
        for f in parts:
            p = out / f["name"]
            want = f.get("computed_md5") or f.get("supplied_md5")
            if not p.exists() or md5(p) != want:
                subprocess.run(["curl", "-L", "--retry", "5", "-o", str(p), f["download_url"]], check=True)
            assert md5(p) == want, f["name"]
        whole = out / "MMRDR.zip"
        with open(whole, "wb") as w:
            for f in parts:
                with open(out / f["name"], "rb") as r:
                    for b in iter(lambda: r.read(1 << 24), b""):
                        w.write(b)
        subprocess.run(["unzip", "-q", "-o", str(whole), "MMRDR-CFP/*", "-d", str(out)], check=True)
    pm = cfg["published_methods"]
    code = ROOT / pm["code_dir"]
    if not code.exists():
        subprocess.run(["git", "clone", "https://github.com/chehx/DGDR", str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "checkout", "-q", pm["code_commit"]], check=True)


if __name__ == "__main__":
    main()
