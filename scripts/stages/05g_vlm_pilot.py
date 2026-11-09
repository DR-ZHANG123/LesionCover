"""视觉语言大模型试点：DDR 开发区按等级分层抽 300 张，请通用视觉语言模型描述病灶并给出 ICDR 等级，
检验其文字信息是否含有分级器之外的信号。接口只经环境变量 OPENAI_BASE_URL / OPENAI_API_KEY 传入。
输出 results/05g_vlm_pilot/{responses.tsv, summary.tsv}。"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

MODEL = "gpt-5.6-terra"
PROMPT = ("Colour fundus photograph. Describe visible diabetic retinopathy lesions (microaneurysms, haemorrhages, "
          "hard exudates, cotton-wool spots, IRMA, venous beading, neovascularisation) and their quadrants in at most "
          "120 words, then give the ICDR grade 0-4 on the last line as 'GRADE: n'.")


def ask(key: str) -> dict:
    img = cv2.imread(str(ROOT / f"data/processed/global1024/{key}.jpg"))
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}]}]}
    for attempt in range(4):
        try:
            req = urllib.request.Request(os.environ["OPENAI_BASE_URL"] + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
            txt = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
            m = re.findall(r"GRADE:\s*([0-4])", txt)
            return {"image_key": key, "vlm_grade": int(m[-1]) if m else -1, "text": txt.replace("\t", " ").replace("\n", " ")}
        except Exception as e:                       # 网络或限流：退避重试
            err = str(e); time.sleep(10 * (attempt + 1))
    return {"image_key": key, "vlm_grade": -1, "text": "ERROR " + err}


def main():
    cfg = load_config()
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    dv = man[(man.role == "ddr_dev") & (man.included == 1)]
    rng = np.random.default_rng(cfg["seed"])
    keys = [k for g, d in dv.groupby("grade") for k in rng.choice(d.image_key.to_numpy(), min(60, len(d)), replace=False)]
    out = ROOT / "results/05g_vlm_pilot"; out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        res = pd.DataFrame(list(ex.map(ask, keys)))
    res = res.merge(dv[["image_key", "grade"]], on="image_key")
    res.to_csv(out / "responses.tsv", sep="\t", index=False)
    write_provenance("05g_vlm_pilot", [ROOT / "results/02_manifest/images.tsv"], [out / "responses.tsv"], seed=cfg["seed"],
                     extra={"model": MODEL, "prompt": PROMPT, "n": len(res)})
    print(res.vlm_grade.value_counts().to_dict())


if __name__ == "__main__":
    main()
