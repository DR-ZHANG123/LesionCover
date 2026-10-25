"""可复现性工具：读配置、固定随机种子、写 PROVENANCE.json。每个分析 stage 都调用。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "run.yaml"


def load_config() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def config_sha() -> str:
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:16]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # GPU 上的确定性计算：同一种子重跑得到相同结果
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_provenance(stage: str, inputs: list[Path], outputs: list[Path], seed: int | None = None,
                     extra: dict | None = None) -> Path:
    """在 results/<stage>/PROVENANCE.json 记录输入输出 MD5、种子、配置 SHA、git commit 与环境。"""
    out_dir = ROOT / "results" / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "stage": stage,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "config_sha": config_sha(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "inputs": {str(Path(p).relative_to(ROOT) if Path(p).is_relative_to(ROOT) else p): md5(Path(p))
                   for p in inputs if Path(p).is_file()},
        "outputs": {str(Path(p).relative_to(ROOT)): md5(Path(p)) for p in outputs if Path(p).is_file()},
    }
    if extra:
        rec.update(extra)
    path = out_dir / "PROVENANCE.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return path
