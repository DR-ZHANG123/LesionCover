"""运行标签中训练设置（variant）字符串的解析。variant 由已知记号依次拼接而成，例如
auxmcsmall = aux + mc + small，auxmcbaseema = aux + mc + base + ema。

记号含义：
  aux    裁块分支辅助分级头          gdrop  训练时遮掉全局特征
  mc     训练集加入 APTOS 2019        art    跨相机伪影增强
  hn     易误读健康照片加权抽样        neg    健康照片阴性监督分割器所选裁块
  small  ConvNeXt-Small               base   ConvNeXt-Base
  effm   EfficientNetV2-M             dino   DINOv2 ViT-B/14（自然图像自监督）
  rf     RETFound ViT-L/16（彩色眼底自监督）  hr  ViT 编码器按原分辨率输入（不缩到 224）
  ema    以权重指数滑动平均模型做选择与预测    ord   加期望等级的平方误差（有序）损失
  hg     全局图用 1024 像素（global_size_large），裁块不变
"""
from __future__ import annotations

TOKENS = ("aux", "gdrop", "mc", "art", "hn", "hg", "neg", "small", "base", "effm", "dino", "rf", "hr", "ema", "ord")


def tokens(var: str) -> set[str]:
    out, i = set(), 0
    while i < len(var):
        for t in TOKENS:
            if var.startswith(t, i):
                out.add(t); i += len(t)
                break
        else:
            raise ValueError(f"unknown variant {var!r} at {var[i:]!r}")
    return out


def encoder_name(var: str, cfg: dict, root) -> str:
    """返回 Grader 的 backbone 参数：timm 模型名，或 RETFound 权重路径（含 hr 时加后缀 @hr）。"""
    cc, t = cfg["classifier"], tokens(var)
    if "rf" in t:
        return str(root / cc["backbone_rf"]) + ("@hr" if "hr" in t else "")
    for key, field in (("small", "backbone_large"), ("base", "backbone_base"), ("effm", "backbone_effm"),
                       ("dino", "backbone_dino")):
        if key in t:
            return cc[field]
    return cc["backbone"]


def global_size(method: str, var: str, cfg: dict) -> int:
    """全局图边长：仅全局的 G1024，或 variant 含 hg 的裁块输入，用 global_size_large；其余用 global_size。"""
    pc = cfg["preprocess"]
    return pc["global_size_large"] if method == "G1024" or "hg" in tokens(var or "") else pc["global_size"]
