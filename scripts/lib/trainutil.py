"""分级器训练的辅助部件：优化器参数组（ViT 逐层学习率衰减）、权重指数滑动平均（EMA）与有序损失。"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def param_groups(model: nn.Module, cc: dict) -> list[dict]:
    """ViT 编码器（有 blocks 属性）：第 i 个 Transformer 块的学习率 = lr × layer_decay^(depth + 1 − i)；
    卷积编码器：整体使用 lr。聚合与分类头使用 head_lr。"""
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    enc = model.enc
    if "layer_decay" in cc and hasattr(enc, "blocks"):
        depth = len(enc.blocks)

        def layer_id(n):
            if n.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token", "mask_token")):
                return 0
            return int(n.split(".")[1]) + 1 if n.startswith("blocks.") else depth + 1
        groups: dict[int, list] = {}
        for n, p in enc.named_parameters():
            groups.setdefault(layer_id(n), []).append(p)
        enc_groups = [{"params": ps, "lr": cc["lr"] * cc["layer_decay"] ** (depth + 1 - i)} for i, ps in groups.items()]
    else:
        enc_groups = [{"params": list(enc.parameters()), "lr": cc["lr"]}]
    return enc_groups + [{"params": head, "lr": cc["head_lr"]}]


class Ema:
    """权重指数滑动平均：衰减率随步数升到 decay（min(decay, (1+t)/(10+t))），缓冲区（如 BN 统计）直接复制。"""

    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.t = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = min(self.decay, (1 + self.t) / (10 + self.t)); self.t += 1
        src = [p for p in model.parameters()]
        dst = [p for p in self.model.parameters()]
        torch._foreach_lerp_(dst, src, 1 - d)
        for b_dst, b_src in zip(self.model.buffers(), model.buffers()):
            b_dst.copy_(b_src)


def ordinal_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """期望等级（softmax 概率 × 等级）与参考等级的平方误差：惩罚远离参考等级的概率质量。"""
    grades = torch.arange(logit.shape[1], device=logit.device, dtype=torch.float32)
    e = (F.softmax(logit.float(), 1) * grades).sum(1)
    return ((e - y.float()) ** 2).mean()
