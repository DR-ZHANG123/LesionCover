"""基础模型编码器：冻结的 ViT（DINOv2 ViT-L/14 等）+ LoRA 适配器（只训练注意力 qkv 的低秩增量）。
输出三层 patch 特征图（中间两层 + 末层），接口与 timm features_only 的 (f1, f2, f3) 相同，供 LesionCover 网络
在其上取全局 token 与病灶局部外观。输入为 0–255 的全局图，内部缩放到 img_size。"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale


class _Info:
    def __init__(self, c):
        self._c = c

    def channels(self):
        return self._c

    def reduction(self):
        return [14, 14, 14]


class FMEncoder(nn.Module):
    def __init__(self, name: str, img_size: int, layers=(11, 17, 23), r: int = 8, weights: str | None = None):
        super().__init__()
        self.net = timm.create_model(name, pretrained=weights is None, num_classes=0, img_size=img_size)
        if weights:                                    # 本地权重（如 RETFound DINOv2 的 teacher 主干），键名与 timm ViT 对齐
            from timm.layers import resample_abs_pos_embed
            sd = torch.load(weights, map_location="cpu", weights_only=False)
            sd = sd.get("teacher", sd.get("model", sd))
            sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.") and "mask_token" not in k}
            g = self.net.patch_embed.grid_size
            sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], g, num_prefix_tokens=self.net.num_prefix_tokens)
            res = self.net.load_state_dict(sd, strict=False)
            assert not res.missing_keys and not res.unexpected_keys, res
        for p in self.net.parameters():
            p.requires_grad_(False)
        for blk in self.net.blocks:                    # r = 0：完全冻结，不加适配器
            if r > 0:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, r)
        self.size, self.layers = img_size, tuple(layers)
        c = self.net.embed_dim
        self.feature_info = _Info([c, c, c])
        self.pretrained_cfg = self.net.pretrained_cfg

    def forward(self, x, with_cls: bool = False):
        """x 已标准化（B,3,S,S）。with_cls 时另返回末层 [CLS]。"""
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        n = self.net
        h = n.norm_pre(n.patch_drop(n._pos_embed(n.patch_embed(x))))
        out = []
        for i, blk in enumerate(n.blocks):
            h = checkpoint(blk, h, use_reentrant=False) if self.training else blk(h)
            if i in self.layers:
                out.append(n.norm(h) if i == len(n.blocks) - 1 else h)
        k = n.num_prefix_tokens; s = self.size // 14
        maps = [o[:, k:].transpose(1, 2).reshape(o.shape[0], -1, s, s).contiguous() for o in out]
        return (maps, out[-1][:, 0]) if with_cls else maps


class FMTokens(nn.Module):
    """双编码器融合中的基础模型分支：基础模型末层 patch 特征池化成 G×G 个语义 token，另加 [CLS] token，
    经瓶颈 adapter（LayerNorm → 降维 → GELU → 升到 d）投影到 LesionCover 的 token 空间。r = 0 时基础模型完全冻结。"""

    def __init__(self, name: str, img_size: int, d: int, G: int, r: int = 0, weights: str | None = None,
                 bottleneck: int = 256):
        super().__init__()
        self.enc = FMEncoder(name, img_size, layers=(23,), r=r, weights=weights)
        self.frozen, self.G = r == 0, G
        c = self.enc.net.embed_dim
        self.adapter = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, bottleneck), nn.GELU(), nn.Linear(bottleneck, d))
        self.cls_adapter = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, bottleneck), nn.GELU(), nn.Linear(bottleneck, d))
        self.type_emb = nn.Parameter(torch.zeros(d))
        dc = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(dc["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(dc["std"]).view(1, 3, 1, 1) * 255)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.enc.eval()
        return self

    def forward(self, g, pos):
        """g：0–255 全局图；pos：(G², d) 位置编码。返回 (B, G²+1, d)。"""
        x = (g.float() - self.mean) / self.std
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            (m,), c = self.enc(x, with_cls=True)
        m = F.avg_pool2d(m.float(), m.shape[-1] // self.G).flatten(2).transpose(1, 2)
        t = self.adapter(m.to(self.adapter[1].weight.dtype)) + pos[None] + self.type_emb
        return torch.cat([self.cls_adapter(c.float()).unsqueeze(1) + self.type_emb, t], 1)
