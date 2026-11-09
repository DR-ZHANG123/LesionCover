"""病灶原型聚合（借鉴空间多组学中的原型感知图聚合思路）：每类病灶设 K 个可学习原型，
每个病灶 token 按余弦相似度软分配到本类原型；原型 token = 分配权重加权的病灶表示 + 原型嵌入 + 对数质量，
作为一组“病灶亚型 token”输入 Transformer。原型对比损失把病灶表示拉向其分配原型、推离其他原型，
使同类病灶内部的不同形态（如真实病灶与反光、斑点等伪影）分到不同原型。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LesionPrototypes(nn.Module):
    def __init__(self, d: int, n_les: int, k: int = 8, tau: float = 0.1):
        super().__init__()
        self.n_les, self.k, self.tau = n_les, k, tau
        self.proto = nn.Parameter(torch.randn(n_les * k, d) * 0.02)
        self.proj = nn.Linear(d, d)
        self.mass = nn.Linear(1, d)

    def forward(self, lt, cls, valid):
        """lt (B,M,d)，cls (B,M)，valid (B,M) → 原型 token (B,n_les*k,d)、原型有效 (B,n_les*k)、对比损失（标量）。"""
        z = F.normalize(self.proj(lt).float(), dim=-1)
        p = F.normalize(self.proto.float(), dim=-1)
        sim = z @ p.t() / self.tau                                               # (B,M,L*K)
        own = (torch.arange(self.n_les * self.k, device=lt.device) // self.k)[None, None] == cls[..., None]
        logit = sim.masked_fill(~own, -1e4)
        a = logit.softmax(-1) * valid[..., None].float()                         # 软分配，只在本类原型之间
        m = a.sum(1)                                                             # (B,L*K) 原型质量
        tok = torch.einsum("bmp,bmd->bpd", a, lt.float()) / m.clamp(min=1e-3)[..., None]
        tok = tok + self.proto[None].float() + self.mass(torch.log1p(m)[..., None])
        # 原型对比：以本类原型中相似度最高者为正例，全部原型为候选（InfoNCE）
        tgt = logit.argmax(-1)
        ce = F.cross_entropy(sim.flatten(0, 1), tgt.flatten(), reduction="none").view(valid.shape)
        loss = (ce * valid.float()).sum() / valid.float().sum().clamp(min=1)
        return tok.to(lt.dtype), m > 0.05, loss
