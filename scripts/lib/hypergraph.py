"""病灶超图：节点 = 预测病灶分量；超边按国际分级标准的组织方式构建——
(1) 同类型 × 同象限（对应重度 NPDR 的“4-2-1”象限规则），(2) 同类型（全视网膜），(3) 空间近邻（每个病灶与其 k 个最近病灶）。
超图卷积采用两步注意力消息传递：节点 → 超边（注意力池化）→ 节点，并带超边类型嵌入。超边嵌入作为“规则 token”输出。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_incidence(cls: torch.Tensor, xy: torch.Tensor, valid: torch.Tensor, n_les: int, k: int):
    """cls (B,M) 整数，xy (B,M,2) ∈ [-1,1]，valid (B,M)。返回 H (B,M,E) 0/1、超边类型 (E,)、超边有效 (B,E)。
    E = n_les*4（类型×象限）+ n_les（类型）+ M（近邻，以每个节点为中心）。"""
    B, M = cls.shape
    dev = cls.device
    v = valid.float()
    quad = (xy[..., 0] >= 0).long() + 2 * (xy[..., 1] >= 0).long()                 # (B,M)
    tq = F.one_hot(cls * 4 + quad, n_les * 4).float() * v[..., None]                 # (B,M,16)
    tt = F.one_hot(cls, n_les).float() * v[..., None]                               # (B,M,4)
    d = torch.cdist(xy.float(), xy.float())                                         # (B,M,M)
    d = d.masked_fill(~valid[:, None, :], 1e4) + torch.eye(M, device=dev)[None] * -1  # 自身最近
    kk = min(k + 1, M)
    idx = d.topk(kk, dim=-1, largest=False).indices                                 # (B,M,kk) 以节点 i 为中心
    nn_ = torch.zeros(B, M, M, device=dev).scatter_(2, idx, 1.0)                    # (B, 中心, 成员)
    nn_ = nn_ * v[:, :, None] * v[:, None, :]
    H = torch.cat([tq, tt, nn_.transpose(1, 2)], 2)                                 # (B,M,E)
    etype = torch.cat([torch.zeros(n_les * 4), torch.ones(n_les), 2 * torch.ones(M)]).long().to(dev)
    evalid = H.sum(1) > 0
    return H, etype, evalid


class HyperConv(nn.Module):
    """节点→超边：超边内节点的注意力加权平均（查询 = 超边类型嵌入 + 超边均值）；超边→节点：节点对其所属超边的注意力加权平均。"""

    def __init__(self, d: int, n_etype: int = 3):
        super().__init__()
        self.etype = nn.Embedding(n_etype, d)
        self.n2e_q, self.n2e_k, self.n2e_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.e2n_q, self.e2n_k, self.e2n_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.scale = d ** -0.5

    def forward(self, x, H, etype, evalid):
        """x (B,M,d)；H (B,M,E)。返回 (节点 (B,M,d), 超边 (B,E,d))。"""
        h = self.ln1(x)
        Hf = H.to(h.dtype)
        mean = torch.einsum("bme,bmd->bed", Hf, h) / Hf.sum(1).clamp(min=1)[..., None]
        q = self.n2e_q(mean + self.etype(etype)[None])                                  # (B,E,d)
        a = torch.einsum("bed,bmd->bem", q, self.n2e_k(h)) * self.scale
        a = a.float().masked_fill(H.transpose(1, 2) <= 0, -1e4).softmax(-1).to(h.dtype)
        e = torch.einsum("bem,bmd->bed", a, self.n2e_v(h)) * evalid[..., None].to(h.dtype)
        g = self.ln2(e)
        b = torch.einsum("bmd,bed->bme", self.e2n_q(h), self.e2n_k(g)) * self.scale
        b = b.float().masked_fill(H <= 0, -1e4).softmax(-1).to(h.dtype)
        has = (H.sum(-1) > 0)[..., None].to(h.dtype)
        x = x + torch.einsum("bme,bed->bmd", b, self.e2n_v(g)) * has
        return x + self.ff(x), e
