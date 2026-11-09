"""分级规则模块：把国际分级标准中“病灶按象限计数”的定级方式写成可微结构。

对每类病灶 c、每个象限 q 的分量数 n_cq，用 K 个可学习软阈值得到“该象限达到阈值”的软指示
s_cqk = sigmoid(β_ck · (log1p n_cq − log1p θ_ck))；把四个象限的软指示从大到小排序，第 j 个值即
“至少 j 个象限达到阈值”的软指示（对应重度 NPDR 的“4-2-1”象限规则：出血在 4 个象限均超过阈值等）。
这些规则特征 (n_les·K·4) 经线性层给出规则 logits（初始化为 0，训练开始时与原网络一致），并作为一个规则 token 输入 Transformer。
另提供按病灶负荷的门控：根据各类病灶总数决定融合输出与全局输出的权重。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrantRule(nn.Module):
    def __init__(self, n_les: int, d: int, n_cls: int = 5, k: int = 3, init=(2.0, 5.0, 20.0), ordinal: bool = False):
        super().__init__()
        self.n_les, self.k = n_les, k
        self.log_t = nn.Parameter(torch.tensor([math.log1p(t) for t in init]).repeat(n_les, 1))   # (n_les, K)
        self.log_b = nn.Parameter(torch.full((n_les, k), math.log(3.0)))
        nf = n_les * k * 4
        self.norm = nn.LayerNorm(nf)
        self.fc = nn.Linear(nf, n_cls)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.tok = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        # 只看规则特征的单调有序分级（可读的“分级标准”）：s = Σ softplus(w)·f，P(级别 > k) = sigmoid(s − c_k)，c_k 递增
        self.ordinal = ordinal
        if ordinal:
            self.ow = nn.Parameter(torch.full((nf,), -2.0))
            self.oc = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0]))
        self.score = None

    def rule_params(self):
        """阈值、斜率与有序分级参数（单独学习率、不加权重衰减）。"""
        ps = [self.log_t, self.log_b]
        return ps + ([self.ow, self.oc] if self.ordinal else [])

    def cut_points(self):
        return self.oc[:1] + torch.cat([self.oc.new_zeros(1), torch.cumsum(F.softplus(self.oc[1:]) + 0.5, 0)])

    def ordinal_logits(self, f):
        """返回 (B, n_cls−1)：logit P(级别 > k)。"""
        s = (F.softplus(self.ow) * f.float()).sum(1, keepdim=True)
        return s - self.cut_points()[None]

    def features(self, counts: torch.Tensor) -> torch.Tensor:
        """counts (B, n_les*4)，类别优先排列 → (B, n_les*K*4)。"""
        c = torch.log1p(counts.float().view(-1, self.n_les, 1, 4))                       # (B,L,1,4)
        s = torch.sigmoid(self.log_b.exp()[None, :, :, None] * (c - self.log_t[None, :, :, None]))  # (B,L,K,4)
        return s.sort(-1, descending=True).values.flatten(1)                            # 至少 j 个象限

    def forward(self, counts):
        f = self.features(counts)
        self.score = self.ordinal_logits(f) if self.ordinal else None
        return self.fc(self.norm(f)), self.tok(f)


class BurdenGate(nn.Module):
    """α = sigmoid(MLP(各类病灶总数 log1p))；最终 logits = (1−α)·融合 + α·全局。偏置初始化使 α≈0.1。"""

    def __init__(self, n_les: int):
        super().__init__()
        self.n_les = n_les
        self.mlp = nn.Sequential(nn.Linear(n_les + 1, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.mlp[2].weight); nn.init.constant_(self.mlp[2].bias, -2.2)

    def forward(self, counts):
        c = counts.float().view(-1, self.n_les, 4).sum(-1)
        x = torch.log1p(torch.cat([c, c.sum(1, keepdim=True)], 1))
        return torch.sigmoid(self.mlp(x))
