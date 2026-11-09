"""LesionCover 网络：一张 1024 全局图 + 病灶 token（每个预测病灶分量一个）经 Transformer 融合后分级。

全局分支：ConvNeXt 第 4 阶段特征池化成 G×G 个全局 token。
病灶分支：每个分量在第 2、3 阶段特征图上做 RoIAlign（高分辨率局部外观），再加类别、得分、面积与位置编码；
另有一个逐类逐象限计数 token（含被截断的分量），对应国际分级标准按病灶类型、数量与象限分布定级。
全局 token 与病灶 token 使用同一坐标系的 Fourier 位置编码。分类头读 [CLS]；两个辅助头分别只读全局特征
与只读病灶 token（门控注意力池化），保证两条通路各自学会分级。"""
from __future__ import annotations

import math

import numpy as np

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clsnet import GatedAttentionMIL
from .hypergraph import HyperConv, build_incidence
from .protoagg import LesionPrototypes
from .rulehead import BurdenGate, QuadrantRule


def fourier(xy: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """xy ∈ [-1, 1]，(..., 2) → (..., 4 n_freq + 2)。"""
    f = (2.0 ** torch.arange(n_freq, device=xy.device, dtype=xy.dtype)) * math.pi
    a = xy[..., None] * f                                   # (..., 2, F)
    return torch.cat([xy, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


def bilinear_matmul(f: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """f: (B,C,H,W)；u, v: (B,Q) 归一化坐标 ∈ [0,1]（像素中心对齐）。返回 (B,Q,C)，等价于双线性 grid_sample（边界夹取）。"""
    B, C, H, W = f.shape
    with torch.no_grad():
        x = (u * W - 0.5).clamp(0, W - 1); y = (v * H - 0.5).clamp(0, H - 1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp(max=W - 1), (y0 + 1).clamp(max=H - 1)
        ax, ay = x - x0, y - y0
        wmat = torch.zeros(B, u.shape[1], H * W, device=f.device, dtype=torch.float32)
        for yy, xx, w in ((y0, x0, (1 - ax) * (1 - ay)), (y0, x1, ax * (1 - ay)), (y1, x0, (1 - ax) * ay), (y1, x1, ax * ay)):
            wmat.scatter_add_(2, (yy * W + xx)[..., None], w[..., None])
    return torch.bmm(wmat.to(f.dtype), f.flatten(2).transpose(1, 2))


class LesionCoverNet(nn.Module):
    def __init__(self, backbone: str, lc: dict, n_cls: int = 5, n_les: int = 4, prob_size: int = 768,
                 drop_path: float = 0.0):
        super().__init__()
        if lc.get("fm"):                                     # 基础模型编码器（冻结 ViT + LoRA，fmencoder.py）
            from .fmencoder import FMEncoder
            self.enc = FMEncoder(lc["fm"], lc.get("fm_size", 560), r=lc.get("fm_rank", 8), weights=lc.get("fm_weights"))
        else:
            self.enc = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=(1, 2, 3),
                                         drop_path_rate=drop_path)
            if lc.get("grad_ckpt"):                          # 只省显存，不改变计算结果
                self.enc.set_grad_checkpointing(True)
        c1, c2, c3 = self.enc.feature_info.channels()
        self.red = self.enc.feature_info.reduction()
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1) * 255)
        d, self.G, self.roi = lc["d_model"], lc["global_grid"], lc["roi"]
        self.lc_fm = bool(lc.get("fm"))
        self.P, self.min_box, self.n_les = prob_size, lc["min_box"], n_les
        self.off = set(lc.get("ablate", ()))                # 消融：app 外观、pos 位置、attr 类别/得分/面积、count 计数 token
        npos = 4 * 8 + 2
        self.gproj = nn.Sequential(nn.LayerNorm(c3), nn.Linear(c3, d))
        self.pos = nn.Linear(npos, d)
        self.rproj = nn.Sequential(nn.LayerNorm((c1 + c2) * self.roi ** 2), nn.Linear((c1 + c2) * self.roi ** 2, d),
                                   nn.GELU(), nn.Linear(d, d))
        self.attr = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.cls_emb = nn.Embedding(n_les, d)
        self.count = nn.Sequential(nn.Linear(n_les * 4 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Embedding(4, d)                  # 0 CLS, 1 全局, 2 病灶, 3 计数
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, lc["heads"], 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, lc["layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(d, n_cls)
        self.fc_global = nn.Linear(c3, n_cls)
        self.mil = GatedAttentionMIL(d, 128)
        self.fc_lesion = nn.Linear(d, n_cls)
        gy, gx = torch.meshgrid(torch.arange(self.G), torch.arange(self.G), indexing="ij")
        # 病灶超图（放在最后创建：不用超图时其余参数的随机初始化与原设置逐位相同）
        self.n_hyper, self.knn = lc.get("hyper_layers", 0), lc.get("hyper_knn", 4)
        self.hyper = nn.ModuleList([HyperConv(d) for _ in range(self.n_hyper)])
        self.rule_emb = nn.Parameter(torch.zeros(d)) if self.n_hyper else None   # 超边（规则 token）类型嵌入
        # 知识引导分类头：分级标准文字定义经 BiomedCLIP 文本编码器得到的类别原型（冻结），去均值后作为余弦分类器的权重
        self.text = None
        if lc.get("text_proto"):
            T = torch.from_numpy(np.load(lc["text_proto"])).float()
            T = F.normalize(T - T.mean(0, keepdim=True), dim=-1)
            self.register_buffer("tproto", T)
            self.text = nn.Linear(d, T.shape[1])
            self.tscale = nn.Parameter(torch.tensor(math.log(10.0)))
        # 分级规则模块与按病灶负荷的门控（同样放在最后创建）
        self.rule = (QuadrantRule(n_les, d, n_cls, init=tuple(lc.get("rule_init", (2.0, 5.0, 20.0))), ordinal=lc.get("rule_aux_weight", 0) > 0)
                     if lc.get("rule") else None)
        self.gate = BurdenGate(n_les) if lc.get("gate") else None
        self.proto = LesionPrototypes(d, n_les, lc.get("proto_k", 8)) if lc.get("proto") else None
        self.proto_loss = None
        self.fmtok = None
        if lc.get("fm_tokens"):                              # 双编码器融合：基础模型语义 token（fmencoder.FMTokens）
            from .fmencoder import FMTokens
            self.fmtok = FMTokens(lc["fm_tokens"], lc.get("fm_size", 560), d, self.G, r=lc.get("fm_rank", 0),
                                  weights=lc.get("fm_weights"))
        self.register_buffer("gxy", torch.stack([gx, gy], -1).view(-1, 2).float().add(0.5).div(self.G).mul(2).sub(1))

    def _lesion_tokens(self, f1, f2, tok, S):
        """tok: (B, M, 7) 字段见 lesiontokens.FIELDS，坐标在概率图坐标系；S：输入边长。"""
        B, M = tok.shape[:2]
        s = S / self.P
        cx, cy = tok[..., 3] * s, tok[..., 4] * s
        hw = torch.clamp(tok[..., 5], min=self.min_box) * s / 2
        hh = torch.clamp(tok[..., 6], min=self.min_box) * s / 2
        # RoI 内 roi×roi 个格点的双线性采样（每格一个采样点的 RoIAlign）。插值写成稀疏权重矩阵 × 特征的矩阵乘法，
        # 反向传播是确定性的（grid_sample / roi_align 的反向依赖原子加，同一种子结果会有微小差异）
        t = (torch.arange(self.roi, device=tok.device, dtype=torch.float32) + 0.5) / self.roi * 2 - 1
        px = (cx[..., None, None] + hw[..., None, None] * t[None, None, None, :]).expand(B, M, self.roi, self.roi)
        py = (cy[..., None, None] + hh[..., None, None] * t[None, None, :, None]).expand(B, M, self.roi, self.roi)
        r = [bilinear_matmul(f, px.reshape(B, -1) / S, py.reshape(B, -1) / S).view(B, M, -1) for f in (f1, f2)]
        app = self.rproj(torch.cat(r, -1).to(f1.dtype))
        attr = torch.stack([tok[..., 1], tok[..., 2], torch.log(tok[..., 5]), torch.log(tok[..., 6])], -1)
        xy = torch.stack([tok[..., 3], tok[..., 4]], -1) / self.P * 2 - 1
        cls = tok[..., 0].long().clamp(0, self.n_les - 1)
        z = self.type_emb.weight[2] + 0 * app
        if "app" not in self.off:
            z = z + app
        if "attr" not in self.off:
            z = z + self.attr(attr) + self.cls_emb(cls)
        if "pos" not in self.off:
            z = z + self.pos(fourier(xy))
        return z

    def forward(self, g, tok, valid, counts, return_aux=False):
        """g: (B,3,S,S) uint8；tok: (B,M,7)；valid: (B,M) bool；counts: (B, n_les*4)。"""
        S = g.shape[-1]
        x = ((g.float() - self.mean) / self.std)
        x = x if self.lc_fm else x.contiguous(memory_format=torch.channels_last)
        f1, f2, f3 = self.enc(x)
        B = g.shape[0]
        gt = F.avg_pool2d(f3, f3.shape[-1] // self.G).flatten(2).transpose(1, 2)            # (B, G², c3)
        gt = self.gproj(gt) + self.pos(fourier(self.gxy))[None] + self.type_emb.weight[1]
        if self.fmtok is not None:
            gt = torch.cat([gt, self.fmtok(g, self.pos(fourier(self.gxy))).to(gt.dtype)], 1)
        lt = self._lesion_tokens(f1, f2, tok, S) if tok.shape[1] else gt.new_zeros(B, 0, gt.shape[-1])
        rt = gt.new_zeros(B, 0, gt.shape[-1]); rv = torch.zeros(B, 0, dtype=torch.bool, device=g.device)
        if self.n_hyper and tok.shape[1]:
            with torch.no_grad():
                xy = torch.stack([tok[..., 3], tok[..., 4]], -1) / self.P * 2 - 1
                H, et, ev = build_incidence(tok[..., 0].long().clamp(0, self.n_les - 1), xy, valid, self.n_les, self.knn)
            for layer in self.hyper:
                lt, e = layer(lt, H, et, ev)
            nr = self.n_les * 5                               # 类型×象限 与 类型 两组超边作为规则 token
            rt, rv = e[:, :nr] + self.rule_emb, ev[:, :nr]
        if self.proto is not None and tok.shape[1]:          # 病灶亚型（原型）token 追加在规则 token 之后
            pt, pv, self.proto_loss = self.proto(lt, tok[..., 0].long().clamp(0, self.n_les - 1), valid)
            rt, rv = torch.cat([rt, pt], 1), torch.cat([rv, pv], 1)
        tot = torch.log1p(counts.sum(1, keepdim=True))
        if "count" in self.off:
            counts, tot = torch.zeros_like(counts), torch.zeros_like(tot)
        ct = (self.count(torch.cat([torch.log1p(counts), tot], 1)) + self.type_emb.weight[3])[:, None]
        if self.rule is not None:                             # 规则 token 追加在计数 token 之后（同类型嵌入）
            rlog, rtok = self.rule(counts)
            ct = torch.cat([ct, (rtok + self.type_emb.weight[3])[:, None]], 1)
        cl = self.cls_tok.expand(B, -1, -1) + self.type_emb.weight[0]
        seq = torch.cat([cl, gt, ct, rt, lt], 1)
        pad = torch.cat([torch.zeros(B, 1 + gt.shape[1] + ct.shape[1], dtype=torch.bool, device=g.device), ~rv, ~valid], 1)
        h = self.norm(self.tf(seq, src_key_padding_mask=pad))
        out = self.fc(self.drop(h[:, 0]))
        if self.text is not None:
            z = F.normalize(self.text(self.drop(h[:, 0])).float(), dim=-1)
            out = out + self.tscale.exp() * z @ self.tproto.t()
        if self.rule is not None:
            out = out + rlog.to(out.dtype)
        aux_g = self.fc_global(self.drop(f3.mean((2, 3))))
        if self.gate is not None:                             # 病灶负荷高时更多依赖全局通路
            a = self.gate(counts).to(out.dtype)
            out = (1 - a) * out + a * aux_g.to(out.dtype)
        if not return_aux:
            return out
        les = torch.cat([ct, rt, lt], 1)
        lv = torch.cat([torch.ones(B, ct.shape[1], dtype=torch.bool, device=g.device), rv, valid], 1)
        a = self.mil.w(torch.tanh(self.mil.v(les)) * torch.sigmoid(self.mil.u(les))).squeeze(-1)
        a = torch.softmax(a.float().masked_fill(~lv, -1e4), 1).to(les.dtype)
        aux_l = self.fc_lesion(self.drop((a[..., None] * les).sum(1)))
        return out, aux_g, aux_l
