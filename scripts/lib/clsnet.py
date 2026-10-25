"""分级模型：共享 ConvNeXt-Tiny 提取全局图与局部裁块特征，门控注意力 MIL 聚合裁块，与全局特征拼接后输出五级概率。"""
from __future__ import annotations

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segnet import MEAN, STD
from .variants import encoder_name

cv2.setNumThreads(1)   # 每个数据加载/多进程 worker 只用单线程，避免 CPU 超订


def to_u8(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 → CHW uint8 张量；归一化在 GPU 上由模型完成。"""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class GatedAttentionMIL(nn.Module):
    """Ilse et al. 2018 的门控注意力池化。"""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.v = nn.Linear(d, h)
        self.u = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x):                       # x: (B, K, d)
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))   # (B, K, 1)
        a = torch.softmax(a, dim=1)
        return (a * x).sum(1), a.squeeze(-1)


def backbone_for(tag: str, cfg: dict, root) -> str:
    """由运行标签末段的 variant 取编码器（见 variants.encoder_name）。"""
    var = tag.rsplit("_", 1)[-1] if "_" in tag else ""
    try:
        return encoder_name(var, cfg, root)
    except ValueError:              # 末段不是 variant（如 G512_s11、带 recipe 的标签）
        return cfg["classifier"]["backbone"]


class Grader(nn.Module):
    def __init__(self, backbone: str, n_cls: int = 5, attn_dim: int = 256, use_crops: bool = True,
                 crop_head: bool = False, drop_path: float = 0.0):
        super().__init__()
        self.in_size, self.patch = None, None
        mean, std = MEAN, STD
        hr = backbone.endswith("@hr"); backbone = backbone.removesuffix("@hr")
        if backbone.endswith(".pth") or backbone.endswith(".bin"):
            # 视网膜基础模型 RETFound（ViT-L/16，在约 90 万张彩色眼底照片上自监督预训练）：缺省输入缩放到 224，
            # 取 patch token 平均（fc_norm 新初始化），与 RETFound 官方微调设置一致；@hr 时按原分辨率输入，位置编码插值
            self.enc = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="avg",
                                         drop_path_rate=drop_path, dynamic_img_size=hr)
            sd = torch.load(backbone, map_location="cpu", weights_only=True)
            sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "norm."))}
            missing = self.enc.load_state_dict(sd, strict=False).missing_keys
            assert set(missing) <= {"fc_norm.weight", "fc_norm.bias"}, missing
            self.enc.set_grad_checkpointing(True)
            self._deterministic_attention()
            self.in_size = None if hr else 224
        elif "dinov2" in backbone:
            # DINOv2 ViT（patch 14）：输入边长取最接近的 14 的倍数，位置编码按输入网格插值
            self.enc = timm.create_model(backbone, pretrained=True, num_classes=0, drop_path_rate=drop_path,
                                         dynamic_img_size=True)
            self.enc.set_grad_checkpointing(True)
            self._deterministic_attention()
            self.patch = 14
            mean, std = self.enc.pretrained_cfg["mean"], self.enc.pretrained_cfg["std"]
        else:
            self.enc = timm.create_model(backbone, pretrained=True, num_classes=0, drop_path_rate=drop_path)
            mean, std = self.enc.pretrained_cfg["mean"], self.enc.pretrained_cfg["std"]   # 与预训练一致的归一化
            if not backbone.startswith(("convnext_tiny", "convnext_small")):
                self.enc.set_grad_checkpointing(True)    # 更大的编码器：重算激活以节省显存
        d = self.enc.num_features
        self.use_crops = use_crops
        if use_crops:
            self.mil = GatedAttentionMIL(d, attn_dim)
        self.drop = nn.Dropout(0.2)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1) * 255)
        self.fc = nn.Linear(2 * d if use_crops else d, n_cls)
        # 只看裁块的辅助分级头（深监督）：保证裁块分支本身学会读取病灶证据
        self.crop_fc = nn.Linear(d, n_cls) if (use_crops and crop_head) else None

    @staticmethod
    def _deterministic_attention():
        # 注意力用确定性的数学实现（flash / memory-efficient 内核的反向传播不确定），保证同一种子可复现
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)

    def norm(self, x):
        x = (x.float() - self.mean) / self.std
        size = self.in_size
        if self.patch is not None:
            size = int(round(x.shape[-1] / self.patch)) * self.patch
        if size is not None and x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        return x.contiguous(memory_format=torch.channels_last)

    def forward(self, g, crops=None, global_mask=None, return_aux=False):
        """global_mask：(B,) 训练时为 0 的样本遮掉全局特征（模态 dropout），迫使网络从裁块分级。"""
        fg = self.enc(self.norm(g))
        if not self.use_crops:
            return self.fc(self.drop(fg)), None
        B, K = crops.shape[:2]
        fc = self.enc(self.norm(crops.flatten(0, 1))).view(B, K, -1)
        z, a = self.mil(fc)
        if global_mask is not None:
            fg = fg * global_mask[:, None].to(fg.dtype)
        out = self.fc(self.drop(torch.cat([fg, z], 1)))
        if return_aux:
            return out, a, (self.crop_fc(self.drop(z)) if self.crop_fc is not None else None)
        return out, a


def artifacts(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """模拟跨相机与成像质量差异：镜头尘点（暗/亮软斑）、反光亮斑、离焦模糊、伽马与色偏、边缘暗角、噪声、JPEG 重压缩。
    每种扰动独立以一定概率施加于一张图块（全局图或裁块）。"""
    t = t.astype(np.float32)
    h, w = t.shape[:2]
    if rng.random() < 0.5:                                    # 尘点与亮斑
        yy, xx = np.mgrid[:h, :w]
        grey = t.mean(axis=2, keepdims=True)
        for _ in range(int(rng.integers(1, 7))):
            # 尘点是去饱和的灰色软斑、反光是亮白斑，半径大于典型微动脉瘤，避免把真实红色小病灶当作伪影
            cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(5, 20) * h / 384
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))[..., None]
            target = 255.0 if rng.random() < 0.4 else grey * rng.uniform(0.5, 0.9)
            a = rng.uniform(0.15, 0.5) * blob
            t = t * (1 - a) + target * a
    if rng.random() < 0.3:                                    # 离焦模糊
        k = float(rng.uniform(0.5, 2.0))
        t = cv2.GaussianBlur(t, (0, 0), k)
    if rng.random() < 0.5:                                    # 伽马与逐通道色偏
        t = 255.0 * (np.clip(t, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
        t = t * rng.uniform(0.85, 1.15, size=3)[None, None]
    if rng.random() < 0.3:                                    # 暗角 / 不均匀照明
        yy, xx = np.mgrid[:h, :w]
        cy, cx = rng.uniform(-0.5, 1.5) * h, rng.uniform(-0.5, 1.5) * w
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / np.hypot(h, w)
        t = t * (1 - rng.uniform(0.1, 0.5) * np.clip(d, 0, 1))[..., None]
    if rng.random() < 0.2:                                    # 传感器噪声
        t = t + rng.normal(0, rng.uniform(2, 8), size=t.shape)
    t = np.clip(t, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                                    # JPEG 重压缩
        ok, enc = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 80))])
        t = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return t


def _aug(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        t = t[:, ::-1]
    if rng.random() < 0.5:
        t = t[::-1]
    return np.rot90(t, int(rng.integers(4)))


class GradeSet(torch.utils.data.Dataset):
    """paths: canon 图路径；crops: (N, K, 2) canon 左上角坐标或 None；返回 (全局, 裁块, 标签, 索引)。"""

    def __init__(self, paths, labels, crops, cfg, global_size: int, train: bool, seed: int, artifact_aug: bool = False):
        self.paths, self.labels, self.crops = list(paths), np.asarray(labels), crops
        self.art = artifact_aug and train
        self.gpaths = [p.replace("/canon/", f"/global{global_size}/") for p in self.paths]
        pc, cc = cfg["preprocess"], cfg["classifier"]
        self.S, self.c, self.gs = pc["canon_size"], cfg["selection"]["crop"], global_size
        self.jit = cc["crop_jitter"]
        self.train, self.seed, self.epoch = train, seed, 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # 注意：DataLoader 使用 persistent_workers，主进程里更新的 self.epoch 不会传到工作进程，工作进程中始终为 0；
        # 因此每张照片的增强（翻转、旋转、亮度、裁块抖动、伪影）只按（种子, 照片）抽一次、各轮相同。Methods 即按此描述。
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        # 亮度/对比度扰动（每张图一组参数，作用于全局图与全部裁块）；在缩放/裁剪之后做以节省计算
        gain, bias = (rng.uniform(0.85, 1.15), rng.uniform(-12, 12)) if self.train else (1.0, 0.0)
        jit = (lambda t: cv2.convertScaleAbs(t, alpha=gain, beta=bias)) if self.train else (lambda t: t)
        g = jit(cv2.imread(self.gpaths[i], cv2.IMREAD_COLOR)[:, :, ::-1])
        if self.art:
            g = artifacts(np.ascontiguousarray(g), rng)
        if self.train:
            g = _aug(g, rng)
        out_g = to_u8(g)
        if self.crops is None:
            return out_g, torch.zeros(0, dtype=torch.uint8), int(self.labels[i]), i
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)[:, :, ::-1]
        tiles = []
        for x0, y0 in self.crops[i]:
            if self.train:
                x0 += int(rng.integers(-self.jit, self.jit + 1)); y0 += int(rng.integers(-self.jit, self.jit + 1))
            x0 = int(np.clip(x0, 0, self.S - self.c)); y0 = int(np.clip(y0, 0, self.S - self.c))
            t = jit(img[y0:y0 + self.c, x0:x0 + self.c])
            if self.art:
                t = artifacts(np.ascontiguousarray(t), rng)
            if self.train:
                t = _aug(t, rng)
            tiles.append(to_u8(t))
        return out_g, torch.stack(tiles), int(self.labels[i]), i


def quadratic_kappa(y, p, n: int = 5) -> float:
    y, p = np.asarray(y, int), np.asarray(p, int)
    O = np.zeros((n, n))
    np.add.at(O, (y, p), 1)
    w = (np.arange(n)[:, None] - np.arange(n)[None]) ** 2 / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else 0.0
