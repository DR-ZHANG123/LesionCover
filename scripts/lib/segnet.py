"""病灶分割器：U-Net（ResNet-18 编码器）四通道（MA/HE/EX/SE），在 canon 坐标的补丁上训练，整图推断。"""
from __future__ import annotations

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

cv2.setNumThreads(1)   # 每个数据加载/多进程 worker 只用单线程，避免 CPU 超订

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_segmenter(encoder: str, n_cls: int, pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, classes=n_cls)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class PatchSet(torch.utils.data.Dataset):
    """每个样本 = 一张图上随机取的 patches_per_image 个补丁；lesion_frac 比例以病灶像素为中心。"""

    def __init__(self, imgs: list, masks: list, fmasks: list, patch: int, per_image: int,
                 lesion_frac: float, seed: int):
        self.imgs, self.masks, self.fmasks = imgs, masks, fmasks
        self.patch, self.per, self.lf = patch, per_image, lesion_frac
        self.seed = seed
        self.lesion_px = [np.argwhere(m.any(0)) for m in masks]
        self.fov_px = [np.argwhere(f > 0)[::97] for f in fmasks]
        self.epoch = 0

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        # 注意：persistent_workers 下工作进程中的 self.epoch 始终为 0，每张照片的八个补丁及其增强各轮相同（Methods 按此描述）。
        rng = np.random.default_rng((self.seed, self.epoch, i))
        img, m = self.imgs[i], self.masks[i]
        S, p = img.shape[0], self.patch
        xs, ys = [], []
        for _ in range(self.per):
            pool = self.lesion_px[i] if (rng.random() < self.lf and len(self.lesion_px[i])) else self.fov_px[i]
            cy, cx = pool[rng.integers(len(pool))]
            y0 = int(np.clip(cy - p // 2 + rng.integers(-p // 4, p // 4 + 1), 0, S - p))
            x0 = int(np.clip(cx - p // 2 + rng.integers(-p // 4, p // 4 + 1), 0, S - p))
            a = img[y0:y0 + p, x0:x0 + p].copy()
            b = m[:, y0:y0 + p, x0:x0 + p].copy()
            if rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, :, ::-1]
            if rng.random() < 0.5:
                a, b = a[::-1], b[:, ::-1]
            k = int(rng.integers(4))
            a, b = np.rot90(a, k), np.rot90(b, k, axes=(1, 2))
            a = cv2.convertScaleAbs(np.ascontiguousarray(a), alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))
            xs.append(torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))))
            ys.append(torch.from_numpy(np.ascontiguousarray(b)))
        return torch.stack(xs), torch.stack(ys)


def gpu_norm(x: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW → ImageNet 归一化 float（在 GPU 上做）。"""
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1) * 255
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1) * 255
    return (x.float() - m) / s


def dice_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight[None, :, None, None])
    p = torch.sigmoid(logits)
    inter = (p * y).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def predict_full(model: torch.nn.Module, img: np.ndarray, device, flip_tta: bool = True) -> np.ndarray:
    """整图推断（canon 尺寸），返回 (C, S, S) 概率 float16。"""
    x = to_tensor(img)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.sigmoid(model(x).float())
        if flip_tta:
            p = (p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)) / 2
    return p[0].float().cpu().numpy()


LOGIT_MIN, LOGIT_STEP = -40.0, 0.25      # 对数几率的 uint8 量化：[-40, 23.75]，步长 0.25（软性渗出的对数几率低至约 -30）


def encode_logit(l: torch.Tensor) -> torch.Tensor:
    return torch.round((l.clamp(LOGIT_MIN, LOGIT_MIN + 255 * LOGIT_STEP) - LOGIT_MIN) / LOGIT_STEP).to(torch.uint8)


def decode_logit(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * LOGIT_STEP + LOGIT_MIN


@torch.no_grad()
def predict_pooled(model: torch.nn.Module, img: np.ndarray, device, out: int) -> np.ndarray:
    """整图推断（水平翻转后概率平均），转回对数几率并在 GPU 上最大池化到 out×out，返回 uint8 编码的对数几率。
    存对数几率而非概率，保留输出数值很小的病灶类型（如软性渗出）的排序信息。"""
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(device)
    x = gpu_norm(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = model(x).float(), model(x.flip(-1)).float().flip(-1)
    # 两次预测概率的平均，在对数域稳定计算：logit(mean(σ(a), σ(b)))
    lp = torch.logaddexp(F.logsigmoid(a), F.logsigmoid(b)) - np.log(2.0)
    ln = torch.logaddexp(F.logsigmoid(-a), F.logsigmoid(-b)) - np.log(2.0)
    lg = F.max_pool2d(lp - ln, (lp - ln).shape[-1] // out)
    return encode_logit(lg[0]).cpu().numpy()


def pool_prob(prob: np.ndarray, out: int) -> np.ndarray:
    """按最大池化下采样到 out×out（保留微小病灶的峰值），返回 uint8（概率×255）。"""
    f = prob.shape[-1] // out
    t = torch.from_numpy(prob[None].astype(np.float32))
    pooled = F.max_pool2d(t, f)[0].numpy()
    return np.round(pooled * 255).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """二值掩膜按面积插值后取 >0.25，尽量保留缩小后的微动脉瘤。"""
    r = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (r > 0.25).astype(np.uint8)
