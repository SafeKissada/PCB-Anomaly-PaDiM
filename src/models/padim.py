"""
PaDiM (Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
Anomaly Detection and Localization", ICPR 2021) —
https://arxiv.org/abs/2011.08785

สรุป pipeline:
  1. ดึง patch feature จาก layer1+layer2+layer3 ของ ImageNet-pretrained CNN
     (เหมือน PatchCore แต่ใช้ 3 layer ไม่ใช่ 2 — PaDiM ใช้ layer ตื้นกว่าด้วย
     เพื่อจับรายละเอียดระดับ texture)
  2. สุ่มเลือก channel ย่อยจาก feature ที่ concat แล้ว (random dimension
     reduction) แทนการทำ PCA เต็มรูปแบบ — ตาม paper วิธีนี้เร็วกว่ามากและ
     ผลไม่ต่างจาก PCA อย่างมีนัยสำคัญ
  3. สำหรับแต่ละตำแหน่ง patch (h, w) สร้าง **Multivariate Gaussian
     distribution** จาก feature vector ของภาพ normal ทั้งหมดที่ตำแหน่งนั้น
     (mean vector + covariance matrix) — ต่างจาก PatchCore ที่เก็บ memory
     bank ของ patch ทั้งหมด, PaDiM สรุปเป็นสถิติ (mean, cov) ต่อตำแหน่งแทน
  4. Inference: patch score = Mahalanobis distance จาก patch feature ของ
     ภาพ test ไปยัง Gaussian ของตำแหน่งเดียวกัน

ข้อแตกต่างสำคัญจาก PatchCore ที่ต้องรู้:
  - PaDiM สนใจ "ตำแหน่ง" (position-aware) — สมมติฐานคือวัตถุ/ตำแหน่งของ
    ชิ้นส่วนในภาพต้อง align กันเป๊ะข้ามภาพ (เพราะ Gaussian ผูกกับตำแหน่ง
    (h,w) ตายตัว) — เหมาะกับ setup ที่ camera/component เดิมทุกครั้ง แบบ
    dataset นี้ (crop จาก AOI ที่ตำแหน่งเดิมของ SMD component) น่าจะเข้ากับ
    สมมติฐานนี้ได้ดี แต่ถ้ามี alignment shift ระหว่างภาพ (เช่น crop ไม่ตรง
    เป๊ะทุกครั้ง) PaDiM จะเสียหายมากกว่า PatchCore ที่ไม่ผูกกับตำแหน่ง
  - ไม่มี "coreset subsampling" เหมือน PatchCore — ต้องเก็บ mean+cov ต่อ
    ตำแหน่งแทน ซึ่งขนาดไม่ได้โตตามจำนวนภาพ train (ต่างจาก PatchCore ที่
    memory bank โตตามข้อมูล) แต่การ invert covariance matrix ต่อตำแหน่ง
    เป็นจุดคอขวดด้านความเร็วแทน
"""
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

logger = logging.getLogger("PaDiM")

_BACKBONE_FACTORY = {
    "wide_resnet50_2": (torchvision.models.wide_resnet50_2,
                         torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "resnet18": (torchvision.models.resnet18,
                 torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (torchvision.models.resnet50,
                 torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
}


class _FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone_name, layers, device, pretrained=True):
        super().__init__()
        ctor, weights = _BACKBONE_FACTORY[backbone_name]
        if not pretrained:
            logger.warning("pretrained=False: random-init weights — smoke "
                            "test/offline dev เท่านั้น ห้ามใช้รันผลจริง")
        self.backbone = ctor(weights=weights if pretrained else None)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.to(device)

        self.layers = layers
        self._features = {}
        self._hooks = [
            dict(self.backbone.named_modules())[name].register_forward_hook(
                self._make_hook(name))
            for name in layers
        ]

    def _make_hook(self, name):
        def hook(_m, _i, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        self.backbone(x)
        return [self._features[name] for name in self.layers]


def _embed_batch(extractor, images, ref_layer_idx=0):
    """Concat feature จากทุก layer ที่ resize ให้เท่ากับ layer อ้างอิง
    (default = layer แรก = ละเอียดสุด) แล้ว flatten เป็น patch vector
    """
    feats = extractor(images)
    ref_h, ref_w = feats[ref_layer_idx].shape[-2:]
    resized = []
    for f in feats:
        if f.shape[-2:] != (ref_h, ref_w):
            f = F.interpolate(f, size=(ref_h, ref_w), mode="bilinear",
                               align_corners=False)
        resized.append(f)
    embedding = torch.cat(resized, dim=1)  # [B, C_total, H, W]
    return embedding, (ref_h, ref_w)


class PaDiM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.extractor = _FeatureExtractor(
            cfg.BACKBONE, cfg.FEATURE_LAYERS, self.device,
            pretrained=getattr(cfg, "PRETRAINED", True))
        self.selected_channels: torch.Tensor = None  # [num_selected]
        self.mean: torch.Tensor = None       # [H*W, C_sel]
        self.cov_inv: torch.Tensor = None    # [H*W, C_sel, C_sel]
        self.spatial_shape: Tuple[int, int] = None

    @torch.no_grad()
    def fit(self, normal_loader) -> None:
        logger.info("PaDiM.fit(): extracting patch features จากภาพ normal ทั้งหมด...")
        all_embeddings = []
        shape = None
        for batch in normal_loader:
            images = batch[0].to(self.device)
            emb, shape = _embed_batch(self.extractor, images)
            all_embeddings.append(emb.cpu())
        self.spatial_shape = shape
        embeddings = torch.cat(all_embeddings, dim=0)  # [N, C_total, H, W]
        N, C_total, H, W = embeddings.shape
        logger.info(f"Feature รวม: N={N} ภาพ, C_total={C_total}, spatial={H}x{W}")

        n_select = min(self.cfg.NUM_SELECTED_CHANNELS, C_total)
        torch.manual_seed(self.cfg.SEED)
        self.selected_channels = torch.randperm(C_total)[:n_select]
        embeddings = embeddings[:, self.selected_channels, :, :].to(self.device)  # [N, C_sel, H, W]

        # reshape เป็น [H*W, N, C_sel] เพื่อคำนวณ mean/cov ต่อตำแหน่งแบบ vectorized
        embeddings = embeddings.permute(2, 3, 0, 1).reshape(H * W, N, n_select)

        self.mean = embeddings.mean(dim=1)  # [H*W, C_sel]
        centered = embeddings - self.mean.unsqueeze(1)  # [H*W, N, C_sel]
        cov = torch.einsum("pnc,pnd->pcd", centered, centered) / max(N - 1, 1)  # [H*W, C_sel, C_sel]
        eps_eye = self.cfg.COV_REG_EPSILON * torch.eye(
            n_select, device=self.device).unsqueeze(0)
        cov = cov + eps_eye

        logger.info("กำลัง invert covariance matrix ต่อตำแหน่ง "
                    f"({H*W} ตำแหน่ง, {n_select}x{n_select} matrix ต่อตัว)...")
        self.cov_inv = torch.linalg.inv(cov)  # [H*W, C_sel, C_sel]
        logger.info("PaDiM fit เสร็จแล้ว")

    @torch.no_grad()
    def score(self, loader):
        from src.models.base import ScoreResult

        if self.mean is None:
            raise RuntimeError("PaDiM.score() ถูกเรียกก่อน fit()")

        H, W = self.spatial_shape
        image_scores, labels, paths, pixel_maps = [], [], [], []

        for batch in loader:
            images, _orig, _preproc, batch_paths, batch_labels, _size = batch
            images = images.to(self.device)
            B = images.shape[0]

            emb, shape = _embed_batch(self.extractor, images)
            assert shape == (H, W), (
                f"Spatial shape เปลี่ยนระหว่าง fit() {(H,W)} กับ score() {shape}")
            emb = emb[:, self.selected_channels, :, :]  # [B, C_sel, H, W]
            emb = emb.permute(2, 3, 0, 1).reshape(H * W, B, -1)  # [H*W, B, C_sel]

            delta = emb - self.mean.unsqueeze(1)  # [H*W, B, C_sel]
            # Mahalanobis^2 = delta^T @ cov_inv @ delta ต่อตำแหน่ง ต่อภาพ
            m_dist_sq = torch.einsum(
                "pbc,pcd,pbd->pb", delta, self.cov_inv, delta)  # [H*W, B]
            m_dist = torch.sqrt(torch.clamp(m_dist_sq, min=0.0))  # [H*W, B]
            m_dist = m_dist.permute(1, 0).view(B, H, W)  # [B, H, W]

            for i in range(B):
                pmap = m_dist[i : i + 1].unsqueeze(0)  # [1,1,H,W]
                pmap = F.interpolate(pmap, size=self.cfg.IMAGE_SIZE,
                                      mode="bilinear", align_corners=False)
                pmap_np = _gaussian_smooth(pmap.squeeze().cpu().numpy(),
                                            self.cfg.HEATMAP_SIGMA)
                pixel_maps.append(pmap_np)
                image_scores.append(float(pmap_np.max()))

            labels.extend([0 if lb == "normal" else 1 for lb in batch_labels])
            paths.extend(batch_paths)

        return ScoreResult(
            image_scores=np.array(image_scores, dtype=np.float64),
            labels=np.array(labels, dtype=np.int64),
            paths=paths,
            pixel_maps=np.stack(pixel_maps, axis=0),
        )


def _gaussian_smooth(arr, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)
