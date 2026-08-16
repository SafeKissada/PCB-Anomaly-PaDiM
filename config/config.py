"""
Config สำหรับ PaDiM — ดู README.md หัวข้อ "การเทียบผลกับ baseline เดิม"
ก่อนรันจริงเสมอ (ต้องตั้ง SPLIT_CACHE_PATH ให้ตรงกับ repo หลัก)
"""
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
import torch


@dataclass
class Config:
    # ── Data (ต้องตรงกับ repo หลักถ้าจะเทียบผลกัน) ──────────────────
    DATA_ROOT: str = "dataset root path (contains good/ and defect/ subfolders)"
    GOOD_DIRNAME: str = "good"
    DEFECT_DIRNAME: str = "defect"
    SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    SPLIT_CACHE_PATH: str = "splits/split_assignment.csv"
    GROUP_ID_REGEX: Optional[str] = None
    VALID_EXT: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    SAVE_PATH: str = "save/logs"
    OUTPUT_PATH: str = "save/results"

    SEED: int = 42
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    EXPERIMENT: str = "PaDiM_Baseline"

    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    USE_AUGMENTATION: bool = False
    AUG_COLOR_JITTER: float = 0.20

    USE_GRAYSCALE: bool = False
    USE_GRAYSCALE_EQUALIZATION: bool = False
    USE_CLAHE: bool = False
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

    THRESHOLD_PERCENTILE: float = 95.0
    HEATMAP_SIGMA: float = 4.0

    BACKBONE: str = "wide_resnet50_2"
    PRETRAINED: bool = True

    # ── PaDiM-specific ────────────────────────────────────────────────
    FEATURE_LAYERS: tuple = ("layer1", "layer2", "layer3")
    # จำนวน channel ที่สุ่มเลือกหลัง concat ทุก layer (paper ใช้ random
    # dimension reduction แทน PCA เพื่อความเร็ว, ค่ามาตรฐาน 100-550)
    NUM_SELECTED_CHANNELS: int = 100
    # ค่า regularization บวกเข้า diagonal ของ covariance matrix แต่ละ patch
    # (กัน matrix singular ตอน invert — จำเป็นมากถ้าจำนวนภาพ train น้อยกว่า
    # จำนวน channel ที่เลือก เช่น group 5 ที่มี normal แค่ ~130 ภาพ)
    COV_REG_EPSILON: float = 0.01


    _DATA_ROOT_PLACEHOLDER = "dataset root path (contains good/ and defect/ subfolders)"

    @property
    def COLOR_MODE(self) -> str:
        if self.USE_GRAYSCALE_EQUALIZATION and self.USE_CLAHE:
            return "GRAYSCALE_EQUALIZATION_CLAHE"
        elif self.USE_GRAYSCALE_EQUALIZATION:
            return "GRAYSCALE_EQUALIZATION"
        elif self.USE_CLAHE:
            return "GRAYSCALE_CLAHE"
        elif self.USE_GRAYSCALE:
            return "GRAYSCALE"
        else:
            return "RGB"

    def __post_init__(self):
        for p in [self.SAVE_PATH, self.OUTPUT_PATH]:
            Path(p).mkdir(parents=True, exist_ok=True)

        ratio_sum = sum(self.SPLIT_RATIOS)
        if not np.isclose(ratio_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Config.SPLIT_RATIOS must sum to 1.0, got {self.SPLIT_RATIOS} "
                f"(sums to {ratio_sum}).")
        if len(self.SPLIT_RATIOS) != 3:
            raise ValueError(
                f"Config.SPLIT_RATIOS must have exactly 3 values, got "
                f"{len(self.SPLIT_RATIOS)}: {self.SPLIT_RATIOS}")

        if self.DATA_ROOT == self._DATA_ROOT_PLACEHOLDER:
            raise ValueError(
                "Config.DATA_ROOT is still the default placeholder string. Set "
                "it to a real folder containing "
                f"{self.GOOD_DIRNAME!r} and {self.DEFECT_DIRNAME!r} subfolders.\n"
                "แนะนำ: ให้ชี้ไปที่ DATA_ROOT เดียวกับ repo หลัก "
                "(Anomaly-Detection-THESIS) และตั้ง SPLIT_CACHE_PATH ให้ชี้ไปที่ "
                "splits/split_assignment.csv ไฟล์เดียวกัน เพื่อให้ train/val/test "
                "membership ตรงกันเป๊ะระหว่างสอง repo")
        if not Path(self.DATA_ROOT).is_dir():
            raise FileNotFoundError(
                f"Config.DATA_ROOT does not exist or is not a directory: "
                f"{self.DATA_ROOT!r}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
