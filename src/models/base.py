"""
Interface กลางที่ทุก baseline model (PatchCore, PaDiM, DRAEM, SimpleNet, RD4AD)
ต้องตาม เพื่อให้ scripts/run_*.py และ evaluate.py เดิมเรียกใช้แบบเดียวกันได้
ทุกตัว ไม่ต้องเขียน training/eval loop ซ้ำในแต่ละ method
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreResult:
    """ผลลัพธ์การให้คะแนนของ 1 split (val หรือ test)

    image_scores : np.ndarray [N]        anomaly score ระดับภาพ (ยิ่งสูง = ยิ่งผิดปกติ)
    labels       : np.ndarray [N]        0 = normal, 1 = anomaly (ตรงกับ evaluate.py เดิม)
    paths        : list[str]  [N]        path ของภาพแต่ละอัน (ใช้ trace กลับตอนทำ error analysis)
    pixel_maps   : np.ndarray [N, H, W]  heatmap ระดับ pixel (optional, None ถ้า method ไม่รองรับ)
    """
    image_scores: np.ndarray
    labels: np.ndarray
    paths: list
    pixel_maps: np.ndarray = None


class BaseAnomalyModel(ABC):
    """ทุก baseline ต้อง implement แค่ fit() กับ score() สองตัวนี้
    ส่วน threshold selection / metric computation ใช้ src/evaluate.py ตัวเดียวกัน
    กับ repo หลัก (Anomaly-Detection-THESIS) เสมอ ไม่เขียนซ้ำในแต่ละ model
    """

    @abstractmethod
    def fit(self, normal_loader) -> None:
        """เทรน/สร้าง memory bank จากภาพ normal (good/false-call) เท่านั้น
        ต้องไม่แตะภาพ defect เลย (เหมือน repo หลัก — unsupervised)"""
        raise NotImplementedError

    @abstractmethod
    def score(self, loader) -> ScoreResult:
        """ให้คะแนน anomaly กับทุกภาพใน loader (val หรือ test)"""
        raise NotImplementedError
