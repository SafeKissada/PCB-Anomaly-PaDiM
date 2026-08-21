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
    y_true       : np.ndarray [N]        ground-truth label เป็น int: 0 = normal, 1 = anomaly
                                          ใช้คำนวณ metric ทุกตัว (AUROC, escape_rate ฯลฯ)
    labels       : list[str]  [N]        ชื่อ class เป็น string เช่น "good" / "defect" —
                                          ใช้สำหรับ display/gallery และ predictions CSV
                                          คนละตัวกับ y_true โดยเจตนา เพื่อให้ตรง schema
                                          ของ repo หลัก (Anomaly-Detection-THESIS) ที่ใช้
                                          key เดียวกันใน scores_{split}.npz
    paths        : list[str]  [N]        path ของภาพแต่ละอัน (ใช้ trace กลับตอนทำ error analysis)
    pixel_maps   : np.ndarray [N, H, W]  heatmap ระดับ pixel — kNN distance map ที่
                                          upsample + Gaussian smooth แล้ว ตรงกับ key
                                          'heatmaps' ใน scores_{split}.npz ของ repo หลัก
                                          ใช้กับ visualize.py เดียวกันได้เลย
    orig_imgs    : np.ndarray [N, H, W, 3]  ภาพ RGB ต้นฉบับก่อน normalize (float32, 0–1)
                                          ใช้ overlay heatmap ตอน visualize ตรงกับ key
                                          'orig_imgs' ใน repo หลักเป๊ะ
    preproc_imgs : np.ndarray [N, H, W, 3]  ภาพหลัง preprocessing จริง (grayscale/CLAHE
                                          ถ้าเปิดใช้) ตรงกับ key 'preproc_imgs' ใน repo
                                          หลัก — ถ้า COLOR_MODE=RGB จะเหมือน orig_imgs

    image_scores : np.ndarray [N]        per-image anomaly score (higher = more anomalous)
    y_true       : np.ndarray [N]        ground-truth label as int: 0 = normal, 1 = anomaly
                                          used for computing every metric (AUROC, escape_rate, etc.)
    labels       : list[str]  [N]        class name as string, e.g. "good" / "defect" —
                                          used for display/gallery and predictions CSV;
                                          intentionally separate from y_true to match the
                                          schema of the main repo (Anomaly-Detection-THESIS),
                                          which uses the same key names in scores_{split}.npz
    paths        : list[str]  [N]        per-image file path (used to trace back during error analysis)
    pixel_maps   : np.ndarray [N, H, W]  pixel-level anomaly heatmap — upsampled + Gaussian-
                                          smoothed kNN distance map, matching the 'heatmaps'
                                          key in the main repo's scores_{split}.npz;
                                          directly usable with the same visualize.py
    orig_imgs    : np.ndarray [N, H, W, 3]  original RGB image before normalization
                                          (float32, 0–1), for heatmap overlay at visualize
                                          time — matches 'orig_imgs' key in the main repo
    preproc_imgs : np.ndarray [N, H, W, 3]  image after real preprocessing (grayscale/CLAHE
                                          if enabled), matching 'preproc_imgs' key in the
                                          main repo — equals orig_imgs when COLOR_MODE=RGB
    """
    image_scores : np.ndarray
    y_true       : np.ndarray
    labels       : list
    paths        : list
    pixel_maps   : np.ndarray = None
    orig_imgs    : np.ndarray = None
    preproc_imgs : np.ndarray = None


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