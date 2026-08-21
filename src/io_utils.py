"""I/O utilities สำหรับ PatchCore baseline — ออกแบบให้ schema ของ output
ตรงกับ repo หลัก (Anomaly-Detection-THESIS) ทุก key เพื่อให้ script เทียบ
ผลข้าม repo ใช้ชื่อไฟล์และ key เดียวกันได้โดยไม่มี silent mismatch

การแบ่ง SAVE_PATH / OUTPUT_PATH:
  SAVE_PATH   — ตัวเลข/log: .npz, .json, .csv  (โหลดกลับคำนวณต่อได้)
  OUTPUT_PATH — ภาพ: .png เท่านั้น             (ดูด้วยตาเท่านั้น)

I/O utilities for the PatchCore baseline — designed so that output schemas
match the main repo (Anomaly-Detection-THESIS) key-for-key, letting
cross-repo comparison scripts use the same filenames and key names without
any silent mismatch.

SAVE_PATH / OUTPUT_PATH split:
  SAVE_PATH   — numeric/log: .npz, .json, .csv  (reloadable for further computation)
  OUTPUT_PATH — images: .png only               (for visual inspection only)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve


# ── SAVE_PATH artifacts (ตัวเลข/log) ─────────────────────────────────────

def save_final_results(cfg, split_name: str, metrics: dict,
                        threshold: float,
                        naive_baselines: dict = None) -> Path:
    """เซฟ final_results_{split_name}.json ลง SAVE_PATH — รูปแบบเดียวกับ
    repo หลัก (config snapshot + metrics + naive_baselines) เพื่อให้ script
    เทียบผลข้าม repo โหลด key เดียวกันได้โดยไม่ต้องแก้ไขอะไรเพิ่ม

    naive_baselines: dict จาก compute_naive_baseline_metrics() —
    เก็บ 3 baseline (always_normal, always_anomaly, random_prior) พร้อม
    seed เพื่อให้ผล random_prior reproduce ได้ข้าม run

    final_results_{split_name}.json goes to SAVE_PATH — same format as the
    main repo (config snapshot + metrics + naive_baselines) so cross-repo
    comparison scripts load the same keys without modification.
    """
    out = {
        "experiment"           : cfg.EXPERIMENT,
        "backbone"             : cfg.BACKBONE,
        "method"               : getattr(cfg, "METHOD_NAME", "Unknown"),
        "split"                : split_name,
        "threshold"            : threshold,
        "threshold_percentile" : cfg.THRESHOLD_PERCENTILE,
        "coreset_ratio"        : getattr(cfg, "CORESET_RATIO", None),  # None for non-PatchCore methods
        # กรอง key ที่เป็น array ขนาดใหญ่ออก เพราะ JSON ไม่รองรับ ndarray
        # และ key เหล่านี้อยู่ใน scores_{split}.npz แล้ว
        # Filter out large-array keys (JSON doesn't support ndarray;
        # those already live in scores_{split}.npz).
        "metrics" : {k: (v.tolist() if hasattr(v, 'tolist') else v)
                     for k, v in metrics.items()
                     if k not in ("cm", "fpr", "tpr", "gt", "pred", "scores")},
        "confusion_matrix" : metrics["cm"].tolist(),
    }
    if naive_baselines is not None:
        out["naive_baselines"] = naive_baselines
    out_path = Path(cfg.SAVE_PATH) / f"final_results_{split_name}.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def save_scores(cfg, split_name: str,
                scores: np.ndarray, y_true: np.ndarray,
                labels: list, paths: list,
                heatmaps: np.ndarray,
                orig_imgs: np.ndarray,
                preproc_imgs: np.ndarray) -> Path:
    """เซฟ scores_{split_name}.npz ลง SAVE_PATH ด้วย schema เดียวกับ repo
    หลัก (7 key) — visualize.py และ script เทียบผลข้าม repo โหลด key
    เดียวกันได้ทันทีโดยไม่ต้องแก้ไขอะไร

    key ที่บันทึก:
      scores       float32 [N]        : anomaly score ต่อภาพ
      y_true       int64   [N]        : 0=normal 1=anomaly (ใช้คำนวณ metric)
      labels       string  [N]        : "good"/"defect" (ใช้ display เท่านั้น)
      paths        string  [N]        : path ต้นฉบับ
      heatmaps     float32 [N,H,W]   : kNN distance map หลัง upsample+smooth
      orig_imgs    float32 [N,H,W,3] : RGB ต้นฉบับก่อน normalize
      preproc_imgs float32 [N,H,W,3] : ภาพหลัง preprocessing จริง

    Saves scores_{split_name}.npz to SAVE_PATH with the same 7-key schema
    as the main repo — visualize.py and cross-repo scripts load the same
    keys immediately without modification.
    """
    out_path = Path(cfg.SAVE_PATH) / f"scores_{split_name}.npz"
    np.savez_compressed(
        out_path,
        scores       = scores.astype(np.float32),
        y_true       = y_true.astype(np.int64),
        labels       = np.array(labels),
        paths        = np.array(paths),
        heatmaps     = heatmaps.astype(np.float32),
        orig_imgs    = orig_imgs.astype(np.float32),
        preproc_imgs = preproc_imgs.astype(np.float32),
    )
    return out_path


def save_roc_csv(cfg, split_name: str,
                  scores: np.ndarray, y_true: np.ndarray) -> Path:
    """เซฟ roc_curve_data_{split_name}.csv ลง SAVE_PATH — จุดดิบทุกจุดบน
    ROC curve (fpr, tpr, threshold) สำหรับ:
      - เปิดดูใน Excel โดยไม่ต้องรัน Python เพิ่ม
      - multi-seed ROC aggregation (vertical averaging) — โหลด fpr/tpr
        ของแต่ละ seed มา np.interp() เข้าแกน FPR ร่วมก่อน average

    Save roc_curve_data_{split_name}.csv to SAVE_PATH — raw ROC curve
    points (fpr, tpr, threshold) for:
      - Direct viewing in Excel without any extra Python
      - Multi-seed ROC aggregation (vertical averaging): load each seed's
        fpr/tpr and np.interp() onto a shared FPR axis before averaging
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    out_path = Path(cfg.SAVE_PATH) / f"roc_curve_data_{split_name}.csv"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("fpr,tpr,threshold\n")
        for a, b, c in zip(fpr, tpr, thr):
            f.write(f"{a},{b},{c}\n")
    return out_path


def write_save_path_readme(cfg) -> Path:
    """สร้าง README.md ใน SAVE_PATH อธิบายไฟล์ทุกตัวที่มีอยู่จริง ณ ตอนเรียก
    (dynamic — list เฉพาะไฟล์ที่ exist) เพื่อให้คนที่เปิดโฟลเดอร์ทราบว่า
    แต่ละไฟล์เก็บอะไรและเอาไปทำอะไรได้บ้าง

    Writes README.md into SAVE_PATH describing every file that actually
    exists at call time (dynamic — only lists files that exist), so anyone
    opening the folder knows what each file holds and what it can be used for.
    """
    import os
    from datetime import datetime

    p = Path(cfg.SAVE_PATH)
    lines = [
        f"# Artifacts ใน SAVE_PATH — PatchCore\n",
        f"สร้างอัตโนมัติเมื่อ {datetime.now().isoformat(timespec='seconds')}\n\n",
        "โฟลเดอร์นี้เก็บ **ตัวเลข/log ทั้งหมด** — ไม่มีไฟล์ภาพ\n",
        f"(ภาพอยู่ที่ `{cfg.OUTPUT_PATH}`)\n\n---\n\n",
    ]

    docs = {
        "scores_val.npz": (
            "7-key array: scores(float32), y_true(int64), labels(str), paths(str), "
            "heatmaps(float32,[N,H,W]), orig_imgs(float32,[N,H,W,3]), "
            "preproc_imgs(float32,[N,H,W,3]) — โหลดด้วย np.load(..., allow_pickle=True)"
        ),
        "scores_test.npz": "เหมือน scores_val.npz แต่สำหรับ test split",
        "final_results_val.json": (
            "config snapshot + metrics ครบชุด (val) — "
            "เทียบ AUROC/escape_rate กับ repo หลักได้โดยตรง"
        ),
        "final_results_test.json": "เหมือน final_results_val.json แต่สำหรับ test split",
        "roc_curve_data_val.csv": (
            "fpr, tpr, threshold ทุกจุดบน ROC curve (val) — "
            "เปิดใน Excel ได้เลย หรือใช้เป็น input ของ multi-seed ROC aggregation"
        ),
        "roc_curve_data_test.csv": "เหมือน roc_curve_data_val.csv แต่สำหรับ test split",
        "README.md": "ไฟล์นี้ — auto-generated",
    }

    for fname, desc in docs.items():
        fpath = p / fname
        if fpath.exists():
            size = fpath.stat().st_size
            size_str = (f"{size/1024:.1f} KB" if size < 1024**2
                        else f"{size/1024**2:.1f} MB")
            lines.append(f"## `{fname}` ({size_str})\n{desc}\n\n---\n\n")

    readme_path = p / "README.md"
    readme_path.write_text("".join(lines), encoding="utf-8")
    return readme_path