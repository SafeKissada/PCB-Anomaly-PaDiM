"""สร้างภาพผลลัพธ์จาก scores_{split}.npz ที่ run_padim.py เซฟไว้ —
reuse ฟังก์ชัน plot/gallery จาก src/visual.py ของ repo หลัก
(Anomaly-Detection-THESIS) ทั้งหมด ไม่เขียน visualize logic ซ้ำ

ต่างจาก scripts/visualize.py ของ repo หลักตรงที่ไม่โหลด:
  - history.json        (PatchCore ไม่มี training loop)
  - extractor_norm_stats.pt  (PatchCore ไม่มี feature normalization)
  - checkpoint .pth     (PatchCore ไม่มี weight เทรน)

ทุกอย่างอื่นเหมือนกันเป๊ะ — scores_{split}.npz มี schema ตรงกับ repo หลัก
ทุก key (scores, y_true, labels, paths, heatmaps, orig_imgs, preproc_imgs)
ทำให้ใช้ visual.py เดิมได้โดยไม่ต้องแก้ไขอะไร

Generates result images from scores_{split}.npz saved by run_padim.py —
reuses all plot/gallery functions from src/visual.py of the main repo
(Anomaly-Detection-THESIS), no duplicated visualize logic.

Differs from the main repo's scripts/visualize.py only in that it does NOT load:
  - history.json        (PatchCore has no training loop)
  - extractor_norm_stats.pt  (PatchCore has no feature normalization step)
  - checkpoint .pth     (PatchCore has no trained weights)

Everything else is identical — scores_{split}.npz has the same schema as the
main repo on every key, so visual.py works as-is without modification.

Usage:
    python scripts/visualize_padim.py
    (configure paths in RUN.py, same as run_padim.py)
"""
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config
from src.evaluate import compute_metrics
from src.io_utils import save_scores  # ใช้เฉพาะ path helper
# เขียน OUTPUT_PATH/README.md ตอนจบ (ดูคำอธิบายเดียวกันใน scripts/run_padim.py
# — output_docs.py เป็นตัวจริงที่ต้องเรียก ไม่ใช่ io_utils.py)
from src.output_docs import write_output_path_readme
# src/visual.py อยู่ใน repo นี้โดยตรง (copy จาก repo หลัก ลบแค่
# plot_training_history ที่ต้องการ history.json ซึ่ง PatchCore ไม่มี)
# src/visual.py lives directly in this repo (copied from main repo,
# with plot_training_history stubbed out since PatchCore has no history.json)
from src.visual import (
    plot_class_distribution, plot_roc_curves, plot_pr_curves,
    plot_confusion_matrices, plot_score_distributions,
    visualize_heatmaps, browse_gallery,
    gallery_original_images, gallery_processed_images,
    gallery_preprocessed_images, gallery_preprocessed_overlay_images,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("visualize_padim")

SPLITS = ["val", "test"]


def _load_scores(split: str, cfg) -> dict:
    """โหลด scores_{split}.npz จาก SAVE_PATH คืนเป็น dict ของ numpy array

    Load scores_{split}.npz from SAVE_PATH, return as a dict of numpy arrays.
    """
    path = Path(cfg.SAVE_PATH) / f"scores_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ {path} — รัน scripts/run_padim.py ก่อนเพื่อสร้างไฟล์นี้\n"
            f"/ {path} not found — run scripts/run_padim.py first to create it."
        )
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _load_threshold(cfg) -> dict:
    """โหลด final_results_val.json เพื่อดึง threshold ที่ใช้จริง

    Load final_results_val.json to retrieve the deployment threshold.
    """
    import json
    path = Path(cfg.SAVE_PATH) / "final_results_val.json"
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ {path} — รัน scripts/run_padim.py ก่อน\n"
            f"/ {path} not found — run scripts/run_padim.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def visualize(cfg: Config):
    """สร้างภาพผลลัพธ์ทั้งหมดจาก artifacts ที่มีอยู่ใน SAVE_PATH

    Generate all result images from artifacts already in SAVE_PATH.
    """
    logger.info(f"โหลด artifacts จาก {cfg.SAVE_PATH}")
    data = {split: _load_scores(split, cfg) for split in SPLITS}

    results_info = _load_threshold(cfg)
    threshold = float(results_info["threshold"])
    logger.info(f"Threshold: {threshold:.6f}")

    # ── metrics จาก score ที่เซฟไว้ ไม่รัน inference ใหม่ ───────────
    metrics = {
        split: compute_metrics(d["scores"], d["y_true"], threshold)
        for split, d in data.items()
    }

    # ── EDA: class distribution ──────────────────────────────────────
    plot_class_distribution({
        "Validation": list(data["val"]["labels"]),
        "Test":       list(data["test"]["labels"]),
    }, cfg)

    # ── ROC / PR / Confusion / Score distribution ────────────────────
    split_meta = [
        ("Validation Set", metrics["val"],  "#4CAF50"),
        ("Test Set",       metrics["test"], "#FF5722"),
    ]
    plot_roc_curves(split_meta, cfg)
    plot_pr_curves(split_meta, cfg)
    plot_confusion_matrices(split_meta, cfg)
    plot_score_distributions(split_meta, threshold, cfg)

    # ── Heatmaps ─────────────────────────────────────────────────────
    for split, split_name in [("val", "Validation"), ("test", "Test")]:
        d = data[split]
        visualize_heatmaps(
            d["paths"], d["orig_imgs"], d["heatmaps"], d["labels"],
            d["scores"], threshold, split_name, cfg,
            n_samples=20, image_kind="rgb")

        if cfg.COLOR_MODE != "RGB":
            visualize_heatmaps(
                d["paths"], d["preproc_imgs"], d["heatmaps"], d["labels"],
                d["scores"], threshold, split_name, cfg,
                n_samples=20, image_kind="preproc")

    # ── Gallery ──────────────────────────────────────────────────────
    split_arrays = {
        split: dict(
            paths=d["paths"], labels=d["labels"], scores=d["scores"],
            hmaps=d["heatmaps"], imgs=d["orig_imgs"],
            preproc_imgs=d["preproc_imgs"],
            gt=metrics[split]["gt"], pred=metrics[split]["pred"],
        )
        for split, d in data.items()
    }

    _rows = []
    for split_name, d in split_arrays.items():
        for i, path in enumerate(d["paths"]):
            _rows.append({
                "split"        : split_name,
                "idx_in_split" : i,
                "filename"     : Path(str(path)).name,
                "path"         : str(path),
                "label_gt"     : str(d["labels"][i]),
                "pred_label"   : "anomaly" if d["pred"][i] == 1 else "normal",
                "score"        : float(d["scores"][i]),
                "correct"      : bool(d["gt"][i] == d["pred"][i]),
            })

    df_gallery = pd.DataFrame(_rows)
    df_gallery.to_csv(
        Path(cfg.SAVE_PATH) / "gallery_index.csv", index=False)

    _ = browse_gallery(df_gallery, split_arrays, cfg,
                        split="test", correct=False, n=100)
    _ = browse_gallery(df_gallery, split_arrays, cfg,
                        split="val",  correct=False, n=100)

    for split_name in SPLITS:
        _ = gallery_original_images(
                df_gallery, split_arrays, cfg, split=split_name, n=20, ncols=5)
        _ = gallery_processed_images(
                df_gallery, split_arrays, cfg, split=split_name, n=20, ncols=5)

    if cfg.COLOR_MODE != "RGB":
        for split_name in SPLITS:
            _ = gallery_preprocessed_images(
                    df_gallery, split_arrays, cfg,
                    split=split_name, n=20, ncols=5)
            _ = gallery_preprocessed_overlay_images(
                    df_gallery, split_arrays, cfg,
                    split=split_name, n=20, ncols=5)

    write_output_path_readme(cfg)
    logger.info(f"ภาพทั้งหมดบันทึกลง {cfg.OUTPUT_PATH}")


if __name__ == "__main__":
    from RUN import OVERRIDES
    cfg = Config(**OVERRIDES)
    visualize(cfg)