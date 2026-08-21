"""
รัน PaDiM แบบ end-to-end บน dataset เดียวกับ repo หลัก
(Anomaly-Detection-THESIS) — metric ที่ได้เทียบกับ EXPERIMENT 0 (ConvNeXt+AE)
ได้โดยตรงเพราะใช้ split/evaluate เดียวกัน

output แบ่งเป็น 2 โฟลเดอร์:
  SAVE_PATH   — ตัวเลข/log: scores_{split}.npz, final_results_{split}.json,
                roc_curve_data_{split}.csv, README.md
  OUTPUT_PATH — ภาพ: (เพิ่มในอนาคต ถ้าต้องการ gallery/heatmap .png)

ฟังก์ชัน run() ถูกเรียกได้จาก:
  RUN.py           → รันรอบเดียว
  RUN_multi_seed.py → รันซ้ำหลาย seed โดย reuse OVERRIDES เดิม
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config, set_seed
from src.data.dataset import build_datasets_and_loaders
from src.evaluate import (compute_metrics, select_percentile_threshold,
                           compute_naive_baseline_metrics)
from src.io_utils import save_final_results, save_scores, save_roc_csv
# write_save_path_readme() มี 2 เวอร์ชันในโค้ดที่ reuse มา: ตัวเก่าใน
# src/io_utils.py (เขียน "PatchCore" ตายตัว, ไม่ list cost_aware_sweep.csv/
# gallery_index.csv) กับตัวใหม่ใน src/output_docs.py (dynamic, ใช้
# cfg.METHOD_NAME, list ไฟล์ทุกตัวที่มีจริง) — ใช้ตัวจาก output_docs.py
# เสมอ เพราะตรงกับ checklist ของ BASELINE_IMPLEMENTATION_GUIDE.md
# ("README.md ถูกสร้างใน SAVE_PATH และ OUTPUT_PATH")
#
# There are 2 versions of write_save_path_readme() in the reused code: the
# old one in src/io_utils.py (hardcodes "PatchCore", doesn't list
# cost_aware_sweep.csv/gallery_index.csv) and the newer one in
# src/output_docs.py (dynamic, uses cfg.METHOD_NAME, lists every file that
# actually exists) — always use the output_docs.py version, since it
# matches BASELINE_IMPLEMENTATION_GUIDE.md's checklist ("README.md is
# created in both SAVE_PATH and OUTPUT_PATH").
from src.output_docs import write_save_path_readme
from src.models.padim import PaDiM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_padim")


def run(cfg: Config):
    """รัน PaDiM 1 รอบเต็ม: split → fit → score → threshold → save

    split caching ทำงานอัตโนมัติผ่าน build_datasets_and_loaders() —
    รอบแรก: compute + save cache ลง cfg.SPLIT_CACHE_PATH
    รอบถัดไป: โหลด cache โดยตรง ไม่ compute ซ้ำ

    Split caching is automatic via build_datasets_and_loaders():
    first run computes and saves the cache; subsequent runs load it directly.
    """
    set_seed(cfg.SEED)

    logger.info(
        f"Loading data from {cfg.DATA_ROOT} "
        f"(split cache: {cfg.SPLIT_CACHE_PATH})"
    )
    data = build_datasets_and_loaders(cfg)
    logger.info(
        f"Train (normal only): {len(data['df_train'])} images | "
        f"Val: {len(data['df_val'])} | Test: {len(data['df_test'])}"
    )

    model = PaDiM(cfg)
    model.fit(data["normal_loader"])

    val_result  = model.score(data["val_loader"])
    test_result = model.score(data["test_loader"])

    # y_true (int 0/1) ใช้คำนวณ metric เสมอ — ไม่ใช้ result.labels (string)
    # ซึ่งเป็นชื่อ class สำหรับ display เท่านั้น (ดู src/models/base.py)
    #
    # Always use y_true (int 0/1) for metric computation, not result.labels
    # (string class name for display only — see src/models/base.py).
    threshold = select_percentile_threshold(
        val_result.image_scores, val_result.y_true, cfg)
    logger.info(
        f"Threshold (percentile={cfg.THRESHOLD_PERCENTILE}): {threshold:.6f}"
    )

    # คำนวณ naive baseline บน val และ test แยกกัน — ใช้ cfg.SEED เดียวกัน
    # กับ pipeline ทั้งหมด เพื่อให้ random_prior reproduce ได้ข้าม run
    # (compute_naive_baseline_metrics() ใช้ RandomState(seed) แยกต่างหาก
    # ไม่แตะ global RNG ที่ set_seed() ตั้งไว้)
    #
    # Compute naive baselines for val and test separately — using the same
    # cfg.SEED as the rest of the pipeline so random_prior reproduces across
    # runs. (compute_naive_baseline_metrics() uses its own RandomState(seed),
    # never touching the global RNG set_seed() configured.)
    naive = {
        'val':  compute_naive_baseline_metrics(val_result.y_true,  cfg.SEED),
        'test': compute_naive_baseline_metrics(test_result.y_true, cfg.SEED),
    }

    for split_name, result in [("val", val_result), ("test", test_result)]:
        metrics = compute_metrics(
            result.image_scores, result.y_true, threshold)
        logger.info(
            f"[{split_name}] AUC={metrics['auc']:.4f}  "
            f"AP={metrics['ap']:.4f}  "
            f"EscapeRate={metrics['escape_rate']:.4f}  "
            f"AutoClearRate={metrics['auto_clear_rate']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )

        # ── SAVE_PATH: ตัวเลข/log ─────────────────────────────────────
        save_scores(
            cfg, split_name,
            result.image_scores, result.y_true,
            result.labels, result.paths,
            result.pixel_maps, result.orig_imgs, result.preproc_imgs,
        )
        save_final_results(cfg, split_name, metrics, threshold,
                           naive_baselines=naive[split_name])
        save_roc_csv(cfg, split_name,
                     result.image_scores, result.y_true)

    # README เขียนท้ายสุดหลังทุก split เสร็จ เพื่อให้ list ไฟล์ครบ
    # (ถ้าเขียนหลัง val เสร็จ ไฟล์ของ test ยังไม่มีจะไม่ถูก list)
    #
    # Write README last, after all splits, so every file is listed.
    # (Writing after val would miss test files that don't exist yet.)
    write_save_path_readme(cfg)

    logger.info(
        f"All artifacts saved → SAVE_PATH: {cfg.SAVE_PATH}"
    )
    return val_result, test_result


if __name__ == "__main__":
    cfg = Config()
    run(cfg)