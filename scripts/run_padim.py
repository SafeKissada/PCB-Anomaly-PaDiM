"""รัน PaDiM แบบ end-to-end บน dataset เดียวกับ repo หลัก"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config, set_seed
from src.data.dataset import build_datasets_and_loaders
from src.evaluate import compute_metrics, select_percentile_threshold
from src.io_utils import save_final_results, save_scores
from src.models.padim import PaDiM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_padim")


def run(cfg: Config):
    set_seed(cfg.SEED)
    logger.info(f"Loading data จาก {cfg.DATA_ROOT} (split cache: {cfg.SPLIT_CACHE_PATH})")
    data = build_datasets_and_loaders(cfg)
    logger.info(f"Train (normal เท่านั้น): {len(data['df_train'])} ภาพ | "
                f"Val: {len(data['df_val'])} | Test: {len(data['df_test'])}")

    model = PaDiM(cfg)
    model.fit(data["normal_loader"])

    val_result = model.score(data["val_loader"])
    test_result = model.score(data["test_loader"])

    threshold = select_percentile_threshold(val_result.image_scores, val_result.labels, cfg)
    logger.info(f"Threshold (percentile={cfg.THRESHOLD_PERCENTILE}): {threshold:.6f}")

    for split_name, result in [("val", val_result), ("test", test_result)]:
        metrics = compute_metrics(result.image_scores, result.labels, threshold)
        logger.info(
            f"[{split_name}] AUC={metrics['auc']:.3f} AP={metrics['ap']:.3f} "
            f"Acc={metrics['acc']:.3f} Recall={metrics['recall']:.3f} "
            f"EscapeRate={metrics['escape_rate']:.3f} AutoClearRate={metrics['auto_clear_rate']:.3f}")
        save_final_results(cfg, split_name, metrics, threshold)
        save_scores(cfg, split_name, result.image_scores, result.labels, result.paths)

    logger.info(f"ผลลัพธ์ทั้งหมดถูกเซฟไว้ที่ {cfg.OUTPUT_PATH}")
    return val_result, test_result


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
