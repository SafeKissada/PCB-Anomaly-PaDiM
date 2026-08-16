import json
from pathlib import Path

import numpy as np


def save_final_results(cfg, split_name: str, metrics: dict, threshold: float,
                        extra: dict = None) -> Path:
    """เซฟ final_results.json ในรูปแบบเดียวกับ repo หลัก (config snapshot +
    metrics) เพื่อให้เขียนสคริปต์เทียบผลข้าม repo ได้ง่าย
    """
    out = {
        "experiment": cfg.EXPERIMENT,
        "backbone": cfg.BACKBONE,
        "split": split_name,
        "threshold": threshold,
        "threshold_percentile": cfg.THRESHOLD_PERCENTILE,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("cm", "fpr", "tpr", "gt", "pred", "scores")},
        "confusion_matrix": metrics["cm"].tolist(),
    }
    if extra:
        out.update(extra)

    out_path = Path(cfg.OUTPUT_PATH) / f"final_results_{split_name}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def save_scores(cfg, split_name: str, scores: np.ndarray, labels: np.ndarray,
                 paths: list) -> Path:
    out_path = Path(cfg.OUTPUT_PATH) / f"scores_{split_name}.npz"
    np.savez(out_path, scores=scores, labels=labels, paths=np.array(paths))
    return out_path
