from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)


def compute_metrics(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict:
    pred = (scores >= threshold).astype(int)
    gt   = y_true
    fpr, tpr, _ = roc_curve(gt, scores)
    auc = roc_auc_score(gt, scores) if len(np.unique(gt)) == 2 else float('nan')
    ap  = average_precision_score(gt, scores) if len(np.unique(gt)) == 2 else float('nan')

    cm = confusion_matrix(gt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_flagged = tn + fp + fn + tp
    auto_clear_rate = float(tn / n_flagged) if n_flagged > 0 else float('nan')
    escape_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else float('nan')
    residual_fcr = float(fp / (fp + tp)) if (fp + tp) > 0 else float('nan')

    return dict(
        auc=auc, ap=ap,
        acc      = float(accuracy_score(gt, pred)),
        precision= float(precision_score(gt, pred, zero_division=0)),
        recall   = float(recall_score(gt, pred, zero_division=0)),
        f1       = float(f1_score(gt, pred, zero_division=0)),
        cm       = cm,
        auto_clear_rate = auto_clear_rate,
        escape_rate     = escape_rate,
        residual_fcr    = residual_fcr,
        fpr=fpr, tpr=tpr,
        gt=gt, pred=pred, scores=scores,
    )


def select_percentile_threshold(val_scores: np.ndarray, val_y: np.ndarray, cfg) -> float:
    """Deployment threshold: percentile of the validation *normal* scores."""
    val_normal_scores = val_scores[val_y == 0]
    return float(np.percentile(val_normal_scores, cfg.THRESHOLD_PERCENTILE))


def oracle_threshold_diagnostic(val_scores: np.ndarray, val_y: np.ndarray) -> Tuple[float, float]:
    """Diagnostic only: max-F1 threshold on Val (uses val anomaly labels,
    NOT used for reported metrics)."""
    precisions, recalls, thresholds_candidates = precision_recall_curve(val_y, val_scores)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    oracle_threshold = float(thresholds_candidates[best_idx])
    oracle_f1        = float(f1_scores[best_idx])
    return oracle_threshold, oracle_f1