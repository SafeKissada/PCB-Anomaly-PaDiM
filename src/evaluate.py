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


def compute_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                                      scores=None) -> Dict:
    """คำนวณ metric จาก prediction ที่กำหนดมาแล้ว (ไม่ต้องมี threshold) —
    ใช้โดย compute_naive_baseline_metrics() ซึ่ง naive baseline แต่ละแบบ
    กำหนด y_pred แบบ hard-coded ไม่ผ่าน threshold เลย

    auc/ap เป็น NaN เสมอถ้า scores=None เพราะ ranking metric ต้องการ
    continuous score ซึ่ง naive baseline ไม่มี

    Computes metrics from already-determined predictions (no threshold
    needed) — used by compute_naive_baseline_metrics(), where each naive
    baseline hard-codes y_pred without going through a threshold.

    auc/ap are always NaN when scores=None because ranking metrics require
    a continuous score, which naive baselines don't have.
    """
    tt = int(np.sum((y_true == 1) & (y_pred == 1)))
    tf = int(np.sum((y_true == 1) & (y_pred == 0)))
    ft = int(np.sum((y_true == 0) & (y_pred == 1)))
    ff = int(np.sum((y_true == 0) & (y_pred == 0)))
    n  = len(y_true)

    precision = tt / (tt + ft) if (tt + ft) > 0 else 0.0
    recall    = tt / (tt + tf) if (tt + tf) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    acc       = (tt + ff) / n if n > 0 else 0.0

    auto_clear_rate = ff / n if n > 0 else 0.0
    escape_rate     = tf / (tt + tf) if (tt + tf) > 0 else 0.0
    residual_fcr    = ft / (tt + ft) if (tt + ft) > 0 else float('nan')

    return dict(
        auc=float('nan'), ap=float('nan'),
        acc=acc, precision=precision, recall=recall, f1=f1,
        tt=tt, tf=tf, ft=ft, ff=ff,
        auto_clear_rate=auto_clear_rate,
        escape_rate=escape_rate,
        residual_fcr=residual_fcr,
    )


def compute_naive_baseline_metrics(y_true: np.ndarray, seed: int) -> Dict:
    """คำนวณ metric ของ naive baseline 3 แบบ สำหรับเทียบกับผลโมเดลจริง
    (เรียกเฉพาะบน val/test เท่านั้น — ไม่เรียกบน train เพราะ train ของ
    repo นี้มีแต่ภาพ normal โดยการออกแบบ ทุก field จะกลายเป็น NaN/0
    ที่ไม่มีความหมาย)

    - always_normal  : pred=0 ทุกภาพ — จำลองสถานการณ์ไม่ตรวจเลย ปล่อยผ่านหมด
    - always_anomaly : pred=1 ทุกภาพ — จำลองสถานการณ์ตีว่าเสียหมดทุกชิ้น
    - random_prior   : สุ่ม pred=1 ด้วยความน่าจะเป็น = สัดส่วน anomaly
                        จริงใน y_true — ใช้ np.random.RandomState(seed)
                        แยกต่างหาก ไม่แตะ global RNG ของ pipeline

    random_prior ต้อง log seed ที่ใช้ไว้เสมอ (ผ่านการเซฟลง
    final_results*.json) เพื่อให้ผลสุ่มสามารถ reproduce ได้ข้าม run

    Compute metrics for 3 naive baselines (val/test only):
    - always_normal  : pred=0 for every image — simulates passing everything
    - always_anomaly : pred=1 for every image — simulates rejecting everything
    - random_prior   : pred=1 at the true anomaly rate in y_true —
                        uses its own np.random.RandomState(seed),
                        never touches the pipeline's global RNG

    The seed used for random_prior must always be logged (via
    final_results*.json) so the random result reproduces across runs.
    """
    n = len(y_true)
    prior = float(np.mean(y_true))
    rng = np.random.RandomState(seed)
    pred_random = (rng.random_sample(n) < prior).astype(int)

    return {
        'seed':         seed,
        'always_normal':  compute_metrics_from_predictions(
                              y_true, np.zeros(n, dtype=int), scores=None),
        'always_anomaly': compute_metrics_from_predictions(
                              y_true, np.ones(n, dtype=int),  scores=None),
        'random_prior':   compute_metrics_from_predictions(
                              y_true, pred_random,            scores=None),
    }