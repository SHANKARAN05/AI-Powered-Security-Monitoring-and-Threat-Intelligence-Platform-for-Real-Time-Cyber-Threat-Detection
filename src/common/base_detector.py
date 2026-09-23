from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class DetectionMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    false_positive_rate: float
    confusion_matrix: list

    def to_dict(self) -> Dict:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "false_positive_rate": self.false_positive_rate,
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> DetectionMetrics:
    """`scores`: higher = more anomalous. `y_true`: 1 = attack, 0 = benign."""
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return DetectionMetrics(
        threshold=float(threshold),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        f1=float(f1_score(y_true, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        false_positive_rate=float(fpr),
        confusion_matrix=cm.tolist(),
    )


def pick_threshold_by_benign_percentile(benign_scores: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(benign_scores, percentile))


def pick_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray, n_steps: int = 200) -> float:
    """Sweep candidate thresholds and pick the one maximizing F1 on labeled
    data. Training itself never sees labels -- only this threshold pick does,
    which is standard practice for semi-supervised anomaly detection."""
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99.5)
    candidates = np.linspace(lo, hi, n_steps)
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


class BaseDetector:
    """Common contract for the three semi-supervised anomaly detectors. Each
    trains on benign-only rows and exposes score(X) where higher = more
    anomalous, so evaluation/threshold logic (above) is shared across all three."""

    name: str

    def fit(self, X_benign: np.ndarray) -> "BaseDetector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path) -> None:
        raise NotImplementedError
