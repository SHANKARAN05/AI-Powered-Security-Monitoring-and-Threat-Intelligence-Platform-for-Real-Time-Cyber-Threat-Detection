from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> Dict:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names), output_dict=True, zero_division=0
    )
    return {
        "per_class": report,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def balanced_class_weight(y: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}


class BasePredictor:
    """Common contract for the three multiclass attack-type classifiers
    (Random Forest, XGBoost, small NN) so the training/eval driver can treat
    them interchangeably."""

    name: str

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, path) -> None:
        raise NotImplementedError
