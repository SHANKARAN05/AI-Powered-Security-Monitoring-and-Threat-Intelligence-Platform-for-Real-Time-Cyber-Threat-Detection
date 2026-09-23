from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.common.base_detector import evaluate_scores, pick_threshold_by_benign_percentile, pick_threshold_by_f1
from src.common.base_preprocessor import BasePreprocessor
from src.common.config import MODELS_DIR, PROCESSED_DIR
from src.models.detection.autoencoder import AutoencoderDetector
from src.models.detection.isolation_forest import IsolationForestDetector
from src.models.detection.one_class_svm import OneClassSVMDetector

BINARY_LABEL_COL = BasePreprocessor.BINARY_LABEL_COL
MULTICLASS_LABEL_COL = BasePreprocessor.MULTICLASS_LABEL_COL


def _load_split(dataset: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / dataset / f"{split}.parquet")


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-9)


def run_for_dataset(dataset: str) -> dict:
    train = _load_split(dataset, "train")
    val = _load_split(dataset, "val")
    test = _load_split(dataset, "test")
    feature_cols = [c for c in train.columns if c not in (BINARY_LABEL_COL, MULTICLASS_LABEL_COL)]

    X_train_benign = train.loc[train[BINARY_LABEL_COL] == 0, feature_cols].to_numpy(dtype="float32")
    X_val = val[feature_cols].to_numpy(dtype="float32")
    y_val = val[BINARY_LABEL_COL].to_numpy()
    X_val_benign = val.loc[val[BINARY_LABEL_COL] == 0, feature_cols].to_numpy(dtype="float32")
    X_test = test[feature_cols].to_numpy(dtype="float32")
    y_test = test[BINARY_LABEL_COL].to_numpy()

    out_dir = MODELS_DIR / dataset / "detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    detectors = {
        "isolation_forest": IsolationForestDetector(),
        "one_class_svm": OneClassSVMDetector(),
        "autoencoder": AutoencoderDetector(input_dim=len(feature_cols)),
    }

    results, val_scores, test_scores = {}, {}, {}
    for key, detector in detectors.items():
        print(f"[{dataset}] training {key} on {len(X_train_benign)} benign rows...", flush=True)
        if key == "autoencoder":
            detector.fit(X_train_benign, X_val=X_val_benign)
        else:
            detector.fit(X_train_benign)

        val_scores[key] = detector.score(X_val)
        test_scores[key] = detector.score(X_test)

        threshold_pct = pick_threshold_by_benign_percentile(detector.score(X_val_benign), percentile=99.0)
        threshold_f1 = pick_threshold_by_f1(y_val, val_scores[key])

        metrics_pct = evaluate_scores(y_test, test_scores[key], threshold_pct)
        metrics_f1 = evaluate_scores(y_test, test_scores[key], threshold_f1)
        results[key] = {
            "benign_99th_percentile_threshold": metrics_pct.to_dict(),
            "f1_optimal_threshold": metrics_f1.to_dict(),
        }

        detector.save(out_dir / ("autoencoder.keras" if key == "autoencoder" else f"{key}.pkl"))

        print(
            f"[{dataset}] {key}: F1-optimal -> "
            f"P={metrics_f1.precision:.3f} R={metrics_f1.recall:.3f} F1={metrics_f1.f1:.3f} "
            f"ROC-AUC={metrics_f1.roc_auc:.3f} PR-AUC={metrics_f1.pr_auc:.3f} FPR={metrics_f1.false_positive_rate:.4f}",
            flush=True,
        )

    ensemble_val = np.mean([_minmax(val_scores[k]) for k in detectors], axis=0)
    ensemble_test = np.mean([_minmax(test_scores[k]) for k in detectors], axis=0)
    threshold_f1 = pick_threshold_by_f1(y_val, ensemble_val)
    metrics_ensemble = evaluate_scores(y_test, ensemble_test, threshold_f1)
    results["ensemble"] = {"f1_optimal_threshold": metrics_ensemble.to_dict()}
    print(
        f"[{dataset}] ensemble: F1-optimal -> "
        f"P={metrics_ensemble.precision:.3f} R={metrics_ensemble.recall:.3f} F1={metrics_ensemble.f1:.3f} "
        f"ROC-AUC={metrics_ensemble.roc_auc:.3f} PR-AUC={metrics_ensemble.pr_auc:.3f}",
        flush=True,
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    datasets = sys.argv[1:] or ["cicids", "unsw"]
    for dataset in datasets:
        run_for_dataset(dataset)


if __name__ == "__main__":
    main()
