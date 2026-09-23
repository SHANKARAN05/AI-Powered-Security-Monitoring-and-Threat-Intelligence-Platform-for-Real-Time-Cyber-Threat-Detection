from __future__ import annotations

import json
import sys

import joblib
import pandas as pd

from src.common.base_predictor import balanced_class_weight, evaluate_predictions
from src.common.base_preprocessor import BasePreprocessor
from src.common.config import MODELS_DIR, PROCESSED_DIR
from src.models.prediction.nn_classifier import NNPredictor
from src.models.prediction.random_forest import RandomForestPredictor
from src.models.prediction.xgboost_model import XGBoostPredictor

BINARY_LABEL_COL = BasePreprocessor.BINARY_LABEL_COL
MULTICLASS_LABEL_COL = BasePreprocessor.MULTICLASS_LABEL_COL


def _load_split(dataset: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / dataset / f"{split}.parquet")


def run_for_dataset(dataset: str) -> dict:
    train = _load_split(dataset, "train")
    val = _load_split(dataset, "val")
    test = _load_split(dataset, "test")
    feature_cols = [c for c in train.columns if c not in (BINARY_LABEL_COL, MULTICLASS_LABEL_COL)]

    encoder = joblib.load(MODELS_DIR / dataset / "multiclass_encoder.pkl")
    class_names = list(encoder.classes_)

    X_train = train[feature_cols].to_numpy(dtype="float32")
    y_train = train[MULTICLASS_LABEL_COL].to_numpy()
    X_val = val[feature_cols].to_numpy(dtype="float32")
    y_val = val[MULTICLASS_LABEL_COL].to_numpy()
    X_test = test[feature_cols].to_numpy(dtype="float32")
    y_test = test[MULTICLASS_LABEL_COL].to_numpy()

    out_dir = MODELS_DIR / dataset / "prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"[{dataset}] training random_forest on {len(X_train)} rows, {len(class_names)} classes...", flush=True)
    rf = RandomForestPredictor().fit(X_train, y_train)
    rf_metrics = evaluate_predictions(y_test, rf.predict(X_test), class_names)
    rf.save(out_dir / "random_forest.pkl")
    results["random_forest"] = rf_metrics
    print(f"[{dataset}] random_forest: macro_f1={rf_metrics['macro_f1']:.3f} weighted_f1={rf_metrics['weighted_f1']:.3f}", flush=True)

    importances = rf.feature_importances(feature_cols)
    with open(out_dir / "feature_importance.json", "w") as f:
        json.dump(importances.to_dict(), f, indent=2)

    print(f"[{dataset}] training xgboost...", flush=True)
    xgb = XGBoostPredictor().fit(X_train, y_train)
    xgb_metrics = evaluate_predictions(y_test, xgb.predict(X_test), class_names)
    xgb.save(out_dir / "xgboost.pkl")
    results["xgboost"] = xgb_metrics
    print(f"[{dataset}] xgboost: macro_f1={xgb_metrics['macro_f1']:.3f} weighted_f1={xgb_metrics['weighted_f1']:.3f}", flush=True)

    print(f"[{dataset}] training nn_classifier...", flush=True)
    class_weight = balanced_class_weight(y_train)
    nn = NNPredictor(input_dim=len(feature_cols), n_classes=len(class_names))
    nn.fit(X_train, y_train, X_val=X_val, y_val=y_val, class_weight=class_weight)
    nn_metrics = evaluate_predictions(y_test, nn.predict(X_test), class_names)
    nn.save(out_dir / "nn_classifier.keras")
    results["nn_classifier"] = nn_metrics
    print(f"[{dataset}] nn_classifier: macro_f1={nn_metrics['macro_f1']:.3f} weighted_f1={nn_metrics['weighted_f1']:.3f}", flush=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    return results


def main() -> None:
    datasets = sys.argv[1:] or ["cicids", "unsw"]
    for dataset in datasets:
        run_for_dataset(dataset)


if __name__ == "__main__":
    main()
