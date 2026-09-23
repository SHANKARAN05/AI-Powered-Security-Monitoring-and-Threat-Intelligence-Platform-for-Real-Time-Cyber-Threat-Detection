from __future__ import annotations

from flask import Blueprint, jsonify

from src.api.pipelines import loaded_datasets
from src.common.config import MODELS_DIR

health_bp = Blueprint("health", __name__, url_prefix="/api")


@health_bp.route("/health", methods=["GET"])
def health():
    status = {}
    for dataset in ("cicids", "unsw"):
        detection_metrics = MODELS_DIR / dataset / "detection" / "metrics.json"
        prediction_metrics = MODELS_DIR / dataset / "prediction" / "metrics.json"
        status[dataset] = {
            "artifacts_present": detection_metrics.exists() and prediction_metrics.exists(),
            "pipeline_loaded": dataset in loaded_datasets(),
        }
    return jsonify({"status": "ok", "datasets": status})
