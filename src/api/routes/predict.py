from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.api.pipelines import get_pipeline

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

_VALID_DATASETS = ("cicids", "unsw")


@predict_bp.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    dataset = payload.get("dataset")
    features = payload.get("features")

    if dataset not in _VALID_DATASETS:
        return jsonify({"error": f"'dataset' must be one of {_VALID_DATASETS}"}), 400
    if not isinstance(features, dict):
        return jsonify({"error": "'features' must be an object of feature_name -> value"}), 400

    row = dict(features)
    row["src_ip"] = payload.get("src_ip")
    row["dst_ip"] = payload.get("dst_ip")
    row["flow_ref"] = payload.get("flow_ref")

    pipeline = get_pipeline(dataset)
    result = pipeline.process(row)
    return jsonify(result)
