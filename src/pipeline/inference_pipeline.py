from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.common.config import MODELS_DIR
from src.db.models import Alert, PredictionLog
from src.db.postgres_client import get_session
from src.models.detection.autoencoder import AutoencoderDetector
from src.models.detection.isolation_forest import IsolationForestDetector
from src.models.detection.one_class_svm import OneClassSVMDetector
from src.models.prediction.random_forest import RandomForestPredictor
from src.models.prediction.risk_scoring import score_risk
from src.models.prediction.xgboost_model import XGBoostPredictor
from src.threat_intel.blacklist_lookup import check_ip

DETECTOR_LOADERS = {
    "isolation_forest": lambda path: IsolationForestDetector.load(path / "isolation_forest.pkl"),
    "one_class_svm": lambda path: OneClassSVMDetector.load(path / "one_class_svm.pkl"),
}
PREDICTOR_LOADERS = {
    "random_forest": lambda path: RandomForestPredictor.load(path / "random_forest.pkl"),
    "xgboost": lambda path: XGBoostPredictor.load(path / "xgboost.pkl"),
}
BENIGN_LABELS = {"BENIGN", "Normal"}


class InferencePipeline:
    """Ties the Detection Phase (anomaly gate, runs on every flow) to the
    Prediction Phase (attack-type classifier + risk score, gated by
    detection) for one dataset. Operates on already-preprocessed feature
    rows -- the schema produced by `src.pipeline.prepare_data` and consumed
    by the replay simulator -- since that is what this demo has available in
    real time (see PROJECT_PLAN.md's real-time scoping decision)."""

    def __init__(
        self,
        dataset: str,
        detector_name: str = "autoencoder",
        predictor_name: str = "xgboost",
        force_predict: bool = False,
    ):
        self.dataset = dataset
        self.detector_name = detector_name
        self.force_predict = force_predict

        model_dir = MODELS_DIR / dataset
        detection_dir = model_dir / "detection"
        prediction_dir = model_dir / "prediction"

        with open(detection_dir / "metrics.json") as f:
            detection_metrics = json.load(f)
        self.threshold = detection_metrics[detector_name]["f1_optimal_threshold"]["threshold"]

        with open(model_dir / "feature_metadata.json") as f:
            self.feature_columns = json.load(f)["selected_features"]

        with open(prediction_dir / "class_names.json") as f:
            self.class_names = json.load(f)

        if detector_name == "autoencoder":
            self.detector = AutoencoderDetector.load(
                detection_dir / "autoencoder.keras", input_dim=len(self.feature_columns)
            )
        else:
            self.detector = DETECTOR_LOADERS[detector_name](detection_dir)

        self.predictor = PREDICTOR_LOADERS[predictor_name](prediction_dir)
        self.benign_label = next((c for c in self.class_names if c in BENIGN_LABELS), self.class_names[0])

    def process(self, row: dict, persist: bool = True) -> dict:
        X = np.array([[row.get(c, 0.0) for c in self.feature_columns]], dtype="float32")

        anomaly_score = float(self.detector.score(X)[0])
        anomaly_flagged = anomaly_score >= self.threshold

        if anomaly_flagged or self.force_predict:
            proba = self.predictor.predict_proba(X)[0]
            class_idx = int(np.argmax(proba))
            predicted_label = self.class_names[class_idx]
            confidence = float(proba[class_idx])
        else:
            predicted_label = self.benign_label
            confidence = 1.0 - min(anomaly_score / (self.threshold + 1e-9), 1.0)

        blacklist_match: Optional[dict] = None
        for ip_field in ("src_ip", "dst_ip"):
            if row.get(ip_field):
                hit = check_ip(row[ip_field])
                if hit:
                    blacklist_match = hit
                    break

        risk = score_risk(
            attack_label=predicted_label,
            confidence=confidence,
            anomaly_flagged=anomaly_flagged,
            blacklist_match=blacklist_match is not None,
        )

        result = {
            "dataset": self.dataset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "anomaly_score": anomaly_score,
            "anomaly_flagged": anomaly_flagged,
            "threshold": self.threshold,
            "threat_intel_match": blacklist_match,
            **risk.to_dict(),
        }

        if persist:
            self._persist(row, result)

        return result

    def _persist(self, row: dict, result: dict) -> None:
        now = datetime.now(timezone.utc)
        with get_session() as session:
            session.add(
                PredictionLog(
                    timestamp=now,
                    dataset_source=self.dataset,
                    predicted_attack_cat=result["attack_label"],
                    binary_flag=int(result["anomaly_flagged"]),
                    anomaly_score=result["anomaly_score"],
                    confidence=result["confidence"],
                    risk_score=result["risk_score"],
                )
            )
            if result["anomaly_flagged"]:
                session.add(
                    Alert(
                        timestamp=now,
                        dataset_source=self.dataset,
                        flow_ref=row.get("flow_ref"),
                        src_ip=row.get("src_ip"),
                        dst_ip=row.get("dst_ip"),
                        predicted_attack_cat=result["attack_label"],
                        severity=result["severity"],
                        risk_score=result["risk_score"],
                        anomaly_score=result["anomaly_score"],
                        detector_scores={self.detector_name: result["anomaly_score"]},
                        threat_intel_match=result["threat_intel_match"],
                        status="new",
                        raw_features={k: float(row[k]) for k in self.feature_columns if k in row},
                    )
                )
