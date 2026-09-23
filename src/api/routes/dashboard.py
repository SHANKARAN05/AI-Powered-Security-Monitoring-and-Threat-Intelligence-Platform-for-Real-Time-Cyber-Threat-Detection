from __future__ import annotations

import json
from collections import Counter

from flask import Blueprint, jsonify

from src.common.config import MODELS_DIR
from src.db.models import Alert
from src.db.postgres_client import get_session

dashboard_bp = Blueprint("dashboard_api", __name__, url_prefix="/api")


@dashboard_bp.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    with get_session() as session:
        alerts = session.query(Alert).order_by(Alert.timestamp.desc()).limit(2000).all()
        severity_counts = Counter(a.severity for a in alerts)
        attack_counts = Counter(a.predicted_attack_cat for a in alerts)
        src_ip_counts = Counter(a.src_ip for a in alerts if a.src_ip)
        recent = [
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "dataset_source": a.dataset_source,
                "predicted_attack_cat": a.predicted_attack_cat,
                "severity": a.severity,
                "risk_score": a.risk_score,
                "status": a.status,
            }
            for a in alerts[:20]
        ]
        total_alerts = len(alerts)

    return jsonify(
        {
            "total_alerts": total_alerts,
            "severity_breakdown": dict(severity_counts),
            "attack_type_breakdown": dict(attack_counts),
            "top_source_ips": dict(src_ip_counts.most_common(10)),
            "recent_alerts": recent,
        }
    )


@dashboard_bp.route("/models", methods=["GET"])
def model_metrics():
    out = {}
    for dataset in ("cicids", "unsw"):
        out[dataset] = {}
        for phase in ("detection", "prediction"):
            path = MODELS_DIR / dataset / phase / "metrics.json"
            if path.exists():
                with open(path) as f:
                    out[dataset][phase] = json.load(f)
    return jsonify(out)
