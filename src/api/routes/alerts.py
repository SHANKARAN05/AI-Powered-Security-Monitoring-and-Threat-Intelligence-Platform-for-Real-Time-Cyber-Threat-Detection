from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.db.models import Alert
from src.db.postgres_client import get_session

alerts_bp = Blueprint("alerts", __name__, url_prefix="/api")

_VALID_STATUSES = ("new", "investigating", "resolved")


def _serialize(alert: Alert) -> dict:
    return {
        "id": alert.id,
        "timestamp": alert.timestamp.isoformat(),
        "dataset_source": alert.dataset_source,
        "flow_ref": alert.flow_ref,
        "src_ip": alert.src_ip,
        "dst_ip": alert.dst_ip,
        "predicted_attack_cat": alert.predicted_attack_cat,
        "severity": alert.severity,
        "risk_score": alert.risk_score,
        "anomaly_score": alert.anomaly_score,
        "detector_scores": alert.detector_scores,
        "threat_intel_match": alert.threat_intel_match,
        "status": alert.status,
    }


@alerts_bp.route("/alerts", methods=["GET"])
def list_alerts():
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 25)), 200)
    severity = request.args.get("severity")
    dataset = request.args.get("dataset")
    status = request.args.get("status")

    with get_session() as session:
        query = session.query(Alert)
        if severity:
            query = query.filter(Alert.severity == severity)
        if dataset:
            query = query.filter(Alert.dataset_source == dataset)
        if status:
            query = query.filter(Alert.status == status)
        total = query.count()
        rows = query.order_by(Alert.timestamp.desc()).offset((page - 1) * per_page).limit(per_page).all()
        data = [_serialize(a) for a in rows]

    return jsonify({"total": total, "page": page, "per_page": per_page, "alerts": data})


@alerts_bp.route("/alerts/<int:alert_id>", methods=["GET"])
def get_alert(alert_id: int):
    with get_session() as session:
        alert = session.get(Alert, alert_id)
        if alert is None:
            return jsonify({"error": "not found"}), 404
        data = _serialize(alert)
        data["raw_features"] = alert.raw_features
    return jsonify(data)


@alerts_bp.route("/alerts/<int:alert_id>", methods=["PATCH"])
def update_alert(alert_id: int):
    payload = request.get_json(force=True, silent=True) or {}
    new_status = payload.get("status")
    if new_status not in _VALID_STATUSES:
        return jsonify({"error": f"'status' must be one of {_VALID_STATUSES}"}), 400

    with get_session() as session:
        alert = session.get(Alert, alert_id)
        if alert is None:
            return jsonify({"error": "not found"}), 404
        alert.status = new_status
        session.add(alert)
        data = _serialize(alert)

    return jsonify(data)
