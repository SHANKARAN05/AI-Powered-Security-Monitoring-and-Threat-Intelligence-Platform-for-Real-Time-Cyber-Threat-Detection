from __future__ import annotations

import threading

from flask import Blueprint, jsonify, request

from src.api.pipelines import get_pipeline
from src.pipeline.replay_simulator import replay

replay_bp = Blueprint("replay", __name__, url_prefix="/api")

_state = {"thread": None, "stop_event": None, "dataset": None}


@replay_bp.route("/replay/start", methods=["POST"])
def start_replay():
    if _state["thread"] and _state["thread"].is_alive():
        return jsonify({"error": f"replay already running for {_state['dataset']}"}), 409

    payload = request.get_json(force=True, silent=True) or {}
    dataset = payload.get("dataset", "cicids")
    n_rows = int(payload.get("n_rows", 500))
    delay = float(payload.get("delay", 1.0))

    stop_event = threading.Event()
    thread = threading.Thread(
        target=replay,
        kwargs={
            "dataset": dataset,
            "n_rows": n_rows,
            "delay_seconds": delay,
            "stop_event": stop_event,
            "pipeline": get_pipeline(dataset),
        },
        daemon=True,
    )
    _state.update(thread=thread, stop_event=stop_event, dataset=dataset)
    thread.start()

    return jsonify({"status": "started", "dataset": dataset, "n_rows": n_rows, "delay": delay})


@replay_bp.route("/replay/stop", methods=["POST"])
def stop_replay():
    if not _state["thread"] or not _state["thread"].is_alive():
        return jsonify({"status": "not running"})
    _state["stop_event"].set()
    return jsonify({"status": "stopping", "dataset": _state["dataset"]})


@replay_bp.route("/replay/status", methods=["GET"])
def replay_status():
    running = bool(_state["thread"] and _state["thread"].is_alive())
    return jsonify({"running": running, "dataset": _state["dataset"] if running else None})
