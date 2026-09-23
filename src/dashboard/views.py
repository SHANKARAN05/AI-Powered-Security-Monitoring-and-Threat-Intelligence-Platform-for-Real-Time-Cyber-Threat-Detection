from __future__ import annotations

from flask import Blueprint, redirect, render_template, url_for

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def index():
    return redirect(url_for("views.dashboard_page"))


@views_bp.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@views_bp.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@views_bp.route("/models")
def models_page():
    return render_template("models.html")
