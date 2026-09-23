from __future__ import annotations

from flask import Flask

from src.db.postgres_client import init_db


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../dashboard/templates", static_folder="../dashboard/static")

    init_db()

    from src.api.routes.alerts import alerts_bp
    from src.api.routes.dashboard import dashboard_bp
    from src.api.routes.health import health_bp
    from src.api.routes.predict import predict_bp
    from src.api.routes.replay import replay_bp
    from src.dashboard.views import views_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(replay_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(views_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
