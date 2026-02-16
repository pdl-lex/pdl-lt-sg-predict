import os
import reflex as rx

# Lokal: port=3001, kein frontend_path
# Docker/Prod: port=3000, frontend_path="/sgpredict"
IS_DOCKER = os.environ.get("REFLEX_ENV") == "prod"

config = rx.Config(
    app_name="pdl_lt_sg_predict_app",
    port=3000 if IS_DOCKER else 3001,
    backend_host="0.0.0.0" if IS_DOCKER else "127.0.0.1",
    frontend_path="/sgpredict" if IS_DOCKER else "",
    db_url="sqlite:///reflex.db",
    plugins=[
        rx.plugins.TailwindV4Plugin(),
    ],
)
