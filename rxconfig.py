import reflex as rx

config = rx.Config(
    app_name="pdl_lt_sg_predict_app",
    backend_host="0.0.0.0",
    db_url="sqlite:///reflex.db",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
)