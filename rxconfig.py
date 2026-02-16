import reflex as rx

config = rx.Config(
    app_name="pdl_lt_sg_predict_app",
    port=3001,  # Different port to avoid conflicts
    db_url="sqlite:///reflex.db",
    plugins=[
        rx.plugins.TailwindV4Plugin(),
    ],
)
