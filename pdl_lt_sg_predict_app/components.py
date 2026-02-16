import reflex as rx


# Setzen Sie ENABLE_TRAINING = True wenn Sie lokal auf einem starken System arbeiten
ENABLE_TRAINING = False

AVAILABLE_MODELS = [
    ("svm", "Linear SVM"),
    ("logistic", "Logistic Regression"),
    ("rf", "Random Forest"),
    ("nn", "Neural Network"),
    ("xgboost", "XGBoost"),
]


def sidebar_item(text: str, url: str, icon: str = "chevron-right"):
    return rx.link(
        rx.hstack(
            rx.icon(tag=icon, size=16, color="var(--jade-12)"),
            rx.text(text, color="var(--gray-12)"),
            spacing="2",
            vertical_align="bottom",
        ),
        href=url,
        width="100%",
    )


def sidebar_left() -> rx.Component:
    items = [
        sidebar_item("Start", "/", "home"),
    ]
    if ENABLE_TRAINING:
        items.append(sidebar_item("Training", "/training", "brain-circuit"))
    items.append(sidebar_item("Analyse", "/analyse", "bar-chart-3"))
    items.append(sidebar_item("Vorhersage", "/vorhersage", "sparkles"))

    return rx.vstack(
        rx.heading("MENÜ", size="4", color="var(--jade-12)", weight="light"),
        *items,
        rx.spacer(),
        rx.text("Version 0.1", size="1", color="gray"),
        width="250px",
        padding="20px",
        spacing="3",
        background_color=rx.color("sand", 1, False),
        border_radius="5px",
        border="1px solid var(--gray-8)",
        margin_left="20px",
        margin_bottom="40px",
        left="0",
        top="100",
    )


def sidebar_right() -> rx.Component:
    return rx.vstack(
        rx.heading("ÜBERSICHT", size="4", color="var(--jade-12)", weight="light"),
        rx.spacer(),
        rx.text("Verfügbare Modelltypen", size="2", weight="bold"),
        *[
            rx.hstack(
                rx.badge(code, color_scheme="jade"),
                rx.text(name, size="2"),
                spacing="2",
            )
            for code, name in AVAILABLE_MODELS
        ],
        width="250px",
        padding="20px",
        background_color=rx.color("sand", 1, False),
        border_radius="5px",
        border="1px solid var(--gray-8)",
        margin_right="20px",
        margin_bottom="40px",
        left="0",
        top="100",
    )


def base_layout(content: rx.Component) -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.box(
                rx.hstack(
                    rx.text(
                        "LexoTerm Sachgruppen-Klassifikation",
                        size="4",
                        weight="light",
                    ),
                    rx.spacer(),
                    rx.color_mode.button(),
                    width="100%",
                    align_items="center",
                ),
                padding="10px",
                background_color="#003835",
                color="white",
                width="100%",
                border_radius="4px",
            ),
            padding="20px",
            padding_bottom="5px",
            width="100%",
        ),
        rx.hstack(
            sidebar_left(),
            rx.box(
                content,
                width="60%",
                padding="20px",
                background_color=rx.color("sand", 1, False),
                border_radius="5px",
                border="1px solid var(--gray-8)",
            ),
            sidebar_right(),
            width="100%",
        ),
        max_width="1400px",
        background_color="var(--jade-2)",
        padding_bottom="40px",
    )
