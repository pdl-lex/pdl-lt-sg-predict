"""
Sachgruppen classification web app.
Machine learning interface for model training, analysis and prediction.

Entry point: registers all pages and contains the index page.
"""
import reflex as rx

from .components import base_layout, ENABLE_TRAINING, BestModelState
from .state import BaseState  # noqa: F401 — registriert BaseState im Reflex-State-Tree


def index() -> rx.Component:
    """Start page."""
    return base_layout(
        rx.vstack(
            rx.heading("START", size="4", color="var(--jade-12)", weight="light"),
            rx.text(
                "Machine Learning Tool für automatische Sachgruppen-Klassifikation von Wörterbuch-Einträgen.",
                size="4",
                color="var(--gray-11)"
            ),
            rx.divider(),

            rx.heading("Features", size="5", margin_top="2rem"),

            rx.card(
                rx.data_list.root(
                    rx.data_list.item(
                        rx.data_list.label(
                            rx.badge(
                                "Training" if ENABLE_TRAINING else "Training (deaktiviert)",
                                variant="soft",
                                color_scheme="jade" if ENABLE_TRAINING else "gray",
                            ),
                        ),
                        rx.data_list.value(
                            "Training neuer Modelle auf eigenen Daten." if ENABLE_TRAINING else
                            "Training ist deaktiviert. Aktivierung: ENABLE_TRAINING=True in der .env-Datei.",
                        ),
                    ),
                    rx.data_list.item(
                        rx.data_list.label(
                            rx.badge("Modellvergleich", variant="soft"),
                        ),
                        rx.data_list.value(
                            "Vergleichen der Performance und Parameter verschiedener trainierter Modelle.",
                        ),
                    ),
                    rx.data_list.item(
                        rx.data_list.label(
                            rx.badge("Analyse", variant="soft"),
                        ),
                        rx.data_list.value(
                            "Übersicht aller trainierten Modelle mit Accuracy, Parametern und Trainingszeiten.",
                        ),
                    ),
                    rx.data_list.item(
                        rx.data_list.label(
                            rx.badge("Vorhersage", variant="soft"),
                        ),
                        rx.data_list.value(
                            "Klassifizierung/Vorhersage für neue Lemmata (einzeln oder im Batch).",
                        ),
                    ),
                ),
                width="100%",
            ),

            spacing="4",
            width="100%",
            max_width="100%"
        )
    )


# ============ App Setup ============

app = rx.App(
    theme=rx.theme(
        appearance="light",
        has_background=True,
        radius="large",
        accent_color="jade",
    )
)


def _register_pages():
    from .training import training_page
    from .analyse import analyse_page, AnalysisState
    from .vorhersage import vorhersage_page
    from .sachgruppen import sachgruppen_page, SachgruppenState
    from .anleitung import anleitung_page

    app.add_page(index, route="/", title="LT Sachgruppen-Vorhersage | Start", on_load=BestModelState.load_best_model)
    app.add_page(training_page, route="/training", title="LT Sachgruppen-Vorhersage | Training", on_load=BestModelState.load_best_model)
    app.add_page(analyse_page, route="/analyse", title="LT Sachgruppen-Vorhersage | Analyse", on_load=[AnalysisState.load_models, BestModelState.load_best_model])
    app.add_page(vorhersage_page, route="/vorhersage", title="LT Sachgruppen-Vorhersage | Vorhersage", on_load=BestModelState.load_best_model)
    app.add_page(sachgruppen_page, route="/sachgruppen", title="LT Sachgruppen-Vorhersage | Sachgruppen", on_load=[SachgruppenState.load_data, BestModelState.load_best_model])
    app.add_page(anleitung_page, route="/anleitung", title="LT Sachgruppen-Vorhersage | Anleitung", on_load=BestModelState.load_best_model)


_register_pages()
