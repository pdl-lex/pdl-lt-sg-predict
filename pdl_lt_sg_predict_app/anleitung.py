"""
Anleitung page – static page with no dedicated state.
"""
import reflex as rx
from pathlib import Path

from .components import base_layout

# Load anleitung text once at startup
_ANLEITUNG_MD = (Path(__file__).parent.parent / "anleitung.md").read_text(encoding="utf-8")


def anleitung_page() -> rx.Component:
    """Anleitung page."""
    return base_layout(
        rx.vstack(
            rx.heading("ANLEITUNG", size="4", color="var(--jade-12)", weight="light"),
            rx.markdown(
                _ANLEITUNG_MD,
                component_map={
                    "h1": lambda text: rx.heading(text, size="6", color="var(--jade-12)", margin_top="1.5rem"),
                    "h2": lambda text: rx.heading(text, size="5", color="var(--jade-11)", margin_top="1.2rem"),
                    "h3": lambda text: rx.heading(text, size="4", margin_top="2.5rem", margin_bottom="0.05rem"),
                    "table": lambda children: rx.el.table(
                        children,
                        border_collapse="collapse",
                        width="100%",
                        font_size="0.875rem",
                    ),
                    "thead": lambda children: rx.el.thead(
                        children,
                        background_color="var(--jade-3)",
                    ),
                    "th": lambda children: rx.el.th(
                        children,
                        text_align="left",
                        padding="6px 12px",
                        border="1px solid var(--gray-6)",
                        font_weight="600",
                    ),
                    "td": lambda children: rx.el.td(
                        children,
                        padding="6px 12px",
                        border="1px solid var(--gray-6)",
                        vertical_align="top",
                    ),
                    "tr": lambda children: rx.el.tr(children),
                    "hr": lambda *_: rx.el.hr(
                        margin_top="2.5rem",
                        margin_bottom="2.5rem",
                        border_color="var(--gray-6)",
                    ),
                },
            ),
            spacing="4",
            width="100%",
            max_width="100%",
        )
    )
