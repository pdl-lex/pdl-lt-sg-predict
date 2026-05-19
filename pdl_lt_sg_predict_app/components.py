import os
import json
import reflex as rx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_models_dir_env = os.getenv("MODELS_DIR", "")
_MODELS_DIR = (
    Path(_models_dir_env)
    if _models_dir_env
    else Path.home() / ".pdl-sg-predict" / "models"
)

ENABLE_TRAINING = os.getenv("ENABLE_TRAINING", "True").strip().lower() in (
    "1",
    "true",
    "yes",
)

AVAILABLE_MODELS = [
    ("svm", "Linear SVM"),
    ("logistic", "Logistic Regression"),
    ("rf", "Random Forest"),
    ("nn", "Neural Network"),
    ("xgboost", "XGBoost"),
]

# ============ Design Constants ============

MAX_PAGE_WIDTH = "100%"
SIDEBAR_WIDTH = ["100%", "100%", "250px"]  # mobile, tablet, desktop (responsive)
PANEL_PADDING = "20px"
PANEL_RADIUS = "5px"
PANEL_BORDER = "1px solid var(--gray-8)"
PANEL_BG = rx.color("sand", 1, False)
PAGE_BG = "var(--jade-2)"
HEADING_COLOR = "var(--jade-12)"


# ============ Best Model State ============


class BestModelState(rx.State):
    """Tracks the best model (by accuracy) and CSV info for the sidebar."""

    best_model_name: str = ""
    best_model_accuracy: str = ""

    csv_filename: str = ""
    csv_num_samples: int = 0
    csv_num_classes: int = 0

    def set_csv_info(self, filename: str, num_samples: int, num_classes: int):
        self.csv_filename = filename
        self.csv_num_samples = num_samples
        self.csv_num_classes = num_classes

    def load_best_model(self):
        best_acc = -1.0
        best_name = ""
        best_acc_str = ""
        try:
            for meta_file in _MODELS_DIR.glob("*_metadata.json"):
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                acc = float(meta.get("accuracy", -1))
                if acc > best_acc:
                    best_acc = acc
                    best_name = meta.get("model_name", meta_file.stem)
                    best_acc_str = f"{acc:.4f}"
        except Exception:
            pass
        self.best_model_name = best_name
        self.best_model_accuracy = best_acc_str


# ============ Navigation ============


def sidebar_item(text: str, url: str, icon: str = "chevron-right") -> rx.Component:
    return rx.link(
        rx.hstack(
            rx.icon(tag=icon, size=16, color=HEADING_COLOR),
            rx.text(text, color="var(--gray-12)"),
            spacing="2",
            align_items="center",
        ),
        href=url,
        width="100%",
    )


def _nav_items() -> list[rx.Component]:
    items = [sidebar_item("Start", "/", "home")]
    items.append(sidebar_item("Training", "/training", "brain-circuit"))
    items.append(sidebar_item("Analyse", "/analyse", "bar-chart-3"))
    items.append(sidebar_item("Vorhersage", "/vorhersage", "sparkles"))
    items.append(sidebar_item("Sachgruppen", "/sachgruppen", "list"))
    items.append(sidebar_item("Anleitung", "/anleitung", "book-open"))
    return items


# ============ Mobile Navigation ============


class MobileNavState(rx.State):
    """State for the mobile navigation menu."""

    is_open: bool = False

    def set_is_open(self, value: bool) -> None:
        self.is_open = value

    def toggle(self) -> None:
        self.is_open = not self.is_open

    def close(self) -> None:
        self.is_open = False


def mobile_nav_drawer() -> rx.Component:
    """Navigation dialog for mobile view (hamburger menu)."""
    items = []
    for item in _nav_items():
        # Wrap each item to close the dialog on click
        items.append(rx.box(item, on_click=MobileNavState.close, width="100%"))
    return rx.dialog.root(
        rx.dialog.content(
            rx.vstack(
                rx.hstack(
                    rx.heading("MENÜ", size="4", color=HEADING_COLOR, weight="light"),
                    rx.spacer(),
                    rx.dialog.close(
                        rx.icon_button(
                            rx.icon("x"),
                            variant="ghost",
                            color=HEADING_COLOR,
                            on_click=MobileNavState.close,
                        ),
                    ),
                    width="100%",
                    align_items="center",
                ),
                *items,
                rx.spacer(),
                rx.text("Version 0.3", size="1", color="gray"),
                spacing="3",
                padding=PANEL_PADDING,
                width="100%",
            ),
            background_color=PANEL_BG,
            max_width="280px",
        ),
        open=MobileNavState.is_open,
        on_open_change=MobileNavState.set_is_open,
    )


# ============ Sidebars ============


def sidebar_left() -> rx.Component:
    return rx.vstack(
        rx.heading("MENÜ", size="4", color=HEADING_COLOR, weight="light"),
        *_nav_items(),
        rx.spacer(),
        rx.text("Version 0.2", size="1", color="gray"),
        width=SIDEBAR_WIDTH,
        min_width=["auto", "auto", "250px"],
        padding=PANEL_PADDING,
        spacing="3",
        background_color=PANEL_BG,
        border_radius=PANEL_RADIUS,
        border=PANEL_BORDER,
        display=[
            "none",
            "none",
            "flex",
        ],  # hidden on mobile/tablet → hamburger used instead
    )


def sidebar_right() -> rx.Component:
    return rx.vstack(
        rx.heading("ÜBERSICHT", size="4", color=HEADING_COLOR, weight="light"),
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
        rx.box(height="0.75rem"),
        rx.text("Daten", size="2", weight="bold"),
        rx.cond(
            BestModelState.csv_filename != "",
            rx.vstack(
                rx.text(BestModelState.csv_filename, size="2"),
                rx.text(
                    BestModelState.csv_num_samples.to_string() + " Samples",
                    size="2",
                    color="var(--gray-11)",
                ),
                rx.text(
                    BestModelState.csv_num_classes.to_string() + " Klassen",
                    size="2",
                    color="var(--gray-11)",
                ),
                spacing="1",
                align_items="start",
            ),
            rx.text("–", size="2", color="var(--gray-9)"),
        ),
        rx.box(height="0.75rem"),
        rx.text("Bestes Modell (Accuracy)", size="2", weight="bold"),
        rx.cond(
            BestModelState.best_model_name != "",
            rx.vstack(
                rx.text(BestModelState.best_model_name, size="2"),
                rx.badge(BestModelState.best_model_accuracy, color_scheme="jade"),
                spacing="1",
                align_items="start",
            ),
            rx.text("–", size="2", color="var(--gray-9)"),
        ),
        width=SIDEBAR_WIDTH,
        min_width=["auto", "auto", "250px"],
        padding=PANEL_PADDING,
        background_color=PANEL_BG,
        border_radius=PANEL_RADIUS,
        border=PANEL_BORDER,
        # Always visible: on mobile it appears below the main content
    )


# ============ Base Layout ============


def base_layout(content: rx.Component) -> rx.Component:
    """Main layout with header, sidebars and flexible content area."""
    return rx.box(
        # Mobile menu dialog (rendered once globally)
        mobile_nav_drawer(),
        # Content container
        rx.vstack(
            # Header
            rx.box(
                rx.hstack(
                    rx.image(
                        src="/lexoterm_logo.svg",
                        height="32px",
                        width="32px",
                        alt="LexoTerm Logo",
                    ),
                    rx.text(
                        "LexoTerm Sachgruppen-Klassifikation",
                        size="4",
                        weight="light",
                    ),
                    rx.spacer(),
                    rx.color_mode.button(),
                    # Hamburger icon: visible on mobile/tablet only
                    rx.icon_button(
                        rx.icon("menu", size=20),
                        variant="ghost",
                        color="white",
                        on_click=MobileNavState.toggle,
                        display=["flex", "flex", "none"],
                    ),
                    width="100%",
                    align_items="center",
                    spacing="3",
                ),
                padding="10px",
                background_color="#003835",
                color="white",
                width="100%",
                border_radius=PANEL_RADIUS,
            ),
            # 3-column layout:
            # Desktop: [sidebar-left | content (flex) | sidebar-right] side by side
            # Mobile:  sidebar-left (hidden) → content → sidebar-right stacked
            rx.flex(
                sidebar_left(),
                rx.box(
                    content,
                    flex="1",
                    min_width="0",  # prevents flex overflow
                    padding=PANEL_PADDING,
                    background_color=PANEL_BG,
                    border_radius=PANEL_RADIUS,
                    border=PANEL_BORDER,
                ),
                sidebar_right(),
                flex_direction=["column", "column", "row"],
                align_items=["stretch", "stretch", "start"],
                width="100%",
                gap=PANEL_PADDING,
            ),
            max_width=MAX_PAGE_WIDTH,
            width="100%",
            margin_x="auto",
            padding=PANEL_PADDING,
            gap=PANEL_PADDING,
        ),
        background_color=PAGE_BG,
        min_height="100vh",
        width="100%",
    )
