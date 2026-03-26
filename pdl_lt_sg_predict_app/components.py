import os
import reflex as rx
from dotenv import load_dotenv

load_dotenv()

ENABLE_TRAINING = os.getenv("ENABLE_TRAINING", "True").strip().lower() in ("1", "true", "yes")

AVAILABLE_MODELS = [
    ("svm", "Linear SVM"),
    ("logistic", "Logistic Regression"),
    ("rf", "Random Forest"),
    ("nn", "Neural Network"),
    ("xgboost", "XGBoost"),
]

# ============ Design-Konstanten ============

MAX_PAGE_WIDTH = "100%"
SIDEBAR_WIDTH = ["100%", "100%", "250px"]   # mobile, tablet, desktop
PANEL_PADDING = "20px"
PANEL_RADIUS = "5px"
PANEL_BORDER = "1px solid var(--gray-8)"
PANEL_BG = rx.color("sand", 1, False)
PAGE_BG = "var(--jade-2)"
HEADING_COLOR = "var(--jade-12)"


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
    if ENABLE_TRAINING:
        items.append(sidebar_item("Training", "/training", "brain-circuit"))
    items.append(sidebar_item("Analyse", "/analyse", "bar-chart-3"))
    items.append(sidebar_item("Vorhersage", "/vorhersage", "sparkles"))
    return items


# ============ Mobile Navigation ============

class MobileNavState(rx.State):
    """State für das mobile Navigations-Menü."""

    is_open: bool = False

    def set_is_open(self, value: bool) -> None:
        self.is_open = value

    def toggle(self) -> None:
        self.is_open = not self.is_open

    def close(self) -> None:
        self.is_open = False


def mobile_nav_drawer() -> rx.Component:
    """Navigations-Dialog für mobile Ansicht (Hamburger-Menü)."""
    items = []
    for item in _nav_items():
        # Wrap jedes Item so dass es den Dialog schließt
        items.append(
            rx.box(item, on_click=MobileNavState.close, width="100%")
        )
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
                rx.text("Version 0.1", size="1", color="gray"),
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
        rx.text("Version 0.1", size="1", color="gray"),
        width=SIDEBAR_WIDTH,
        min_width=["auto", "auto", "250px"],
        padding=PANEL_PADDING,
        spacing="3",
        background_color=PANEL_BG,
        border_radius=PANEL_RADIUS,
        border=PANEL_BORDER,
        display=["none", "none", "flex"],   # auf Mobile/Tablet versteckt → Hamburger
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
        width=SIDEBAR_WIDTH,
        min_width=["auto", "auto", "250px"],
        padding=PANEL_PADDING,
        background_color=PANEL_BG,
        border_radius=PANEL_RADIUS,
        border=PANEL_BORDER,
        # Immer sichtbar: auf Mobile erscheint sie unterhalb des Hauptinhalts
    )


# ============ Base Layout ============

def base_layout(content: rx.Component) -> rx.Component:
    """Haupt-Layout mit Header, Sidebars und flexiblem Inhaltsbereich."""
    return rx.box(
        # Mobile-Menü-Dialog (einmal global gerendert)
        mobile_nav_drawer(),
        # Inhalts-Container
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
                    # Hamburger-Icon: nur auf Mobile/Tablet sichtbar
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
            # 3-Spalten-Layout:
            # Desktop: [Sidebar-left | Content (flex) | Sidebar-right] nebeneinander
            # Mobile:  Sidebar-left (hidden) → Content → Sidebar-right untereinander
            rx.flex(
                sidebar_left(),
                rx.box(
                    content,
                    flex="1",
                    min_width="0",      # verhindert Flex-Overflow
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
