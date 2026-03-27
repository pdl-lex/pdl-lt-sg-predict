"""
Sachgruppen page: overview of all Sachgruppen with classification metrics
from the best trained model.
"""
import json
from pathlib import Path

import pandas as pd
import reflex as rx
from pdl_lt_reflex_aggrid_wrapper import ag_grid

from .state import BaseState, MODELS_DIR
from .components import base_layout


def _parse_report(report_text: str) -> dict[str, dict]:
    """
    Parse an sklearn classification_report string.

    Expects the standard format:
        precision    recall  f1-score   support
    1000      0.85      0.85      0.85        26

    Returns dict: label → {"precision", "recall", "f1", "support"}.
    Skips header, blank lines, and summary lines (accuracy / macro avg / weighted avg).
    """
    result: dict[str, dict] = {}
    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped or "precision" in stripped:
            continue
        if any(stripped.startswith(s) for s in ("accuracy", "macro avg", "weighted avg")):
            continue
        # Split from right into max. 5 fields: label  prec  rec  f1  support
        parts = stripped.rsplit(None, 4)
        if len(parts) == 5:
            label = parts[0].strip()
            try:
                result[label] = {
                    "precision": float(parts[1]),
                    "recall":    float(parts[2]),
                    "f1":        float(parts[3]),
                    "support":   int(parts[4]),
                }
            except ValueError:
                pass
    return result


def _fmt(val: float) -> str:
    return f"{val:.4f}"


class SachgruppenState(BaseState):
    """State for the Sachgruppen overview page."""

    rows: list[dict] = []
    is_loading: bool = False
    active_model_name: str = ""   # filename (without _metadata.json) of the active model
    active_model_accuracy: str = ""

    def load_data(self):
        """Load sachgruppen.csv and the best model's report, then combine them."""
        self.is_loading = True
        yield

        try:
            # ── 1. Load sachgruppen.csv ───────────────────────────────────────
            sg_csv = Path(__file__).parent.parent / "sachgruppen.csv"
            sg_df = pd.read_csv(sg_csv, sep=";", dtype=str)
            sg_map: dict[str, str] = dict(
                zip(sg_df["Nummer"].str.strip(), sg_df["Sachgruppe"].str.strip())
            )

            # ── 2. Find best model (max. accuracy) ────────────────────────────
            report_data: dict[str, dict] = {}
            model_name = ""
            model_acc = ""
            best_acc = -1.0

            for meta_file in MODELS_DIR.glob("*_metadata.json"):
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        meta = json.load(f)
                    acc = float(meta.get("accuracy", -1))
                    if acc > best_acc:
                        best_acc = acc
                        model_name = meta_file.stem.replace("_metadata", "")
                        model_acc = f"{acc:.4f}"
                except Exception:
                    pass

            # ── 3. Parse best model's report ──────────────────────────────────
            if model_name:
                report_path = MODELS_DIR / f"{model_name}_report.txt"
                if report_path.exists():
                    report_data = _parse_report(report_path.read_text(encoding="utf-8"))

            self.active_model_name = model_name
            self.active_model_accuracy = model_acc

            # ── 4. Rows: all Sachgruppen from sachgruppen.csv ─────────────────
            #    (sorted by number, numerically)
            rows: list[dict] = []
            seen_labels: set[str] = set()

            for nummer in sorted(sg_map.keys(), key=lambda x: (len(x), x)):
                seen_labels.add(nummer)
                beschreibung = sg_map[nummer]
                metrics = report_data.get(nummer, {})
                rows.append({
                    "nummer":     nummer,
                    "sachgruppe": beschreibung,
                    "support":    metrics.get("support", 0),
                    "precision":  _fmt(metrics["precision"]) if "precision" in metrics else "-",
                    "recall":     _fmt(metrics["recall"])    if "recall"    in metrics else "-",
                    "f1":         _fmt(metrics["f1"])        if "f1"        in metrics else "-",
                })

            # ── 5. Report labels without entry in sachgruppen.csv ────────────
            #    (e.g. class "0" or undocumented numbers)
            for label in sorted(report_data.keys(), key=lambda x: (len(x), x)):
                if label in seen_labels:
                    continue
                metrics = report_data[label]
                rows.append({
                    "nummer":     label,
                    "sachgruppe": "–",
                    "support":    metrics.get("support", 0),
                    "precision":  _fmt(metrics["precision"]),
                    "recall":     _fmt(metrics["recall"]),
                    "f1":         _fmt(metrics["f1"]),
                })

            self.rows = rows

        except Exception as e:
            print(f"SachgruppenState.load_data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_loading = False


def sachgruppen_page() -> rx.Component:
    """Sachgruppen page."""
    column_defs = [
        ag_grid.column_def(
            field="nummer", header_name="Nummer",
            sortable=True, filter=True, width=110,
        ),
        ag_grid.column_def(
            field="sachgruppe", header_name="Sachgruppe",
            sortable=True, filter=True, flex=2,
        ),
        ag_grid.column_def(
            field="support", header_name="Samples",
            sortable=True, filter=True, width=105,
        ),
        ag_grid.column_def(
            field="precision", header_name="Precision",
            sortable=True, filter=True, width=110,
        ),
        ag_grid.column_def(
            field="recall", header_name="Recall",
            sortable=True, filter=True, width=105,
        ),
        ag_grid.column_def(
            field="f1", header_name="F1-Score",
            sortable=True, filter=True, width=110,
        ),
    ]

    return base_layout(
        rx.vstack(
            # ── Header ────────────────────────────────────────────────────────
            rx.hstack(
                rx.heading(
                    "SACHGRUPPEN",
                    size="4", color="var(--jade-12)", weight="light",
                ),
                rx.icon_button(
                    rx.icon("refresh-cw", size=16),
                    on_click=SachgruppenState.load_data,
                    loading=SachgruppenState.is_loading,
                    variant="ghost",
                    size="2",
                    title="Neu laden",
                ),
                align_items="center",
                spacing="3",
            ),

            # ── Taxonomy notice ───────────────────────────────────────────────
            rx.callout(
                rx.vstack(
                    rx.text(
                        "Für die Sachgruppen wird die klassische Taxonomie nach "
                        "Hallig-Wartburg verwendet.",
                        weight="bold",
                        size="2",
                    ),
                    rx.text(
                        "Hinweis: Diese traditionelle, vielfach verwendete Liste "
                        "entspricht nicht mehr einem modernen Sprachgebrauch und "
                        "wird nur aus Gründen der Kompatibilität verwendet.",
                        size="2",
                    ),
                    spacing="1",
                ),
                icon="info",
                color_scheme="jade",
                width="100%",
            ),

            # ── Model info ────────────────────────────────────────────────────
            rx.cond(
                SachgruppenState.active_model_name != "",
                rx.hstack(
                    rx.text("Metriken aus:", size="2", color="var(--gray-11)"),
                    rx.badge(SachgruppenState.active_model_name, color_scheme="jade"),
                    rx.text("·", size="2", color="var(--gray-9)"),
                    rx.text("Accuracy:", size="2", color="var(--gray-11)"),
                    rx.badge(SachgruppenState.active_model_accuracy, color_scheme="jade"),
                    spacing="2",
                    align_items="center",
                    flex_wrap="wrap",
                ),
                rx.text(
                    "Kein trainiertes Modell gefunden – Metriken nicht verfügbar.",
                    size="2", color="var(--amber-11)",
                ),
            ),

            # ── Grid ──────────────────────────────────────────────────────────
            rx.cond(
                SachgruppenState.is_loading,
                rx.text("Lade Daten…", color="var(--gray-11)"),
                ag_grid(
                    id="sachgruppen_grid",
                    row_data=SachgruppenState.rows,
                    column_defs=column_defs,
                    default_col_def={"minWidth": 80},
                    dom_layout="autoHeight",
                    height="None",
                    column_size="sizeToFit",
                ),
            ),

            spacing="4",
            width="100%",
            max_width="100%",
        )
    )
