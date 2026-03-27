"""
Analyse-Seite: AnalysisState + analyse_page.
"""
import json
from datetime import datetime
from pathlib import Path

import reflex as rx
from pdl_lt_reflex_aggrid_wrapper import ag_grid

from .state import BaseState, MODELS_DIR, MODEL_DISPLAY_NAMES
from .components import base_layout


class AnalysisState(BaseState):
    """State for model analysis."""
    models_list: list[dict] = []
    is_loading: bool = False
    selected_row_model_file: str = ""
    report_text: str = ""
    report_dialog_open: bool = False

    @rx.var
    def has_models(self) -> bool:
        return len(self.models_list) > 0

    @rx.var
    def models_count(self) -> int:
        return len(self.models_list)

    def load_models(self):
        """Load list of all saved models."""
        self.is_loading = True
        yield

        try:
            models_dir = MODELS_DIR
            models = []

            if not models_dir.exists():
                self.is_loading = False
                return

            for pkl_file in models_dir.glob("model_*.pkl"):
                metadata_file = models_dir / pkl_file.name.replace(".pkl", "_metadata.json")

                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Fallback: extract basic info from filename
                    model_type = pkl_file.stem.split('_')[1] if '_' in pkl_file.stem else "unknown"
                    metadata = {
                        "model_file": pkl_file.name,
                        "model_type": model_type,
                        "accuracy": 0.0,
                        "timestamp": datetime.fromtimestamp(pkl_file.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
                    }

                # Ensure display name
                model_name = metadata.get("model_name") or MODEL_DISPLAY_NAMES.get(
                    metadata.get("model_type", ""), metadata.get("model_type", "")
                )

                # Format values
                accuracy = round(metadata.get("accuracy", 0.0), 3)

                training_time = ""
                if "training_time" in metadata:
                    secs = int(metadata["training_time"])
                    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
                    training_time = f"{h:02d}:{m:02d}:{s:02d}"

                date = metadata.get("timestamp", "")
                try:
                    dt = datetime.strptime(date, "%Y%m%d_%H%M%S")
                    date = dt.strftime("%d.%m.%y")
                except ValueError:
                    pass

                test_size_raw = metadata.get("test_size", None)
                test_size_str = f"{test_size_raw * 100:.0f}%" if test_size_raw is not None else "–"
                stopwords_str = "ja" if metadata.get("remove_stopwords", False) else "nein"

                min_len = metadata.get("min_word_length", 1)
                min_len_str = f"≥ {min_len}" if min_len > 1 else "1 (alle)"

                analyzer_raw = metadata.get("analyzer", "char_wb")
                ngram_max = metadata.get("word_ngram_max", 1)
                if analyzer_raw == "word":
                    analyzer_str = f"word-(1,{ngram_max})"
                else:
                    analyzer_str = "char_wb"

                model_file_name = metadata.get("model_file", pkl_file.name)
                report_file = models_dir / model_file_name.replace(".pkl", "_report.txt")
                models.append({
                    "model_file": model_file_name,
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "training_time": training_time,
                    "date": date,
                    "num_samples": metadata.get("num_samples", 0),
                    "num_classes": metadata.get("num_classes", 0),
                    "test_size": test_size_str,
                    "stopwords_removed": stopwords_str,
                    "min_word_len": min_len_str,
                    "analyzer": analyzer_str,
                    "has_report": report_file.exists(),
                })

            # Sort by date descending; reorder dd.mm.yy → yy.mm.dd for correct comparison
            models.sort(
                key=lambda x: ".".join(reversed(x.get("date", "").split("."))),
                reverse=True
            )

            self.models_list = models

        except Exception as e:
            print(f"Error loading models: {e}")
        finally:
            self.is_loading = False

    def handle_selection_changed(self, selected_rows: list, source: str, event_type: str):
        """Called when a row in the model grid is selected."""
        if selected_rows:
            self.selected_row_model_file = selected_rows[0].get("model_file", "")
        else:
            self.selected_row_model_file = ""

    def go_to_vorhersage_with_model(self):
        """Pass the selected model to the prediction page and navigate there."""
        from .vorhersage import PredictionState
        yield PredictionState.preselect_and_load(self.selected_row_model_file)
        return rx.redirect("/vorhersage")

    def open_report_for_selected(self):
        """Load the classification report of the selected model and open the dialog."""
        report_path = MODELS_DIR / self.selected_row_model_file.replace(".pkl", "_report.txt")
        if report_path.exists():
            self.report_text = report_path.read_text(encoding="utf-8")
        else:
            self.report_text = "Kein Report vorhanden für dieses Modell."
        self.report_dialog_open = True

    def close_report_dialog(self):
        self.report_dialog_open = False


def analyse_page() -> rx.Component:
    """Analysis page."""
    models_column_defs = [
        ag_grid.column_def(field="model_file", header_name="Datei", sortable=True, filter=True),
        ag_grid.column_def(field="model_name", header_name="Model", sortable=True, filter=True),
        ag_grid.column_def(field="accuracy", header_name="Accuracy", sortable=True, filter=True),
        ag_grid.column_def(field="training_time", header_name="Training Time", sortable=True, filter=True),
        ag_grid.column_def(field="date", header_name="Date", sortable=True, filter=True),
        ag_grid.column_def(field="num_samples", header_name="Samples", sortable=True, filter=True),
        ag_grid.column_def(field="num_classes", header_name="Classes", sortable=True, filter=True),
        ag_grid.column_def(field="test_size", header_name="Test-Split", sortable=True, filter=True),
        ag_grid.column_def(field="stopwords_removed", header_name="Stopwords entf.", sortable=True, filter=True),
        ag_grid.column_def(field="min_word_len", header_name="Min. Wortlänge", sortable=True, filter=True),
        ag_grid.column_def(field="analyzer", header_name="Analyzer", sortable=True, filter=True),
    ]

    return base_layout(
        rx.vstack(
            rx.hstack(
                rx.heading("ANALYSE", size="4", color="var(--jade-12)", weight="light"),
                rx.icon_button(
                    rx.icon("refresh-cw", size=16),
                    on_click=AnalysisState.load_models,
                    loading=AnalysisState.is_loading,
                    variant="ghost",
                    size="2",
                    title="Modelle neu laden",
                ),
                align_items="center",
                spacing="3",
            ),
            rx.text("Vergleich aller trainierten Modelle", color="var(--gray-11)"),

            # Report dialog
            rx.dialog.root(
                rx.dialog.content(
                    rx.vstack(
                        rx.hstack(
                            rx.heading("Klassifikations-Report", size="5"),
                            rx.spacer(),
                            rx.dialog.close(
                                rx.icon_button(
                                    rx.icon("x"),
                                    variant="ghost",
                                    on_click=AnalysisState.close_report_dialog,
                                ),
                            ),
                            width="100%",
                            align_items="center",
                        ),
                        rx.text(
                            AnalysisState.selected_row_model_file,
                            size="2", color="var(--gray-11)",
                        ),
                        rx.el.pre(
                            AnalysisState.report_text,
                            style={
                                "fontFamily": "monospace",
                                "fontSize": "0.78rem",
                                "whiteSpace": "pre",
                                "overflowX": "auto",
                                "padding": "1rem",
                                "borderRadius": "4px",
                                "border": "1px solid var(--gray-6)",
                                "background": "var(--gray-2)",
                                "width": "100%",
                            },
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    max_width="700px",
                    width="90vw",
                ),
                open=AnalysisState.report_dialog_open,
                on_open_change=AnalysisState.close_report_dialog,
            ),

            rx.cond(
                AnalysisState.is_loading,
                rx.text("Modelle werden geladen…", color="var(--gray-11)"),
                rx.cond(
                    AnalysisState.has_models,
                    rx.vstack(
                        rx.heading(f"{AnalysisState.models_count} Modelle gefunden", size="5"),

                        ag_grid(
                            id="models_grid",
                            row_data=AnalysisState.models_list,
                            column_defs=models_column_defs,
                            default_col_def={"flex": 1, "minWidth": 80},
                            row_selection={"mode": "singleRow", "checkboxes": True, "enableClickSelection": True},
                            on_selection_changed=AnalysisState.handle_selection_changed,
                            resizable=True,
                            dom_layout="autoHeight",
                            height="None",
                            column_size="sizeToFit",
                        ),

                        rx.cond(
                            AnalysisState.selected_row_model_file,
                            rx.hstack(
                                rx.icon("sparkles", size=16, color="var(--jade-11)"),
                                rx.text(
                                    AnalysisState.selected_row_model_file,
                                    size="2", color="var(--gray-11)",
                                ),
                                rx.button(
                                    rx.icon("chart-bar", size=14),
                                    "Klassifikations-Report",
                                    on_click=AnalysisState.open_report_for_selected,
                                    color_scheme="jade",
                                    variant="soft",
                                ),
                                rx.button(
                                    "Modell für Vorhersage auswählen",
                                    on_click=AnalysisState.go_to_vorhersage_with_model,
                                    color_scheme="jade",
                                ),
                                spacing="3",
                                align_items="center",
                            ),
                        ),

                        spacing="3",
                        width="100%"
                    ),
                    rx.text("Keine trainierten Modelle gefunden", color="var(--gray-11)")
                ),
            ),

            spacing="4",
            width="100%",
            max_width="100%"
        )
    )
