"""
Vorhersage-Seite: PredictionState + SHAP-UI + vorhersage_page.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import reflex as rx
from pdl_lt_reflex_aggrid_wrapper import ag_grid

from .state import (
    BaseState,
    MODELS_DIR,
    SACHGRUPPEN_MAP,
    _get_model,
    _shap_pairs_to_dicts,
)
from .components import base_layout


class PredictionState(BaseState):
    """State for predictions."""
    # Single prediction
    input_lemma: str = ""
    input_bedeutung: str = ""
    prediction_result: str = ""
    prediction_result_description: str = ""
    prediction_proba: float = 0.0
    top_predictions: list[dict] = []  # [{"label": str, "description": str, "proba": str}, ...]

    # Batch prediction
    batch_filename: str = ""
    batch_file_path: str = ""
    batch_upload_error: str = ""
    batch_results: list[dict] = []
    is_predicting: bool = False

    # Model selection
    selected_model_file: str = ""
    available_models: list[str] = []

    # SHAP explanation
    shap_lemma_scores: list[dict] = []     # [{"word": str, "score": float}, ...]
    shap_bedeutung_scores: list[dict] = []
    shap_error: str = ""
    shap_is_computing: bool = False
    shap_model_is_nn: bool = False         # MLP: SHAP only on demand
    shap_filter_stopwords: bool = True     # hide stopwords (default: on)

    @rx.var
    def has_model(self) -> bool:
        return bool(self.selected_model_file)

    @rx.var
    def can_predict(self) -> bool:
        return self.has_model and bool(self.input_bedeutung)

    @rx.var
    def can_batch_predict(self) -> bool:
        return self.has_model and bool(self.batch_filename)

    @rx.var
    def has_batch_results(self) -> bool:
        return len(self.batch_results) > 0

    @rx.var
    def batch_results_count(self) -> int:
        """Number of batch results."""
        return len(self.batch_results)

    @rx.var
    def prediction_proba_formatted(self) -> str:
        """Format probability as percentage string."""
        if self.prediction_proba > 0:
            return f"{self.prediction_proba * 100:.2f}%"
        return ""

    @rx.var
    def shap_top_words(self) -> list[dict]:
        """Top 10 words by absolute SHAP influence (for bar chart)."""
        combined = self.shap_lemma_scores + self.shap_bedeutung_scores
        return sorted(combined, key=lambda x: abs(x["score"]), reverse=True)[:10]

    @rx.var
    def has_shap_results(self) -> bool:
        return bool(self.shap_lemma_scores or self.shap_bedeutung_scores)

    # Explicit setters (required since Reflex 0.9.0)
    def set_selected_model_file(self, value: str):
        self.selected_model_file = value

    def set_input_lemma(self, value: str):
        self.input_lemma = value

    def set_input_bedeutung(self, value: str):
        self.input_bedeutung = value

    def load_available_models(self):
        """Load list of available models."""
        model_files = [f.name for f in MODELS_DIR.glob("*.pkl")]
        self.available_models = sorted(model_files, reverse=True)

        if model_files and not self.selected_model_file:
            self.selected_model_file = self.available_models[0]

    def preselect_and_load(self, model_file: str):
        """Load available models and set the given model as active."""
        self.available_models = sorted([f.name for f in MODELS_DIR.glob("*.pkl")], reverse=True)
        self.selected_model_file = model_file

    def predict_single(self):
        """Single prediction."""
        if not self.can_predict:
            return

        self.is_predicting = True
        self.prediction_result = ""
        self.prediction_result_description = ""
        self.prediction_proba = 0.0
        self.top_predictions = []
        self.shap_lemma_scores = []
        self.shap_bedeutung_scores = []
        self.shap_error = ""
        self.shap_model_is_nn = False
        yield

        model_path_str = ""
        clf = None
        X_pred = None

        try:
            model_path = MODELS_DIR / self.selected_model_file
            model_path_str = str(model_path)

            # Load model from cache (loaded once per process)
            clf = _get_model(model_path_str)

            X_pred = pd.DataFrame({
                'lemma': [self.input_lemma],
                'bedeutung': [self.input_bedeutung]
            })

            prediction = clf.predict(X_pred)[0]
            self.prediction_result = str(prediction)
            self.prediction_result_description = SACHGRUPPEN_MAP.get(
                str(prediction), "(unbekannt)"
            )

            # Top-3 predictions
            classifier_step = clf.pipeline.named_steps["classifier"]
            classes = classifier_step.classes_
            if clf.label_encoder is not None:
                classes = clf.label_encoder.inverse_transform(classes)

            try:
                # Models with predict_proba (Logistic, RF, NN, XGBoost)
                probas = np.array(clf.predict_proba(X_pred)[0], dtype=float)
                top3_idx = probas.argsort()[::-1][:3]
                self.prediction_proba = float(probas[top3_idx[0]])
                self.top_predictions = [
                    {
                        "label": str(classes[i]),
                        "description": SACHGRUPPEN_MAP.get(str(classes[i]), "(unbekannt)"),
                        "proba": f"{probas[i] * 100:.1f}%",
                        "has_proba": True,
                        "is_best": rank == 0,
                    }
                    for rank, i in enumerate(top3_idx)
                ]
            except Exception:
                # SVM: decision_function as ranking substitute
                try:
                    scores = np.array(clf.pipeline.named_steps["classifier"].decision_function(
                        clf.pipeline[:-1].transform(X_pred)
                    )[0], dtype=float)
                    top3_idx = scores.argsort()[::-1][:3]
                    self.top_predictions = [
                        {
                            "label": str(classes[i]),
                            "description": SACHGRUPPEN_MAP.get(str(classes[i]), "(unbekannt)"),
                            "proba": "",
                            "has_proba": False,
                            "is_best": rank == 0,
                        }
                        for rank, i in enumerate(top3_idx)
                    ]
                except Exception:
                    self.top_predictions = [
                        {
                            "label": self.prediction_result,
                            "description": self.prediction_result_description,
                            "proba": "",
                            "has_proba": False,
                            "is_best": True,
                        }
                    ]
                self.prediction_proba = 0.0

        except Exception as e:
            self.prediction_result = f"Fehler: {str(e)}"
        finally:
            self.is_predicting = False

        # Compute SHAP explanation (only when prediction succeeded)
        if clf is not None and X_pred is not None and self.prediction_result and not self.prediction_result.startswith("Fehler"):
            self.shap_model_is_nn = (clf.model_type == "nn")

            if clf.model_type != "nn":
                yield from self._run_shap_computation(clf, X_pred, model_path_str)

    async def handle_batch_upload(self, files: list[rx.UploadFile]):
        """Batch CSV upload: validate file and store path; does not run prediction."""
        self.batch_upload_error = ""
        self.batch_filename = ""
        self.batch_file_path = ""
        self.batch_results = []
        yield

        try:
            if len(files) != 1:
                self.batch_upload_error = "Bitte genau eine CSV-Datei hochladen"
                return

            file = files[0]

            if not file.filename.lower().endswith('.csv'):
                self.batch_upload_error = "Nur CSV-Dateien erlaubt"
                return

            # Save file
            session_path = self._create_session_dir()
            safe_filename = Path(file.filename).name
            file_path = session_path / safe_filename

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # Validate – try common separators if auto-detection fails
            df = None
            for sep in [None, ";", ",", "\t"]:
                try:
                    kwargs = {"engine": "python"} if sep is None else {}
                    df = pd.read_csv(file_path, sep=sep, **kwargs)
                    if len(df.columns) >= 1:
                        break
                except Exception:
                    continue
            if df is None or df.empty:
                self.batch_upload_error = "CSV konnte nicht gelesen werden"
                return

            if 'bedeutung' not in df.columns:
                self.batch_upload_error = "CSV muss mindestens eine 'bedeutung' Spalte enthalten"
                return

            self.batch_filename = safe_filename
            self.batch_file_path = str(file_path)

        except Exception as e:
            self.batch_upload_error = f"Fehler: {str(e)}"

    def run_batch_prediction(self):
        """Run batch prediction on the uploaded file with the currently selected model."""
        if not self.can_batch_predict:
            return

        self.batch_upload_error = ""
        self.batch_results = []
        self.is_predicting = True
        yield

        try:
            # Re-read the saved file
            file_path = Path(self.batch_file_path)
            df = None
            for sep in [None, ";", ",", "\t"]:
                try:
                    kwargs = {"engine": "python"} if sep is None else {}
                    df = pd.read_csv(file_path, sep=sep, **kwargs)
                    if len(df.columns) >= 1:
                        break
                except Exception:
                    continue
            if df is None or df.empty:
                self.batch_upload_error = "CSV konnte nicht gelesen werden"
                return
            if 'lemma' not in df.columns:
                df['lemma'] = ""

            clf = _get_model(str(MODELS_DIR / self.selected_model_file))

            X_batch = df[['lemma', 'bedeutung']]
            predictions = clf.predict(X_batch)

            # Class labels (decoded for XGBoost/NN)
            classifier_step = clf.pipeline.named_steps["classifier"]
            classes = classifier_step.classes_
            if clf.label_encoder is not None:
                classes = clf.label_encoder.inverse_transform(classes)

            # Top-3 via predict_proba or decision_function
            has_proba = False
            top3_indices = None
            proba_matrix = None

            try:
                proba_matrix = np.array(clf.predict_proba(X_batch), dtype=float)
                has_proba = True
                top3_indices = np.argsort(proba_matrix, axis=1)[:, ::-1][:, :3]
            except Exception:
                try:
                    scores = np.array(
                        clf.pipeline.named_steps["classifier"].decision_function(
                            clf.pipeline[:-1].transform(X_batch)
                        ),
                        dtype=float,
                    )
                    if scores.ndim == 2:
                        top3_indices = np.argsort(scores, axis=1)[:, ::-1][:, :3]
                except Exception:
                    pass

            # Collect results
            results = []
            for idx, (_, row) in enumerate(df.iterrows()):
                sg = str(predictions[idx])
                result = {
                    'lemma': row['lemma'],
                    'bedeutung': row['bedeutung'],
                    'sachgruppe': sg,
                    'beschreibung': SACHGRUPPEN_MAP.get(sg, "(unbekannt)"),
                    'wahrscheinlichkeit': "",
                    'sachgruppe_2': "",
                    'beschreibung_2': "",
                    'wahrscheinlichkeit_2': "",
                    'sachgruppe_3': "",
                    'beschreibung_3': "",
                    'wahrscheinlichkeit_3': "",
                }

                if top3_indices is not None:
                    top3 = top3_indices[idx]
                    if has_proba:
                        result['wahrscheinlichkeit'] = f"{proba_matrix[idx, top3[0]] * 100:.1f}%"
                    if len(top3) > 1:
                        sg2 = str(classes[top3[1]])
                        result['sachgruppe_2'] = sg2
                        result['beschreibung_2'] = SACHGRUPPEN_MAP.get(sg2, "(unbekannt)")
                        if has_proba:
                            result['wahrscheinlichkeit_2'] = f"{proba_matrix[idx, top3[1]] * 100:.1f}%"
                    if len(top3) > 2:
                        sg3 = str(classes[top3[2]])
                        result['sachgruppe_3'] = sg3
                        result['beschreibung_3'] = SACHGRUPPEN_MAP.get(sg3, "(unbekannt)")
                        if has_proba:
                            result['wahrscheinlichkeit_3'] = f"{proba_matrix[idx, top3[2]] * 100:.1f}%"

                results.append(result)

            self.batch_results = results

        except Exception as e:
            self.batch_upload_error = f"Fehler: {str(e)}"
        finally:
            self.is_predicting = False

    def _run_shap_computation(self, clf, X_pred, model_path_str: str):
        """Shared SHAP computation logic for predict_single and compute_shap_nn."""
        self.shap_is_computing = True
        yield
        try:
            shap_result = clf.explain(
                X_pred, self.prediction_result, model_path_str,
                filter_stopwords=self.shap_filter_stopwords,
            )
            self.shap_lemma_scores = _shap_pairs_to_dicts(shap_result.get("lemma", []))
            self.shap_bedeutung_scores = _shap_pairs_to_dicts(shap_result.get("bedeutung", []))
        except Exception as e:
            self.shap_error = str(e)
        finally:
            self.shap_is_computing = False

    def compute_shap_nn(self):
        """SHAP explanation for neural network model (slow, triggered manually)."""
        if not self.prediction_result or self.prediction_result.startswith("Fehler"):
            return

        self.shap_lemma_scores = []
        self.shap_bedeutung_scores = []
        self.shap_error = ""

        model_path = MODELS_DIR / self.selected_model_file
        model_path_str = str(model_path)
        clf = _get_model(model_path_str)
        X_pred = pd.DataFrame({
            'lemma': [self.input_lemma],
            'bedeutung': [self.input_bedeutung]
        })
        yield from self._run_shap_computation(clf, X_pred, model_path_str)

    def toggle_shap_stopwords(self):
        """Stoppwort-Filter umschalten und SHAP neu berechnen."""
        self.shap_filter_stopwords = not self.shap_filter_stopwords

        if not self.prediction_result or self.prediction_result.startswith("Fehler"):
            return

        self.shap_lemma_scores = []
        self.shap_bedeutung_scores = []
        self.shap_error = ""

        model_path = MODELS_DIR / self.selected_model_file
        model_path_str = str(model_path)
        clf = _get_model(model_path_str)

        if clf.model_type == "nn":
            # MLP: do not recompute automatically (too slow)
            return

        X_pred = pd.DataFrame({
            'lemma': [self.input_lemma],
            'bedeutung': [self.input_bedeutung]
        })
        yield from self._run_shap_computation(clf, X_pred, model_path_str)

    def download_batch_csv(self):
        """Download batch results as CSV."""
        if not self.batch_results:
            return

        df = pd.DataFrame(self.batch_results)
        csv_content = df.to_csv(index=False, sep=";")
        return rx.download(
            data=csv_content,
            filename="vorhersage_ergebnisse.csv",
        )


# ============ SHAP UI Components ============

def shap_word_badge(word_data: dict) -> rx.Component:
    """Badge for a single word with color-coded SHAP influence.
    Color ('jade'/'red'/'gray') is pre-computed in the state.
    """
    return rx.badge(word_data["word"], color_scheme=word_data["color"], variant="soft")


def shap_card() -> rx.Component:
    """SHAP explanation card shown after a prediction."""
    return rx.vstack(
        # Loading indicator
        rx.cond(
            PredictionState.shap_is_computing,
            rx.callout(
                "SHAP-Erklärung wird berechnet...",
                icon="loader",
                color_scheme="gray",
            ),
        ),

        # MLP: manual trigger button
        rx.cond(
            PredictionState.shap_model_is_nn & ~PredictionState.shap_is_computing & ~PredictionState.has_shap_results,
            rx.button(
                "Erklärung anzeigen (Neural Network – dauert ~30–60 Sek.)",
                on_click=PredictionState.compute_shap_nn,
                color_scheme="amber",
                variant="soft",
            ),
        ),

        # Error display
        rx.cond(
            PredictionState.shap_error,
            rx.callout(
                rx.hstack(
                    rx.text("SHAP-Fehler: ", weight="bold"),
                    rx.text(PredictionState.shap_error),
                    spacing="1",
                ),
                icon="triangle_alert",
                color_scheme="red",
            ),
        ),

        # Main card with results
        rx.cond(
            PredictionState.has_shap_results,
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.hstack(
                            rx.icon("sparkle", size=18, color="var(--jade-11)"),
                            rx.heading("Vorhersage-Erklärung (SHAP)", size="4"),
                            align_items="center",
                            spacing="2",
                        ),
                        rx.spacer(),
                        rx.hstack(
                            rx.switch(
                                checked=PredictionState.shap_filter_stopwords,
                                on_change=PredictionState.toggle_shap_stopwords,
                                size="1",
                            ),
                            rx.text("Stoppwörter ausblenden", size="2", color="var(--gray-11)"),
                            align_items="center",
                            spacing="2",
                        ),
                        width="100%",
                        align_items="center",
                    ),
                    rx.text(
                        "Grün = unterstützt die Vorhersage  |  Rot = widerspricht der Vorhersage  |  Grau = neutral",
                        size="2",
                        color="var(--gray-10)",
                    ),

                    # Lemma words
                    rx.cond(
                        PredictionState.shap_lemma_scores,
                        rx.vstack(
                            rx.text("Lemma:", weight="bold", size="2"),
                            rx.hstack(
                                rx.foreach(PredictionState.shap_lemma_scores, shap_word_badge),
                                flex_wrap="wrap",
                                gap="2",
                            ),
                            align_items="start",
                            spacing="1",
                        ),
                    ),

                    # Bedeutung words
                    rx.cond(
                        PredictionState.shap_bedeutung_scores,
                        rx.vstack(
                            rx.text("Bedeutung:", weight="bold", size="2"),
                            rx.hstack(
                                rx.foreach(PredictionState.shap_bedeutung_scores, shap_word_badge),
                                flex_wrap="wrap",
                                gap="2",
                            ),
                            align_items="start",
                            spacing="1",
                        ),
                    ),

                    # Bar chart: top 10 words (CSS-based for color control)
                    rx.cond(
                        PredictionState.shap_top_words,
                        rx.vstack(
                            rx.text("Top-Wörter nach Einfluss:", weight="bold", size="2"),
                            rx.hstack(
                                rx.foreach(
                                    PredictionState.shap_top_words,
                                    lambda item: rx.vstack(
                                        # Positive area (bar grows upward)
                                        rx.box(
                                            rx.cond(
                                                item["is_positive"],
                                                rx.box(
                                                    height=item["bar_height"],
                                                    width="100%",
                                                    background_color=item["fill"],
                                                    border_radius="3px 3px 0 0",
                                                ),
                                                rx.box(),
                                            ),
                                            height="100px",
                                            width="100%",
                                            display="flex",
                                            align_items="flex-end",
                                        ),
                                        # Baseline line
                                        rx.box(
                                            height="1px",
                                            width="100%",
                                            background_color="var(--gray-8)",
                                        ),
                                        # Negative area (bar grows downward)
                                        rx.box(
                                            rx.cond(
                                                ~item["is_positive"],
                                                rx.box(
                                                    height=item["bar_height"],
                                                    width="100%",
                                                    background_color=item["fill"],
                                                    border_radius="0 0 3px 3px",
                                                ),
                                                rx.box(),
                                            ),
                                            height="100px",
                                            width="100%",
                                            display="flex",
                                            align_items="flex-start",
                                            padding_bottom="8px",
                                        ),
                                        # Word label
                                        rx.text(
                                            item["word"],
                                            size="1",
                                            text_align="center",
                                            width="100%",
                                            color="var(--gray-11)",
                                        ),
                                        spacing="0",
                                        align_items="center",
                                        flex="1",
                                        min_width="0",
                                    ),
                                ),
                                width="100%",
                                align_items="stretch",
                                gap="4px",
                            ),
                            align_items="start",
                            spacing="1",
                            width="100%",
                        ),
                    ),

                    spacing="4",
                    align_items="start",
                ),
                padding="1.5rem",
                width="100%",
            ),
        ),

        width="100%",
        spacing="3",
    )


def vorhersage_page() -> rx.Component:
    """Prediction page."""
    return base_layout(
        rx.vstack(
            rx.heading("VORHERSAGE", size="4", color="var(--jade-12)", weight="light"),

            # Model selection
            rx.card(
                rx.vstack(
                    rx.heading("Modell auswählen", size="5"),
                    rx.button(
                        "Verfügbare Modelle laden",
                        on_click=PredictionState.load_available_models
                    ),
                    rx.cond(
                        PredictionState.available_models,
                        rx.select(
                            PredictionState.available_models,
                            value=PredictionState.selected_model_file,
                            on_change=PredictionState.set_selected_model_file
                        )
                    ),
                    spacing="3"
                ),
                padding="1.5rem",
                width="100%"
            ),

            # Single prediction
            rx.card(
                rx.vstack(
                    rx.heading("Einzelvorhersage", size="5"),

                    rx.vstack(
                        rx.input(
                            placeholder="Lemma (z.B. 'Waggala') – optional",
                            value=PredictionState.input_lemma,
                            on_change=PredictionState.set_input_lemma,
                            width="100%"
                        ),
                        rx.input(
                            placeholder="Bedeutung (z.B. 'kleines Kind; Kind, das noch wackelig auf den Beinen ist')",
                            value=PredictionState.input_bedeutung,
                            on_change=PredictionState.set_input_bedeutung,
                            width="100%"
                        ),
                        spacing="2",
                        width="100%"
                    ),

                    rx.button(
                        "Vorhersagen",
                        on_click=PredictionState.predict_single,
                        disabled=~PredictionState.can_predict,
                        loading=PredictionState.is_predicting,
                        color_scheme="jade"
                    ),

                    rx.cond(
                        PredictionState.prediction_result,
                        rx.hstack(
                            rx.foreach(
                                PredictionState.top_predictions,
                                lambda item: rx.cond(
                                    item["is_best"],
                                    rx.box(
                                        rx.hstack(
                                            rx.icon("sparkles", size=16, color="var(--jade-11)"),
                                            rx.text(
                                                "Sachgruppe: ", item["label"],
                                                weight="bold", size="3", color="var(--jade-11)",
                                            ),
                                            spacing="2",
                                            align_items="center",
                                        ),
                                        rx.text("Beschreibung: ", item["description"], size="2", color="var(--jade-11)"),
                                        rx.cond(
                                            item["has_proba"],
                                            rx.text("Wahrscheinlichkeit: ", item["proba"], size="2", color="var(--jade-11)"),
                                        ),
                                        padding="12px",
                                        background_color="var(--jade-2)",
                                        border="1px solid var(--jade-6)",
                                        border_radius="5px",
                                        flex="1",
                                    ),
                                    rx.box(
                                        rx.text(item["label"], weight="bold", size="3", color="var(--gray-11)"),
                                        rx.text(item["description"], size="2", color="var(--gray-10)"),
                                        rx.cond(
                                            item["has_proba"],
                                            rx.text(item["proba"], size="2", color="var(--gray-10)"),
                                        ),
                                        padding="12px",
                                        background_color="var(--gray-3)",
                                        border_radius="5px",
                                        flex="1",
                                    ),
                                ),
                            ),
                            align_items="stretch",
                            spacing="3",
                            width="100%",
                        )
                    ),

                    # SHAP explanation
                    rx.cond(
                        PredictionState.prediction_result,
                        shap_card(),
                    ),

                    spacing="3"
                ),
                padding="1.5rem",
                width="100%"
            ),

            # Batch Prediction
            rx.card(
                rx.vstack(
                    rx.heading("Batch-Vorhersage", size="5"),
                    rx.text("CSV mit Spalte 'bedeutung' (Spalte 'lemma' optional)", color="var(--gray-11)"),

                    rx.hstack(
                        rx.upload(
                            rx.button("CSV hochladen"),
                            id="batch_upload",
                            accept={".csv": ["text/csv"]},
                            max_files=1,
                            on_drop=PredictionState.handle_batch_upload,
                        ),
                        rx.cond(
                            PredictionState.batch_filename,
                            rx.text(
                                PredictionState.batch_filename,
                                size="2",
                                color="var(--gray-11)",
                            ),
                        ),
                        align_items="center",
                        spacing="3",
                    ),

                    rx.button(
                        "Vorhersagen",
                        on_click=PredictionState.run_batch_prediction,
                        disabled=~PredictionState.can_batch_predict,
                        loading=PredictionState.is_predicting,
                        color_scheme="jade",
                    ),

                    rx.cond(
                        PredictionState.batch_upload_error,
                        rx.callout(
                            PredictionState.batch_upload_error,
                            icon="triangle_alert",
                            color_scheme="red"
                        )
                    ),

                    rx.cond(
                        PredictionState.has_batch_results,
                        rx.vstack(
                            rx.hstack(
                                rx.heading(f"{PredictionState.batch_results_count} Vorhersagen", size="4"),
                                rx.spacer(),
                                rx.button(
                                    rx.icon("download", size=16),
                                    "CSV herunterladen",
                                    on_click=PredictionState.download_batch_csv,
                                    variant="outline",
                                    color_scheme="jade",
                                    size="2",
                                ),
                                width="100%",
                                align_items="center",
                            ),
                            ag_grid(
                                id="batch_results_grid",
                                row_data=PredictionState.batch_results,
                                column_defs=[
                                    ag_grid.column_def(field="lemma", header_name="Lemma", sortable=True, filter=True),
                                    ag_grid.column_def(field="bedeutung", header_name="Bedeutung", sortable=True, filter=True),
                                    ag_grid.column_def(field="sachgruppe", header_name="SG 1", sortable=True, filter=True),
                                    ag_grid.column_def(field="beschreibung", header_name="Beschreibung 1", sortable=True, filter=True),
                                    ag_grid.column_def(field="wahrscheinlichkeit", header_name="W. 1", sortable=True, filter=True),
                                    ag_grid.column_def(field="sachgruppe_2", header_name="SG 2", sortable=True, filter=True),
                                    ag_grid.column_def(field="beschreibung_2", header_name="Beschreibung 2", sortable=True, filter=True),
                                    ag_grid.column_def(field="wahrscheinlichkeit_2", header_name="W. 2", sortable=True, filter=True),
                                    ag_grid.column_def(field="sachgruppe_3", header_name="SG 3", sortable=True, filter=True),
                                    ag_grid.column_def(field="beschreibung_3", header_name="Beschreibung 3", sortable=True, filter=True),
                                    ag_grid.column_def(field="wahrscheinlichkeit_3", header_name="W. 3", sortable=True, filter=True),
                                ],
                                default_col_def={"flex": 1, "minWidth": 100},
                                resizable=True,
                                dom_layout="autoHeight",
                                height="None",
                                column_size="sizeToFit",
                            ),
                            spacing="3",
                            width="100%"
                        )
                    ),

                    spacing="3"
                ),
                padding="1.5rem",
                width="100%"
            ),

            spacing="4",
            width="100%",
            max_width="100%"
        )
    )
