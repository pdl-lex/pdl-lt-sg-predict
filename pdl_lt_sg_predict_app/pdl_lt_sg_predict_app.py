"""
Sachgruppen-Klassifikation Web-App
Machine Learning Interface für Modelltraining, Analyse und Vorhersage
"""

import reflex as rx
import pandas as pd
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
import sys

# Füge Parent-Verzeichnis zum Path hinzu um sachgruppen_classifier zu importieren
# __file__ -> pdl_lt_sg_predict_app.py
# parent -> pdl_lt_sg_predict_app/ (package)
# parent.parent -> pdl_lt_sg_predict_app/ (project root)
# parent.parent.parent -> pdl-lt-sg-predict/ (wo sachgruppen_classifier.py liegt)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sachgruppen_classifier import SachgruppenClassifier, train_and_evaluate

# ============ Configuration ============

# WICHTIG: Training-Seite ist standardmäßig deaktiviert (für schwache VMs)
# Setzen Sie ENABLE_TRAINING = True wenn Sie lokal auf einem starken System arbeiten
ENABLE_TRAINING = True

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Verfügbare Modelle
AVAILABLE_MODELS = [
    ("svm", "Linear SVM (schnell, gut)"),
    ("logistic", "Logistic Regression (sehr schnell)"),
    ("rf", "Random Forest (mittel)"),
    ("nn", "Neural Network (langsam)"),
    ("xgboost", "XGBoost (sehr langsam, beste Accuracy)"),
]

# ============ States ============

class BaseState(rx.State):
    """Basis-State für gemeinsame Funktionen"""
    session_dir: str = ""

    def _create_session_dir(self):
        """Erstellt Session-Verzeichnis"""
        import tempfile

        if not self.session_dir:
            session_id = self.router.session.client_token
            self.session_dir = str(Path(tempfile.gettempdir()) / f"ml_session_{session_id}")

        session_path = Path(self.session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path


class TrainingState(BaseState):
    """State für Model-Training"""
    # File Upload
    uploaded_filename: str = ""
    upload_error: str = ""
    is_uploading: bool = False

    # Training Config
    selected_model: str = "svm"
    test_size: float = 0.2
    use_gpu: bool = False

    # Training Status
    is_training: bool = False
    training_progress: str = ""
    training_error: str = ""

    # Training Results
    accuracy: float = 0.0
    training_time: float = 0.0
    saved_model_path: str = ""

    # Data Info
    total_samples: int = 0
    num_classes: int = 0

    @rx.var
    def has_data(self) -> bool:
        return bool(self.uploaded_filename)

    @rx.var
    def can_train(self) -> bool:
        return self.has_data and not self.is_training

    @rx.var
    def test_size_formatted(self) -> str:
        """Formatiert test_size als Prozent-String"""
        return f"{self.test_size * 100:.0f}%"

    async def handle_csv_upload(self, files: list[rx.UploadFile]):
        """Verarbeitet CSV-Upload"""
        self.is_uploading = True
        self.upload_error = ""
        self.uploaded_filename = ""
        yield

        try:
            if len(files) != 1:
                self.upload_error = "Bitte genau eine CSV-Datei hochladen"
                return

            file = files[0]

            # Check file type
            if not file.filename.lower().endswith('.csv'):
                self.upload_error = "Nur CSV-Dateien erlaubt"
                return

            # Save file
            session_path = self._create_session_dir()
            safe_filename = Path(file.filename).name
            file_path = session_path / safe_filename

            content = await file.read()

            # Size check
            if len(content) > MAX_FILE_SIZE:
                self.upload_error = f"Datei zu groß (max {MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"
                return

            with open(file_path, "wb") as f:
                f.write(content)

            # Validate CSV structure
            try:
                df = pd.read_csv(file_path)

                required_cols = ['lemma', 'bedeutung', 'sachgruppe']
                if not all(col in df.columns for col in required_cols):
                    self.upload_error = f"CSV muss Spalten enthalten: {', '.join(required_cols)}"
                    return

                self.uploaded_filename = safe_filename
                self.total_samples = len(df)
                self.num_classes = df['sachgruppe'].nunique()

            except Exception as e:
                self.upload_error = f"Fehler beim Lesen der CSV: {str(e)}"
                return

        except Exception as e:
            self.upload_error = f"Upload-Fehler: {str(e)}"
        finally:
            self.is_uploading = False

    def handle_test_size_change(self, value: list[float]):
        """Handler für Slider (erwartet Liste)"""
        if value:
            self.test_size = value[0]

    def handle_model_selection(self, display_name: str):
        """Konvertiert Display-Name zu Model-Type"""
        for model_type, name in AVAILABLE_MODELS:
            if name == display_name:
                self.selected_model = model_type
                return

    @rx.var
    def selected_model_display(self) -> str:
        """Gibt Display-Name für selected_model zurück"""
        for model_type, name in AVAILABLE_MODELS:
            if model_type == self.selected_model:
                return name
        return "Linear SVM (schnell, gut)"

    async def start_training(self):
        """Startet Model-Training"""
        self.is_training = True
        self.training_error = ""
        self.training_progress = "Starte Training..."
        self.accuracy = 0.0
        self.training_time = 0.0
        yield

        try:
            session_path = self._create_session_dir()
            csv_path = session_path / self.uploaded_filename

            if not csv_path.exists():
                self.training_error = "CSV-Datei nicht gefunden"
                return

            # Model-Namen für Speicherung
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_{self.selected_model}_{timestamp}.pkl"
            model_path = Path(__file__).parent.parent.parent / model_filename

            self.training_progress = f"Trainiere {self.selected_model.upper()}-Modell..."
            yield

            # Training starten
            start_time = time.time()

            clf, accuracy = train_and_evaluate(
                str(csv_path),
                model_type=self.selected_model,
                test_size=self.test_size,
                tune=False,
                save_path=str(model_path),
                use_gpu=self.use_gpu
            )

            self.training_time = time.time() - start_time
            self.accuracy = accuracy
            self.saved_model_path = str(model_path.name)
            self.training_progress = "✓ Training abgeschlossen!"

            # Speichere Metadaten
            metadata = {
                "model_type": self.selected_model,
                "accuracy": accuracy,
                "training_time": self.training_time,
                "timestamp": timestamp,
                "num_samples": self.total_samples,
                "num_classes": self.num_classes,
                "model_file": model_filename
            }

            metadata_filename = model_filename.replace(".pkl", "_metadata.json")
            metadata_path = Path(__file__).parent.parent.parent / metadata_filename
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.training_error = f"Training-Fehler: {str(e)}"
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False


class AnalysisState(BaseState):
    """State für Model-Analyse"""
    models_list: list[dict] = []
    is_loading: bool = False

    @rx.var
    def has_models(self) -> bool:
        return len(self.models_list) > 0

    @rx.var
    def models_count(self) -> int:
        return len(self.models_list)

    def load_models(self):
        """Lädt Liste aller gespeicherten Modelle"""
        self.is_loading = True
        yield

        try:
            models_dir = Path(__file__).parent.parent.parent
            models = []

            # Finde alle .pkl Dateien
            for pkl_file in models_dir.glob("model_*.pkl"):
                # Versuche Metadaten zu laden
                metadata_file = models_dir / pkl_file.name.replace(".pkl", "_metadata.json")

                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Fallback: Basisdaten aus Dateiname extrahieren
                    metadata = {
                        "model_file": pkl_file.name,
                        "model_type": pkl_file.stem.split('_')[1] if '_' in pkl_file.stem else "unknown",
                        "accuracy": 0.0,
                        "timestamp": datetime.fromtimestamp(pkl_file.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
                    }

                models.append(metadata)

            # Sortiere nach Timestamp (neueste zuerst)
            models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            self.models_list = models

        except Exception as e:
            print(f"Fehler beim Laden der Modelle: {e}")
        finally:
            self.is_loading = False


class PredictionState(BaseState):
    """State für Vorhersagen"""
    # Single Prediction
    input_lemma: str = ""
    input_bedeutung: str = ""
    prediction_result: str = ""
    prediction_proba: float = 0.0

    # Batch Prediction
    batch_filename: str = ""
    batch_upload_error: str = ""
    batch_results: list[dict] = []
    is_predicting: bool = False

    # Model Selection
    selected_model_file: str = ""
    available_models: list[str] = []

    @rx.var
    def has_model(self) -> bool:
        return bool(self.selected_model_file)

    @rx.var
    def can_predict(self) -> bool:
        return self.has_model and bool(self.input_lemma) and bool(self.input_bedeutung)

    @rx.var
    def has_batch_results(self) -> bool:
        return len(self.batch_results) > 0

    @rx.var
    def batch_results_count(self) -> int:
        """Anzahl der Batch-Ergebnisse"""
        return len(self.batch_results)

    @rx.var
    def prediction_proba_formatted(self) -> str:
        """Formatiert Wahrscheinlichkeit als Prozent-String"""
        if self.prediction_proba > 0:
            return f"{self.prediction_proba * 100:.2f}%"
        return ""

    # Explicit setters (erforderlich ab Reflex 0.9.0)
    def set_selected_model_file(self, value: str):
        """Setter für selected_model_file"""
        self.selected_model_file = value

    def set_input_lemma(self, value: str):
        """Setter für input_lemma"""
        self.input_lemma = value

    def set_input_bedeutung(self, value: str):
        """Setter für input_bedeutung"""
        self.input_bedeutung = value

    def load_available_models(self):
        """Lädt Liste verfügbarer Modelle"""
        # __file__ -> pdl_lt_sg_predict_app.py
        # parent -> pdl_lt_sg_predict_app/ (package)
        # parent.parent -> pdl_lt_sg_predict_app/ (project)
        # parent.parent.parent -> pdl-lt-sg-predict/ (wo Modelle liegen)
        models_dir = Path(__file__).parent.parent.parent
        model_files = [f.name for f in models_dir.glob("model_*.pkl")]
        self.available_models = sorted(model_files, reverse=True)

        if model_files and not self.selected_model_file:
            self.selected_model_file = model_files[0]

    def predict_single(self):
        """Einzelvorhersage"""
        if not self.can_predict:
            return

        self.is_predicting = True
        self.prediction_result = ""
        self.prediction_proba = 0.0
        yield

        try:
            models_dir = Path(__file__).parent.parent.parent
            model_path = models_dir / self.selected_model_file

            # Modell laden
            clf = SachgruppenClassifier.load(str(model_path))

            # Vorhersage
            X_pred = pd.DataFrame({
                'lemma': [self.input_lemma],
                'bedeutung': [self.input_bedeutung]
            })

            prediction = clf.predict(X_pred)[0]
            self.prediction_result = str(prediction)

            # Wahrscheinlichkeit wenn verfügbar
            try:
                probas = clf.predict_proba(X_pred)[0]
                self.prediction_proba = float(max(probas))
            except:
                self.prediction_proba = 0.0

        except Exception as e:
            self.prediction_result = f"Fehler: {str(e)}"
        finally:
            self.is_predicting = False

    async def handle_batch_upload(self, files: list[rx.UploadFile]):
        """Batch-CSV Upload"""
        self.batch_upload_error = ""
        self.batch_filename = ""
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

            # Validate
            df = pd.read_csv(file_path)
            if 'lemma' not in df.columns or 'bedeutung' not in df.columns:
                self.batch_upload_error = "CSV muss 'lemma' und 'bedeutung' Spalten enthalten"
                return

            self.batch_filename = safe_filename

            # Batch Prediction
            self.is_predicting = True
            yield

            models_dir = Path(__file__).parent.parent.parent
            model_path = models_dir / self.selected_model_file

            clf = SachgruppenClassifier.load(str(model_path))

            predictions = clf.predict(df[['lemma', 'bedeutung']])

            # Ergebnisse sammeln
            results = []
            for idx, (_, row) in enumerate(df.iterrows()):
                results.append({
                    'lemma': row['lemma'],
                    'bedeutung': row['bedeutung'],
                    'sachgruppe': predictions[idx]
                })

            self.batch_results = results

        except Exception as e:
            self.batch_upload_error = f"Fehler: {str(e)}"
        finally:
            self.is_predicting = False


# ============ UI Components ============

def navbar() -> rx.Component:
    """Navigation Bar"""
    nav_items = [
        ("Start", "/"),
        ("Analyse", "/analyse"),
        ("Vorhersage", "/vorhersage"),
    ]

    if ENABLE_TRAINING:
        nav_items.insert(1, ("Training", "/training"))

    return rx.hstack(
        rx.heading("Sachgruppen-Klassifikation", size="6", color="var(--jade-11)"),
        rx.spacer(),
        rx.hstack(
            *[
                rx.link(
                    rx.button(
                        name,
                        variant="ghost",
                        color_scheme="jade"
                    ),
                    href=href
                )
                for name, href in nav_items
            ],
            spacing="2"
        ),
        padding="1rem",
        width="100%",
        border_bottom="1px solid var(--gray-6)"
    )


def base_layout(*children) -> rx.Component:
    """Base Layout mit Navigation"""
    return rx.vstack(
        navbar(),
        rx.container(
            *children,
            padding_top="2rem",
            padding_bottom="2rem",
        ),
        spacing="0",
        width="100%",
        min_height="100vh"
    )


# ============ Pages ============

def index() -> rx.Component:
    """Start-Seite"""
    return base_layout(
        rx.vstack(
            rx.heading("Willkommen", size="7", color="var(--jade-12)"),
            rx.text(
                "Machine Learning Tool für automatische Sachgruppen-Klassifikation von Wörterbuch-Einträgen.",
                size="4",
                color="var(--gray-11)"
            ),
            rx.divider(),

            rx.heading("Features", size="5", margin_top="2rem"),

            rx.vstack(
                rx.card(
                    rx.hstack(
                        rx.icon("brain_circuit", size=32, color="var(--jade-11)"),
                        rx.vstack(
                            rx.heading("Training" if ENABLE_TRAINING else "Training (deaktiviert)", size="4"),
                            rx.text(
                                "Trainiere Modelle auf Ihren eigenen Daten" if ENABLE_TRAINING else
                                "Training ist für schwache VMs deaktiviert. Setzen Sie ENABLE_TRAINING = True im Code.",
                                color="var(--gray-11)"
                            ),
                            align_items="start",
                            spacing="1"
                        ),
                        align_items="center",
                        spacing="4"
                    ),
                    padding="1.5rem",
                    width="100%"
                ),

                rx.card(
                    rx.hstack(
                        rx.icon("bar_chart_3", size=32, color="var(--jade-11)"),
                        rx.vstack(
                            rx.heading("Analyse", size="4"),
                            rx.text(
                                "Vergleichen Sie Performance verschiedener trainierter Modelle",
                                color="var(--gray-11)"
                            ),
                            align_items="start",
                            spacing="1"
                        ),
                        align_items="center",
                        spacing="4"
                    ),
                    padding="1.5rem",
                    width="100%"
                ),

                rx.card(
                    rx.hstack(
                        rx.icon("sparkles", size=32, color="var(--jade-11)"),
                        rx.vstack(
                            rx.heading("Vorhersage", size="4"),
                            rx.text(
                                "Klassifizieren Sie neue Einträge (einzeln oder im Batch)",
                                color="var(--gray-11)"
                            ),
                            align_items="start",
                            spacing="1"
                        ),
                        align_items="center",
                        spacing="4"
                    ),
                    padding="1.5rem",
                    width="100%"
                ),

                spacing="3",
                width="100%"
            ),

            rx.divider(margin_top="2rem"),

            rx.heading("Verfügbare Modelle", size="5", margin_top="2rem"),
            rx.vstack(
                *[
                    rx.hstack(
                        rx.badge(code, color_scheme="jade"),
                        rx.text(name),
                        spacing="2"
                    )
                    for code, name in AVAILABLE_MODELS
                ],
                align_items="start",
                spacing="2"
            ),

            spacing="4",
            width="100%",
            max_width="800px"
        )
    )


def training_page() -> rx.Component:
    """Training-Seite"""
    if not ENABLE_TRAINING:
        return base_layout(
            rx.vstack(
                rx.heading("Training deaktiviert", size="7", color="var(--red-11)"),
                rx.text(
                    "Das Training ist standardmäßig deaktiviert, um schwache VMs zu schonen.",
                    size="4"
                ),
                rx.text(
                    "Um Training zu aktivieren, setzen Sie ENABLE_TRAINING = True in der App-Konfiguration.",
                    size="3",
                    color="var(--gray-11)"
                ),
                spacing="4"
            )
        )

    return base_layout(
        rx.vstack(
            rx.heading("Model Training", size="7", color="var(--jade-12)"),

            # Data Upload
            rx.card(
                rx.vstack(
                    rx.heading("1. Daten hochladen", size="5"),
                    rx.text("CSV mit Spalten: lemma, bedeutung, sachgruppe", color="var(--gray-11)"),

                    rx.upload(
                        rx.button(
                            "CSV-Datei auswählen",
                            loading=TrainingState.is_uploading
                        ),
                        id="csv_upload",
                        accept={".csv": ["text/csv"]},
                        max_files=1,
                        on_drop=TrainingState.handle_csv_upload
                    ),

                    rx.cond(
                        TrainingState.uploaded_filename,
                        rx.hstack(
                            rx.icon("check-check", color="var(--jade-11)"),
                            rx.text(TrainingState.uploaded_filename),
                            rx.badge(f"{TrainingState.total_samples} Samples, {TrainingState.num_classes} Klassen"),
                            spacing="2"
                        )
                    ),

                    rx.cond(
                        TrainingState.upload_error,
                        rx.callout(
                            TrainingState.upload_error,
                            icon="triangle_alert",
                            color_scheme="red"
                        )
                    ),

                    spacing="3"
                ),
                padding="1.5rem",
                width="100%"
            ),

            # Training Config
            rx.card(
                rx.vstack(
                    rx.heading("2. Modell & Parameter", size="5"),

                    rx.hstack(
                        rx.vstack(
                            rx.text("Modell-Typ:", font_weight="bold"),
                            rx.select(
                                [name for _, name in AVAILABLE_MODELS],
                                value=TrainingState.selected_model_display,
                                on_change=TrainingState.handle_model_selection
                            ),
                            align_items="start",
                            spacing="1"
                        ),

                        rx.vstack(
                            rx.text("Test-Anteil:", font_weight="bold"),
                            rx.slider(
                                default_value=[0.2],
                                value=[TrainingState.test_size],
                                min=0.1,
                                max=0.4,
                                step=0.05,
                                on_change=TrainingState.handle_test_size_change
                            ),
                            rx.text(TrainingState.test_size_formatted, color="var(--gray-11)"),
                            align_items="start",
                            spacing="1"
                        ),

                        rx.vstack(
                            rx.text("GPU-Training:", font_weight="bold"),
                            rx.switch(
                                checked=TrainingState.use_gpu,
                                on_change=TrainingState.set_use_gpu
                            ),
                            align_items="start",
                            spacing="1"
                        ),

                        spacing="4",
                        width="100%"
                    ),

                    spacing="3"
                ),
                padding="1.5rem",
                width="100%"
            ),

            # Start Training
            rx.card(
                rx.vstack(
                    rx.heading("3. Training starten", size="5"),

                    rx.button(
                        "Modell trainieren",
                        on_click=TrainingState.start_training,
                        disabled=~TrainingState.can_train,
                        loading=TrainingState.is_training,
                        size="3",
                        color_scheme="jade"
                    ),

                    rx.cond(
                        TrainingState.is_training,
                        rx.vstack(
                            rx.text(TrainingState.training_progress),
                            rx.progress(value=50, is_indeterminate=True),
                            spacing="2"
                        )
                    ),

                    rx.cond(
                        TrainingState.accuracy > 0,
                        rx.callout(
                            rx.box(
                                rx.heading(f"Accuracy: {TrainingState.accuracy:.4f}", size="4"),
                                rx.text(f"Trainingszeit: {TrainingState.training_time:.1f}s"),
                                rx.text(f"Modell gespeichert: {TrainingState.saved_model_path}"),
                            ),
                            icon="check-check",
                            color_scheme="jade"
                        )
                    ),

                    rx.cond(
                        TrainingState.training_error,
                        rx.callout(
                            TrainingState.training_error,
                            icon="triangle_alert",
                            color_scheme="red"
                        )
                    ),

                    spacing="3"
                ),
                padding="1.5rem",
                width="100%"
            ),

            spacing="4",
            width="100%",
            max_width="1000px"
        )
    )


def analyse_page() -> rx.Component:
    """Analyse-Seite"""
    return base_layout(
        rx.vstack(
            rx.heading("Model-Analyse", size="7", color="var(--jade-12)"),
            rx.text("Vergleich aller trainierten Modelle", color="var(--gray-11)"),

            rx.button(
                "Modelle neu laden",
                on_click=AnalysisState.load_models,
                loading=AnalysisState.is_loading
            ),

            rx.cond(
                AnalysisState.has_models,
                rx.vstack(
                    rx.heading(f"{AnalysisState.models_count} Modelle gefunden", size="5"),

                    rx.data_table(
                        data=AnalysisState.models_list,
                        columns=["model_type", "accuracy", "training_time", "timestamp", "num_samples", "num_classes"],
                        sort=True
                    ),

                    spacing="3",
                    width="100%"
                ),
                rx.text("Keine trainierten Modelle gefunden", color="var(--gray-11)")
            ),

            spacing="4",
            width="100%",
            max_width="1200px"
        )
    )


def vorhersage_page() -> rx.Component:
    """Vorhersage-Seite"""
    return base_layout(
        rx.vstack(
            rx.heading("Vorhersage", size="7", color="var(--jade-12)"),

            # Model Selection
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

            # Single Prediction
            rx.card(
                rx.vstack(
                    rx.heading("Einzelvorhersage", size="5"),

                    rx.vstack(
                        rx.input(
                            placeholder="Lemma (z.B. #Kreuz)",
                            value=PredictionState.input_lemma,
                            on_change=PredictionState.set_input_lemma,
                            width="100%"
                        ),
                        rx.input(
                            placeholder="Bedeutung (z.B. Kreuz in Redensart)",
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
                        rx.callout(
                            rx.box(
                                rx.text(f"Sachgruppe: {PredictionState.prediction_result}", font_weight="bold", size="4"),
                                rx.cond(
                                    PredictionState.prediction_proba > 0,
                                    rx.text(f"Wahrscheinlichkeit: {PredictionState.prediction_proba_formatted}")
                                ),
                            ),
                            icon="sparkles",
                            color_scheme="jade"
                        )
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
                    rx.text("CSV mit Spalten: lemma, bedeutung", color="var(--gray-11)"),

                    rx.upload(
                        rx.button("CSV hochladen"),
                        id="batch_upload",
                        accept={".csv": ["text/csv"]},
                        max_files=1,
                        on_drop=PredictionState.handle_batch_upload
                    ),

                    rx.cond(
                        PredictionState.batch_filename,
                        rx.text(f"Datei: {PredictionState.batch_filename}")
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
                            rx.heading(f"{PredictionState.batch_results_count} Vorhersagen", size="4"),
                            rx.data_table(
                                data=PredictionState.batch_results,
                                columns=["lemma", "bedeutung", "sachgruppe"],
                                pagination=True,
                                search=True,
                                sort=True
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
            max_width="1000px"
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

app.add_page(index, route="/")
if ENABLE_TRAINING:
    app.add_page(training_page, route="/training")
app.add_page(analyse_page, route="/analyse")
app.add_page(vorhersage_page, route="/vorhersage")
