"""
Sachgruppen-Klassifikation Web-App
Machine Learning Interface für Modelltraining, Analyse und Vorhersage
"""

import asyncio
import os
import reflex as rx
import pandas as pd
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
import sys

from dotenv import load_dotenv
load_dotenv()

# Füge Parent-Verzeichnis zum Path hinzu um sachgruppen_classifier zu importieren
# __file__ -> pdl_lt_sg_predict_app/pdl_lt_sg_predict_app.py
# parent -> pdl_lt_sg_predict_app/ (package)
# parent.parent -> pdl-lt-sg-predict/ (project root, wo sachgruppen_classifier.py liegt)
sys.path.insert(0, str(Path(__file__).parent.parent))

from sachgruppen_classifier import SachgruppenClassifier, train_and_evaluate

# Modelle außerhalb des Projektordners speichern (konfigurierbar via .env)
_models_dir_env = os.getenv("MODELS_DIR", "")
MODELS_DIR = Path(_models_dir_env) if _models_dir_env else Path.home() / ".pdl-sg-predict" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _next_model_filename(model_type: str) -> str:
    """Gibt den nächsten verfügbaren Dateinamen zurück, z.B. model_svm_003.pkl."""
    import re
    pattern = re.compile(rf"model_{re.escape(model_type)}_(\d+)\.pkl", re.IGNORECASE)
    max_n = 0
    for f in MODELS_DIR.glob(f"model_{model_type}_*.pkl"):
        m = pattern.match(f.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"model_{model_type}_{max_n + 1:03d}.pkl"
from pdl_lt_reflex_aggrid_wrapper import ag_grid
from .components import base_layout, ENABLE_TRAINING

# ============ Configuration ============

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Verfügbare Modelle
AVAILABLE_MODELS = [
    ("svm", "Linear SVM (schnell, gut)"),
    ("logistic", "Logistic Regression (sehr schnell)"),
    ("rf", "Random Forest (mittel)"),
    ("nn", "Neural Network (langsam)"),
    ("xgboost", "XGBoost (sehr langsam, beste Accuracy)"),
]

# Sachgruppen-Mapping laden (Nummer -> Beschreibung)
SACHGRUPPEN_CSV = Path(__file__).parent.parent / "sachgruppen.csv"
SACHGRUPPEN_MAP: dict[str, str] = {}
if SACHGRUPPEN_CSV.exists():
    _sg_df = pd.read_csv(SACHGRUPPEN_CSV, sep=";", dtype=str)
    SACHGRUPPEN_MAP = dict(zip(_sg_df["Nummer"], _sg_df["Sachgruppe"]))

# Mapping von internem model_type zu schönem Display-Namen
MODEL_DISPLAY_NAMES = {
    "svm": "Linear SVM",
    "logistic": "Logistic Regression",
    "rf": "Random Forest",
    "nn": "Neural Network",
    "xgboost": "XGBoost",
}

# Fallback-Trainingszeiten (Sekunden) für ~113 127 Samples, gemessen auf Entwicklungsrechner.
# Werden durch historische Metadaten überschrieben sobald ein Modell des jeweiligen Typs trainiert wurde.
_TIME_FALLBACKS: dict[str, float] = {
    "svm": 120.0,
    "logistic": 4286.0,
    "rf": 30.0,
    "nn": 111.0,
    "xgboost": 6112.0,
}
_TIME_FALLBACK_SAMPLES = 113_127

# ============ SHAP Helpers ============

def _shap_score_to_dict(word: str, score: float) -> dict:
    """Erzeugt ein serialisierbares Dict mit vorberechneter Badge-Farbe."""
    color = "jade" if score > 0.1 else ("red" if score < -0.1 else "gray")
    return {"word": word, "score": round(score, 4), "color": color}


def _shap_pairs_to_dicts(pairs: list) -> list[dict]:
    return [_shap_score_to_dict(w, s) for w, s in pairs]


# ============ Model Cache ============

_MODEL_CACHE: dict[str, "SachgruppenClassifier"] = {}

def _get_model(model_path: str) -> "SachgruppenClassifier":
    """Lädt ein Modell einmalig und hält es im Speicher vor."""
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = SachgruppenClassifier.load(model_path)
    return _MODEL_CACHE[model_path]


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
    use_stopword_removal: bool = False
    min_word_length: int = 1
    analyzer_mode: str = "char_wb"   # "char_wb" oder "word"
    word_ngram_max: int = 1          # 1=(1,1), 2=(1,2) — nur bei analyzer_mode="word"

    # Batch-Training Config
    batch_model_types: list[str] = ["svm"]
    batch_use_stopwords: list[bool] = [False]   # [False], [True], oder [False, True]
    batch_min_lengths: list[int] = [1]
    batch_analyzers: list[str] = ["char_wb"]

    # Historische Trainingszeiten: model_type → Sekunden (skaliert auf aktuelle Sample-Anzahl)
    time_per_type: dict[str, float] = {}

    # Batch-Training Status
    batch_is_running: bool = False
    batch_total: int = 0
    batch_done: int = 0
    batch_current_config: str = ""
    batch_errors: list[str] = []

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
        return self.has_data and not self.is_training and not self.batch_is_running

    @rx.var
    def test_size_formatted(self) -> str:
        """Formatiert test_size als Prozent-String"""
        return f"{self.test_size * 100:.0f}%"

    @rx.var
    def min_word_length_formatted(self) -> str:
        return f"≥ {self.min_word_length} Zeichen"

    @rx.var
    def word_ngram_max_str(self) -> str:
        return str(self.word_ngram_max)

    @rx.var
    def batch_preview_count(self) -> int:
        """Anzahl Modelle, die beim Batch-Training trainiert werden."""
        return (
            len(self.batch_model_types)
            * len(self.batch_use_stopwords)
            * len(self.batch_min_lengths)
            * len(self.batch_analyzers)
        )

    @rx.var
    def batch_preview_label(self) -> str:
        n = (
            len(self.batch_model_types)
            * len(self.batch_use_stopwords)
            * len(self.batch_min_lengths)
            * len(self.batch_analyzers)
        )
        return f"{n} Modelle werden trainiert"

    @rx.var
    def batch_estimated_time_str(self) -> str:
        """Geschätzte Gesamtdauer des Batch-Trainings basierend auf historischen Messungen."""
        if not self.batch_model_types:
            return ""
        # Anzahl Konfigurationen je Modelltyp (alle Preprocessing-Kombinationen)
        configs_per_type = (
            len(self.batch_use_stopwords)
            * len(self.batch_min_lengths)
            * len(self.batch_analyzers)
        )
        n_current = max(self.total_samples, 1)
        total_secs = 0.0
        for mt in self.batch_model_types:
            if mt in self.time_per_type and self.time_per_type[mt] > 0:
                est = self.time_per_type[mt]
            else:
                # Fallback: auf aktuelle Sample-Anzahl skalieren
                fallback = _TIME_FALLBACKS.get(mt, 120.0)
                est = fallback / _TIME_FALLBACK_SAMPLES * n_current
            total_secs += est * configs_per_type
        if total_secs <= 0:
            return ""
        t = int(total_secs)
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        source = "gemessen" if self.time_per_type else "geschätzt"
        if h > 0:
            return f"ca. {h}h {m:02d}min ({source})"
        elif m > 0:
            return f"ca. {m}min {s:02d}s ({source})"
        else:
            return f"ca. {s}s ({source})"

    @rx.var
    def batch_progress_pct(self) -> int:
        if self.batch_total == 0:
            return 0
        return int(self.batch_done / self.batch_total * 100)

    @rx.var
    def batch_progress_label(self) -> str:
        if self.batch_total == 0:
            return ""
        return f"Trainiere {self.batch_done + 1}/{self.batch_total}: {self.batch_current_config}"

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
                df = pd.read_csv(file_path, sep=None, engine="python")

                required_cols = ['lemma', 'bedeutung', 'sachgruppe']
                if not all(col in df.columns for col in required_cols):
                    self.upload_error = f"CSV muss Spalten enthalten: {', '.join(required_cols)}"
                    return

                self.uploaded_filename = safe_filename
                self.total_samples = len(df)
                self.num_classes = df['sachgruppe'].nunique()
                self._refresh_time_estimates()

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

    def set_use_stopword_removal(self, value: bool):
        self.use_stopword_removal = value

    def _refresh_time_estimates(self):
        """
        Liest historische Trainingszeiten aus Metadaten-Dateien und skaliert sie
        auf die aktuell hochgeladene Datenmenge. Wird nach jedem CSV-Upload aufgerufen.
        Für jeden Modelltyp wird der aktuellste Eintrag verwendet.
        """
        models_dir = MODELS_DIR
        if not models_dir.exists() or self.total_samples == 0:
            return
        # Aktuellste Messung pro Typ suchen (höchster Timestamp gewinnt)
        best: dict[str, tuple[str, float, int]] = {}  # type → (timestamp, time, samples)
        for mf in models_dir.glob("*_metadata.json"):
            try:
                with open(mf) as f:
                    m = json.load(f)
                mt = m.get("model_type", "")
                t = float(m.get("training_time", 0))
                n = int(m.get("num_samples", 0))
                ts = m.get("timestamp", "")
                if mt and t > 0 and n > 0:
                    if mt not in best or ts > best[mt][0]:
                        best[mt] = (ts, t, n)
            except Exception:
                pass
        estimates = {}
        for mt, (_, t, n) in best.items():
            estimates[mt] = t / n * self.total_samples
        self.time_per_type = estimates

    def handle_min_word_length_change(self, value: list[float]):
        if value:
            self.min_word_length = int(value[0])

    def handle_analyzer_mode_change(self, value: str):
        self.analyzer_mode = value

    def handle_word_ngram_max_change(self, value: str):
        self.word_ngram_max = int(value)

    # ---- Batch-Konfiguration ----

    def toggle_batch_model(self, model_type: str):
        """Fügt einen Modelltyp zur Batch-Liste hinzu oder entfernt ihn."""
        if model_type in self.batch_model_types:
            if len(self.batch_model_types) > 1:
                self.batch_model_types = [m for m in self.batch_model_types if m != model_type]
        else:
            self.batch_model_types = self.batch_model_types + [model_type]

    def toggle_batch_stopwords(self, value: bool):
        """Schaltet True/False-Wert in batch_use_stopwords an/aus."""
        if value in self.batch_use_stopwords:
            if len(self.batch_use_stopwords) > 1:
                self.batch_use_stopwords = [v for v in self.batch_use_stopwords if v != value]
        else:
            self.batch_use_stopwords = self.batch_use_stopwords + [value]

    def toggle_batch_min_length(self, length: int):
        """Fügt Mindest-Wortlänge zur Batch-Liste hinzu oder entfernt sie."""
        if length in self.batch_min_lengths:
            if len(self.batch_min_lengths) > 1:
                self.batch_min_lengths = [l for l in self.batch_min_lengths if l != length]
        else:
            self.batch_min_lengths = sorted(self.batch_min_lengths + [length])

    def toggle_batch_analyzer(self, analyzer: str):
        """Fügt Analyzer zur Batch-Liste hinzu oder entfernt ihn."""
        if analyzer in self.batch_analyzers:
            if len(self.batch_analyzers) > 1:
                self.batch_analyzers = [a for a in self.batch_analyzers if a != analyzer]
        else:
            self.batch_analyzers = self.batch_analyzers + [analyzer]

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
            model_filename = _next_model_filename(self.selected_model)
            model_path = MODELS_DIR / model_filename

            self.training_progress = f"Trainiere {self.selected_model.upper()}-Modell..."
            yield

            # Training starten
            start_time = time.time()

            # State-Vars vor dem Executor-Aufruf auslesen (Thread-Sicherheit)
            word_ngram_max = self.word_ngram_max if self.analyzer_mode == "word" else 1
            _model = self.selected_model
            _test_size = self.test_size
            _remove_sw = self.use_stopword_removal
            _min_len = self.min_word_length
            _analyzer = self.analyzer_mode
            _csv = str(csv_path)
            _mp = str(model_path)

            loop = asyncio.get_running_loop()
            clf, accuracy = await loop.run_in_executor(
                None,
                lambda: train_and_evaluate(
                    _csv,
                    model_type=_model,
                    test_size=_test_size,
                    tune=False,
                    save_path=_mp,
                    remove_stopwords=_remove_sw,
                    min_word_length=_min_len,
                    analyzer=_analyzer,
                    word_ngram_max=word_ngram_max,
                )
            )

            self.training_time = time.time() - start_time
            self.accuracy = accuracy
            self.saved_model_path = str(model_path.name)
            self.training_progress = "✓ Training abgeschlossen!"

            # Speichere Metadaten
            metadata = {
                "model_type": self.selected_model,
                "model_name": MODEL_DISPLAY_NAMES.get(self.selected_model, self.selected_model),
                "accuracy": accuracy,
                "training_time": self.training_time,
                "timestamp": timestamp,
                "num_samples": self.total_samples,
                "num_classes": self.num_classes,
                "model_file": model_filename,
                "remove_stopwords": self.use_stopword_removal,
                "test_size": self.test_size,
                "min_word_length": self.min_word_length,
                "analyzer": self.analyzer_mode,
                "word_ngram_max": word_ngram_max,
            }

            metadata_filename = model_filename.replace(".pkl", "_metadata.json")
            metadata_path = MODELS_DIR / metadata_filename
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.training_error = f"Training-Fehler: {str(e)}"
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False

    async def start_batch_training(self):
        """Trainiert alle Kombinationen aus batch_* Konfigurationslisten."""
        if not self.uploaded_filename or self.is_training or self.batch_is_running:
            return

        session_path = self._create_session_dir()
        csv_path = session_path / self.uploaded_filename
        if not csv_path.exists():
            self.training_error = "CSV-Datei nicht gefunden"
            return

        # Kartesisches Produkt aller Konfigurationen
        configs = [
            {
                "model_type": mt,
                "remove_stopwords": sw,
                "min_word_length": ml,
                "analyzer": an,
                "word_ngram_max": 2 if an == "word-(1,2)" else 1,
                "analyzer_clean": "word" if an.startswith("word") else "char_wb",
            }
            for mt in self.batch_model_types
            for sw in self.batch_use_stopwords
            for ml in self.batch_min_lengths
            for an in self.batch_analyzers
        ]

        self.batch_total = len(configs)
        self.batch_done = 0
        self.batch_is_running = True
        self.batch_errors = []
        self.training_error = ""
        yield

        for cfg in configs:
            self.batch_current_config = " | ".join([
                cfg["model_type"].upper(),
                f"sw={'ja' if cfg['remove_stopwords'] else 'nein'}",
                f"len≥{cfg['min_word_length']}",
                cfg["analyzer"],
            ])
            yield

            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = _next_model_filename(cfg["model_type"])
                model_path = MODELS_DIR / model_filename

                # State-Vars vor dem Executor-Aufruf auslesen (Thread-Sicherheit)
                test_size_val = self.test_size
                total_samples_val = self.total_samples
                num_classes_val = self.num_classes
                csv_path_str = str(csv_path)
                model_path_str = str(model_path)

                start_time = time.time()
                loop = asyncio.get_running_loop()
                clf, accuracy = await loop.run_in_executor(
                    None,
                    lambda c=cfg, ts=test_size_val, cp=csv_path_str, mp=model_path_str: train_and_evaluate(
                        cp,
                        model_type=c["model_type"],
                        test_size=ts,
                        tune=False,
                        save_path=mp,
                        remove_stopwords=c["remove_stopwords"],
                        min_word_length=c["min_word_length"],
                        analyzer=c["analyzer_clean"],
                        word_ngram_max=c["word_ngram_max"],
                    )
                )
                training_time = time.time() - start_time

                metadata = {
                    "model_type": cfg["model_type"],
                    "model_name": MODEL_DISPLAY_NAMES.get(cfg["model_type"], cfg["model_type"]),
                    "accuracy": accuracy,
                    "training_time": training_time,
                    "timestamp": timestamp,
                    "num_samples": total_samples_val,
                    "num_classes": num_classes_val,
                    "model_file": model_filename,
                    "remove_stopwords": cfg["remove_stopwords"],
                    "test_size": test_size_val,
                    "min_word_length": cfg["min_word_length"],
                    "analyzer": cfg["analyzer_clean"],
                    "word_ngram_max": cfg["word_ngram_max"],
                }
                metadata_path = MODELS_DIR / model_filename.replace(".pkl", "_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            except Exception as e:
                self.batch_errors = self.batch_errors + [f"{self.batch_current_config}: {e}"]
                import traceback
                traceback.print_exc()

            self.batch_done += 1
            yield

        self.batch_is_running = False
        self.batch_current_config = ""


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
            models_dir = MODELS_DIR
            models = []

            if not models_dir.exists():
                self.is_loading = False
                return

            # Finde alle .pkl Dateien
            for pkl_file in models_dir.glob("model_*.pkl"):
                # Versuche Metadaten zu laden
                metadata_file = models_dir / pkl_file.name.replace(".pkl", "_metadata.json")

                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Fallback: Basisdaten aus Dateiname extrahieren
                    model_type = pkl_file.stem.split('_')[1] if '_' in pkl_file.stem else "unknown"
                    metadata = {
                        "model_file": pkl_file.name,
                        "model_type": model_type,
                        "accuracy": 0.0,
                        "timestamp": datetime.fromtimestamp(pkl_file.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
                    }

                # Display-Name sicherstellen
                model_name = metadata.get("model_name") or MODEL_DISPLAY_NAMES.get(
                    metadata.get("model_type", ""), metadata.get("model_type", "")
                )

                # Werte sinnvoll formatieren
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

                models.append({
                    "model_file": metadata.get("model_file", pkl_file.name),
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
                })

            # Sortiere nach Datum (neueste zuerst), dd.mm.yy -> yy.mm.dd für korrekte Sortierung
            models.sort(
                key=lambda x: ".".join(reversed(x.get("date", "").split("."))),
                reverse=True
            )

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
    prediction_result_description: str = ""
    prediction_proba: float = 0.0

    # Batch Prediction
    batch_filename: str = ""
    batch_upload_error: str = ""
    batch_results: list[dict] = []
    is_predicting: bool = False

    # Model Selection
    selected_model_file: str = ""
    available_models: list[str] = []

    # SHAP-Erklärung
    shap_lemma_scores: list[dict] = []     # [{"word": str, "score": float}, ...]
    shap_bedeutung_scores: list[dict] = []
    shap_error: str = ""
    shap_is_computing: bool = False
    shap_model_is_nn: bool = False         # MLP: SHAP nur auf Anfrage
    shap_filter_stopwords: bool = True     # Stoppwörter ausblenden (Standard: an)

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

    @rx.var
    def shap_top_words(self) -> list[dict]:
        """Top 10 Wörter nach absolutem SHAP-Einfluss (für Balkendiagramm)"""
        combined = self.shap_lemma_scores + self.shap_bedeutung_scores
        return sorted(combined, key=lambda x: abs(x["score"]), reverse=True)[:10]

    @rx.var
    def has_shap_results(self) -> bool:
        return bool(self.shap_lemma_scores or self.shap_bedeutung_scores)

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
        models_dir = MODELS_DIR
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
        self.prediction_result_description = ""
        self.prediction_proba = 0.0
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

            # Modell aus Cache laden (einmaliges Laden pro Prozess)
            clf = _get_model(model_path_str)

            # Vorhersage
            X_pred = pd.DataFrame({
                'lemma': [self.input_lemma],
                'bedeutung': [self.input_bedeutung]
            })

            prediction = clf.predict(X_pred)[0]
            self.prediction_result = str(prediction)
            self.prediction_result_description = SACHGRUPPEN_MAP.get(
                str(prediction), "(unbekannt)"
            )

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

        # SHAP-Erklärung berechnen (nur wenn Vorhersage erfolgreich)
        if clf is not None and X_pred is not None and self.prediction_result and not self.prediction_result.startswith("Fehler"):
            self.shap_model_is_nn = (clf.model_type == "nn")

            if clf.model_type != "nn":
                yield from self._run_shap_computation(clf, X_pred, model_path_str)

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
            df = pd.read_csv(file_path, sep=None, engine="python")
            if 'lemma' not in df.columns or 'bedeutung' not in df.columns:
                self.batch_upload_error = "CSV muss 'lemma' und 'bedeutung' Spalten enthalten"
                return

            self.batch_filename = safe_filename

            # Batch Prediction
            self.is_predicting = True
            yield

            model_path = MODELS_DIR / self.selected_model_file

            clf = _get_model(str(model_path))

            predictions = clf.predict(df[['lemma', 'bedeutung']])

            # Ergebnisse sammeln
            results = []
            for idx, (_, row) in enumerate(df.iterrows()):
                sg = str(predictions[idx])
                results.append({
                    'lemma': row['lemma'],
                    'bedeutung': row['bedeutung'],
                    'sachgruppe': sg,
                    'beschreibung': SACHGRUPPEN_MAP.get(sg, "(unbekannt)"),
                })

            self.batch_results = results

        except Exception as e:
            self.batch_upload_error = f"Fehler: {str(e)}"
        finally:
            self.is_predicting = False

    def _run_shap_computation(self, clf, X_pred, model_path_str: str):
        """Gemeinsame SHAP-Berechnungslogik für predict_single und compute_shap_nn."""
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
        """SHAP-Erklärung für Neural-Network-Modell (langsam, manuell ausgelöst)."""
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
            # MLP nicht automatisch neu berechnen (zu langsam)
            return

        X_pred = pd.DataFrame({
            'lemma': [self.input_lemma],
            'bedeutung': [self.input_bedeutung]
        })
        yield from self._run_shap_computation(clf, X_pred, model_path_str)

    def download_batch_csv(self):
        """Batch-Ergebnisse als CSV herunterladen"""
        if not self.batch_results:
            return

        df = pd.DataFrame(self.batch_results)
        csv_content = df.to_csv(index=False, sep=";")
        return rx.download(
            data=csv_content,
            filename="vorhersage_ergebnisse.csv",
        )


# ============ Pages ============

def index() -> rx.Component:
    """Start-Seite"""
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

            rx.vstack(
                rx.card(
                    rx.hstack(
                        rx.icon("brain_circuit", size=32, color="var(--jade-11)"),
                        rx.vstack(
                            rx.heading("Training" if ENABLE_TRAINING else "Training (deaktiviert)", size="4"),
                            rx.text(
                                "Training neuer Modelle auf eigenen Daten" if ENABLE_TRAINING else
                                "Training ist für (schwache) VMs deaktiviert. Aktivierung: ENABLE_TRAINING = True in components.py.",
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
                            rx.heading("Modellvergleich", size="4"),
                            rx.text(
                                "Vergleichen der Performance und Parameter verschiedener trainierter Modelle",
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
                                "Klassifizierung/Vorhersage für neue Lemmata (einzeln oder im Batch)",
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
                rx.heading("TRAINING", size="4", color="var(--jade-12)", weight="light"),
                rx.callout(
                    rx.vstack(
                        rx.text(
                            "Das Training ist im VM-Betrieb deaktiviert.",
                            font_weight="bold",
                        ),
                        rx.text(
                            "Der Trainingsmodus steht nur bei lokaler Ausführung zur Verfügung, "
                            "da das Training rechenintensiv ist und auf einer VM zu langsam wäre.",
                        ),
                        rx.text(
                            "Um Training zu aktivieren, setzen Sie ENABLE_TRAINING = True "
                            "in pdl_lt_sg_predict_app.py.",
                            size="2",
                            color="var(--gray-11)",
                        ),
                        spacing="2",
                    ),
                    icon="info",
                    color_scheme="amber",
                ),
                spacing="4",
            )
        )

    return base_layout(
        rx.vstack(
            rx.heading("TRAINING", size="4", color="var(--jade-12)", weight="light"),

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

                    # Jede Option in einer eigenen Zeile: links Steuerelement, rechts Erklärtext
                    # Zeile 1: Modell-Typ
                    rx.hstack(
                        rx.vstack(
                            rx.text("Modell-Typ", weight="bold", size="2"),
                            rx.select(
                                [name for _, name in AVAILABLE_MODELS],
                                value=TrainingState.selected_model_display,
                                on_change=TrainingState.handle_model_selection,
                            ),
                            align_items="start",
                            spacing="1",
                            min_width="220px",
                        ),
                        rx.text(
                            "Algorithmus für die Klassifikation. SVM ist ein guter Standard – schnell und genau. "
                            "XGBoost erreicht oft die höchste Accuracy, braucht aber deutlich länger.",
                            size="2", color="var(--gray-11)", flex="1",
                        ),
                        align_items="center",
                        spacing="6",
                        width="100%",
                    ),

                    rx.divider(),

                    # Zeile 2: Test-Anteil
                    rx.hstack(
                        rx.vstack(
                            rx.text("Test-Anteil", weight="bold", size="2"),
                            rx.slider(
                                default_value=[0.2],
                                value=[TrainingState.test_size],
                                min=0.1,
                                max=0.4,
                                step=0.05,
                                on_change=TrainingState.handle_test_size_change,
                                width="180px",
                            ),
                            rx.text(TrainingState.test_size_formatted, size="2", color="var(--gray-11)"),
                            align_items="start",
                            spacing="1",
                            min_width="220px",
                        ),
                        rx.text(
                            "Anteil der Daten, der für die Evaluation zurückgehalten wird (nicht zum Training genutzt). "
                            "20% ist ein üblicher Standardwert.",
                            size="2", color="var(--gray-11)", flex="1",
                        ),
                        align_items="center",
                        spacing="6",
                        width="100%",
                    ),

                    rx.divider(),

                    # Zeile 3: Stoppwörter
                    rx.hstack(
                        rx.vstack(
                            rx.text("Stoppwörter entfernen", weight="bold", size="2"),
                            rx.hstack(
                                rx.switch(
                                    checked=TrainingState.use_stopword_removal,
                                    on_change=TrainingState.set_use_stopword_removal,
                                    size="2",
                                ),
                                rx.text(
                                    rx.cond(TrainingState.use_stopword_removal, "an", "aus"),
                                    size="2", color="var(--gray-11)",
                                ),
                                align_items="center",
                                spacing="2",
                            ),
                            align_items="start",
                            spacing="1",
                            min_width="220px",
                        ),
                        rx.text(
                            "Entfernt häufige Funktionswörter (Artikel, Präpositionen, Hilfsverben) "
                            "aus stopwords_de.txt vor der TF-IDF-Vektorisierung. "
                            "Ermöglicht Vergleich mit/ohne Stoppwörter.",
                            size="2", color="var(--gray-11)", flex="1",
                        ),
                        align_items="center",
                        spacing="6",
                        width="100%",
                    ),

                    rx.divider(),

                    # Zeile 4: Min. Wortlänge
                    rx.hstack(
                        rx.vstack(
                            rx.text("Min. Wortlänge", weight="bold", size="2"),
                            rx.slider(
                                default_value=[1],
                                value=[TrainingState.min_word_length],
                                min=1,
                                max=5,
                                step=1,
                                on_change=TrainingState.handle_min_word_length_change,
                                width="180px",
                            ),
                            rx.text(TrainingState.min_word_length_formatted, size="2", color="var(--gray-11)"),
                            align_items="start",
                            spacing="1",
                            min_width="220px",
                        ),
                        rx.text(
                            "Wörter kürzer als dieser Wert werden vor der Vektorisierung entfernt. "
                            "Wert 1 bedeutet: alle Wörter bleiben. "
                            "Ab 2 fallen Einzelbuchstaben weg, ab 3 auch zweistellige Abkürzungen.",
                            size="2", color="var(--gray-11)", flex="1",
                        ),
                        align_items="center",
                        spacing="6",
                        width="100%",
                    ),

                    rx.divider(),

                    # Zeile 5: Analyzer / N-Gramm
                    rx.hstack(
                        rx.vstack(
                            rx.text("Analyzer", weight="bold", size="2"),
                            rx.select(
                                ["char_wb", "word"],
                                value=TrainingState.analyzer_mode,
                                on_change=TrainingState.handle_analyzer_mode_change,
                            ),
                            rx.cond(
                                TrainingState.analyzer_mode == "word",
                                rx.hstack(
                                    rx.text("N-Gramm:", size="2", color="var(--gray-11)"),
                                    rx.select(
                                        ["1", "2"],
                                        value=TrainingState.word_ngram_max_str,
                                        on_change=TrainingState.handle_word_ngram_max_change,
                                        size="1",
                                    ),
                                    align_items="center",
                                    spacing="2",
                                ),
                                rx.fragment(),
                            ),
                            align_items="start",
                            spacing="2",
                            min_width="220px",
                        ),
                        rx.cond(
                            TrainingState.analyzer_mode == "word",
                            rx.text(
                                "Wort-Ebene: Features sind ganze Wörter statt Zeichenketten. "
                                "N-Gramm 1 = nur Einzelwörter; "
                                "N-Gramm 2 = Einzelwörter + Wortpaare (z.B. \"kleines Kind\" als Feature).",
                                size="2", color="var(--gray-11)", flex="1",
                            ),
                            rx.text(
                                "Zeichen-N-Gramme (char_wb): Features sind Zeichenfolgen innerhalb von Wörtern. "
                                "Robuster bei Tippfehlern und morphologischen Varianten (Flexion, Komposita).",
                                size="2", color="var(--gray-11)", flex="1",
                            ),
                        ),
                        align_items="center",
                        spacing="6",
                        width="100%",
                    ),

                    spacing="4",
                ),
                padding="1.5rem",
                width="100%"
            ),

            # Batch-Training
            rx.card(
                rx.vstack(
                    rx.heading("3. Batch-Training (optional)", size="5"),
                    rx.text(
                        "Trainiere mehrere Parameterkombinationen in einem Durchlauf. "
                        "Alle Kombinationen aus den gewählten Optionen werden kreuzweise trainiert.",
                        color="var(--gray-11)",
                        size="2",
                    ),

                    rx.hstack(
                        # Modelltypen
                        rx.vstack(
                            rx.text("Modell-Typen:", weight="bold", size="2"),
                            *[
                                rx.hstack(
                                    rx.checkbox(
                                        checked=TrainingState.batch_model_types.contains(mt),
                                        on_change=lambda v, m=mt: TrainingState.toggle_batch_model(m),
                                        size="1",
                                    ),
                                    rx.text(name, size="2"),
                                    align_items="center",
                                    spacing="2",
                                )
                                for mt, name in AVAILABLE_MODELS
                            ],
                            align_items="start",
                            spacing="1",
                        ),

                        # Stoppwörter
                        rx.vstack(
                            rx.text("Stoppwörter:", weight="bold", size="2"),
                            rx.hstack(
                                rx.checkbox(
                                    checked=TrainingState.batch_use_stopwords.contains(False),
                                    on_change=lambda v: TrainingState.toggle_batch_stopwords(False),
                                    size="1",
                                ),
                                rx.text("nicht entfernen", size="2"),
                                align_items="center",
                                spacing="2",
                            ),
                            rx.hstack(
                                rx.checkbox(
                                    checked=TrainingState.batch_use_stopwords.contains(True),
                                    on_change=lambda v: TrainingState.toggle_batch_stopwords(True),
                                    size="1",
                                ),
                                rx.text("entfernen", size="2"),
                                align_items="center",
                                spacing="2",
                            ),
                            align_items="start",
                            spacing="1",
                        ),

                        # Min. Wortlänge
                        rx.vstack(
                            rx.text("Min. Wortlänge:", weight="bold", size="2"),
                            *[
                                rx.hstack(
                                    rx.checkbox(
                                        checked=TrainingState.batch_min_lengths.contains(length),
                                        on_change=lambda v, l=length: TrainingState.toggle_batch_min_length(l),
                                        size="1",
                                    ),
                                    rx.text(f"≥ {length}", size="2"),
                                    align_items="center",
                                    spacing="2",
                                )
                                for length in [1, 2, 3]
                            ],
                            align_items="start",
                            spacing="1",
                        ),

                        # Analyzer
                        rx.vstack(
                            rx.text("Analyzer:", weight="bold", size="2"),
                            *[
                                rx.hstack(
                                    rx.checkbox(
                                        checked=TrainingState.batch_analyzers.contains(an),
                                        on_change=lambda v, a=an: TrainingState.toggle_batch_analyzer(a),
                                        size="1",
                                    ),
                                    rx.text(an, size="2"),
                                    align_items="center",
                                    spacing="2",
                                )
                                for an in ["char_wb", "word-(1,1)", "word-(1,2)"]
                            ],
                            align_items="start",
                            spacing="1",
                        ),

                        spacing="6",
                        align_items="start",
                        flex_wrap="wrap",
                    ),

                    rx.hstack(
                        rx.vstack(
                            rx.text(
                                TrainingState.batch_preview_label,
                                size="2",
                                color="var(--jade-11)",
                                weight="bold",
                            ),
                            rx.cond(
                                TrainingState.batch_estimated_time_str,
                                rx.text(
                                    TrainingState.batch_estimated_time_str,
                                    size="1",
                                    color="var(--gray-10)",
                                ),
                            ),
                            spacing="1",
                            align_items="start",
                        ),
                        rx.button(
                            "Batch Training starten",
                            on_click=TrainingState.start_batch_training,
                            disabled=~TrainingState.can_train | TrainingState.batch_is_running,
                            loading=TrainingState.batch_is_running,
                            color_scheme="jade",
                            variant="soft",
                        ),
                        align_items="center",
                        spacing="4",
                    ),

                    # Batch-Fortschritt
                    rx.cond(
                        TrainingState.batch_is_running,
                        rx.vstack(
                            rx.text(TrainingState.batch_progress_label, size="2"),
                            rx.progress(value=TrainingState.batch_progress_pct),
                            spacing="2",
                        ),
                    ),

                    # Batch-Fehler
                    rx.cond(
                        TrainingState.batch_errors,
                        rx.callout(
                            rx.vstack(
                                rx.text("Fehler bei einzelnen Konfigurationen:", weight="bold"),
                                rx.foreach(
                                    TrainingState.batch_errors,
                                    lambda e: rx.text(e, size="1"),
                                ),
                                spacing="1",
                            ),
                            icon="triangle_alert",
                            color_scheme="amber",
                        ),
                    ),

                    spacing="3",
                ),
                padding="1.5rem",
                width="100%",
            ),

            # Start Training
            rx.card(
                rx.vstack(
                    rx.heading("4. Einzeltraining starten", size="5"),

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
            rx.heading("ANALYSE", size="4", color="var(--jade-12)", weight="light"),
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

                    ag_grid(
                        id="models_grid",
                        row_data=AnalysisState.models_list,
                        column_defs=models_column_defs,
                        default_col_def={"flex": 1, "minWidth": 80},
                        resizable=True,
                        dom_layout="autoHeight",
                        height="None",
                        column_size="sizeToFit",
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


def shap_word_badge(word_data: dict) -> rx.Component:
    """Badge-Komponente für ein einzelnes Wort mit farbcodiertem SHAP-Einfluss.
    Die Farbe ('jade'/'red'/'gray') wird bereits im State berechnet.
    """
    return rx.badge(word_data["word"], color_scheme=word_data["color"], variant="soft")


def shap_card() -> rx.Component:
    """SHAP-Erklärungskarte, die nach einer Vorhersage angezeigt wird."""
    return rx.vstack(
        # Ladeindikator
        rx.cond(
            PredictionState.shap_is_computing,
            rx.callout(
                "SHAP-Erklärung wird berechnet...",
                icon="loader",
                color_scheme="gray",
            ),
        ),

        # MLP: manueller Trigger-Button
        rx.cond(
            PredictionState.shap_model_is_nn & ~PredictionState.shap_is_computing & ~PredictionState.has_shap_results,
            rx.button(
                "Erklärung anzeigen (Neural Network – dauert ~30–60 Sek.)",
                on_click=PredictionState.compute_shap_nn,
                color_scheme="amber",
                variant="soft",
            ),
        ),

        # Fehleranzeige
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

        # Hauptkarte mit Ergebnissen
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

                    # Lemma-Wörter
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

                    # Bedeutung-Wörter
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

                    # Balkendiagramm: Top 10 Wörter
                    rx.cond(
                        PredictionState.shap_top_words,
                        rx.vstack(
                            rx.text("Top-Wörter nach Einfluss:", weight="bold", size="2"),
                            rx.recharts.responsive_container(
                                rx.recharts.bar_chart(
                                    rx.recharts.bar(
                                        data_key="score",
                                        fill="#30a46c",  # Jade-Grün als Hex
                                    ),
                                    rx.recharts.x_axis(data_key="word"),
                                    rx.recharts.y_axis(),
                                    rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                                    rx.recharts.reference_line(y=0, stroke="#888"),
                                    data=PredictionState.shap_top_words,
                                ),
                                width="100%",
                                height=220,
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
    """Vorhersage-Seite"""
    return base_layout(
        rx.vstack(
            rx.heading("VORHERSAGE", size="4", color="var(--jade-12)", weight="light"),

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
                            placeholder="Lemma (z.B. 'Waggala')",
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
                        rx.callout(
                            rx.box(
                                rx.text(f"Sachgruppe: {PredictionState.prediction_result}", font_weight="bold", size="4"),
                                rx.text(f"Beschreibung: {PredictionState.prediction_result_description}", size="3"),
                                rx.cond(
                                    PredictionState.prediction_proba > 0,
                                    rx.text(f"Wahrscheinlichkeit: {PredictionState.prediction_proba_formatted}")
                                ),
                            ),
                            icon="sparkles",
                            color_scheme="jade"
                        )
                    ),

                    # SHAP-Erklärung
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
                                    ag_grid.column_def(field="sachgruppe", header_name="Sachgruppe", sortable=True, filter=True),
                                    ag_grid.column_def(field="beschreibung", header_name="Beschreibung", sortable=True, filter=True),
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

app.add_page(index, route="/", title="LT Sachgruppen-Vorhersage | Start")
app.add_page(training_page, route="/training", title="LT Sachgruppen-Vorhersage | Training")
app.add_page(analyse_page, route="/analyse", title="LT Sachgruppen-Vorhersage | Analyse")
app.add_page(vorhersage_page, route="/vorhersage", title="LT Sachgruppen-Vorhersage | Vorhersage")
