"""
Training-Seite: TrainingState + training_page.
"""
import asyncio
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import reflex as rx
import pandas as pd

from .state import (
    BaseState,
    AVAILABLE_MODELS,
    MODELS_DIR,
    MAX_FILE_SIZE,
    _TIME_FALLBACKS,
    _TIME_FALLBACK_SAMPLES,
)
from .components import base_layout, ENABLE_TRAINING, BestModelState


class TrainingState(BaseState):
    """State for model training."""
    # File upload
    uploaded_filename: str = ""
    upload_error: str = ""
    is_uploading: bool = False

    # Training config
    selected_model: str = "svm"
    test_size: float = 0.2
    use_stopword_removal: bool = False
    min_word_length: int = 1
    analyzer_mode: str = "char_wb"   # "char_wb" or "word"
    word_ngram_max: int = 1          # 1=(1,1), 2=(1,2) — word analyzer only

    # Hyperparameter tuning config
    tune_mode: str = "standard"      # "standard", "auto", "manual"
    tune_n_iter_str: str = "20"
    tune_cv_str: str = "3"
    svm_c_str: str = "1.0"
    xgb_n_estimators_str: str = "300"
    xgb_max_depth_str: str = "6"
    xgb_learning_rate_str: str = "0.05"
    xgb_subsample_str: str = "0.8"
    nn_hidden_layers_str: str = "200,100,50"
    nn_alpha_str: str = "0.0001"
    nn_learning_rate_init_str: str = "0.001"

    # Batch training config
    batch_model_types: list[str] = ["svm"]
    batch_use_stopwords: list[bool] = [False]   # [False], [True], or [False, True]
    batch_min_lengths: list[int] = [1]
    batch_analyzers: list[str] = ["char_wb"]

    # Historical training times: model_type → seconds (scaled to current sample count)
    time_per_type: dict[str, float] = {}

    # Batch training status
    batch_is_running: bool = False
    batch_total: int = 0
    batch_done: int = 0
    batch_current_config: str = ""
    batch_errors: list[str] = []

    # Training status
    is_training: bool = False
    training_progress: str = ""
    training_progress_pct: int = 0
    training_error: str = ""

    # Training results
    accuracy: float = 0.0
    training_time: float = 0.0
    saved_model_path: str = ""
    best_params_str: str = ""    # Best params after auto-tune (empty for standard/manual)
    best_cv_score: float = 0.0

    # Data info
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
        """Format test_size as percentage string."""
        return f"{self.test_size * 100:.0f}%"

    @rx.var
    def min_word_length_formatted(self) -> str:
        return f"≥ {self.min_word_length} Zeichen"

    @rx.var
    def word_ngram_max_str(self) -> str:
        return str(self.word_ngram_max)

    @rx.var
    def batch_preview_count(self) -> int:
        """Number of models to be trained in batch mode."""
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
        """Estimated total batch training duration based on historical measurements."""
        if not self.batch_model_types:
            return ""
        # Number of configurations per model type (all preprocessing combinations)
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
                # Fallback: scale to current sample count
                fallback = _TIME_FALLBACKS.get(mt, 120.0)
                est = fallback / _TIME_FALLBACK_SAMPLES * n_current
            total_secs += est * configs_per_type
        if total_secs <= 0:
            return ""
        t = int(total_secs)
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        source = "measured" if self.time_per_type else "estimated"
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
        """Handle CSV upload."""
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
                yield BestModelState.set_csv_info(safe_filename, self.total_samples, self.num_classes)

            except Exception as e:
                self.upload_error = f"Fehler beim Lesen der CSV: {str(e)}"
                return

        except Exception as e:
            self.upload_error = f"Upload-Fehler: {str(e)}"
        finally:
            self.is_uploading = False

    def handle_test_size_change(self, value: list[float]):
        """Handle slider change (expects a list)."""
        if value:
            self.test_size = value[0]

    def set_use_stopword_removal(self, value: bool):
        self.use_stopword_removal = value

    def _refresh_time_estimates(self):
        """
        Read historical training times from metadata files and scale to current dataset size.
        Called after every CSV upload. Most recent entry per model type is used.
        """
        models_dir = MODELS_DIR
        if not models_dir.exists() or self.total_samples == 0:
            return
        # Find most recent measurement per type (highest timestamp wins)
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

    def handle_tune_mode_change(self, label: str):
        mapping = {
            "Standard-Werte": "standard",
            "Auto Tune": "auto",
            "Parameter definieren": "manual",
        }
        self.tune_mode = mapping.get(label, "standard")

    @rx.var
    def tune_mode_label(self) -> str:
        mapping = {
            "standard": "Standard-Werte",
            "auto": "Auto Tune",
            "manual": "Parameter definieren",
        }
        return mapping.get(self.tune_mode, "Standard-Werte")

    def set_tune_n_iter_str(self, v: str): self.tune_n_iter_str = v
    def set_tune_cv_str(self, v: str): self.tune_cv_str = v

    def set_svm_c_str(self, v: str): self.svm_c_str = v
    def set_xgb_n_estimators_str(self, v: str): self.xgb_n_estimators_str = v
    def set_xgb_max_depth_str(self, v: str): self.xgb_max_depth_str = v
    def set_xgb_learning_rate_str(self, v: str): self.xgb_learning_rate_str = v
    def set_xgb_subsample_str(self, v: str): self.xgb_subsample_str = v
    def set_nn_hidden_layers_str(self, v: str): self.nn_hidden_layers_str = v
    def set_nn_alpha_str(self, v: str): self.nn_alpha_str = v
    def set_nn_learning_rate_init_str(self, v: str): self.nn_learning_rate_init_str = v

    # ---- Batch configuration ----

    def toggle_batch_model(self, model_type: str):
        """Add or remove a model type from the batch list."""
        if model_type in self.batch_model_types:
            if len(self.batch_model_types) > 1:
                self.batch_model_types = [m for m in self.batch_model_types if m != model_type]
        else:
            self.batch_model_types = self.batch_model_types + [model_type]

    def toggle_batch_stopwords(self, value: bool):
        """Toggle a True/False value in batch_use_stopwords."""
        if value in self.batch_use_stopwords:
            if len(self.batch_use_stopwords) > 1:
                self.batch_use_stopwords = [v for v in self.batch_use_stopwords if v != value]
        else:
            self.batch_use_stopwords = self.batch_use_stopwords + [value]

    def toggle_batch_min_length(self, length: int):
        """Add or remove a minimum word length from the batch list."""
        if length in self.batch_min_lengths:
            if len(self.batch_min_lengths) > 1:
                self.batch_min_lengths = [l for l in self.batch_min_lengths if l != length]
        else:
            self.batch_min_lengths = sorted(self.batch_min_lengths + [length])

    def toggle_batch_analyzer(self, analyzer: str):
        """Add or remove an analyzer from the batch list."""
        if analyzer in self.batch_analyzers:
            if len(self.batch_analyzers) > 1:
                self.batch_analyzers = [a for a in self.batch_analyzers if a != analyzer]
        else:
            self.batch_analyzers = self.batch_analyzers + [analyzer]

    def handle_model_selection(self, display_name: str):
        """Convert display name to model type."""
        for model_type, name in AVAILABLE_MODELS:
            if name == display_name:
                self.selected_model = model_type
                return

    @rx.var
    def selected_model_display(self) -> str:
        """Return display name for selected_model."""
        for model_type, name in AVAILABLE_MODELS:
            if model_type == self.selected_model:
                return name
        return "Linear SVM (schnell, gut)"

    async def start_training(self):
        """Start model training as a separate subprocess (survives worker restarts)."""
        self.is_training = True
        self.training_error = ""
        self.training_progress = "Starte Training…"
        self.training_progress_pct = 0
        self.accuracy = 0.0
        self.training_time = 0.0
        yield

        try:
            session_path = self._create_session_dir()
            csv_path = session_path / self.uploaded_filename

            if not csv_path.exists():
                self.training_error = "CSV-Datei nicht gefunden"
                return

            # Read parameters from state
            word_ngram_max = self.word_ngram_max if self.analyzer_mode == "word" else 1
            _tune = self.tune_mode == "auto"
            _tune_n_iter = max(1, int(self.tune_n_iter_str))
            _tune_cv = max(2, int(self.tune_cv_str))
            _estimated_train_sec = self.time_per_type.get(
                self.selected_model, _TIME_FALLBACKS.get(self.selected_model, 120.0)
            )
            # Auto-tune takes significantly longer
            if _tune:
                _estimated_train_sec = _estimated_train_sec * max(_tune_n_iter * _tune_cv / 5, 1.0)
            _estimated_fit_sec = max(_estimated_train_sec * 0.70, 1.0)

            # Progress file: subprocess writes progress, we read it
            progress_file = session_path / "training_progress.json"
            progress_file.write_text('{"pct": 0, "msg": "Starte…", "done": false, "error": ""}')

            # Path to the classifier script (project root, one level above this module)
            classifier_script = Path(__file__).parent.parent / "sachgruppen_classifier.py"

            cmd = [
                sys.executable, str(classifier_script),
                "--csv", str(csv_path),
                "--model", self.selected_model,
                "--test-size", str(self.test_size),
                "--analyzer", self.analyzer_mode,
                "--word-ngram-max", str(word_ngram_max),
                "--min-length", str(self.min_word_length),
                "--stopwords", "true" if self.use_stopword_removal else "false",
                "--svm-c", self.svm_c_str,
                "--xgb-n-estimators", self.xgb_n_estimators_str,
                "--xgb-max-depth", self.xgb_max_depth_str,
                "--xgb-learning-rate", self.xgb_learning_rate_str,
                "--xgb-subsample", self.xgb_subsample_str,
                "--nn-hidden-layers", self.nn_hidden_layers_str,
                "--nn-alpha", self.nn_alpha_str,
                "--nn-learning-rate-init", self.nn_learning_rate_init_str,
                "--output-dir", str(MODELS_DIR),
                "--progress-file", str(progress_file),
            ]
            if _tune:
                cmd += ["--tune",
                        "--tune-n-iter", str(_tune_n_iter),
                        "--tune-cv", str(_tune_cv)]

            start_time = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)

            # Poll progress and forward to UI
            fit_started_at = None
            while proc.poll() is None:
                await asyncio.sleep(0.4)
                try:
                    prog = json.loads(progress_file.read_text())
                except (OSError, json.JSONDecodeError):
                    prog = {}

                phase_pct = prog.get("pct", 0)
                phase_msg = prog.get("msg", "…")

                if phase_pct > 35:
                    # Real progress (e.g. XGBoost round callback): use directly
                    self.training_progress_pct = phase_pct
                    self.training_progress = phase_msg
                    fit_started_at = None  # reset timer
                elif phase_pct == 35:
                    # Training running but no real-time progress (SVM/LR/RF/NN/Tune).
                    # Sqrt-compressed time estimate: bar always moves, never freezes.
                    if fit_started_at is None:
                        fit_started_at = time.time()
                    elapsed = time.time() - fit_started_at
                    ratio = min(elapsed / _estimated_fit_sec, 0.99)
                    sqrt_pct = int(50 * math.sqrt(ratio))
                    self.training_progress_pct = 35 + sqrt_pct
                    self.training_progress = phase_msg
                else:
                    self.training_progress_pct = phase_pct
                    self.training_progress = phase_msg
                yield

            # Subprocess finished – read result
            if proc.returncode != 0:
                stdout = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"Subprocess exit {proc.returncode}:\n{stdout}")

            try:
                prog = json.loads(progress_file.read_text())
            except (OSError, json.JSONDecodeError):
                raise RuntimeError("Progress file not readable after training.")

            if prog.get("error"):
                raise RuntimeError(prog["error"])

            model_file = prog.get("model_file", "")
            accuracy = prog.get("accuracy", 0.0)
            training_time = prog.get("training_time", time.time() - start_time)

            self.training_time = training_time
            self.accuracy = accuracy
            self.saved_model_path = Path(model_file).name if model_file else ""

            # Read best params from metadata JSON
            meta_path = model_file.replace(".pkl", "_metadata.json") if model_file else ""
            self.best_params_str = ""
            self.best_cv_score = 0.0
            if meta_path and Path(meta_path).exists():
                try:
                    meta = json.loads(Path(meta_path).read_text())
                    best_params = meta.get("best_params", {})
                    self.best_cv_score = meta.get("best_cv_score", 0.0)
                    if best_params:
                        def _shorten(k: str) -> str:
                            parts = k.split("__")
                            return " ".join(parts[1:]) if len(parts) > 1 else k
                        self.best_params_str = "  ·  ".join(
                            f"{_shorten(k)} = {v}" for k, v in sorted(best_params.items())
                        )
                except (OSError, json.JSONDecodeError):
                    pass

            self.training_progress = "✓ Training abgeschlossen!"
            self.training_progress_pct = 100
            yield

        except Exception as e:
            self.training_error = f"Training-Fehler: {str(e)}"
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False

    async def start_batch_training(self):
        """Train all combinations from batch_* config lists (as subprocess)."""
        if not self.uploaded_filename or self.is_training or self.batch_is_running:
            return

        session_path = self._create_session_dir()
        csv_path = session_path / self.uploaded_filename
        if not csv_path.exists():
            self.training_error = "CSV-Datei nicht gefunden"
            return

        # Pre-compute configuration count for the UI
        sw_vals = self.batch_use_stopwords
        total = (len(self.batch_model_types) * len(sw_vals)
                 * len(self.batch_min_lengths) * len(self.batch_analyzers))

        self.batch_total = total
        self.batch_done = 0
        self.batch_is_running = True
        self.batch_errors = []
        self.training_error = ""
        yield

        try:
            _tune = self.tune_mode == "auto"
            _tune_n_iter = max(1, int(self.tune_n_iter_str))
            _tune_cv = max(2, int(self.tune_cv_str))

            progress_file = session_path / "batch_progress.json"
            progress_file.write_text('{"pct": 0, "msg": "Starte…", "done": false, '
                                     '"config_idx": 0, "config_total": 0, "error": ""}')

            classifier_script = Path(__file__).parent.parent / "sachgruppen_classifier.py"

            # Normalize analyzer names (web-internal → CLI names)
            analyzers_clean = list({
                "word" if a.startswith("word") else "char_wb"
                for a in self.batch_analyzers
            })
            # word_ngram_max: derive maximum from all batch_analyzers
            word_ngram_max = max(
                (2 if a == "word-(1,2)" else 1) for a in self.batch_analyzers
            )

            cmd = [
                sys.executable, str(classifier_script),
                "--csv", str(csv_path),
                "--model", *self.batch_model_types,
                "--analyzer", *analyzers_clean,
                "--min-length", *[str(m) for m in self.batch_min_lengths],
                "--stopwords", *["true" if s else "false" for s in sw_vals],
                "--word-ngram-max", str(word_ngram_max),
                "--test-size", str(self.test_size),
                "--svm-c", self.svm_c_str,
                "--xgb-n-estimators", self.xgb_n_estimators_str,
                "--xgb-max-depth", self.xgb_max_depth_str,
                "--xgb-learning-rate", self.xgb_learning_rate_str,
                "--xgb-subsample", self.xgb_subsample_str,
                "--nn-hidden-layers", self.nn_hidden_layers_str,
                "--nn-alpha", self.nn_alpha_str,
                "--nn-learning-rate-init", self.nn_learning_rate_init_str,
                "--output-dir", str(MODELS_DIR),
                "--progress-file", str(progress_file),
            ]
            if _tune:
                cmd += ["--tune",
                        "--tune-n-iter", str(_tune_n_iter),
                        "--tune-cv", str(_tune_cv)]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)

            while proc.poll() is None:
                await asyncio.sleep(0.5)
                try:
                    prog = json.loads(progress_file.read_text())
                except (OSError, json.JSONDecodeError):
                    prog = {}

                self.batch_done = prog.get("config_idx", self.batch_done)
                self.batch_current_config = prog.get("msg", "…")
                yield

            if proc.returncode != 0:
                stdout = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"Subprocess exit {proc.returncode}:\n{stdout}")

            try:
                prog = json.loads(progress_file.read_text())
            except (OSError, json.JSONDecodeError):
                prog = {}

            if prog.get("error"):
                self.batch_errors = [prog["error"]]

            self.batch_done = total

        except Exception as e:
            self.batch_errors = self.batch_errors + [str(e)]
            import traceback
            traceback.print_exc()
        finally:
            self.batch_is_running = False
            self.batch_current_config = ""


def training_page() -> rx.Component:
    """Training page."""
    return base_layout(
        rx.vstack(
            rx.heading("TRAINING", size="4", color="var(--jade-12)", weight="light"),

            # Notice when training is disabled
            *([rx.callout(
                rx.vstack(
                    rx.text("Training ist deaktiviert.", font_weight="bold"),
                    rx.text(
                        "Um Training zu aktivieren, setzen Sie ENABLE_TRAINING=True in der .env-Datei "
                        "und starten Sie die Anwendung neu.",
                        size="2",
                    ),
                    spacing="1",
                ),
                icon="octagon_alert",
                color_scheme="amber",
                width="100%",
            )] if not ENABLE_TRAINING else []),

            # Data upload
            rx.card(
                rx.vstack(
                    rx.heading("Daten hochladen", size="5"),
                    rx.text("CSV mit Spalten: lemma, bedeutung, sachgruppe", color="var(--gray-11)"),

                    rx.upload(
                        rx.button(
                            "CSV-Datei auswählen",
                            loading=TrainingState.is_uploading,
                            disabled=not ENABLE_TRAINING,
                        ),
                        id="csv_upload",
                        accept={".csv": ["text/csv"]},
                        max_files=1,
                        on_drop=TrainingState.handle_csv_upload,
                        disabled=not ENABLE_TRAINING,
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

            # Training config
            rx.card(
                rx.vstack(
                    rx.heading("Einzeltraining", size="5"),

                    # Each option on its own row: control on the left, description on the right
                    # Row 1: model type
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

                    # Row 2: test fraction
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

                    # Row 3: stopwords
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

                    # Row 4: min. word length
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

                    # Row 5: analyzer / n-gram
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

                    rx.button(
                        "Modell trainieren",
                        on_click=TrainingState.start_training,
                        disabled=not ENABLE_TRAINING or ~TrainingState.can_train,
                        loading=TrainingState.is_training,
                        size="3",
                        color_scheme="jade"
                    ),

                    rx.cond(
                        TrainingState.is_training,
                        rx.vstack(
                            rx.hstack(
                                rx.text(TrainingState.training_progress, size="2"),
                                rx.spacer(),
                                rx.text(
                                    TrainingState.training_progress_pct.to_string() + "%",
                                    size="2",
                                    color="var(--gray-11)",
                                ),
                                width="100%",
                            ),
                            rx.progress(value=TrainingState.training_progress_pct),
                            spacing="2",
                            width="100%",
                        )
                    ),

                    rx.cond(
                        TrainingState.accuracy > 0,
                        rx.callout(
                            rx.vstack(
                                rx.heading(f"Accuracy: {TrainingState.accuracy:.4f}", size="4"),
                                rx.text(f"Trainingszeit: {TrainingState.training_time:.1f}s"),
                                rx.text(f"Modell gespeichert: {TrainingState.saved_model_path}"),
                                rx.cond(
                                    TrainingState.best_params_str != "",
                                    rx.vstack(
                                        rx.divider(),
                                        rx.text(
                                            f"Beste Parameter (Auto Tune, CV-Score: {TrainingState.best_cv_score:.4f}):",
                                            size="2", weight="bold",
                                        ),
                                        rx.text(
                                            TrainingState.best_params_str,
                                            size="2", font_family="monospace",
                                        ),
                                        spacing="1",
                                        width="100%",
                                    ),
                                ),
                                spacing="1",
                            ),
                            icon="check-check",
                            color_scheme="jade",
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

                    spacing="4",
                ),
                padding="1.5rem",
                width="100%"
            ),

            # Hyperparameter tuning
            rx.card(
                rx.vstack(
                    rx.heading("Hyperparameter-Tuning", size="5"),
                    rx.text(
                        "Legt fest, welche Modellparameter beim Training verwendet werden (Einzel- und Batchtraining).",
                        color="var(--gray-11)", size="2",
                    ),

                    rx.radio_group(
                        ["Standard-Werte", "Auto Tune", "Parameter definieren"],
                        value=TrainingState.tune_mode_label,
                        on_change=TrainingState.handle_tune_mode_change,
                        direction="column",
                        gap="2",
                    ),

                    # Standard mode: note on current defaults
                    rx.cond(
                        TrainingState.tune_mode == "standard",
                        rx.callout(
                            rx.vstack(
                                rx.text("Verwendete Standardparameter:", weight="bold", size="2"),
                                rx.cond(
                                    TrainingState.selected_model == "svm",
                                    rx.text(
                                        "SVM: C=1.0 · max_iter=5000 · dual=False · class_weight=balanced",
                                        size="2", font_family="monospace",
                                    ),
                                    rx.cond(
                                        TrainingState.selected_model == "xgboost",
                                        rx.text(
                                            "XGBoost: n_estimators=300 · max_depth=6 · learning_rate=0.05 "
                                            "· subsample=0.8 · colsample_bytree=0.8",
                                            size="2", font_family="monospace",
                                        ),
                                        rx.text(
                                            "Die Standardparameter des gewählten Modells werden verwendet.",
                                            size="2", color="var(--gray-11)",
                                        ),
                                    ),
                                ),
                                spacing="1",
                            ),
                            icon="info",
                            color_scheme="gray",
                            width="100%",
                        ),
                    ),

                    # Auto tune: configuration + warning
                    rx.cond(
                        TrainingState.tune_mode == "auto",
                        rx.vstack(
                            rx.hstack(
                                rx.vstack(
                                    rx.text("Kombinationen (n_iter)", weight="bold", size="2"),
                                    rx.input(
                                        value=TrainingState.tune_n_iter_str,
                                        on_change=TrainingState.set_tune_n_iter_str,
                                        type="number",
                                        min="1",
                                        width="120px",
                                    ),
                                    rx.text(
                                        "Wie viele zufällige Parameterkombinationen getestet werden. "
                                        "Weniger = schneller, mehr = gründlicher. Standard: 20",
                                        size="1", color="var(--gray-11)",
                                    ),
                                    spacing="1",
                                ),
                                rx.vstack(
                                    rx.text("CV-Folds", weight="bold", size="2"),
                                    rx.input(
                                        value=TrainingState.tune_cv_str,
                                        on_change=TrainingState.set_tune_cv_str,
                                        type="number",
                                        min="2",
                                        width="120px",
                                    ),
                                    rx.text(
                                        "Jede Kombination wird k-fach kreuzvalidiert. "
                                        "Mindestens 2. Standard: 3",
                                        size="1", color="var(--gray-11)",
                                    ),
                                    spacing="1",
                                ),
                                spacing="6",
                                align="start",
                            ),
                            rx.callout(
                                rx.text(
                                    "Das Training dauert ca. n_iter × cv × Einzeltraining. "
                                    "Die besten gefundenen Parameter werden automatisch übernommen.",
                                    size="2",
                                ),
                                icon="timer",
                                color_scheme="amber",
                                width="100%",
                            ),
                            spacing="3",
                            width="100%",
                        ),
                    ),

                    # Manual: input fields for SVM and XGBoost
                    rx.cond(
                        TrainingState.tune_mode == "manual",
                        rx.vstack(
                            rx.callout(
                                rx.text(
                                    "SVM, XGBoost und neurale Netze erzielen in der Regel die höchste Accuracy und bieten "
                                    "daher erweiterte Tuning-Optionen. Für andere Modelle werden Standardwerte verwendet.",
                                    size="2",
                                ),
                                icon="info",
                                color_scheme="blue",
                                width="100%",
                            ),

                            # SVM parameters
                            rx.box(
                                rx.vstack(
                                    rx.text("Linear SVM", weight="bold", size="3"),
                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("C (Regularisierung)", weight="bold", size="2"),
                                            rx.select(
                                                ["0.01", "0.1", "0.5", "1.0", "5.0", "10.0", "50.0", "100.0"],
                                                value=TrainingState.svm_c_str,
                                                on_change=TrainingState.set_svm_c_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Trade-off zwischen Margin-Maximierung und Fehlklassifikationen. "
                                            "Kleinere Werte = stärkere Regularisierung, robuster bei Rauschen. "
                                            "Größere Werte = engere Anpassung an die Trainingsdaten. Standard: 1.0",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),
                                    spacing="3",
                                ),
                                padding="1rem",
                                border="1px solid var(--gray-6)",
                                border_radius="var(--radius-2)",
                                width="100%",
                            ),

                            # XGBoost parameters
                            rx.box(
                                rx.vstack(
                                    rx.text("XGBoost", weight="bold", size="3"),

                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("Anzahl Bäume (n_estimators)", weight="bold", size="2"),
                                            rx.select(
                                                ["100", "200", "300", "500", "800"],
                                                value=TrainingState.xgb_n_estimators_str,
                                                on_change=TrainingState.set_xgb_n_estimators_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Anzahl der Entscheidungsbäume im Ensemble. Mehr Bäume verbessern oft "
                                            "die Accuracy, erhöhen aber die Trainingszeit linear. Standard: 300",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),

                                    rx.divider(),

                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("Baumtiefe (max_depth)", weight="bold", size="2"),
                                            rx.select(
                                                ["3", "4", "5", "6", "7", "8", "10"],
                                                value=TrainingState.xgb_max_depth_str,
                                                on_change=TrainingState.set_xgb_max_depth_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Maximale Tiefe jedes Baums. Tiefere Bäume erfassen komplexere Muster, "
                                            "neigen aber zu Overfitting. Flache Bäume (3–4) regularisieren stärker. Standard: 6",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),

                                    rx.divider(),

                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("Lernrate (learning_rate)", weight="bold", size="2"),
                                            rx.select(
                                                ["0.01", "0.03", "0.05", "0.1", "0.15", "0.2"],
                                                value=TrainingState.xgb_learning_rate_str,
                                                on_change=TrainingState.set_xgb_learning_rate_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Schrittgröße beim Hinzufügen jedes Baums. Kleinere Werte generalisieren "
                                            "besser, benötigen aber mehr Bäume. Kombiniere niedrige Lernrate mit "
                                            "hohen n_estimators. Standard: 0.05",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),

                                    rx.divider(),

                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("Zeilen-Sampling (subsample)", weight="bold", size="2"),
                                            rx.select(
                                                ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                                                value=TrainingState.xgb_subsample_str,
                                                on_change=TrainingState.set_xgb_subsample_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Anteil der Trainingsdaten, der pro Baum zufällig gezogen wird. "
                                            "Werte < 1.0 reduzieren Overfitting und beschleunigen das Training. Standard: 0.8",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),

                                    spacing="3",
                                ),
                                padding="1rem",
                                border="1px solid var(--gray-6)",
                                border_radius="var(--radius-2)",
                                width="100%",
                            ),

                            # Neural Network parameters
                            rx.box(
                                rx.vstack(
                                    rx.text("Neural Network (MLP)", weight="bold", size="3"),

                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("Hidden Layers", weight="bold", size="2"),
                                            rx.select(
                                                ["100", "200,100", "200,100,50", "300,150,75", "400,200,100,50"],
                                                value=TrainingState.nn_hidden_layers_str,
                                                on_change=TrainingState.set_nn_hidden_layers_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Größe und Anzahl der versteckten Schichten. Mehr/größere Schichten "
                                            "erhöhen die Modellkapazität, benötigen aber mehr Trainingszeit und "
                                            "neigen eher zu Overfitting. Standard: 200,100,50",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),

                                    rx.divider(),

                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("L2-Regularisierung (alpha)", weight="bold", size="2"),
                                            rx.select(
                                                ["0.00001", "0.0001", "0.001", "0.01", "0.1"],
                                                value=TrainingState.nn_alpha_str,
                                                on_change=TrainingState.set_nn_alpha_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Stärke der L2-Regularisierung der Gewichte. Größere Werte "
                                            "reduzieren Overfitting, können aber die Modellkapazität einschränken. "
                                            "Standard: 0.0001",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),

                                    rx.divider(),

                                    rx.hstack(
                                        rx.vstack(
                                            rx.text("Lernrate (learning_rate_init)", weight="bold", size="2"),
                                            rx.select(
                                                ["0.0001", "0.0005", "0.001", "0.005", "0.01"],
                                                value=TrainingState.nn_learning_rate_init_str,
                                                on_change=TrainingState.set_nn_learning_rate_init_str,
                                            ),
                                            spacing="1",
                                            min_width="220px",
                                        ),
                                        rx.text(
                                            "Initiale Lernrate des Adam-Optimierers. Kleinere Werte konvergieren "
                                            "stabiler aber langsamer; größere Werte können instabil werden. "
                                            "Standard: 0.001",
                                            size="2", color="var(--gray-11)", flex="1",
                                        ),
                                        align_items="center",
                                        spacing="6",
                                        width="100%",
                                    ),

                                    spacing="3",
                                ),
                                padding="1rem",
                                border="1px solid var(--gray-6)",
                                border_radius="var(--radius-2)",
                                width="100%",
                            ),

                            spacing="4",
                            width="100%",
                        ),
                    ),

                    spacing="4",
                ),
                padding="1.5rem",
                width="100%",
            ),

            # Batch training
            rx.card(
                rx.vstack(
                    rx.heading("Batch-Training", size="5"),
                    rx.text(
                        "Trainiere mehrere Parameterkombinationen in einem Durchlauf. "
                        "Alle Kombinationen aus den gewählten Optionen werden kreuzweise trainiert.",
                        color="var(--gray-11)",
                        size="2",
                    ),

                    rx.hstack(
                        # Model types
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

                        # Stopwords
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

                        # Min. word length
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
                        align_items="start",
                        spacing="4",
                    ),
                    rx.button(
                        "Batch-Training starten",
                        on_click=TrainingState.start_batch_training,
                        disabled=not ENABLE_TRAINING or ~TrainingState.can_train | TrainingState.batch_is_running | (TrainingState.tune_mode == "auto"),
                        loading=TrainingState.batch_is_running,
                        color_scheme="jade",
                        variant="soft",
                        size="3",
                    ),

                    rx.cond(
                        TrainingState.tune_mode == "auto",
                        rx.callout(
                            rx.text(
                                "Batch-Training ist bei aktiviertem Auto Tune deaktiviert. "
                                "Auto Tune würde jede Konfiguration einzeln tunen und damit "
                                "die Gesamtdauer vervielfachen. Bitte 'Standard-Werte' oder "
                                "'Parameter definieren' wählen.",
                                size="2",
                            ),
                            icon="octagon_alert",
                            color_scheme="amber",
                            width="100%",
                        ),
                    ),

                    # Batch progress
                    rx.cond(
                        TrainingState.batch_is_running,
                        rx.vstack(
                            rx.text(TrainingState.batch_progress_label, size="2"),
                            rx.progress(value=TrainingState.batch_progress_pct),
                            spacing="2",
                        ),
                    ),

                    # Batch errors
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

            spacing="4",
            width="100%",
            max_width="100%"
        )
    )
