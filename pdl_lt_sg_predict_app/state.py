"""
Shared state definitions, constants and helper functions.
All other app modules import from here.
"""
import os
import re
import json
import sys
import reflex as rx
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path so sachgruppen_classifier is importable.
# __file__ → pdl_lt_sg_predict_app/state.py → parent.parent → pdl-lt-sg-predict/
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============ Directories ============

_models_dir_env = os.getenv("MODELS_DIR", "")
MODELS_DIR = Path(_models_dir_env) if _models_dir_env else Path.home() / ".pdl-sg-predict" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============ Constants ============

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

AVAILABLE_MODELS = [
    ("svm", "Linear SVM"),
    ("logistic", "Logistic Regression"),
    ("rf", "Random Forest"),
    ("nn", "Neural Network"),
    ("xgboost", "XGBoost"),
]

# Load Sachgruppen mapping (number → description)
_SACHGRUPPEN_CSV = Path(__file__).parent.parent / "sachgruppen.csv"
SACHGRUPPEN_MAP: dict[str, str] = {}
if _SACHGRUPPEN_CSV.exists():
    _sg_df = pd.read_csv(_SACHGRUPPEN_CSV, sep=";", dtype=str)
    SACHGRUPPEN_MAP = dict(zip(_sg_df["Nummer"], _sg_df["Sachgruppe"]))

MODEL_DISPLAY_NAMES = {
    "svm": "Linear SVM",
    "logistic": "Logistic Regression",
    "rf": "Random Forest",
    "nn": "Neural Network",
    "xgboost": "XGBoost",
}

# Fallback training times (seconds) for ~113 127 samples, measured on dev machine.
# Overridden by historical metadata once a model of the given type has been trained.
_TIME_FALLBACKS: dict[str, float] = {
    "svm": 120.0,
    "logistic": 4286.0,
    "rf": 30.0,
    "nn": 111.0,
    "xgboost": 6112.0,
}
_TIME_FALLBACK_SAMPLES = 113_127


# ============ Helpers ============

def _next_model_filename(model_type: str) -> str:
    """Return the next available filename, e.g. model_svm_003.pkl."""
    pattern = re.compile(rf"model_{re.escape(model_type)}_(\d+)\.pkl", re.IGNORECASE)
    max_n = 0
    for f in MODELS_DIR.glob(f"model_{model_type}_*.pkl"):
        m = pattern.match(f.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"model_{model_type}_{max_n + 1:03d}.pkl"


# ============ SHAP Helpers ============

def _shap_score_to_dict(word: str, score: float) -> dict:
    """Build a serializable dict with pre-computed badge color and bar height."""
    if score > 0.1:
        color, fill = "jade", "#30a46c"
    elif score < -0.1:
        color, fill = "red", "#e5484d"
    else:
        color, fill = "gray", "#8d8d8d"
    bar_height = f"{int(abs(score) * 100)}px"
    return {
        "word": word,
        "score": round(score, 4),
        "color": color,
        "fill": fill,
        "bar_height": bar_height,
        "is_positive": score >= 0,
    }


def _shap_pairs_to_dicts(pairs: list) -> list[dict]:
    return [_shap_score_to_dict(w, s) for w, s in pairs]


# ============ Model Cache ============

_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_path: str):
    """Load a model once and keep it in memory."""
    if model_path not in _MODEL_CACHE:
        from sachgruppen_classifier import SachgruppenClassifier
        _MODEL_CACHE[model_path] = SachgruppenClassifier.load(model_path)
    return _MODEL_CACHE[model_path]


# ============ BaseState ============

class BaseState(rx.State):
    """Base state with shared functionality."""
    session_dir: str = ""

    def _create_session_dir(self):
        """Create session directory."""
        import tempfile
        if not self.session_dir:
            session_id = self.router.session.client_token
            self.session_dir = str(Path(tempfile.gettempdir()) / f"ml_session_{session_id}")
        session_path = Path(self.session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path
