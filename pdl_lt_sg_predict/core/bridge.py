"""Brücke zur ML-Schicht + gemeinsame Ressourcen (Pfade, Modell-Cache, Mapping).

Die ML-Pipeline (``sachgruppen_classifier`` / ``shap_utils``) liegt im Projekt-Root
und wird hier per ``sys.path``-Eintrag importierbar gemacht. So bleiben bestehende
gepickelte Modelle (Modulname ``sachgruppen_classifier`` bzw. ``__main__``) ladbar.
"""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# ── Projekt-Root bestimmen und in den Importpfad legen ──────────────────────
# bridge.py -> core -> pdl_lt_sg_predict -> <repo-root>
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

# Pfad zum Classifier-Skript (dient auch als Trainings-Worker im Subprozess).
CLASSIFIER_SCRIPT = REPO_ROOT / "sachgruppen_classifier.py"


def _resolve_dir(env_value: str, default: Path) -> Path:
    if not env_value:
        return default
    p = Path(env_value).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


# ── Konfiguration (aus .env) ────────────────────────────────────────────────
MODELS_DIR = _resolve_dir(os.getenv("MODELS_DIR", ""), REPO_ROOT / "models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ENABLE_TRAINING = os.getenv("ENABLE_TRAINING", "True").strip().lower() in (
    "1", "true", "yes",
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Session-Verzeichnis für Uploads (Batch-CSV, Trainings-CSV, Fortschrittsdateien).
SESSIONS_DIR = _resolve_dir(os.getenv("SESSIONS_DIR", ""), REPO_ROOT / ".sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Modelltypen (Code -> Anzeigename) ───────────────────────────────────────
AVAILABLE_MODELS: list[tuple[str, str]] = [
    ("svm", "Linear SVM"),
    ("logistic", "Logistic Regression"),
    ("rf", "Random Forest"),
    ("nn", "Neural Network"),
    ("xgboost", "XGBoost"),
]
MODEL_DISPLAY_NAMES: dict[str, str] = dict(AVAILABLE_MODELS)


@lru_cache(maxsize=1)
def sachgruppen_map() -> dict[str, str]:
    """Nummer -> Sachgruppen-Bezeichnung (aus data/sachgruppen.csv)."""
    import pandas as pd

    csv_path = REPO_ROOT / "data" / "sachgruppen.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, sep=";", dtype=str)
    return dict(zip(df["Nummer"].str.strip(), df["Sachgruppe"].str.strip()))


def describe(label: str) -> str:
    """Sachgruppen-Bezeichnung zu einer Nummer (oder '(unbekannt)')."""
    return sachgruppen_map().get(str(label), "(unbekannt)")


# ── Modell-Cache (LRU, max. 2 — Modelle sind 100–330 MB groß) ───────────────
_MODEL_CACHE: OrderedDict[str, object] = OrderedDict()
_MODEL_CACHE_MAX = 2
_MODEL_CACHE_LOCK = threading.Lock()


def get_model(model_path: str | Path):
    """Ein gespeichertes Modell laden und im Prozess vorhalten (LRU).

    Der Lock verhindert, dass parallele Erst-Requests dasselbe Modell doppelt
    laden (Sync-Endpunkte laufen im FastAPI-Threadpool).
    """
    key = str(model_path)
    with _MODEL_CACHE_LOCK:
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
            return _MODEL_CACHE[key]

        from sachgruppen_classifier import SachgruppenClassifier

        try:
            model = SachgruppenClassifier.load(key)
        except FileNotFoundError:
            # Durchreichen: die API-Handler mappen das auf HTTP 404.
            raise
        except Exception as e:  # noqa: BLE001 — inkompatible/beschädigte Pickles
            name = Path(key).name
            raise RuntimeError(
                f"Modell '{name}' konnte nicht geladen werden "
                f"(evtl. inkompatible oder beschädigte Datei): {e}"
            ) from e
        _MODEL_CACHE[key] = model
        while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
            _MODEL_CACHE.popitem(last=False)
        return model
