"""Auflistung und Metadaten trainierter Modelle."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .bridge import MODELS_DIR, MODEL_DISPLAY_NAMES


def _fmt_training_time(secs: float | int) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_date(raw: str) -> str:
    # Metadaten nutzen entweder "%Y%m%d_%H%M%S" oder "%Y-%m-%d %H:%M:%S".
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).strftime("%d.%m.%y")
        except (ValueError, TypeError):
            continue
    return raw or ""


def _row_from_metadata(pkl_file: Path, meta: dict) -> dict:
    model_name = meta.get("model_name") or MODEL_DISPLAY_NAMES.get(
        meta.get("model_type", ""), meta.get("model_type", "")
    )

    min_len = meta.get("min_word_length", 1)
    analyzer_raw = meta.get("analyzer", "char_wb")
    ngram_max = meta.get("word_ngram_max", 1)
    analyzer_str = f"word-(1,{ngram_max})" if analyzer_raw == "word" else "char_wb"

    test_size = meta.get("test_size")
    report_file = MODELS_DIR / pkl_file.name.replace(".pkl", "_report.txt")

    return {
        "model_file": pkl_file.name,
        "model_name": model_name,
        "model_type": meta.get("model_type", ""),
        "accuracy": round(float(meta.get("accuracy", 0.0)), 4),
        "training_time": _fmt_training_time(meta["training_time"]) if "training_time" in meta else "",
        "date": _fmt_date(meta.get("timestamp", "")),
        "num_samples": int(meta.get("num_samples", 0)),
        "num_classes": int(meta.get("num_classes", 0)),
        "test_size": f"{test_size * 100:.0f}%" if test_size is not None else "–",
        "stopwords_removed": "ja" if meta.get("remove_stopwords", False) else "nein",
        "use_lemma": "ja" if meta.get("use_lemma", True) else "nein",
        "use_spacy": "ja" if meta.get("use_spacy", False) else "nein",
        "use_dornseiff": "ja" if meta.get("use_dornseiff", False) else "nein",
        "min_word_len": f"≥ {min_len}" if min_len > 1 else "1 (alle)",
        "analyzer": analyzer_str,
        "has_report": report_file.exists(),
        "uses_lemma": bool(meta.get("use_lemma", True)),
    }


def list_models() -> list[dict]:
    """Alle gespeicherten Modelle mit aufbereiteten Metadaten (neueste zuerst)."""
    rows: list[dict] = []
    if not MODELS_DIR.exists():
        return rows

    for pkl_file in MODELS_DIR.glob("*.pkl"):
        meta_file = MODELS_DIR / pkl_file.name.replace(".pkl", "_metadata.json")
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        else:
            model_type = pkl_file.stem.split("_")[1] if "_" in pkl_file.stem else "unknown"
            meta = {
                "model_type": model_type,
                "accuracy": 0.0,
                "timestamp": datetime.fromtimestamp(pkl_file.stat().st_mtime).strftime("%Y%m%d_%H%M%S"),
            }
        rows.append(_row_from_metadata(pkl_file, meta))

    # Nach Datum absteigend; dd.mm.yy -> yy.mm.dd für korrekten Vergleich.
    rows.sort(key=lambda x: ".".join(reversed(x.get("date", "").split("."))), reverse=True)
    return rows


def best_model() -> dict | None:
    """Modell mit höchster Accuracy (für Sidebar/Statusbar und Sachgruppen-Metriken)."""
    best: dict | None = None
    best_acc = -1.0
    if not MODELS_DIR.exists():
        return None
    for meta_file in MODELS_DIR.glob("*_metadata.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        acc = float(meta.get("accuracy", -1))
        if acc > best_acc:
            best_acc = acc
            best = {
                "model_file": meta_file.name.replace("_metadata.json", ".pkl"),
                "model_name": meta.get("model_name")
                or MODEL_DISPLAY_NAMES.get(meta.get("model_type", ""), meta.get("model_type", "")),
                "accuracy": round(acc, 4),
                "stem": meta_file.stem.replace("_metadata", ""),
            }
    return best


def available_model_files() -> list[str]:
    """Dateinamen aller gespeicherten Modelle (neueste zuerst)."""
    return sorted((f.name for f in MODELS_DIR.glob("*.pkl")), reverse=True)


def model_uses_lemma(model_file: str) -> bool:
    """use_lemma aus den Metadaten lesen (Default True)."""
    meta_path = MODELS_DIR / model_file.replace(".pkl", "_metadata.json")
    try:
        return bool(json.loads(meta_path.read_text(encoding="utf-8")).get("use_lemma", True))
    except (OSError, json.JSONDecodeError):
        return True


def report_text(model_file: str) -> str | None:
    """Klassifikations-Report eines Modells als Text (oder None)."""
    report_path = MODELS_DIR / model_file.replace(".pkl", "_report.txt")
    if report_path.exists():
        return report_path.read_text(encoding="utf-8")
    return None
