"""Auflistung und Metadaten trainierter Modelle."""
from __future__ import annotations

import json
import re
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


def _fmt_bool(v: bool) -> str:
    return "ja" if v else "nein"


def _fmt_min_len(min_len) -> str:
    return f"≥ {min_len}" if min_len and min_len > 1 else "1 (alle)"


def _fmt_analyzer(meta: dict) -> str:
    analyzer_raw = meta.get("analyzer", "char_wb")
    ngram_max = meta.get("word_ngram_max", 1)
    return f"word-(1,{ngram_max})" if analyzer_raw == "word" else "char_wb"


def _fmt_test_size(test_size) -> str:
    return f"{test_size * 100:.0f}%" if test_size is not None else "–"


def _parse_weighted_avg(report_text: str) -> dict | None:
    """"weighted avg"-Zeile eines sklearn-classification_report parsen."""
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("weighted avg"):
            parts = stripped.rsplit(None, 4)  # "weighted avg" prec rec f1 support
            if len(parts) == 5:
                try:
                    return {
                        "precision": float(parts[1]),
                        "recall": float(parts[2]),
                        "f1": float(parts[3]),
                    }
                except ValueError:
                    return None
    return None


def _row_from_metadata(pkl_file: Path, meta: dict) -> dict:
    model_name = meta.get("model_name") or MODEL_DISPLAY_NAMES.get(
        meta.get("model_type", ""), meta.get("model_type", "")
    )

    report_file = MODELS_DIR / pkl_file.name.replace(".pkl", "_report.txt")
    weighted = None
    if report_file.exists():
        try:
            weighted = _parse_weighted_avg(report_file.read_text(encoding="utf-8"))
        except OSError:
            weighted = None

    topk = meta.get("topk_metrics") or {}
    cv = meta.get("cross_validation") or {}
    group_kfold_accuracy = (
        round(float(cv["mean"]), 4) if cv.get("ok") and cv.get("mode") == "group" else "–"
    )

    return {
        "model_file": pkl_file.name,
        "model_name": model_name,
        "model_type": meta.get("model_type", ""),
        "accuracy": round(float(meta.get("accuracy", 0.0)), 4),
        "group_kfold_accuracy": group_kfold_accuracy,
        "precision": round(weighted["precision"], 4) if weighted else "–",
        "recall": round(weighted["recall"], 4) if weighted else "–",
        "f1_score": round(weighted["f1"], 4) if weighted else "–",
        "top1_confidence": round(float(topk["top1"]), 4) if "top1" in topk else "–",
        "top3_confidence": round(float(topk["top3"]), 4) if "top3" in topk else "–",
        "training_time": _fmt_training_time(meta["training_time"]) if "training_time" in meta else "",
        "date": _fmt_date(meta.get("timestamp", "")),
        "num_samples": int(meta.get("num_samples", 0)),
        "num_classes": int(meta.get("num_classes", 0)),
        "test_size": _fmt_test_size(meta.get("test_size")),
        "stopwords_removed": _fmt_bool(meta.get("remove_stopwords", False)),
        "use_lemma": _fmt_bool(meta.get("use_lemma", True)),
        "use_spacy": _fmt_bool(meta.get("use_spacy", False)),
        "use_dornseiff": _fmt_bool(meta.get("use_dornseiff", False)),
        "min_word_len": _fmt_min_len(meta.get("min_word_length", 1)),
        "analyzer": _fmt_analyzer(meta),
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
            # Dateinamen beginnen mit dem Modelltyp (z. B. nn_char_wb_…).
            model_type = pkl_file.stem.split("_")[0]
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
    """Dateinamen aller gespeicherten Modelle (neueste zuerst).

    Dateinamen folgen dem Adjektiv-Frucht-Schema (z. B. ``nn_sunny_orange.pkl``)
    und sind daher nicht chronologisch sortierbar — es wird nach mtime sortiert.
    """
    return [
        f.name for f in sorted(
            MODELS_DIR.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True
        )
    ]


def model_uses_lemma(model_file: str) -> bool:
    """use_lemma aus den Metadaten lesen (Default True)."""
    meta_path = MODELS_DIR / model_file.replace(".pkl", "_metadata.json")
    try:
        return bool(json.loads(meta_path.read_text(encoding="utf-8")).get("use_lemma", True))
    except (OSError, json.JSONDecodeError):
        return True


def _config_header(meta: dict) -> str:
    """Trainings-Konfiguration, die aus der Ergebnistabelle entfernt wurde, für den Report."""
    lines = [
        "=" * 60,
        "TRAININGS-KONFIGURATION",
        "=" * 60,
        f"Test-Split:            {_fmt_test_size(meta.get('test_size'))}",
        f"Trainingsdauer:        {_fmt_training_time(meta['training_time']) if 'training_time' in meta else '–'}",
        f"Lemma:                 {_fmt_bool(meta.get('use_lemma', True))}",
        f"Min. Wortlänge:        {_fmt_min_len(meta.get('min_word_length', 1))}",
        f"Analyzer:              {_fmt_analyzer(meta)}",
        f"Stoppwörter entfernt:  {_fmt_bool(meta.get('remove_stopwords', False))}",
    ]
    return "\n".join(lines)


_SECTION_RULE = "=" * 60
_TOPK_MARKER = "TOP-k / HIERARCHIE / KONFIDENZ"
_CV_MARKER = "CROSS-VALIDIERUNG"
_SECTION_HEADER_RE = re.compile(rf"{re.escape(_SECTION_RULE)}\n(.+)\n{re.escape(_SECTION_RULE)}\n")


def _split_report_sections(text: str) -> tuple[str, str, str]:
    """Rohen Report-Text in (Klassifikationstabelle, Top-k-Block, CV-Block) aufteilen.

    Das Trainingsskript hängt beide Zusatzblöcke ans Ende der sklearn-
    ``classification_report``-Tabelle an, eingeleitet durch eine
    ``{_SECTION_RULE}\\n<Titel>\\n{_SECTION_RULE}\\n``-Kopfzeile. Fehlt ein Block
    (z. B. weil Cross-Validation nicht lief), ist der jeweilige Rückgabewert leer.
    """
    marks = [(m.start(), m.group(1).strip()) for m in _SECTION_HEADER_RE.finditer(text)]
    boundaries = [(pos, label) for pos, label in marks if label in (_TOPK_MARKER, _CV_MARKER)]

    if not boundaries:
        return text.rstrip("\n"), "", ""

    table = text[:boundaries[0][0]].rstrip("\n")
    sections = {_TOPK_MARKER: "", _CV_MARKER: ""}
    for i, (pos, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        sections[label] = text[pos:end].strip("\n")
    return table, sections[_TOPK_MARKER], sections[_CV_MARKER]


def report_text(model_file: str) -> str | None:
    """Klassifikations-Report eines Modells als Text, oder None.

    Reihenfolge: Trainings-Konfiguration, Top-k/Hierarchie/Konfidenz,
    Cross-Validierung, dann erst die klassenweise Auswertungstabelle
    (in der Rohdatei steht die Tabelle zuerst, die Zusatzblöcke werden hier
    nach vorne verschoben).
    """
    report_path = MODELS_DIR / model_file.replace(".pkl", "_report.txt")
    if not report_path.exists():
        return None
    table, topk_block, cv_block = _split_report_sections(report_path.read_text(encoding="utf-8"))

    parts: list[str] = []
    meta_path = MODELS_DIR / model_file.replace(".pkl", "_metadata.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            parts.append(_config_header(meta))
        except (OSError, json.JSONDecodeError):
            pass
    if topk_block:
        parts.append(topk_block)
    if cv_block:
        parts.append(cv_block)
    parts.append(f"{_SECTION_RULE}\nKOMPLETTE AUSWERTUNG\n{_SECTION_RULE}\n\n{table}")

    return "\n\n".join(parts)
