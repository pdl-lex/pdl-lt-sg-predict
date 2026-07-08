"""Sachgruppen-Übersicht: Taxonomie + Klassifikationsmetriken des besten Modells."""
from __future__ import annotations

from .bridge import MODELS_DIR, sachgruppen_map
from .models import best_model


def _parse_report(report_text: str) -> dict[str, dict]:
    """sklearn-classification_report parsen -> label -> {precision, recall, f1, support}."""
    result: dict[str, dict] = {}
    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped or "precision" in stripped:
            continue
        if any(stripped.startswith(s) for s in ("accuracy", "macro avg", "weighted avg")):
            continue
        parts = stripped.rsplit(None, 4)  # label prec rec f1 support
        if len(parts) == 5:
            try:
                result[parts[0].strip()] = {
                    "precision": float(parts[1]),
                    "recall": float(parts[2]),
                    "f1": float(parts[3]),
                    "support": int(parts[4]),
                }
            except ValueError:
                pass
    return result


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def sachgruppen_overview() -> dict:
    """Alle Sachgruppen (aus sachgruppen.csv) plus Metriken aus dem besten Modell."""
    sg_map = sachgruppen_map()

    best = best_model()
    report_data: dict[str, dict] = {}
    if best:
        report_path = MODELS_DIR / f"{best['stem']}_report.txt"
        if report_path.exists():
            report_data = _parse_report(report_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    seen: set[str] = set()
    for nummer in sorted(sg_map.keys(), key=lambda x: (len(x), x)):
        seen.add(nummer)
        m = report_data.get(nummer, {})
        rows.append({
            "nummer": nummer,
            "sachgruppe": sg_map[nummer],
            "support": m.get("support", 0),
            "precision": _fmt(m["precision"]) if "precision" in m else "-",
            "recall": _fmt(m["recall"]) if "recall" in m else "-",
            "f1": _fmt(m["f1"]) if "f1" in m else "-",
        })

    # Report-Labels ohne Eintrag in sachgruppen.csv (z. B. Klasse "0").
    for label in sorted(report_data.keys(), key=lambda x: (len(x), x)):
        if label in seen:
            continue
        m = report_data[label]
        rows.append({
            "nummer": label,
            "sachgruppe": "–",
            "support": m.get("support", 0),
            "precision": _fmt(m["precision"]),
            "recall": _fmt(m["recall"]),
            "f1": _fmt(m["f1"]),
        })

    return {
        "rows": rows,
        "model_name": best["model_name"] if best else "",
        "model_file": best["model_file"] if best else "",
        "accuracy": f"{best['accuracy']:.4f}" if best else "",
    }
