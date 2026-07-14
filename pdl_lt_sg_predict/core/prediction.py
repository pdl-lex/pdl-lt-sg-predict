"""Vorhersage-Logik: Einzel- und Batch-Vorhersage sowie SHAP-Erklärung.

Kapselt die Aufrufe an den ``SachgruppenClassifier`` und liefert reine
Python-Datenstrukturen zurück (von der API in JSON gewandelt).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .bridge import MODELS_DIR, describe, get_model


def _decoded_classes(clf) -> np.ndarray:
    classifier_step = clf.pipeline.named_steps["classifier"]
    classes = classifier_step.classes_
    if clf.label_encoder is not None:
        classes = clf.label_encoder.inverse_transform(classes)
    return classes


def _predict_with_topk(clf, X, k: int = 3):
    """(predictions, indices, proba_matrix|None, classes) je Zeile.

    Transformiert die Eingabe genau EINMAL durch die Pipeline und leitet daraus
    sowohl die Vorhersage (Argmax) als auch das Top-k-Ranking ab — statt predict
    und predict_proba getrennt (= doppelte TF-IDF-Transformation) aufzurufen.
    Ranking über predict_proba, sonst decision_function (SVM); Fallback ohne
    Ranking: nur clf.predict.
    """
    classes = _decoded_classes(clf)
    classifier = clf.pipeline.named_steps["classifier"]
    X_transformed = clf.pipeline[:-1].transform(X)

    proba = None
    try:
        proba = np.asarray(classifier.predict_proba(X_transformed), dtype=float)
        scores = proba
    except Exception:
        try:
            scores = np.asarray(classifier.decision_function(X_transformed), dtype=float)
            if scores.ndim == 1:  # binär – unwahrscheinlich, aber abgesichert
                scores = np.column_stack([-scores, scores])
        except Exception:
            scores = None

    if scores is None:
        return [str(p) for p in clf.predict(X)], None, None, classes

    idx = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    predictions = [str(classes[row[0]]) for row in idx]
    return predictions, idx, proba, classes


def predict_single(model_file: str, lemma: str, bedeutung: str, k: int = 3) -> dict:
    """Einzelvorhersage. Liefert Top-k Sachgruppen mit optionaler Wahrscheinlichkeit."""
    clf = get_model(str(MODELS_DIR / model_file))
    X = pd.DataFrame({"lemma": [lemma or ""], "bedeutung": [bedeutung or ""]})

    predictions, idx, proba, classes = _predict_with_topk(clf, X, k)
    prediction = predictions[0]

    top: list[dict] = []
    if idx is not None:
        for rank, i in enumerate(idx[0]):
            label = str(classes[i])
            top.append({
                "label": label,
                "description": describe(label),
                "proba": round(float(proba[0, i]) * 100, 1) if proba is not None else None,
                "is_best": rank == 0,
            })
    else:
        top.append({"label": prediction, "description": describe(prediction), "proba": None, "is_best": True})

    return {
        "prediction": prediction,
        "description": describe(prediction),
        "top": top,
        "model_type": clf.model_type,
        "uses_lemma": bool(clf.use_lemma),
    }


def explain(model_file: str, lemma: str, bedeutung: str, predicted_label: str,
            filter_stopwords: bool = True) -> dict:
    """SHAP-Worterklärung für eine Einzelvorhersage."""
    model_path = str(MODELS_DIR / model_file)
    clf = get_model(model_path)
    X = pd.DataFrame({"lemma": [lemma or ""], "bedeutung": [bedeutung or ""]})
    result = clf.explain(X, predicted_label, model_path, filter_stopwords=filter_stopwords)

    def _pairs(pairs) -> list[dict]:
        return [{"word": w, "score": round(float(s), 4)} for w, s in pairs]

    return {
        "lemma": _pairs(result.get("lemma", [])),
        "bedeutung": _pairs(result.get("bedeutung", [])),
        "is_nn": clf.model_type == "nn",
    }


def _read_csv_flexible(source) -> pd.DataFrame | None:
    """CSV mit unbekanntem Trenner robust einlesen."""
    for sep in (None, ";", ",", "\t"):
        try:
            buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
            kwargs = {"engine": "python"} if sep is None else {}
            df = pd.read_csv(buf, sep=sep, **kwargs)
            if len(df.columns) >= 1:
                df.columns = [c.lstrip("﻿").strip() for c in df.columns]
                return df
        except Exception:
            continue
    return None


def predict_batch(model_file: str, csv_bytes: bytes, k: int = 3) -> dict:
    """Batch-Vorhersage über eine hochgeladene CSV (Spalte 'bedeutung' erforderlich)."""
    df = _read_csv_flexible(csv_bytes)
    if df is None or df.empty:
        raise ValueError("CSV konnte nicht gelesen werden.")
    if "bedeutung" not in df.columns:
        raise ValueError("CSV muss mindestens eine Spalte 'bedeutung' enthalten.")
    if "lemma" not in df.columns:
        df["lemma"] = ""

    clf = get_model(str(MODELS_DIR / model_file))
    X = df[["lemma", "bedeutung"]].fillna("")
    predictions, idx, proba, classes = _predict_with_topk(clf, X, k)

    rows: list[dict] = []
    for n, (_, row) in enumerate(df.iterrows()):
        sg = str(predictions[n])
        entry = {
            "lemma": str(row.get("lemma", "")),
            "bedeutung": str(row.get("bedeutung", "")),
            "sachgruppe": sg,
            "beschreibung": describe(sg),
            "wahrscheinlichkeit": "",
            "sachgruppe_2": "", "beschreibung_2": "", "wahrscheinlichkeit_2": "",
            "sachgruppe_3": "", "beschreibung_3": "", "wahrscheinlichkeit_3": "",
        }
        if idx is not None:
            top = idx[n]
            if proba is not None:
                entry["wahrscheinlichkeit"] = f"{proba[n, top[0]] * 100:.1f}%"
            for rank, suffix in ((1, "_2"), (2, "_3")):
                if len(top) > rank:
                    lbl = str(classes[top[rank]])
                    entry[f"sachgruppe{suffix}"] = lbl
                    entry[f"beschreibung{suffix}"] = describe(lbl)
                    if proba is not None:
                        entry[f"wahrscheinlichkeit{suffix}"] = f"{proba[n, top[rank]] * 100:.1f}%"
        rows.append(entry)

    return {
        "rows": rows,
        "count": len(rows),
        "uses_lemma": bool(clf.use_lemma),
    }
