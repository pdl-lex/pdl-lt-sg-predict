"""Evaluiert NN-Modelle auf dem Goldstandard (test/shuffle_pfwb1000_mit_sg.txt).

Der Goldstandard dient hier als Test-Set: die Bedeutungen werden unverändert durch
den Classifier geschickt, die Vorhersagen mit den bekannten Sachgruppen abgeglichen
und die üblichen Klassifikationsmetriken ausgegeben.

Vergleicht zwei Modelle:
  - bestes NN-Modell MIT Lemma  (Goldstandard hat kein Lemma -> leerer String)
  - bestes NN-Modell OHNE Lemma (use_lemma=False; kein Lemma-Nachteil)

Separates Analyse-Skript, unabhängig vom Haupt-Code der App.

Ausgabe: scripts/goldstandard_nn_report.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402
from pdl_lt_sg_predict.core.bridge import MODELS_DIR  # noqa: E402

GOLD_FILE = REPO_ROOT / "test" / "shuffle_pfwb1000_mit_sg.txt"
OUT_REPORT = Path(__file__).resolve().parent / "goldstandard_nn_report.txt"


def best_nn_model(use_lemma: bool) -> dict:
    """Bestes Modell vom Typ 'nn' mit passendem use_lemma, nach Trainings-Accuracy."""
    best: dict | None = None
    for meta_file in MODELS_DIR.glob("*_metadata.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("model_type") != "nn" or bool(meta.get("use_lemma", True)) != use_lemma:
            continue
        acc = float(meta.get("accuracy", -1))
        if best is None or acc > best["accuracy"]:
            best = {
                "model_file": meta_file.name.replace("_metadata.json", ".pkl"),
                "accuracy": acc,
                "use_lemma": use_lemma,
            }
    if best is None:
        raise SystemExit(f"Kein trainiertes NN-Modell mit use_lemma={use_lemma} gefunden.")
    return best


def load_goldstandard() -> pd.DataFrame:
    rows = []
    for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            continue
        bedeutung, sg = parts
        rows.append({"lemma": "", "bedeutung": bedeutung.strip(), "sachgruppe": sg.strip()})
    return pd.DataFrame(rows)


def decoded_classes(clf) -> np.ndarray:
    classes = clf.pipeline.named_steps["classifier"].classes_
    if clf.label_encoder is not None:
        classes = clf.label_encoder.inverse_transform(classes)
    return np.asarray(classes).astype(str)


def evaluate(model_info: dict, df: pd.DataFrame) -> dict:
    model_path = MODELS_DIR / model_info["model_file"]
    clf = SachgruppenClassifier.load(str(model_path))

    X = df[["lemma", "bedeutung"]]
    y_true = df["sachgruppe"].to_numpy()
    y_pred = np.asarray(clf.predict(X)).astype(str)

    proba = None
    classes = None
    try:
        proba = np.asarray(clf.predict_proba(X), dtype=float)
        classes = decoded_classes(clf)
    except Exception as e:  # noqa: BLE001
        print(f"Hinweis: predict_proba nicht verfügbar für {model_info['model_file']} ({e})")

    result = {
        "model_file": model_info["model_file"],
        "train_accuracy": model_info["accuracy"],
        "use_lemma": model_info["use_lemma"],
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }

    if proba is not None:
        order = np.argsort(proba, axis=1)[:, ::-1]
        ranked = classes[order]
        for k in (1, 3, 5):
            hit = (ranked[:, :k] == y_true[:, None]).any(axis=1)
            result[f"top{k}"] = hit.mean()

    return result


def main() -> None:
    df = load_goldstandard()
    print(f"Goldstandard geladen: {len(df)} Fälle, {df['sachgruppe'].nunique()} distinkte Sachgruppen")

    models = [best_nn_model(use_lemma=True), best_nn_model(use_lemma=False)]
    for m in models:
        print(f"  {'mit' if m['use_lemma'] else 'ohne'} Lemma: {m['model_file']} "
              f"(Trainings-Accuracy {m['accuracy']:.4f})")

    results = [evaluate(m, df) for m in models]

    r_lemma, r_no_lemma = results
    col = 18
    lines = [
        "=" * 70,
        "GOLDSTANDARD-EVALUATION (test/shuffle_pfwb1000_mit_sg.txt)",
        f"n = {len(df)}",
        "=" * 70,
        f"MIT Lemma:  {r_lemma['model_file']}",
        f"OHNE Lemma: {r_no_lemma['model_file']}",
        "",
        f"{'Metrik':<20}{'MIT Lemma':>{col}}{'OHNE Lemma':>{col}}",
        "-" * 56,
    ]
    lines.append(f"{'Trainings-Acc.':<20}{r_lemma['train_accuracy']:>{col}.4f}{r_no_lemma['train_accuracy']:>{col}.4f}")
    for k in (1, 3, 5):
        key = f"top{k}"
        if key in r_lemma and key in r_no_lemma:
            lines.append(f"{'Goldstandard Top-' + str(k):<20}{r_lemma[key]:>{col}.4f}{r_no_lemma[key]:>{col}.4f}")
    lines.append(f"{'F1 (macro)':<20}{r_lemma['f1_macro']:>{col}.4f}{r_no_lemma['f1_macro']:>{col}.4f}")
    lines.append(f"{'F1 (weighted)':<20}{r_lemma['f1_weighted']:>{col}.4f}{r_no_lemma['f1_weighted']:>{col}.4f}")

    for r in results:
        lines += [
            "",
            "=" * 70,
            f"Classification Report - {'MIT' if r['use_lemma'] else 'OHNE'} Lemma ({r['model_file']})",
            "=" * 70,
            r["report"],
        ]

    report = "\n".join(lines)
    print("\n" + report)

    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\nVollständiger Report gespeichert: {OUT_REPORT}")


if __name__ == "__main__":
    main()
