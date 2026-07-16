#!/usr/bin/env python3
"""
Misst an einem (oder mehreren) trainierten Modellen, wie ehrlich die
Konfidenzwerte sind — also ob "Modell meldet p" auch "p ist richtig" bedeutet:

  * ECE (Expected Calibration Error, 15 Bins) und Brier-Score der Top-1-Konfidenz
  * Reliability-Tabelle (gemeldete Konfidenz vs. tatsaechliche Trefferquote)
  * Schwellen-Tabelle fuer den Automatik-Betrieb: Abdeckung + Accuracy bei p >= t

Bei kalibrierten Modellen (--calibrate beim Training) werden zusaetzlich die
ROHEN Scores des Basis-Modells ausgewertet (NN: unkalibrierte Softmax), so dass
kalibriert vs. unkalibriert direkt vergleichbar ist — aus EINEM Modell.

Der Test-Split wird exakt wie in train_and_evaluate() reproduziert
(random_state=42), es wird NICHT neu trainiert.

Beispiel:
    python scripts/evaluate_calibration.py \
        --models models/svm_*.pkl models/nn_*.pkl \
        --csv data/woerterbuch_daten_124217.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Projekt-Root importierbar machen (Skript liegt in scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402
from scripts.evaluate_topk import reproduce_test_split  # noqa: E402

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
N_BINS = 15


def probability_metrics(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray) -> dict:
    """Kalibrierungs-Kennzahlen der Top-1-Konfidenz einer Wahrscheinlichkeitsmatrix."""
    pred_idx = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(proba)), pred_idx]
    correct = classes[pred_idx] == y_true

    # ECE: mittlere |Konfidenz - Trefferquote| ueber gleichbreite Bins,
    # gewichtet mit dem Anteil der Faelle im Bin.
    bins = np.clip((conf * N_BINS).astype(int), 0, N_BINS - 1)
    ece = 0.0
    reliability = []
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        acc_b, conf_b, frac = correct[mask].mean(), conf[mask].mean(), mask.mean()
        ece += frac * abs(acc_b - conf_b)
        reliability.append((conf_b, acc_b, int(mask.sum())))

    thresholds = []
    for t in THRESHOLDS:
        mask = conf >= t
        thresholds.append((t, float(mask.mean()),
                           float(correct[mask].mean()) if mask.any() else float("nan")))

    return {
        "top1": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "ece": float(ece),
        "brier_top1": float(np.mean((conf - correct) ** 2)),
        "reliability": reliability,
        "thresholds": thresholds,
    }


def print_metrics(title: str, m: dict) -> None:
    print(f"\n--- {title}")
    print(f"  Top-1: {m['top1']:.4f}   mittlere Konfidenz: {m['mean_conf']:.4f}   "
          f"(Gap: {m['mean_conf'] - m['top1']:+.4f})")
    print(f"  ECE: {m['ece']:.4f}   Brier (Top-1): {m['brier_top1']:.4f}")
    print(f"  {'Schwelle':>9} | {'Abdeckung':>9} | {'Accuracy':>8}")
    for t, cov, acc in m["thresholds"]:
        acc_s = f"{acc:.4f}" if acc == acc else "   —"
        print(f"  {t:>8.0%} | {cov:>9.1%} | {acc_s:>8}")
    print(f"  Reliability (gemeldet -> getroffen, n):")
    for conf_b, acc_b, n in m["reliability"]:
        print(f"    {conf_b:.2f} -> {acc_b:.2f}  (n={n})")


def evaluate_model(model_path: str, X_test, y_true: np.ndarray) -> None:
    print("\n" + "=" * 68)
    print(f"MODELL: {model_path}")
    print("=" * 68)
    clf = SachgruppenClassifier.load(model_path)

    classifier = clf.pipeline.named_steps["classifier"]
    classes = classifier.classes_
    if clf.label_encoder is not None:
        classes = clf.label_encoder.inverse_transform(classes)
    classes = np.asarray(classes).astype(str)

    # Eingabe EINMAL durch die Feature-Pipeline (Kalibriert- und Basis-Scores
    # unterscheiden sich nur im letzten Schritt).
    X_transformed = clf.pipeline[:-1].transform(X_test)

    if hasattr(classifier, "predict_proba"):
        label = "kalibriert" if isinstance(classifier, CalibratedClassifierCV) else "predict_proba (roh)"
        proba = np.asarray(classifier.predict_proba(X_transformed), dtype=float)
        print_metrics(f"{clf.model_type.upper()} — {label}", probability_metrics(proba, classes, y_true))

    if isinstance(classifier, CalibratedClassifierCV):
        # ensemble=False: genau EIN Basis-Modell, auf allen Trainingsdaten gefittet.
        base = classifier.calibrated_classifiers_[0].estimator
        if hasattr(base, "predict_proba"):
            proba = np.asarray(base.predict_proba(X_transformed), dtype=float)
            print_metrics(f"{clf.model_type.upper()} — Basis unkalibriert (Softmax)",
                          probability_metrics(proba, classes, y_true))
        else:
            print(f"\n--- {clf.model_type.upper()} — Basis ohne predict_proba "
                  f"({type(base).__name__}): keine rohen Wahrscheinlichkeiten.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, nargs="+", help="Pfad(e) zu .pkl-Modelldateien")
    ap.add_argument("--csv", default="data/woerterbuch_daten_124217.csv", help="Trainings-CSV")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    print(f"Reproduziere Test-Split aus {args.csv} ...")
    X_test, y_test, _ = reproduce_test_split(args.csv, args.test_size)
    y_true = y_test.to_numpy().astype(str)
    print(f"Test-Samples: {len(y_true)}")

    for model_path in args.models:
        evaluate_model(model_path, X_test, y_true)


if __name__ == "__main__":
    main()
