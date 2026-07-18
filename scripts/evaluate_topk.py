#!/usr/bin/env python3
"""
Misst an einem bereits trainierten Modell die für die beiden Betriebsarten
relevanten Kennzahlen:

  * Top-1 / Top-3 / Top-5 Accuracy            (Mensch waehlt aus Vorschlaegen)
  * Hierarchische Accuracy (richtige Zweisteller-Gruppe)
  * Konfidenz-/Abdeckungs-Kurve               (halbautomatischer Betrieb)

Der Test-Split wird exakt wie in train_and_evaluate() reproduziert
(random_state=42), es wird NICHT neu trainiert.

Beispiel:
    python scripts/evaluate_topk.py \
        --model models/svm_blue_banana.pkl \
        --csv data/woerterbuch_daten.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Projekt-Root importierbar machen (Skript liegt in scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402


def reproduce_test_split(csv_file: str, test_size: float = 0.2):
    """Reproduziert den Test-Split aus train_and_evaluate() Zeichen fuer Zeichen."""
    df = pd.read_csv(csv_file)
    df_clean = df.dropna(subset=["lemma", "bedeutung", "sachgruppe"]).copy()
    df_clean["lemma"] = df_clean["lemma"].astype(str).replace("", "LEER")
    df_clean["bedeutung"] = df_clean["bedeutung"].astype(str).replace("", "LEER")

    X = df_clean[["lemma", "bedeutung"]]
    y = df_clean["sachgruppe"].astype(str)

    class_counts = y.value_counts()
    single_sample_classes = class_counts[class_counts == 1].index

    if len(single_sample_classes) > 0:
        mask_single = y.isin(single_sample_classes)
        X_multi, y_multi = X[~mask_single], y[~mask_single]
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi,
                y_multi,
                test_size=test_size,
                random_state=42,
                stratify=y_multi,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi,
                y_multi,
                test_size=test_size,
                random_state=42,
            )
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
            )
    return X_test, y_test, y_train


def print_baseline(y_train, y_true, ks=(1, 3, 5)):
    """Input-unabhaengige Baseline: sagt fuer jeden Testfall die k in y_train
    haeufigsten Sachgruppen voraus (Menge, kein Ranking). Treffer = wahres
    Label liegt in dieser fixen Menge. Das ist die bestmoegliche Strategie
    ohne jede Information ueber lemma/bedeutung und damit die Messlatte, die
    ein Modell schlagen muss, um ueberhaupt etwas gelernt zu haben.
    """
    freq_order = y_train.value_counts().index.to_numpy().astype(str)
    print("\n" + "=" * 60)
    print("BASELINE (haeufigste Sachgruppen aus Training, ohne lemma/bedeutung)")
    print("=" * 60)
    for k in ks:
        top_k = set(freq_order[:k])
        hit = np.array([label in top_k for label in y_true])
        print(f"  Top-{k}: {hit.mean():.4f}")


def class_scores(clf, X_test):
    """(n_samples, n_classes)-Score-Matrix + zugehoerige String-Labels.

    Nutzt predict_proba falls vorhanden, sonst decision_function (SVM).
    """
    classifier = clf.pipeline.named_steps["classifier"]
    classes = classifier.classes_
    if clf.label_encoder is not None:  # XGBoost/NN: int -> Original-String
        classes = clf.label_encoder.inverse_transform(classes)
    classes = np.asarray(classes).astype(str)

    # Nackte LinearSVC hat kein predict_proba (hasattr False -> decision_function);
    # eine kalibrierte SVM hat eins, und dann zaehlen die Wahrscheinlichkeiten.
    if hasattr(clf.pipeline, "predict_proba"):
        try:
            scores = np.asarray(clf.pipeline.predict_proba(X_test), dtype=float)
            return scores, classes
        except Exception:
            pass
    scores = np.asarray(clf.pipeline.decision_function(X_test), dtype=float)
    return scores, classes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="Pfad zur .pkl-Modelldatei")
    ap.add_argument("--csv", default="data/woerterbuch_daten.csv", help="Trainings-CSV")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument(
        "--group-len",
        type=int,
        default=2,
        help="Stellen des Sachgruppen-Codes fuer die Hierarchie-Gruppe",
    )
    args = ap.parse_args()

    print(f"Lade Modell: {args.model}")
    clf = SachgruppenClassifier.load(args.model)

    print(f"Reproduziere Test-Split aus {args.csv} ...")
    X_test, y_test, y_train = reproduce_test_split(args.csv, args.test_size)
    y_true = y_test.to_numpy().astype(str)
    print(f"Test-Samples: {len(y_true)}")

    print_baseline(y_train, y_true)

    print("Berechne Scores ...")
    scores, classes = class_scores(clf, X_test)

    # Ranking pro Zeile (absteigend)
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranked_labels = classes[order]  # (n, n_classes)
    pred1 = ranked_labels[:, 0]
    correct1 = pred1 == y_true

    print("\n" + "=" * 60)
    print("TOP-k ACCURACY")
    print("=" * 60)
    for k in (1, 3, 5):
        hit = (ranked_labels[:, :k] == y_true[:, None]).any(axis=1)
        print(f"  Top-{k}: {hit.mean():.4f}")

    # Hierarchische Accuracy: richtige Gruppe (erste N Stellen)
    g = args.group_len
    grp_pred = np.array([s[:g] for s in pred1])
    grp_true = np.array([s[:g] for s in y_true])
    hier = (grp_pred == grp_true).mean()
    # Von den Top-1-FEHLERN: wie viele immerhin in richtiger Gruppe?
    wrong = ~correct1
    near_miss = ((grp_pred == grp_true) & wrong).sum()
    print("\n" + "=" * 60)
    print(f"HIERARCHIE (erste {g} Stellen = Zweisteller-Gruppe)")
    print("=" * 60)
    print(f"  Top-1 richtige Gruppe:        {hier:.4f}")
    if wrong.sum() > 0:
        print(
            f"  davon Top-1-Fehler, aber richtige Gruppe: "
            f"{near_miss}/{wrong.sum()} = {near_miss / wrong.sum():.1%} der Fehler sind Nachbar-Treffer"
        )

    # Konfidenz-/Abdeckungs-Kurve
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]  # Abstand #1 zu #2
    idx = np.argsort(margin)[::-1]  # sicherste zuerst
    correct_sorted = correct1[idx]
    n = len(correct_sorted)
    print("\n" + "=" * 60)
    print("KONFIDENZ / ABDECKUNG")
    print("=" * 60)
    print(f"  {'Abdeckung':>10} | {'Top-1-Accuracy auf akzept. Teil':>32}")
    print(f"  {'-' * 10} | {'-' * 32}")
    for cov in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        m = max(1, int(n * cov))
        print(f"  {cov:>9.0%} | {correct_sorted[:m].mean():>32.4f}")


if __name__ == "__main__":
    main()
