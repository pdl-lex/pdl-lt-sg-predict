#!/usr/bin/env python3
"""
Bewertet ein trainiertes Modell zusätzlich gegen die VOLLSTÄNDIGE Codemenge
einer Bedeutung statt nur gegen den einen Code, der bei der Extraktion übrig
blieb.

Hintergrund: 15,3 % der Bedeutungen tragen mehrere Sachgruppen; `extract_data.py`
nimmt als Trainingslabel den ersten (= kleinsten) Code. Eine Vorhersage, die
einen der übrigen Codes nennt, ist fachlich richtig, wird von der strikten
Metrik aber als Fehler gezählt. Gemessen an nn_char_wb_ml1_sw0: 8,9 % der
"Fehler" sind keine.

Es wird NICHT neu trainiert und NICHTS am Training geändert — dies ist reine
Berichterstattung. Die strikte Zahl bleibt immer mit ausgewiesen.

Zwei Metriken, die die Wahrheit einklammern:

  * set-aware Accuracy (Weg A): Treffer, wenn die Vorhersage in der Codemenge
    liegt. Obere Schranke — bei drei Goldcodes hat das Modell drei Chancen.
  * micro-F1 auf Indikatormatrix (Weg B): sklearn-nativ multilabel. Untere
    Schranke — ein Single-Label-Modell kann pro Zeile nur einmal feuern, bei
    zwei Goldcodes ist der Recall auf 0,5 gedeckelt.

Beispiel:
    python scripts/evaluate_setaware.py \
        --model models/nn_bright_cherry.pkl \
        --csv data/woerterbuch_daten.csv

Wenn die Trainings-CSV die Spalte `sachgruppen_alle` noch nicht hat (Modelle von
vor dem Umbau), liefert eine neuere Extraktion die Mengen per --mengen-csv; der
Abgleich läuft dann über (lemma, bedeutung). Wichtig: --csv muss dieselbe Datei
sein, auf der das Modell trainiert wurde, sonst reproduziert der Split den
Test-Teil nicht und die Zahlen sind durch Leckage geschönt.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Skript liegt in scripts/; Projekt-Root und scripts/ importierbar machen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_topk import class_scores, reproduce_test_split  # noqa: E402
from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402

MENGEN_SPALTE = 'sachgruppen_alle'
MENGEN_TRENNER = ';'


def mengen_aus_spalte(csv_file, index, y_true):
    """Codemengen aus der Spalte `sachgruppen_alle` der Trainings-CSV."""
    df = pd.read_csv(csv_file)
    if MENGEN_SPALTE not in df.columns:
        return None
    roh = df.loc[index, MENGEN_SPALTE]
    return [
        set(str(s).split(MENGEN_TRENNER)) if isinstance(s, str) and s else {t}
        for s, t in zip(roh, y_true)
    ]


def mengen_aus_fremd_csv(pfad, X_test, y_true):
    """
    Codemengen aus einer anderen (neueren) Extraktion, verknüpft über
    (lemma, bedeutung). Für Modelle, deren Trainings-CSV die Spalte noch nicht
    hat.
    """
    df = pd.read_csv(pfad)
    fehlend = {'lemma', 'bedeutung', MENGEN_SPALTE} - set(df.columns)
    if fehlend:
        raise SystemExit(f"{pfad}: Spalten fehlen: {', '.join(sorted(fehlend))}")

    tabelle = {}
    for lemma, bed, alle in zip(df['lemma'], df['bedeutung'], df[MENGEN_SPALTE]):
        if not isinstance(alle, str) or not alle:
            continue
        tabelle.setdefault((str(lemma).strip(), str(bed).strip()), set()).update(
            alle.split(MENGEN_TRENNER))

    mengen, treffer = [], 0
    for lemma, bed, t in zip(X_test['lemma'], X_test['bedeutung'], y_true):
        gefunden = tabelle.get((str(lemma).strip(), str(bed).strip()))
        if gefunden:
            treffer += 1
            mengen.append(gefunden | {t})
        else:
            mengen.append({t})
    print(f"  Mengen zugeordnet: {treffer}/{len(mengen)} Testzeilen "
          f"({treffer/len(mengen):.1%}); der Rest wird strikt gewertet")
    return mengen


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--model', required=True, help='Pfad zur .pkl-Modelldatei')
    ap.add_argument('--csv', default='data/woerterbuch_daten.csv',
                    help='Trainings-CSV (dieselbe, auf der das Modell trainiert wurde)')
    ap.add_argument('--mengen-csv', default=None,
                    help='Alternative Quelle für die Codemengen, verknüpft über '
                         '(lemma, bedeutung) — falls --csv die Spalte '
                         f'"{MENGEN_SPALTE}" nicht hat.')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--report', action='store_true',
                    help='Zusätzlich den klassenweisen Bericht auf der '
                         'Indikatormatrix ausgeben (468 Zeilen).')
    args = ap.parse_args()

    print(f"Lade Modell: {args.model}")
    clf = SachgruppenClassifier.load(args.model)

    print(f"Reproduziere Test-Split aus {args.csv} ...")
    X_test, y_test, _y_train = reproduce_test_split(args.csv, args.test_size)
    y_true = y_test.to_numpy().astype(str)
    print(f"Test-Samples: {len(y_true)}")

    if args.mengen_csv:
        print(f"Codemengen aus {args.mengen_csv} (Abgleich über lemma+bedeutung)")
        mengen = mengen_aus_fremd_csv(args.mengen_csv, X_test, y_true)
    else:
        mengen = mengen_aus_spalte(args.csv, X_test.index, y_true)
        if mengen is None:
            raise SystemExit(
                f"{args.csv} hat keine Spalte '{MENGEN_SPALTE}'.\n"
                "Entweder neu extrahieren (scripts/extract_data.py --quelle db)\n"
                "oder die Mengen aus einer neueren Extraktion holen: --mengen-csv PFAD"
            )

    mehrdeutig = sum(1 for m in mengen if len(m) > 1)
    print(f"Mehrdeutige Testzeilen: {mehrdeutig} ({mehrdeutig/len(mengen):.1%})")

    print("Berechne Scores ...")
    scores, classes = class_scores(clf, X_test)
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranked = classes[order]
    pred1 = ranked[:, 0]

    # --- Weg A: y_true umschreiben, accuracy_score bleibt unangetastet -------
    in_menge = np.array([p in m for p, m in zip(pred1, mengen)])
    y_true_lenient = np.where(in_menge, pred1, y_true)

    strikt = accuracy_score(y_true, pred1)
    lenient = accuracy_score(y_true_lenient, pred1)
    fehler = int((pred1 != y_true).sum())
    gerettet = int((in_menge & (pred1 != y_true)).sum())

    print("\n" + "=" * 64)
    print("TOP-1 ACCURACY")
    print("=" * 64)
    print(f"  strikt (ein Goldcode)          {strikt:.4f}")
    print(f"  set-aware (in der Codemenge)   {lenient:.4f}   {lenient - strikt:+.4f}")
    if fehler:
        print(f"  Fehler gesamt: {fehler} — davon gültige Zweitcodes: "
              f"{gerettet} ({gerettet/fehler:.1%})")

    print("\n" + "=" * 64)
    print("TOP-k ACCURACY")
    print("=" * 64)
    print(f"  {'k':>3} | {'strikt':>8} | {'set-aware':>10}")
    print(f"  {'-'*3} | {'-'*8} | {'-'*10}")
    for k in (1, 3, 5):
        s = (ranked[:, :k] == y_true[:, None]).any(axis=1).mean()
        a = np.array([bool(set(r[:k]) & m) for r, m in zip(ranked, mengen)]).mean()
        print(f"  {k:>3} | {s:>8.4f} | {a:>10.4f}")

    # --- Weg B: Indikatormatrix, sklearn-nativ multilabel --------------------
    alle_klassen = sorted({c for m in mengen for c in m} | set(pred1))
    mlb = MultiLabelBinarizer(classes=alle_klassen)
    Y_true = mlb.fit_transform(mengen)
    Y_pred = mlb.transform([[p] for p in pred1])

    print("\n" + "=" * 64)
    print("MULTILABEL (Indikatormatrix, ein Vorschlag pro Zeile)")
    print("=" * 64)
    print(f"  Klassen mit Belegen im Testteil: {int((Y_true.sum(axis=0) > 0).sum())}")
    print(f"  micro-F1  {f1_score(Y_true, Y_pred, average='micro', zero_division=0):.4f}")
    print(f"  macro-F1  {f1_score(Y_true, Y_pred, average='macro', zero_division=0):.4f}")
    print("  (Recall ist bauartbedingt gedeckelt: bei n Goldcodes höchstens 1/n)")

    if args.report:
        print("\n" + classification_report(
            Y_true, Y_pred, target_names=[str(c) for c in mlb.classes_],
            zero_division=0))


if __name__ == '__main__':
    main()
