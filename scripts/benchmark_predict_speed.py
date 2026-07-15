#!/usr/bin/env python3
"""
Misst die Vorhersage-Geschwindigkeit (Latenz) der trainierten Modelle.

Fuer jeden Modelltyp (svm, logistic, rf, xgboost, nn) wird automatisch das
neueste Modell aus models/ geladen und auf zwei Beispiel-Eingaben getimt:

    waggala : kleines kind; kind, das noch wackelig auf den Beinen ist
    Datschi : belegter, flacher Obstkuchen

Gemessen werden zwei Szenarien, jeweils nach einem Warmup-Lauf (der erste
predict()-Aufruf zahlt einmalige Kosten wie das Laden von spaCy/Dornseiff):

  * Einzel-Latenz : ein Sample pro predict()-Aufruf  (typischer API-Fall)
  * Batch-Latenz  : beide Samples in einem predict()-Aufruf

Beispiel:
    python scripts/benchmark_predict_speed.py
    python scripts/benchmark_predict_speed.py --repeats 200
    python scripts/benchmark_predict_speed.py --models svm nn
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_TYPES = ["svm", "logistic", "rf", "xgboost", "nn"]

# Beispiel-Eingaben (lemma + bedeutung)
SAMPLES = pd.DataFrame(
    {
        "lemma": ["waggala", "Datschi"],
        "bedeutung": [
            "kleines kind; kind, das noch wackelig auf den Beinen ist",
            "belegter, flacher Obstkuchen",
        ],
    }
)


def newest_model_for(model_type: str) -> Path | None:
    """Neueste .pkl-Datei fuer einen Modelltyp (Dateiname beginnt mit '<typ>_')."""
    candidates = [
        p for p in MODELS_DIR.glob(f"{model_type}_*.pkl")
        if p.stem.split("_")[0] == model_type
    ]
    if not candidates:
        return None
    # Zeitstempel steckt im Namen; mtime als robuster Fallback.
    return max(candidates, key=lambda p: (p.stem, p.stat().st_mtime))


def time_predict(clf: SachgruppenClassifier, X: pd.DataFrame, repeats: int) -> list[float]:
    """predict() 'repeats' mal ausfuehren, Zeiten in Millisekunden zurueckgeben."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        clf.predict(X)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(times: list[float]) -> dict:
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=MODEL_TYPES,
                    help=f"Zu testende Modelltypen (Default: {' '.join(MODEL_TYPES)})")
    ap.add_argument("--repeats", type=int, default=100,
                    help="Messwiederholungen pro Szenario (Default: 100)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Warmup-Laeufe vor der Messung (Default: 3)")
    args = ap.parse_args()

    single = SAMPLES.iloc[[0]]  # ein Sample (waggala)
    batch = SAMPLES              # beide Samples

    print("=" * 78)
    print(f"Vorhersage-Geschwindigkeit  ({args.repeats} Wiederholungen, "
          f"{args.warmup} Warmup)")
    print("=" * 78)

    results = []
    for model_type in args.models:
        model_path = newest_model_for(model_type)
        if model_path is None:
            print(f"\n[{model_type:8}] kein Modell in {MODELS_DIR} gefunden – uebersprungen.")
            continue

        print(f"\n[{model_type:8}] {model_path.name}")
        clf = SachgruppenClassifier.load(str(model_path))

        # Warmup (einmalige Init-Kosten wie spaCy-Laden nicht mitmessen)
        for _ in range(max(1, args.warmup)):
            clf.predict(batch)

        # Korrektheits-Sichtprobe: was sagt das Modell auf die zwei Beispiele?
        preds = clf.predict(batch)
        for lemma, pred in zip(SAMPLES["lemma"], preds):
            print(f"           {lemma:10} -> Sachgruppe {pred}")

        single_ms = summarize(time_predict(clf, single, args.repeats))
        batch_ms = summarize(time_predict(clf, batch, args.repeats))

        print(f"           Einzel : {single_ms['median']:7.2f} ms/Aufruf "
              f"(min {single_ms['min']:.2f}, max {single_ms['max']:.2f}, "
              f"stdev {single_ms['stdev']:.2f})")
        print(f"           Batch-2: {batch_ms['median']:7.2f} ms/Aufruf "
              f"(= {batch_ms['median'] / 2:.2f} ms/Sample)")

        results.append((model_type, single_ms, batch_ms))

    if not results:
        print("\nKeine Modelle getestet.")
        return

    # Uebersichtstabelle (nach Einzel-Median sortiert, schnellstes zuerst)
    print("\n" + "=" * 78)
    print("UEBERSICHT  (Median-Latenz, schnellstes zuerst)")
    print("=" * 78)
    header = f"{'Modell':10} {'Einzel [ms]':>14} {'Batch-2 [ms]':>14} {'Durchsatz [1/s]':>18}"
    print(header)
    print("-" * len(header))
    for model_type, single_ms, batch_ms in sorted(results, key=lambda r: r[1]["median"]):
        throughput = 1000.0 / single_ms["median"] if single_ms["median"] > 0 else float("inf")
        print(f"{model_type:10} {single_ms['median']:14.2f} {batch_ms['median']:14.2f} "
              f"{throughput:18.1f}")


if __name__ == "__main__":
    main()
