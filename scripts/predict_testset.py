#!/usr/bin/env python3
"""
Erzeugt Vorhersagen von einem oder mehreren trainierten Modellen auf
demselben Test-Set und speichert sie zusammen mit dem wahren Label in
einer CSV -- Grundlage fuer gepaarte Signifikanztests (siehe
scripts/significance.py, Funktion mcnemar_test()).

Wichtig: Alle uebergebenen Modelle muessen mit demselben Test-Split
(gleicher Seed) trainiert worden sein, sonst ist der paarweise Vergleich
pro Zeile ungueltig.

Beispiel:
    python scripts/predict_testset.py \
        --models models/nn_humble_peach.pkl \
                 models/nn_humble_coconut.pkl \
        --csv data/baseline_test.csv \
        --out data/predictions_nn_addons.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402


def check_same_training_setup(model_paths: list[str]) -> None:
    """Bricht ab, wenn die Modelle nicht mit demselben Datensatz/Split trainiert
    wurden (Metadaten-Abgleich von csv_file, num_samples, test_size).

    Ohne diesen Check waere ein Mischvergleich (z.B. 113126- gegen
    124217-Sample-Modell auf data/baseline_test.csv) stilles Leakage:
    Testzeilen des einen Modells lagen im Training des anderen.
    """
    setups: dict[str, tuple] = {}
    for model_path in model_paths:
        meta_path = Path(model_path).with_name(Path(model_path).stem + "_metadata.json")
        if not meta_path.exists():
            print(f"WARNUNG: Keine Metadaten fuer {Path(model_path).name} -- "
                  f"Split-Vergleichbarkeit kann nicht geprueft werden.")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        setups[Path(model_path).name] = (
            meta.get("csv_file"), meta.get("num_samples"), meta.get("test_size"),
        )
    if len(set(setups.values())) > 1:
        print("FEHLER: Die Modelle wurden nicht mit demselben Datensatz/Split trainiert.")
        for name, (csv_file, n, ts) in setups.items():
            print(f"  {name}: csv_file={csv_file}, num_samples={n}, test_size={ts}")
        print("Ein gepaarter Vergleich (McNemar) waere ungueltig. Abbruch.")
        sys.exit(1)


def load_testset(csv_file: str) -> pd.DataFrame:
    """Laedt ein Test-CSV und wendet dieselbe Bereinigung wie beim Training
    an (dropna + Leerstring -> 'LEER'), damit Zeilen/Reihenfolge zu den
    Modellen passen."""
    df = pd.read_csv(csv_file)
    required_cols = ["lemma", "bedeutung", "sachgruppe"]
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean["lemma"] = df_clean["lemma"].astype(str).replace("", "LEER")
    df_clean["bedeutung"] = df_clean["bedeutung"].astype(str).replace("", "LEER")
    df_clean["sachgruppe"] = df_clean["sachgruppe"].astype(str)
    return df_clean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                     help="Ein oder mehrere Pfade zu .pkl-Modelldateien")
    ap.add_argument("--csv", default="data/baseline_test.csv",
                     help="Test-CSV mit Spalten lemma, bedeutung, sachgruppe")
    ap.add_argument("--out", required=True,
                     help="Ziel-CSV: sachgruppe (wahr) + eine Spalte pro Modell")
    args = ap.parse_args()

    check_same_training_setup(args.models)

    print(f"Lade Test-Set: {args.csv}")
    df_test = load_testset(args.csv)
    X_test = df_test[["lemma", "bedeutung"]]
    print(f"Test-Samples: {len(df_test)}")

    out = pd.DataFrame({"sachgruppe": df_test["sachgruppe"].to_numpy()})

    for model_path in args.models:
        name = Path(model_path).stem
        print(f"\nLade Modell: {name}")
        clf = SachgruppenClassifier.load(model_path)
        print(f"  Vorhersage auf {len(X_test)} Test-Beispielen ...")
        preds = clf.predict(X_test)
        out[name] = preds
        acc = (out[name].astype(str) == out["sachgruppe"]).mean()
        print(f"  Accuracy: {acc:.4f}")

    args_out = Path(args.out)
    args_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args_out, index=False)
    print(f"\nGespeichert: {args_out}  ({len(out)} Zeilen, {len(out.columns) - 1} Modell(e))")


if __name__ == "__main__":
    main()
