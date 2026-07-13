"""
Baseline-Accuracy fuer die Sachgruppen-Klassifikation.

Baseline = "wie oft treffe ich richtig, wenn ich immer die im Trainingsset
haeufigste(n) Klasse(n) vorhersage?" Gemessen wird der Treffer nicht auf dem
Trainingsset selbst, sondern auf zwei Zielsets:
  1. baseline_test.csv  - der Test-Split (gleiche Verteilung wie das Training,
     stratifiziert, kein nennenswerter Shift zu erwarten)
  2. goldstandard.csv   - unabhaengiges Set, hier kann die Verteilung
     (Distributional Shift) tatsaechlich von der Trainingsverteilung abweichen

Split repliziert exakt train_and_evaluate() in sachgruppen_classifier.py:
test_size=0.2, random_state=42, stratifiziert, Singleton-Klassen (1 Beispiel)
werden komplett dem Trainingsset zugeschlagen.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_FILE = "data/woerterbuch_daten_124217.csv"
GOLDSTANDARD_CSV = "data/goldstandard.csv"
BASELINE_TRAINING_CSV = "data/baseline_training.csv"
BASELINE_TEST_CSV = "data/baseline_test.csv"
RESULTS_TXT = "data/baseline_results.txt"


def split(csv_file: str):
    df = pd.read_csv(csv_file, sep=None, engine="python")
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]

    required_cols = ["lemma", "bedeutung", "sachgruppe"]
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean["lemma"] = df_clean["lemma"].astype(str).replace("", "LEER")
    df_clean["bedeutung"] = df_clean["bedeutung"].astype(str).replace("", "LEER")

    X = df_clean[["lemma", "bedeutung"]]
    y = df_clean["sachgruppe"].astype(str)

    class_counts = y.value_counts()
    single_sample_classes = class_counts[class_counts == 1].index

    mask_single = y.isin(single_sample_classes)
    X_single, y_single = X[mask_single], y[mask_single]
    X_multi, y_multi = X[~mask_single], y[~mask_single]

    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    X_train = pd.concat([X_train, X_single])
    y_train = pd.concat([y_train, y_single])

    return X_train, X_test, y_train, y_test


def load_goldstandard(csv_file: str) -> pd.Series:
    df = pd.read_csv(csv_file, sep="\t", header=None, names=["bedeutung", "sachgruppe"])
    return df["sachgruppe"].astype(str)


def baseline_report(name: str, y_target: pd.Series, top1_class: str,
                     top3_classes: list[str]) -> list[str]:
    n = len(y_target)
    top1_hits = (y_target == top1_class).sum()
    top3_hits = y_target.isin(top3_classes).sum()
    top1_pct = top1_hits / n * 100
    top3_pct = top3_hits / n * 100

    lines = [
        f"Ziel: {name} ({n} Datensaetze)",
        "-" * 50,
        f"  Top-1 (Klasse {top1_class}): {top1_hits}/{n} = {top1_pct:.2f}%",
        f"  Top-3 (Klassen {', '.join(top3_classes)}): {top3_hits}/{n} = {top3_pct:.2f}%",
        "",
    ]
    return lines


def main():
    X_train, X_test, y_train, y_test = split(CSV_FILE)

    train_set = X_train.copy()
    train_set["sachgruppe"] = y_train
    train_set.to_csv(BASELINE_TRAINING_CSV, index=False)

    test_set = X_test.copy()
    test_set["sachgruppe"] = y_test
    test_set.to_csv(BASELINE_TEST_CSV, index=False)

    y_gold = load_goldstandard(GOLDSTANDARD_CSV)

    n_train = len(y_train)

    # Haeufigste Klasse(n) IM TRAINING bestimmen (Basis fuer beide Auswertungen)
    train_counts = y_train.value_counts()
    top1_class = train_counts.index[0]
    top3_classes = list(train_counts.index[:3])
    top1_train_share = train_counts.iloc[0] / n_train * 100
    top3_train_share = train_counts.iloc[:3].sum() / n_train * 100

    lines = []
    lines.append("Baseline-Accuracy Sachgruppen-Klassifikation")
    lines.append("=" * 50)
    lines.append(f"Datenquelle: {CSV_FILE}")
    lines.append(f"Split: test_size=0.2, random_state=42, stratifiziert "
                 f"(identisch zu train_and_evaluate() in sachgruppen_classifier.py)")
    lines.append(f"Trainingsset: {n_train} Datensaetze -> {BASELINE_TRAINING_CSV}")
    lines.append(f"Testset: {len(y_test)} Datensaetze -> {BASELINE_TEST_CSV}")
    lines.append(f"Goldstandard: {len(y_gold)} Datensaetze <- {GOLDSTANDARD_CSV}")
    lines.append("")
    lines.append(f"Haeufigste Klasse(n) im TRAINING (Basis der Vorhersage):")
    lines.append(f"  Top-1: {top1_class} ({train_counts.iloc[0]} Beispiele, {top1_train_share:.2f}% des Trainingssets)")
    top3_detail = ", ".join(
        f"{cls} ({train_counts[cls]}, {train_counts[cls]/n_train*100:.2f}%)"
        for cls in top3_classes
    )
    lines.append(f"  Top-3: {top3_detail} (Summe {top3_train_share:.2f}%)")
    lines.append("")
    lines.append("Baseline = Trefferquote dieser trainingsbasierten Klasse(n), gemessen je Zielset:")
    lines.append("=" * 50)
    lines += baseline_report("baseline_test.csv (Test-Split)", y_test, top1_class, top3_classes)
    lines += baseline_report("goldstandard.csv", y_gold, top1_class, top3_classes)
    lines.append("Hinweis: Die Baseline ist die Trefferquote AUF DEM JEWEILIGEN ZIELSET, nicht "
                 "der Trainingsset-Anteil. Weichen Testset- und Goldstandard-Werte voneinander "
                 "ab, zeigt das einen Distributional Shift zwischen den beiden Zielsets.")

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
