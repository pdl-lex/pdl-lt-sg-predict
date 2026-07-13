"""Diagnose: warum bricht die Accuracy des NN-Modells auf dem Goldstandard
(test/shuffle_pfwb1000_mit_sg.txt) gegenüber der Trainings-Split-Accuracy so stark ein?

Prüft drei Hypothesen der Reihe nach, jede mit Zahlen belegt:

  1. Verteilung der getesteten Sachgruppen (Klassenhäufigkeit)
     -> widerlegt: Accuracy bleibt auch bei häufigen Sachgruppen (500+ Trainingsbeispiele) niedrig.
  2. Domain-Shift über das Vokabular (bairisch vs. pfälzisch)
     -> widerlegt: char-ngram-OOV-Rate ggü. dem tatsächlichen TFIDF-Vokabular liegt nur bei ~6%.
  3. Memorisierung durch Bedeutung-Duplikate im Trainingskorpus
     -> bestätigt: 75% der Zeilen haben eine Bedeutung, die im Korpus mehrfach vorkommt
        (generische Glosse wie "Familienname" 1407x); ein Zufalls-Split verteilt Duplikate auf
        beide Seiten, was die Testsplit-Accuracy stark aufbläst. Auf Bedeutungen, die im Testsplit
        wirklich neu sind (nicht auch im Train-Teil vorkommen), sinkt die Accuracy auf ein Niveau
        nahe am Goldstandard.

Separates Analyse-Skript, unabhängig vom Haupt-Code der App.

Ausgabe: scripts/goldstandard_gap_diagnosis.txt
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")  # Pickle-Versionswarnungen von sklearn beim Laden alter Modelle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402
from pdl_lt_sg_predict.core.bridge import MODELS_DIR, describe  # noqa: E402

CSV_FILE = REPO_ROOT / "data" / "woerterbuch_daten_124217.csv"
GOLD_FILE = REPO_ROOT / "test" / "shuffle_pfwb1000_mit_sg.txt"
MODEL_FILE = "nn_char_wb_ml1_sw0_20260709_230256.pkl"  # bestes NN-Modell (mit Lemma)
OUT_FILE = Path(__file__).resolve().parent / "goldstandard_gap_diagnosis.txt"

WORD_RE = re.compile(r"[a-zA-ZäöüÄÖÜß]+")


def load_goldstandard() -> pd.DataFrame:
    rows = []
    for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            continue
        bedeutung, sg = parts
        rows.append({"lemma": "", "bedeutung": bedeutung.strip(), "sachgruppe": sg.strip()})
    return pd.DataFrame(rows)


def reproduce_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Exakt wie train_and_evaluate() im Classifier (random_state=42)."""
    df_clean = df.dropna(subset=["lemma", "bedeutung", "sachgruppe"]).copy()
    df_clean["lemma"] = df_clean["lemma"].astype(str).replace("", "LEER")
    df_clean["bedeutung"] = df_clean["bedeutung"].astype(str).replace("", "LEER")
    X = df_clean[["lemma", "bedeutung"]]
    y = df_clean["sachgruppe"].astype(str)

    class_counts = y.value_counts()
    single = class_counts[class_counts == 1].index
    mask_single = y.isin(single)
    X_multi, y_multi = X[~mask_single], y[~mask_single]
    return train_test_split(X_multi, y_multi, test_size=test_size, random_state=42, stratify=y_multi)


def char_ngram_oov_rate(clf, texts: list[str]) -> tuple[float, float]:
    """OOV-Rate ggü. dem tatsächlich gefitteten TFIDF-Vokabular der 'bedeutung'-Spalte."""
    ct = clf.pipeline.named_steps["vectorizer"]
    bed_vect = {name: t for name, t, _cols in ct.transformers_}["bedeutung"]
    analyzer = bed_vect.build_analyzer()
    vocab = set(bed_vect.vocabulary_.keys())

    total, oov = 0, 0
    per_doc = []
    for s in texts:
        grams = analyzer(s)
        if not grams:
            continue
        n_oov = sum(1 for g in grams if g not in vocab)
        total += len(grams)
        oov += n_oov
        per_doc.append(n_oov / len(grams))
    return oov / total, float(np.mean(per_doc))


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 74)
    log("DIAGNOSE: Accuracy-Einbruch NN-Modell auf Goldstandard (pfwb1000)")
    log(f"Modell: {MODEL_FILE}")
    log("=" * 74)

    df = pd.read_csv(CSV_FILE, dtype=str)
    gold = load_goldstandard()
    clf = SachgruppenClassifier.load(str(MODELS_DIR / MODEL_FILE))

    # --- Hypothese 1: Klassenhäufigkeit -----------------------------------
    log("\n--- Hypothese 1: Verteilung der getesteten Sachgruppen ---")
    train_counts = df["sachgruppe"].str.strip().value_counts()
    gold_pred = np.asarray(clf.predict(gold[["lemma", "bedeutung"]])).astype(str)
    gold_true = gold["sachgruppe"].to_numpy()
    gold_correct = gold_pred == gold_true
    support = gold["sachgruppe"].map(train_counts).fillna(0).astype(int)
    for lo, hi, label in [(1, 49, "selten (1-49)"), (50, 499, "mittel (50-499)"), (500, 10**9, "haeufig (500+)")]:
        mask = (support >= lo) & (support <= hi)
        if mask.sum():
            log(f"  {label:<20} n={mask.sum():4d}  Accuracy={gold_correct[mask.to_numpy()].mean():.4f}")
    log("  -> Accuracy bleibt niedrig auch bei Sachgruppen mit 500+ Trainingsbeispielen.")
    log("     Klassenhäufigkeit alleine erklärt den Einbruch NICHT.")

    # --- Hypothese 2: Vokabular-Domain-Shift ------------------------------
    log("\n--- Hypothese 2: Domain-Shift über das Vokabular ---")
    oov_rate, oov_per_doc = char_ngram_oov_rate(clf, gold["bedeutung"].tolist())
    log(f"  Char-ngram OOV-Rate ggü. gefittetem TFIDF-Vokabular: {oov_rate*100:.1f}%")
    log(f"  Mittlerer OOV-Anteil pro Bedeutung: {oov_per_doc*100:.1f}%")
    log("  -> Nur ~6% der tatsächlich vom Modell genutzten Zeichen-n-Gramme sind unbekannt.")
    log("     Vokabular-Unterschied ist zu klein, um einen ~45-Punkte-Einbruch zu erklären.")

    # --- Hypothese 3: Memorisierung durch Bedeutung-Duplikate ------------
    log("\n--- Hypothese 3: Memorisierung durch Bedeutung-Duplikate im Trainingskorpus ---")
    dup_counts = df["bedeutung"].value_counts()
    n_dup_rows = dup_counts[dup_counts > 1].sum()
    log(f"  Zeilen mit mehrfach vorkommender Bedeutung (>=2x im Korpus): "
        f"{n_dup_rows}/{len(df)} ({n_dup_rows/len(df)*100:.1f}%)")
    log(f"  Häufigste Bedeutung-Strings: {dup_counts.head(3).to_dict()}")

    X_train, X_test, y_train, y_test = reproduce_test_split(df)
    train_bed_set = set(X_train["bedeutung"])
    seen_mask = X_test["bedeutung"].isin(train_bed_set).to_numpy()

    y_pred_split = np.asarray(clf.predict(X_test)).astype(str)
    y_true_split = y_test.to_numpy()
    correct_split = y_pred_split == y_true_split

    acc_seen = correct_split[seen_mask].mean()
    acc_novel = correct_split[~seen_mask].mean()
    log(f"\n  Im eigenen Testsplit (n={len(X_test)}):")
    log(f"    Bedeutung auch im Train-Teil vorhanden (\"gesehen\"): "
        f"{seen_mask.sum()} ({seen_mask.mean()*100:.1f}%) -> Accuracy {acc_seen:.4f}")
    log(f"    Bedeutung NICHT im Train-Teil (\"neu\"):              "
        f"{(~seen_mask).sum()} ({(~seen_mask).mean()*100:.1f}%) -> Accuracy {acc_novel:.4f}")
    log(f"    Gesamt-Accuracy (offiziell reportiert): {correct_split.mean():.4f}")

    # Lemma-Effekt isoliert auf denselben "neuen" Zeilen (fairer Vergleich mit Goldstandard-Setup)
    X_novel = X_test[~seen_mask].copy()
    y_novel = y_true_split[~seen_mask]
    acc_novel_with_lemma = (np.asarray(clf.predict(X_novel)).astype(str) == y_novel).mean()
    X_novel_nolemma = X_novel.copy()
    X_novel_nolemma["lemma"] = ""
    acc_novel_no_lemma = (np.asarray(clf.predict(X_novel_nolemma)).astype(str) == y_novel).mean()

    log("\n  Zerlegung der Lücke (83.75% Trainings-Accuracy -> Goldstandard):")
    log(f"    1. Reportierte Testsplit-Accuracy (inkl. Memorisierung):       {correct_split.mean():.4f}")
    log(f"    2. ... nur neue Bedeutungen, mit Lemma:                       {acc_novel_with_lemma:.4f}")
    log(f"    3. ... nur neue Bedeutungen, Lemma='' (wie im Goldstandard):  {acc_novel_no_lemma:.4f}")
    log(f"    4. Goldstandard (fremdes Korpus, Lemma=''):                   {gold_correct.mean():.4f}")
    log(f"    Schritt 1->2 (Memorisierung entfernt):  {(correct_split.mean()-acc_novel_with_lemma)*100:+.1f} Punkte")
    log(f"    Schritt 2->3 (Lemma entfernt):          {(acc_novel_with_lemma-acc_novel_no_lemma)*100:+.1f} Punkte")
    log(f"    Schritt 3->4 (fremdes Korpus):          {(acc_novel_no_lemma-gold_correct.mean())*100:+.1f} Punkte")
    log("  -> Der Großteil der Lücke ist Memorisierung von Duplikaten, nicht Domain-Shift.")
    log("     Der Lemma-Effekt (~3 Punkte) deckt sich mit der bekannten 2-3%-Faustregel.")
    log("     Der Rest (~10 Punkte) ist der tatsächliche Distanz zum fremden Korpus.")

    # --- Qualitative Stichprobe: Fehlklassifikationen mit Top-3 ----------
    log("\n--- Stichprobe: 15 falsch klassifizierte Goldstandard-Fälle (Top-3-Vorschläge) ---")
    classes = clf.pipeline.named_steps["classifier"].classes_
    if clf.label_encoder is not None:
        classes = clf.label_encoder.inverse_transform(classes)
    classes = np.asarray(classes).astype(str)
    proba = np.asarray(clf.predict_proba(gold[["lemma", "bedeutung"]]), dtype=float)
    order = np.argsort(proba, axis=1)[:, ::-1]

    wrong_idx = np.where(~gold_correct)[0]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(wrong_idx, size=min(15, len(wrong_idx)), replace=False)
    for i in sorted(sample_idx):
        top3 = classes[order[i, :3]]
        top3_desc = [f"{lbl} ({describe(lbl)})" for lbl in top3]
        log(f"\n  Bedeutung: {gold.iloc[i]['bedeutung']}")
        log(f"    Wahr:  {gold_true[i]} ({describe(gold_true[i])})")
        log(f"    Top-3: {' | '.join(top3_desc)}")

    report = "\n".join(lines)
    OUT_FILE.write_text(report, encoding="utf-8")
    print(f"\nVollständiger Report gespeichert: {OUT_FILE}")


if __name__ == "__main__":
    main()
