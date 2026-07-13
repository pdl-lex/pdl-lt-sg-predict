"""Vergleicht die Sachgruppen-Verteilung des Goldstandards (test/shuffle_pfwb1000_mit_sg.txt)
mit (a) der Verteilung aller Trainingsdaten (CSV des besten Modells) und (b) der
Verteilung im Testdaten-Split desselben Modells (classification_report-Support,
dieselbe Quelle wie die "Sachgruppen"-Seite der App).

Ausgabe: scripts/goldstandard_vs_testdaten.png
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pdl_lt_sg_predict.core.bridge import MODELS_DIR, sachgruppen_map  # noqa: E402
from pdl_lt_sg_predict.core.models import best_model  # noqa: E402
from pdl_lt_sg_predict.core.sachgruppen import _parse_report  # noqa: E402

GOLD_FILE = REPO_ROOT / "test" / "shuffle_pfwb1000_mit_sg.txt"
OUT_FILE = Path(__file__).resolve().parent / "goldstandard_vs_testdaten.png"

COLOR_ALL = "#2a78d6"    # blue   - alle Trainingsdaten
COLOR_TEST = "#1baf7a"   # aqua   - Testdaten-Split
COLOR_GOLD = "#eb6834"   # orange - Goldstandard


def load_goldstandard_counts() -> Counter:
    counts: Counter = Counter()
    for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            continue
        counts[parts[1].strip()] += 1
    return counts


def load_testdata_support(best: dict) -> dict[str, int]:
    report_path = MODELS_DIR / f"{best['stem']}_report.txt"
    report = _parse_report(report_path.read_text(encoding="utf-8"))
    return {label: m["support"] for label, m in report.items()}


def load_all_data_counts(best: dict) -> dict[str, int]:
    """Sachgruppen-Häufigkeit über die komplette Trainings-CSV des besten Modells."""
    csv_name = Path(best.get("csv_file", "")).name
    candidates = [
        REPO_ROOT / "data" / csv_name,
        REPO_ROOT / csv_name,
        REPO_ROOT / ".sessions" / "training" / csv_name,
    ]
    csv_path = next((p for p in candidates if csv_name and p.exists()), None)
    if csv_path is None:
        raise SystemExit(f"Trainings-CSV '{csv_name}' des besten Modells nicht gefunden.")
    df = pd.read_csv(csv_path, sep=None, engine="python", dtype={"sachgruppe": str})
    return df["sachgruppe"].str.strip().value_counts().to_dict()


def main() -> None:
    best = best_model()
    if not best:
        raise SystemExit("Kein trainiertes Modell gefunden.")

    # best_model() liefert kein csv_file -> Metadaten direkt nachladen.
    import json
    meta = json.loads((MODELS_DIR / f"{best['stem']}_metadata.json").read_text(encoding="utf-8"))
    best["csv_file"] = meta.get("csv_file", "")

    gold_counts = load_goldstandard_counts()
    test_support = load_testdata_support(best)
    all_counts = load_all_data_counts(best)
    sg_names = sachgruppen_map()

    total_gold = sum(gold_counts.values())
    total_test = sum(test_support.values())
    total_all = sum(all_counts.values())

    # Sachgruppen nach Häufigkeit über ALLE Trainingsdaten sortiert (Rang 1 = häufigste).
    all_labels = sorted(all_counts, key=lambda l: all_counts[l], reverse=True)
    ranks = list(range(1, len(all_labels) + 1))
    all_share = [all_counts[l] / total_all * 100 for l in all_labels]
    test_share = [test_support.get(l, 0) / total_test * 100 if l in test_support else None for l in all_labels]
    gold_share = [gold_counts.get(l, 0) / total_gold * 100 for l in all_labels]

    fig, (ax_rank, ax_bar) = plt.subplots(2, 1, figsize=(12, 10))

    # --- Panel 1: Rang-Häufigkeits-Kurven + Goldstandard-Fälle darauf ---
    ax_rank.plot(
        ranks, all_share, color=COLOR_ALL, lw=2,
        label=f"Alle Trainingsdaten (n={total_all:,})".replace(",", "."),
    )
    test_ranks = [r for r, v in zip(ranks, test_share) if v is not None]
    test_vals = [v for v in test_share if v is not None]
    ax_rank.plot(
        test_ranks, test_vals, color=COLOR_TEST, lw=1.5, ls="--",
        label=f"Testdaten-Split ({best['model_name']}, n={total_test:,}".replace(",", ".") + ")",
    )
    hit_ranks = [r for r, l in zip(ranks, all_labels) if gold_counts.get(l, 0) > 0]
    hit_share = [gold_share[r - 1] for r in hit_ranks]
    ax_rank.scatter(
        hit_ranks, hit_share, color=COLOR_GOLD, s=26, zorder=3, edgecolors="none",
        label=f"Goldstandard (n={total_gold})",
    )
    miss = len(all_labels) - len(hit_ranks)
    ax_rank.set_yscale("log")
    ax_rank.set_xlabel("Sachgruppen-Rang (sortiert nach Häufigkeit über alle Trainingsdaten, häufigste zuerst)")
    ax_rank.set_ylabel("Anteil an allen Fällen (%, log)")
    ax_rank.set_title("Wo liegen die Goldstandard-Fälle auf der Sachgruppen-Verteilung der Daten?")
    ax_rank.grid(alpha=0.3, which="both")
    ax_rank.legend()
    ax_rank.annotate(
        f"{miss} Sachgruppen ohne Goldstandard-Beleg",
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=9, color="#52514e",
    )

    # --- Panel 2: Balkendiagramm Top-N Sachgruppen, Anteile im Vergleich ---
    top_n = 25
    top_labels = all_labels[:top_n]
    x = list(range(len(top_labels)))
    width = 0.27
    ax_bar.bar(
        [i - width for i in x], [all_counts[l] / total_all * 100 for l in top_labels],
        width=width, color=COLOR_ALL, label="Alle Trainingsdaten",
    )
    ax_bar.bar(
        x, [test_support.get(l, 0) / total_test * 100 for l in top_labels],
        width=width, color=COLOR_TEST, label="Testdaten-Split",
    )
    ax_bar.bar(
        [i + width for i in x], [gold_counts.get(l, 0) / total_gold * 100 for l in top_labels],
        width=width, color=COLOR_GOLD, label="Goldstandard",
    )
    labels = [f"{l}\n{sg_names.get(l, '')[:14]}" for l in top_labels]
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax_bar.set_ylabel("Anteil an allen Fällen (%)")
    ax_bar.set_title(f"Top {top_n} Sachgruppen: Anteil in allen Daten vs. Testdaten-Split vs. Goldstandard")
    ax_bar.grid(alpha=0.3, axis="y")
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"Diagramm gespeichert: {OUT_FILE}")

    covered = sum(1 for l in gold_counts if l in all_counts)
    print(f"Goldstandard: {len(gold_counts)} distinkte Sachgruppen, {covered} davon in den Trainingsdaten vertreten.")
    print(f"Alle Trainingsdaten: {len(all_labels)} distinkte Sachgruppen, {miss} davon ohne Goldstandard-Beleg (meist selten/im Tail).")


if __name__ == "__main__":
    main()
