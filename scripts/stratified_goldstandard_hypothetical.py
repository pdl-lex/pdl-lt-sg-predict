"""Hypothetischer Vergleich: Wie sähe der obere Plot aus goldstandard_vs_testdaten.png aus,
wenn der Goldstandard ein *stratifizierter* 1000er-Sample der Trainingsverteilung wäre
(statt ein fremdes Korpus, pfwb1000)?

Erzeugt eine ZWEI-Panel-Grafik (Original wird NICHT überschrieben):
  oben:  Ist-Zustand  - echter Goldstandard auf der Rang-Häufigkeits-Kurve (wie im Original)
  unten: Hypothese     - ein aus der Trainingsverteilung multinomial gezogener 1000er-Sample

Kernaussagen, die der untere Plot sichtbar macht:
  1. Stratifiziert liegen die Punkte ENG auf der Kurve (kein Streuen ± wie beim fremden Korpus).
  2. Quantisierungs-Boden bei 1/1000 = 0,1 %: die Kurve fällt rechts weit darunter, ein 1000er-
     Sample kann dem nicht folgen -> ab dem Rang, wo die Kurve 0,1 % unterschreitet, entsteht
     ein flaches Band bei 0,1 % (genau 1 Beleg) plus viele Lücken (0 Belege).
  3. Die Zahl der Sachgruppen ohne Beleg steigt eher, statt zu sinken.

Separates Analyse-Skript, unabhängig vom Haupt-Code der App.

Ausgabe: scripts/stratified_goldstandard_hypothetical.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pdl_lt_sg_predict.core.bridge import MODELS_DIR  # noqa: E402
from pdl_lt_sg_predict.core.models import best_model  # noqa: E402
from compare_goldstandard_distribution import (  # noqa: E402
    COLOR_ALL,
    COLOR_GOLD,
    load_all_data_counts,
    load_goldstandard_counts,
)

OUT_FILE = Path(__file__).resolve().parent / "stratified_goldstandard_hypothetical.png"

COLOR_STRAT = "#8e44ad"  # violett - stratifizierter Hypothesen-Sample
N_SAMPLE = 1000
SEED = 42


def _rank_panel(ax, ranks, all_share, floor_pct, title):
    ax.plot(ranks, all_share, color=COLOR_ALL, lw=2,
            label=f"Alle Trainingsdaten (n={sum_all:,})".replace(",", "."))
    ax.axhline(floor_pct, color="#999", lw=1, ls=":",
               label=f"Boden bei n={N_SAMPLE}: 1 Beleg = {floor_pct:g} %")
    ax.set_yscale("log")
    ax.set_xlabel("Sachgruppen-Rang (sortiert nach Häufigkeit über alle Trainingsdaten, häufigste zuerst)")
    ax.set_ylabel("Anteil an allen Fällen (%, log)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")


def main() -> None:
    best = best_model()
    if not best:
        raise SystemExit("Kein trainiertes Modell gefunden.")
    meta = json.loads((MODELS_DIR / f"{best['stem']}_metadata.json").read_text(encoding="utf-8"))
    best["csv_file"] = meta.get("csv_file", "")

    all_counts = load_all_data_counts(best)
    gold_counts = load_goldstandard_counts()

    global sum_all
    all_labels = sorted(all_counts, key=lambda l: all_counts[l], reverse=True)
    sum_all = sum(all_counts.values())
    ranks = list(range(1, len(all_labels) + 1))
    all_share = [all_counts[l] / sum_all * 100 for l in all_labels]
    p = np.array([all_counts[l] / sum_all for l in all_labels])

    floor_pct = 100.0 / N_SAMPLE  # 0,1 %

    # Rang, an dem die Trainingskurve den 0,1 %-Boden unterschreitet.
    below = [r for r, s in zip(ranks, all_share) if s < floor_pct]
    cross_rank = below[0] if below else None

    # --- Ist: echter Goldstandard --------------------------------------------
    total_gold = sum(gold_counts.values())
    gold_hit = [(r, gold_counts[l] / total_gold * 100)
                for r, l in zip(ranks, all_labels) if gold_counts.get(l, 0) > 0]
    gold_miss = len(all_labels) - len(gold_hit)

    # --- Hypothese: stratifizierter multinomial-Sample -----------------------
    rng = np.random.default_rng(SEED)
    strat_counts = rng.multinomial(N_SAMPLE, p)
    strat_hit = [(r, c / N_SAMPLE * 100) for r, c in zip(ranks, strat_counts) if c > 0]
    strat_miss = len(all_labels) - len(strat_hit)

    fig, (ax_ist, ax_hyp) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    _rank_panel(ax_ist, ranks, all_share, floor_pct,
                "IST: Goldstandard (fremdes Korpus pfwb1000) — Punkte streuen ± um die Kurve")
    gx, gy = zip(*gold_hit)
    ax_ist.scatter(gx, gy, color=COLOR_GOLD, s=26, zorder=3, edgecolors="none",
                   label=f"Goldstandard (n={total_gold})")
    ax_ist.legend(loc="lower left")
    ax_ist.annotate(f"{gold_miss} Sachgruppen ohne Beleg",
                    xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color="#52514e")

    _rank_panel(ax_hyp, ranks, all_share, floor_pct,
                f"HYPOTHESE: stratifizierter {N_SAMPLE}er-Sample der Trainingsverteilung "
                "— Punkte liegen auf der Kurve")
    sx, sy = zip(*strat_hit)
    ax_hyp.scatter(sx, sy, color=COLOR_STRAT, s=26, zorder=3, edgecolors="none",
                   label=f"Stratifizierter Sample (n={N_SAMPLE}, seed={SEED})")
    if cross_rank is not None:
        ax_hyp.axvline(cross_rank, color=COLOR_STRAT, lw=1, ls="--", alpha=0.6)
        ax_hyp.annotate(f"ab hier Kurve < {floor_pct:g} %\n→ flaches Band + Lücken",
                        xy=(cross_rank, floor_pct), xytext=(cross_rank + 12, floor_pct * 3),
                        fontsize=9, color=COLOR_STRAT,
                        arrowprops=dict(arrowstyle="->", color=COLOR_STRAT, lw=1))
    ax_hyp.legend(loc="lower left")
    ax_hyp.annotate(f"{strat_miss} Sachgruppen ohne Beleg",
                    xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color="#52514e")

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"Diagramm gespeichert: {OUT_FILE}")

    print(f"Trainings-Sachgruppen gesamt:        {len(all_labels)}")
    print(f"Kurve unterschreitet 0,1 % ab Rang:  {cross_rank}")
    print(f"Ist  (Goldstandard):   {len(gold_hit)} belegt, {gold_miss} ohne Beleg")
    print(f"Hyp. (stratifiziert):  {len(strat_hit)} belegt, {strat_miss} ohne Beleg")
    n_at_floor = int((strat_counts == 1).sum())
    print(f"Hyp.: davon genau 1 Beleg (auf der 0,1 %-Linie): {n_at_floor}")


if __name__ == "__main__":
    main()
