"""
Gepaarte Signifikanztests fuer zwei Klassifikator-Varianten auf demselben Testset.

Zwei Tests, je nach Metrik:
  - mcnemar_test():      Top-1-Accuracy (binaer richtig/falsch pro Beispiel)
  - paired_bootstrap():  beliebige Metrik (z.B. macro-F1), per Testset-Resampling

Beide nutzen aus, dass Baseline und Variante auf denselben Testbeispielen
ausgewertet werden (gepaarter Vergleich) statt sie wie unabhaengige Stichproben
zu behandeln -- das ist der uebliche Fehler, der reale Unterschiede als
"Rauschen" erscheinen laesst.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def mcnemar_test(y_true, pred_a, pred_b):
    """Exakter McNemar-Test auf den diskordanten Faellen (a richtig/b falsch vs. umgekehrt)."""
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    p_value = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value}


def paired_bootstrap(y_true, pred_a, pred_b, metric_fn, n_boot=2000, seed=42):
    """95%-CI und p-Wert fuer die Differenz metric(b) - metric(a) per Testset-Resampling."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    observed_diff = metric_fn(y_true, pred_b) - metric_fn(y_true, pred_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(y_true[idx], pred_b[idx]) - metric_fn(y_true[idx], pred_a[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_value = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)
    return {"observed_diff": float(observed_diff), "ci95": (float(lo), float(hi)),
            "p_value": float(p_value)}


def print_significance(label_a, label_b, y_true, pred_a, pred_b, macro_f1_fn, n_boot=2000):
    """Druckt McNemar (Top-1) + paired Bootstrap (macro-F1) fuer ein Variantenpaar."""
    mc = mcnemar_test(y_true, pred_a, pred_b)
    print(f"\n  McNemar Top-1  [{label_a} vs. {label_b}]: "
          f"diskordant b={mc['b']} (nur {label_a} richtig), c={mc['c']} (nur {label_b} richtig), "
          f"p={mc['p_value']:.4g}"
          + ("  *** signifikant (p<0.05)" if mc['p_value'] < 0.05 else "  n.s."))

    bs = paired_bootstrap(y_true, pred_a, pred_b, macro_f1_fn, n_boot=n_boot)
    print(f"  Paired Bootstrap macro-F1 [{label_b} - {label_a}], n_boot={n_boot}: "
          f"D={bs['observed_diff']*100:+.2f}pp  95%-CI=[{bs['ci95'][0]*100:+.2f}, {bs['ci95'][1]*100:+.2f}]pp  "
          f"p={bs['p_value']:.4g}"
          + ("  *** signifikant (p<0.05)" if bs['p_value'] < 0.05 else "  n.s."))
    return mc, bs


def format_mcnemar_report(label_a, label_b, acc_a, acc_b, mc, n_total, testset_desc):
    """Formatiert das McNemar-Ergebnis als lesbaren Textblock mit Erklaerung
    (fuer data/statistical_significance.txt)."""
    b, c, n_disc, p = mc["b"], mc["c"], mc["n_discordant"], mc["p_value"]
    diff_pp = (acc_b - acc_a) * 100
    sig = p < 0.05

    lines = []
    lines.append("=" * 78)
    lines.append(f"Statistischer Signifikanztest (McNemar) -- {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Vergleich:")
    lines.append(f"  Modell A (Baseline):   {label_a}")
    lines.append(f"                          Accuracy: {acc_a:.4f}")
    lines.append(f"  Modell B (Vergleich):  {label_b}")
    lines.append(f"                          Accuracy: {acc_b:.4f}")
    lines.append(f"  Testset:               {testset_desc} ({n_total} Beispiele)")
    lines.append(f"  Accuracy-Differenz:    {diff_pp:+.2f} Prozentpunkte")
    lines.append("")
    lines.append("Was misst McNemar?")
    lines.append("  Der Test prueft, ob zwei Modelle auf DENSELBEN Testbeispielen unter-")
    lines.append("  schiedlich gut abschneiden. Faelle, in denen beide Modelle einig sind")
    lines.append("  (beide richtig oder beide falsch), tragen nichts zur Aussage bei -- nur")
    lines.append("  die Faelle, in denen sie sich widersprechen (\"diskordant\"), zaehlen.")
    lines.append("  Unter der Nullhypothese (kein echter Unterschied) sollten sich diese")
    lines.append("  Faelle etwa 50/50 auf \"nur A richtig\" und \"nur B richtig\" verteilen.")
    lines.append("")
    lines.append("Diskordante Faelle:")
    lines.append(f"  b = {b:>6}  (nur {label_a} richtig)")
    lines.append(f"  c = {c:>6}  (nur {label_b} richtig)")
    lines.append(f"  n = b + c = {n_disc}")
    lines.append("")
    lines.append("Ergebnis:")
    lines.append(f"  p-Wert = {p:.4g}")
    if sig:
        lines.append("  *** SIGNIFIKANT (p < 0.05) ***")
        lines.append(f"  Die Aufteilung {b} vs. {c} weicht so stark von 50/50 ab, dass die")
        lines.append(f"  gemessene Differenz von {diff_pp:+.2f} PP nicht durch Zufall/Testset-")
        lines.append(f"  Rauschen erklaerbar ist -- {label_b} ist mit hoher Sicherheit echt")
        lines.append(f"  {'besser' if diff_pp > 0 else 'schlechter'} als {label_a}.")
    else:
        lines.append("  n.s. (p >= 0.05)")
        lines.append(f"  Die Aufteilung {b} vs. {c} ist mit einer fairen 50/50-Verteilung noch")
        lines.append(f"  vertraeglich -- die gemessene Differenz von {diff_pp:+.2f} PP koennte")
        lines.append("  Zufall/Testset-Rauschen sein, statt eines echten Unterschieds.")
    lines.append("")
    lines.append("=" * 78)
    lines.append("")
    return "\n".join(lines)


REPORT_HEADER = """\
================================================================================
McNemar-Signifikanztests -- automatisch generierte Reports
================================================================================

Workflow zur Erzeugung eines neuen Eintrags in dieser Datei:

  python scripts/predict_testset.py --models <a.pkl> <b.pkl> \\
      --csv data/baseline_test.csv --out data/predictions_XY.csv
  python scripts/significance.py --predictions data/predictions_XY.csv \\
      --model-a <spaltenname_a> --model-b <spaltenname_b>

Jeder Aufruf haengt unten einen neuen Report an (ueberschreibt nichts).

================================================================================

"""


def main():
    ap = argparse.ArgumentParser(
        description="Fuehrt den McNemar-Test fuer zwei Modellspalten einer "
                     "predict_testset.py-Ausgabe aus und haengt einen lesbaren "
                     "Report an eine Textdatei an.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--predictions", required=True,
                     help="CSV von scripts/predict_testset.py (sachgruppe + eine Spalte je Modell)")
    ap.add_argument("--model-a", required=True, help="Spaltenname der Baseline in der CSV")
    ap.add_argument("--model-b", required=True, help="Spaltenname der Vergleichsvariante in der CSV")
    ap.add_argument("--out", default="data/statistical_significance.txt",
                     help="Zieldatei fuer den Report (wird angehaengt, nicht ueberschrieben)")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    y_true = df["sachgruppe"].astype(str).to_numpy()
    pred_a = df[args.model_a].astype(str).to_numpy()
    pred_b = df[args.model_b].astype(str).to_numpy()

    acc_a = float((pred_a == y_true).mean())
    acc_b = float((pred_b == y_true).mean())
    mc = mcnemar_test(y_true, pred_a, pred_b)

    report = format_mcnemar_report(
        args.model_a, args.model_b, acc_a, acc_b, mc,
        n_total=len(y_true), testset_desc=args.predictions,
    )

    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    with open(out_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(REPORT_HEADER)
        f.write(report)
    print(f"Report angehaengt an: {out_path}")


if __name__ == "__main__":
    main()
