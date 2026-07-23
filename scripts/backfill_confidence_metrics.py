#!/usr/bin/env python3
"""
Einmalig: berechnet fuer bereits trainierte Modelle die echten Top-k-Konfidenz-
Werte (top{k}_conf) nach und schreibt sie in die vorhandene *_metadata.json und
den Top-k-Block in *_report.txt -- ohne Neutraining.

Grund: _extended_metrics() lieferte bisher nur Top-k-*Accuracy* unter dem
Schluessel "top{k}", die im Frontend faelschlich als "Top{k}-Konfidenz"
angezeigt wurde (identisch zur Accuracy-Spalte). Der Fix berechnet zusaetzlich
echte Konfidenz (mittlere Wahrscheinlichkeitsmasse auf den Top-k-Vorschlaegen).

Reproduziert den Test-Split exakt wie beim Training (siehe reproduce_test_split)
und ruft dieselbe _extended_metrics()/_format_extended_metrics()-Logik auf, die
auch beim naechsten regulaeren Training verwendet wird.

Ordnet dabei auch die Reihenfolge im Report neu: TOP-k-Block (und Cross-Validierung,
falls vorhanden) vor die klassenweise Tabelle unter "KOMPLETTE AUSWERTUNG" -- vorher
stand die lange Tabelle zuerst, die Kurzuebersicht am Ende der Datei.

Beispiel:
    python scripts/backfill_confidence_metrics.py --models-dir models
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402
from scripts.evaluate_topk import reproduce_test_split  # noqa: E402

TOPK_BLOCK_RE = re.compile(
    # Header-Titel matcht per Substring "KONFIDENZ" statt exaktem String, damit das
    # Skript alte Report-Titel (vor Umbenennung) findet und ersetzt, und bei
    # kuenftigen Titelaenderungen nicht erneut angepasst werden muss.
    r"={60}\n[^\n]*KONFIDENZ[^\n]*\n={60}\n.*?(?=\n\n={60}\n|\Z)",
    re.DOTALL,
)
CV_BLOCK_RE = re.compile(
    r"={60}\nCROSS-VALIDIERUNG\n={60}\n.*?(?=\n\n={60}\n|\Z)",
    re.DOTALL,
)
TABLE_HEADER_RE = re.compile(r"^={60}\nKOMPLETTE AUSWERTUNG\n={60}\n+", re.DOTALL)


def resolve_csv(recorded_path: str, repo_root: Path) -> Path | None:
    """Der in der Metadata gespeicherte Pfad stammt oft von einer anderen
    Maschine/Session (z. B. /tmp/ml_session_.../datei.csv). Der Dateiname
    allein reicht: dieselbe Datei liegt unveraendert unter data/ oder
    .sessions/training/."""
    name = Path(recorded_path).name
    for candidate in (repo_root / "data" / name, repo_root / ".sessions" / "training" / name):
        if candidate.exists():
            return candidate
    direct = Path(recorded_path)
    if direct.exists():
        return direct
    return None


def backfill_one(pkl_path: Path, repo_root: Path, dry_run: bool) -> None:
    meta_path = pkl_path.with_name(pkl_path.stem + "_metadata.json")
    report_path = pkl_path.with_name(pkl_path.stem + "_report.txt")

    if not meta_path.exists():
        print(f"  -> uebersprungen (keine Metadata-Datei)")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    csv_path = resolve_csv(meta.get("csv_file", ""), repo_root)
    if csv_path is None:
        print(f"  -> uebersprungen (CSV nicht auffindbar: {meta.get('csv_file')})")
        return

    test_size = meta.get("test_size", 0.2)
    X_test, y_test, _ = reproduce_test_split(str(csv_path), test_size)

    clf = SachgruppenClassifier.load(str(pkl_path))
    new_metrics = clf._extended_metrics(X_test, y_test)

    old_top1 = (meta.get("topk_metrics") or {}).get("top1")
    recorded_acc = meta.get("accuracy")
    if old_top1 is not None and abs(new_metrics["top1"] - old_top1) > 1e-6:
        print(f"  !! WARNUNG: neu berechnetes top1 ({new_metrics['top1']:.4f}) weicht von "
              f"gespeichertem top1 ({old_top1:.4f}) ab -- CSV/Split evtl. nicht identisch, "
              f"Ergebnis wird trotzdem geschrieben (bitte pruefen).")
    elif recorded_acc is not None and abs(new_metrics["top1"] - recorded_acc) > 1e-6:
        print(f"  !! WARNUNG: neu berechnete Top-1-Accuracy ({new_metrics['top1']:.4f}) weicht "
              f"von gespeicherter accuracy ({recorded_acc:.4f}) ab.")

    conf_str = (
        f"top1_conf={new_metrics.get('top1_conf'):.4f} top3_conf={new_metrics.get('top3_conf'):.4f} "
        f"top5_conf={new_metrics.get('top5_conf'):.4f}"
        if new_metrics.get("has_proba")
        else "keine Wahrscheinlichkeiten (has_proba=False)"
    )
    print(f"  -> {conf_str}")

    if dry_run:
        return

    meta_path.with_suffix(".json.bak").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta["topk_metrics"] = new_metrics
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        topk_match = TOPK_BLOCK_RE.search(text)
        if topk_match is None:
            print(f"  !! TOP-k-Block nicht in {report_path.name} gefunden, Report unveraendert.")
            return
        cv_match = CV_BLOCK_RE.search(text)

        # Tabelle = Restlicher Text nach Entfernen von TOPK-/CV-Block (Positionen
        # koennen je nach Dateistand vor/nach der Tabelle liegen -- absteigend nach
        # Startposition entfernen, damit sich Indizes nicht verschieben) und eines
        # evtl. schon vorhandenen "KOMPLETTE AUSWERTUNG"-Headers (idempotent bei
        # erneutem Lauf nach einer frueheren Migration).
        remainder = text
        for m in sorted((x for x in (topk_match, cv_match) if x), key=lambda x: x.start(), reverse=True):
            remainder = remainder[:m.start()] + remainder[m.end():]
        table_str = TABLE_HEADER_RE.sub("", remainder.strip("\n")).strip("\n")

        new_topk_block = SachgruppenClassifier._format_extended_metrics(new_metrics)
        cv_block = cv_match.group(0).strip("\n") if cv_match else ""
        labeled_table = f"{'=' * 60}\nKOMPLETTE AUSWERTUNG\n{'=' * 60}\n\n{table_str}"
        new_text = "\n\n".join(p for p in (new_topk_block, cv_block, labeled_table) if p)

        report_path.with_suffix(".txt.bak").write_text(text, encoding="utf-8")
        report_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--dry-run", action="store_true",
                     help="Nur berechnen und anzeigen, nichts schreiben")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / args.models_dir

    for pkl_path in sorted(models_dir.glob("*.pkl")):
        print(f"\n{pkl_path.name}")
        try:
            backfill_one(pkl_path, repo_root, args.dry_run)
        except Exception as e:
            print(f"  !! Fehler: {e}")


if __name__ == "__main__":
    main()
