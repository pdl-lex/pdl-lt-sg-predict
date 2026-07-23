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
    r"={60}\nTOP-k / HIERARCHIE / KONFIDENZ\n={60}\n.*?(?=\n\n={60}\n|\Z)",
    re.DOTALL,
)


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
        new_block = SachgruppenClassifier._format_extended_metrics(new_metrics)
        if TOPK_BLOCK_RE.search(text):
            report_path.with_suffix(".txt.bak").write_text(text, encoding="utf-8")
            text = TOPK_BLOCK_RE.sub(lambda _m: new_block, text, count=1)
            report_path.write_text(text, encoding="utf-8")
        else:
            print(f"  !! TOP-k-Block nicht in {report_path.name} gefunden, Report unveraendert.")


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
