"""Trainings-Orchestrierung.

Startet ``sachgruppen_classifier.py`` als Subprozess (überlebt Reloads des
Backends) und verfolgt den Fortschritt über eine JSON-Fortschrittsdatei. Es läuft
maximal ein Job (Einzel- oder Batch-Training) gleichzeitig.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .bridge import (
    CLASSIFIER_SCRIPT,
    ENABLE_TRAINING,
    MAX_FILE_SIZE,
    MODELS_DIR,
    SESSIONS_DIR,
)

# Fallback-Trainingszeiten (Sekunden) für ~113 127 Samples (Referenzmaschine).
_TIME_FALLBACKS: dict[str, float] = {
    "svm": 120.0, "logistic": 4286.0, "rf": 30.0, "nn": 111.0, "xgboost": 6112.0,
}
_TIME_FALLBACK_SAMPLES = 113_127

_TRAIN_DIR = SESSIONS_DIR / "training"
_TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def _historical_time_per_type(total_samples: int) -> dict[str, float]:
    """Jüngste gemessene Trainingszeit je Modelltyp, skaliert auf die aktuelle Datenmenge."""
    if total_samples <= 0 or not MODELS_DIR.exists():
        return {}
    best: dict[str, tuple[str, float, int]] = {}  # type -> (timestamp, time, samples)
    for mf in MODELS_DIR.glob("*_metadata.json"):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mt, t, n, ts = m.get("model_type", ""), float(m.get("training_time", 0)), int(m.get("num_samples", 0)), m.get("timestamp", "")
        if mt and t > 0 and n > 0 and (mt not in best or ts > best[mt][0]):
            best[mt] = (ts, t, n)
    return {mt: t / n * total_samples for mt, (_, t, n) in best.items()}


class TrainingManager:
    """Verwaltet den (einzelnen) laufenden Trainings-Subprozess."""

    def __init__(self) -> None:
        self.csv_path: Path | None = None
        self.csv_info: dict = {}
        self.time_per_type: dict[str, float] = {}
        self._proc: subprocess.Popen | None = None
        self._job: dict | None = None

    # ── Daten-Upload ────────────────────────────────────────────────────────
    def upload_csv(self, filename: str, content: bytes) -> dict:
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(f"Datei zu groß (max {MAX_FILE_SIZE // 1024 // 1024} MB).")
        safe = Path(filename).name
        path = _TRAIN_DIR / safe
        path.write_bytes(content)

        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"CSV konnte nicht gelesen werden: {e}") from e
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
        for col in ("bedeutung", "sachgruppe"):
            if col not in df.columns:
                raise ValueError("CSV muss die Spalten 'bedeutung' und 'sachgruppe' enthalten (Spalte 'lemma' optional).")

        self.csv_path = path
        self.csv_info = {
            "filename": safe,
            "num_samples": int(len(df)),
            "num_classes": int(df["sachgruppe"].nunique()),
        }
        self.time_per_type = _historical_time_per_type(self.csv_info["num_samples"])
        return {**self.csv_info, "time_per_type": self.time_per_type}

    # ── Job-Status ──────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _base_cmd(self, progress_file: Path, cfg: dict) -> list[str]:
        # word_ngram_max kommt vorbereitet aus start_single/start_batch — hier
        # nicht mehr an analyzer_mode koppeln (im Batch laufen char_wb UND word).
        word_ngram_max = cfg.get("word_ngram_max", 1)
        cmd = [
            sys.executable, str(CLASSIFIER_SCRIPT),
            "--csv", str(self.csv_path),
            "--test-size", str(cfg.get("test_size", 0.2)),
            "--word-ngram-max", str(word_ngram_max),
            "--svm-c", str(cfg.get("svm_c", 1.0)),
            "--xgb-n-estimators", str(cfg.get("xgb_n_estimators", 300)),
            "--xgb-max-depth", str(cfg.get("xgb_max_depth", 6)),
            "--xgb-learning-rate", str(cfg.get("xgb_learning_rate", 0.05)),
            "--xgb-subsample", str(cfg.get("xgb_subsample", 0.8)),
            "--nn-hidden-layers", str(cfg.get("nn_hidden_layers", "100")),
            "--nn-alpha", str(cfg.get("nn_alpha", 0.0001)),
            "--nn-learning-rate-init", str(cfg.get("nn_learning_rate_init", 0.0005)),
            "--output-dir", str(MODELS_DIR),
            "--progress-file", str(progress_file),
        ]
        if cfg.get("tune_mode") == "auto":
            cmd += ["--tune", "--tune-n-iter", str(cfg.get("tune_n_iter", 20)),
                    "--tune-cv", str(cfg.get("tune_cv", 3))]
        if cfg.get("use_spacy"):
            cmd.append("--use-spacy")
        if cfg.get("use_dornseiff"):
            cmd.append("--use-dornseiff")
        if cfg.get("calibrate"):
            cmd.append("--calibrate")
        return cmd

    def start_single(self, cfg: dict) -> dict:
        self._guard_start()
        progress_file = _TRAIN_DIR / "single_progress.json"
        progress_file.write_text('{"pct": 0, "msg": "Starte…", "done": false, "error": ""}')

        analyzer = cfg.get("analyzer_mode", "char_wb")
        if analyzer != "word":
            cfg = {**cfg, "word_ngram_max": 1}
        cmd = self._base_cmd(progress_file, cfg) + [
            "--model", cfg.get("model", "svm"),
            "--analyzer", analyzer,
            "--min-length", str(cfg.get("min_word_length", 1)),
            "--stopwords", "true" if cfg.get("use_stopword_removal") else "false",
        ]
        cv_folds = max(2, int(cfg.get("cv_folds", 5)))
        if cfg.get("cross_validate"):
            cv_mode = cfg.get("cv_mode", "stratified")
            cv_mode = cv_mode if cv_mode in ("stratified", "group") else "stratified"
            cmd += ["--cross-validate", "--cv-folds", str(cv_folds), "--cv-mode", cv_mode]

        # Zeitschätzung für die geglättete Fortschrittsanzeige.
        est = self.time_per_type.get(cfg.get("model", "svm"),
                                     _TIME_FALLBACKS.get(cfg.get("model", "svm"), 120.0))
        if cfg.get("calibrate") and cfg.get("model", "svm") in ("svm", "nn"):
            # CalibratedClassifierCV (cv=3, ensemble=False): 3 CV-Fits + 1 finaler Fit.
            est *= 3
        if cfg.get("tune_mode") == "auto":
            est *= max(cfg.get("tune_n_iter", 20) * cfg.get("tune_cv", 3) / 5, 1.0)
        if cfg.get("cross_validate"):
            # CV trainiert das Modell zusätzlich cv_folds-mal auf (annähernd) allen Daten.
            est *= 1 + cv_folds
        self._launch(cmd, progress_file, mode="single", estimated_fit_sec=max(est * 0.7, 1.0))
        return {"status": "started", "mode": "single"}

    def start_batch(self, cfg: dict) -> dict:
        self._guard_start()
        model_types = cfg.get("batch_model_types") or ["svm"]
        sw_vals = cfg.get("batch_use_stopwords", [False])
        min_lengths = cfg.get("batch_min_lengths", [1])
        analyzers_raw = cfg.get("batch_analyzers", ["char_wb"])
        analyzers = sorted({("word" if a.startswith("word") else "char_wb") for a in analyzers_raw})
        word_ngram_max = max((2 if a == "word-(1,2)" else 1) for a in analyzers_raw)
        total = len(model_types) * len(sw_vals) * len(min_lengths) * len(analyzers)

        progress_file = _TRAIN_DIR / "batch_progress.json"
        progress_file.write_text('{"pct": 0, "msg": "Starte…", "done": false, "config_idx": 0, "config_total": 0, "error": ""}')

        cfg = {**cfg, "word_ngram_max": word_ngram_max}
        cmd = self._base_cmd(progress_file, cfg)
        # --model/--analyzer/--min-length/--stopwords als Mehrfachwerte anhängen.
        cmd += ["--model", *model_types, "--analyzer", *analyzers,
                "--min-length", *[str(m) for m in min_lengths],
                "--stopwords", *["true" if s else "false" for s in sw_vals]]
        self._launch(cmd, progress_file, mode="batch", total=total)
        return {"status": "started", "mode": "batch", "total": total}

    def _guard_start(self) -> None:
        if not ENABLE_TRAINING:
            raise PermissionError("Training ist deaktiviert (ENABLE_TRAINING=False).")
        if self.running:
            raise RuntimeError("Es läuft bereits ein Training.")
        if not self.csv_path or not self.csv_path.exists():
            raise ValueError("Keine Trainingsdaten hochgeladen.")

    def _launch(self, cmd: list[str], progress_file: Path, *, mode: str,
                estimated_fit_sec: float = 1.0, total: int = 1) -> None:
        log_file = _TRAIN_DIR / f"{mode}_train.log"
        self._log_handle = log_file.open("w")
        self._proc = subprocess.Popen(cmd, stdout=self._log_handle, stderr=subprocess.STDOUT, text=True)
        self._job = {
            "mode": mode, "progress_file": progress_file, "log_file": log_file,
            "started_at": time.time(), "estimated_fit_sec": estimated_fit_sec,
            "total": total, "fit_started_at": None,
        }

    def status(self) -> dict:
        if self._job is None:
            return {"state": "idle"}
        job = self._job
        try:
            prog = json.loads(job["progress_file"].read_text())
        except (OSError, json.JSONDecodeError):
            prog = {}

        finished = self._proc is None or self._proc.poll() is not None
        if not finished:
            return self._running_status(job, prog)

        # Prozess beendet – Endergebnis auswerten.
        self._close_log()
        rc = self._proc.returncode if self._proc else 0
        if prog.get("error"):
            return self._finish({"state": "error", "error": prog["error"], "mode": job["mode"]})
        if rc not in (0, None):
            tail = self._log_tail(job["log_file"])
            return self._finish({"state": "error", "error": f"Subprozess-Exit {rc}:\n{tail}", "mode": job["mode"]})

        if job["mode"] == "batch":
            return self._finish({
                "state": "done", "mode": "batch",
                "done": job["total"], "total": job["total"], "pct": 100,
                "msg": "Batch-Training abgeschlossen.",
            })

        model_file = prog.get("model_file", "")
        result = {
            "state": "done", "mode": "single", "pct": 100,
            "msg": "Training abgeschlossen.",
            "model_file": Path(model_file).name if model_file else "",
            "accuracy": prog.get("accuracy", 0.0),
            "training_time": prog.get("training_time", 0.0),
        }
        if prog.get("cross_validation"):
            result["cross_validation"] = prog["cross_validation"]
        # Beste Parameter (Auto-Tune) aus Metadaten ergänzen.
        if model_file:
            meta_path = Path(model_file.replace(".pkl", "_metadata.json"))
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    bp = meta.get("best_params", {})
                    if bp:
                        result["best_cv_score"] = meta.get("best_cv_score", 0.0)
                        result["best_params"] = {k.split("__")[-1]: v for k, v in bp.items()}
                except (OSError, json.JSONDecodeError):
                    pass
        return self._finish(result)

    def _running_status(self, job: dict, prog: dict) -> dict:
        phase_pct = prog.get("pct", 0)
        phase_msg = prog.get("msg", "…")
        if job["mode"] == "batch":
            return {"state": "running", "mode": "batch",
                    "done": prog.get("config_idx", 0), "total": job["total"],
                    "msg": phase_msg}

        # Einzel-Training: geglätteter Fortschritt, wenn keine Echtzeit-Info (pct==35).
        if phase_pct == 35:
            if job["fit_started_at"] is None:
                job["fit_started_at"] = time.time()
            ratio = min((time.time() - job["fit_started_at"]) / job["estimated_fit_sec"], 0.99)
            pct = 35 + int(50 * math.sqrt(ratio))
        elif phase_pct > 35:
            job["fit_started_at"] = None
            pct = phase_pct
        else:
            pct = phase_pct
        return {"state": "running", "mode": "single", "pct": pct, "msg": phase_msg}

    def _finish(self, payload: dict) -> dict:
        self._proc = None
        self._job = None
        return payload

    def _close_log(self) -> None:
        handle = getattr(self, "_log_handle", None)
        if handle and not handle.closed:
            handle.close()

    @staticmethod
    def _log_tail(log_file: Path, lines: int = 30) -> str:
        try:
            return "\n".join(log_file.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


# Prozessweit ein Manager (Single-User-Werkzeug).
MANAGER = TrainingManager()
