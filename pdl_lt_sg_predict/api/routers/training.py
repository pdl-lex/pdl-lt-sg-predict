"""Trainings-Endpunkte: Daten-Upload, Start (Einzel/Batch) und Status-Polling."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ...core.bridge import ENABLE_TRAINING
from ...core.training import MANAGER

router = APIRouter(prefix="/training", tags=["training"])


class HyperParams(BaseModel):
    test_size: float = 0.2
    tune_mode: str = "standard"  # standard | auto | manual
    tune_n_iter: int = 20
    tune_cv: int = 3
    svm_c: float = 1.0
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    nn_hidden_layers: str = "100"
    nn_alpha: float = 0.0001
    nn_learning_rate_init: float = 0.0005
    use_spacy: bool = True
    use_dornseiff: bool = True
    calibrate: bool = True  # Konfidenz-Kalibrierung (wirkt nur bei svm; beim nn Softmax bereits gut kalibriert)


class SingleConfig(HyperParams):
    model: str = "svm"
    analyzer_mode: str = "char_wb"  # char_wb | word
    word_ngram_max: int = 1
    min_word_length: int = 1
    use_stopword_removal: bool = False
    cross_validate: bool = False  # zusätzliche split-unabhängige k-fold-Bewertung
    cv_folds: int = 5
    cv_mode: str = "stratified"  # stratified | group (GroupKFold nach bedeutung)


class BatchConfig(HyperParams):
    batch_model_types: list[str] = Field(default_factory=lambda: ["svm"])
    batch_use_stopwords: list[bool] = Field(default_factory=lambda: [False])
    batch_min_lengths: list[int] = Field(default_factory=lambda: [1])
    batch_analyzers: list[str] = Field(default_factory=lambda: ["char_wb"])


class UploadResponse(BaseModel):
    filename: str
    num_samples: int
    num_classes: int
    time_per_type: dict[str, float]


class TrainingInfo(BaseModel):
    enable_training: bool
    csv: dict | None
    running: bool


@router.get("/info", response_model=TrainingInfo)
def info() -> TrainingInfo:
    return TrainingInfo(
        enable_training=ENABLE_TRAINING,
        csv={**MANAGER.csv_info, "time_per_type": MANAGER.time_per_type} if MANAGER.csv_info else None,
        running=MANAGER.running,
    )


@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> UploadResponse:
    # Bewusst sync (Threadpool): upload_csv schreibt die Datei und parst sie
    # mit pandas — als async-Handler würde das den Event-Loop blockieren.
    if not ENABLE_TRAINING:
        raise HTTPException(403, "Training ist deaktiviert.")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(422, "Nur CSV-Dateien erlaubt.")
    content = file.file.read()
    try:
        return UploadResponse(**MANAGER.upload_csv(file.filename or "daten.csv", content))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/start")
def start(cfg: SingleConfig) -> dict:
    try:
        return MANAGER.start_single(cfg.model_dump())
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e)) from e


@router.post("/batch")
def batch(cfg: BatchConfig) -> dict:
    try:
        return MANAGER.start_batch(cfg.model_dump())
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e)) from e


@router.get("/status")
def status() -> dict:
    return MANAGER.status()
