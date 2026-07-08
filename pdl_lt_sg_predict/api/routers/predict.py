"""Vorhersage-Endpunkte: Einzel-, Batch-Vorhersage und SHAP-Erklärung.

Dies ist zugleich die Grundlage für die geplante öffentliche Vorhersage-API.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ...core import prediction

router = APIRouter(prefix="/predict", tags=["predict"])


class SingleRequest(BaseModel):
    model_file: str
    lemma: str = ""
    bedeutung: str
    top_k: int = Field(3, ge=1, le=10)


class TopEntry(BaseModel):
    label: str
    description: str
    proba: float | None = None
    is_best: bool = False


class SingleResponse(BaseModel):
    prediction: str
    description: str
    top: list[TopEntry]
    model_type: str
    uses_lemma: bool


class ShapRequest(BaseModel):
    model_file: str
    lemma: str = ""
    bedeutung: str
    predicted_label: str
    filter_stopwords: bool = True


class ShapWord(BaseModel):
    word: str
    score: float


class ShapResponse(BaseModel):
    lemma: list[ShapWord]
    bedeutung: list[ShapWord]
    is_nn: bool


class BatchResponse(BaseModel):
    rows: list[dict]
    count: int
    uses_lemma: bool


@router.post("/single", response_model=SingleResponse)
def single(req: SingleRequest) -> SingleResponse:
    if not req.bedeutung.strip():
        raise HTTPException(422, "Feld 'bedeutung' darf nicht leer sein.")
    try:
        return SingleResponse(**prediction.predict_single(req.model_file, req.lemma, req.bedeutung, req.top_k))
    except FileNotFoundError as e:
        raise HTTPException(404, f"Modell nicht gefunden: {req.model_file}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Vorhersage fehlgeschlagen: {e}") from e


@router.post("/shap", response_model=ShapResponse)
def shap(req: ShapRequest) -> ShapResponse:
    try:
        return ShapResponse(**prediction.explain(
            req.model_file, req.lemma, req.bedeutung, req.predicted_label, req.filter_stopwords,
        ))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"SHAP-Berechnung fehlgeschlagen: {e}") from e


@router.post("/batch", response_model=BatchResponse)
async def batch(model_file: str = Form(...), file: UploadFile = File(...)) -> BatchResponse:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(422, "Nur CSV-Dateien erlaubt.")
    content = await file.read()
    try:
        return BatchResponse(**prediction.predict_batch(model_file, content))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Batch-Vorhersage fehlgeschlagen: {e}") from e
