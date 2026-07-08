"""Modell-Liste, bestes Modell und Klassifikations-Report."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...core import models as core_models

router = APIRouter(prefix="/models", tags=["models"])


class ModelsResponse(BaseModel):
    models: list[dict]
    files: list[str]
    best: dict | None
    count: int


class ReportResponse(BaseModel):
    model_file: str
    report: str


@router.get("", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    rows = core_models.list_models()
    return ModelsResponse(
        models=rows,
        files=core_models.available_model_files(),
        best=core_models.best_model(),
        count=len(rows),
    )


@router.get("/{model_file}/report", response_model=ReportResponse)
def get_report(model_file: str) -> ReportResponse:
    text = core_models.report_text(model_file)
    if text is None:
        raise HTTPException(404, f"Kein Report vorhanden für {model_file}.")
    return ReportResponse(model_file=model_file, report=text)
