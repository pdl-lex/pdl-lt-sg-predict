"""Modell-Liste, bestes Modell, Klassifikations-Report und Modell-Download."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

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


@router.get("/{model_file}/bundle")
def download_bundle(model_file: str) -> FileResponse:
    """Modell samt Metadaten und Report als ZIP herunterladen.

    Das ZIP wird ungepackt (ZIP_STORED) in einer Temp-Datei aufgebaut — die
    .pkl sind bereits binär und 100–330 MB groß, Deflate brächte kaum Gewinn
    bei hohem CPU-Aufwand. Die Temp-Datei wird nach dem Senden gelöscht.
    """
    files = core_models.bundle_files(model_file)
    if not files:
        raise HTTPException(404, f"Modell '{model_file}' nicht gefunden.")

    tmp = tempfile.NamedTemporaryFile(prefix="model_bundle_", suffix=".zip", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
    except OSError as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Bundle konnte nicht erstellt werden: {e}") from e

    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=f"{model_file[:-4]}_bundle.zip",
        background=BackgroundTask(tmp_path.unlink, missing_ok=True),
    )
