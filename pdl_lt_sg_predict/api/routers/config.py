"""Laufzeit-Konfiguration für das Frontend."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...core.bridge import AVAILABLE_MODELS, ENABLE_TRAINING, MODELS_DIR

router = APIRouter(prefix="/config", tags=["config"])


class ModelType(BaseModel):
    code: str
    name: str


class AppConfig(BaseModel):
    enable_training: bool
    models_dir: str
    model_types: list[ModelType]


@router.get("", response_model=AppConfig)
def get_config() -> AppConfig:
    return AppConfig(
        enable_training=ENABLE_TRAINING,
        models_dir=str(MODELS_DIR),
        model_types=[ModelType(code=c, name=n) for c, n in AVAILABLE_MODELS],
    )
