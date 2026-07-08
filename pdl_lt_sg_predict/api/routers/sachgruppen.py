"""Sachgruppen-Übersicht mit Klassifikationsmetriken."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...core.sachgruppen import sachgruppen_overview

router = APIRouter(prefix="/sachgruppen", tags=["sachgruppen"])


class SachgruppenResponse(BaseModel):
    rows: list[dict]
    model_name: str
    model_file: str
    accuracy: str


@router.get("", response_model=SachgruppenResponse)
def get_sachgruppen() -> SachgruppenResponse:
    return SachgruppenResponse(**sachgruppen_overview())
