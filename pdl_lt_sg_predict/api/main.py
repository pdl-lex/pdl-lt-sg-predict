"""FastAPI-Einstiegspunkt der Sachgruppen-Vorhersage-API.

Start:  uv run uvicorn pdl_lt_sg_predict.api.main:app --reload
Docs:   http://localhost:8000/docs

Die API liegt unter ``/api``; in Produktion wird das gebaute Frontend
(``frontend/dist``) unter ``/`` ausgeliefert.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import config, models, predict, sachgruppen, training

app = FastAPI(
    title="LexoTerm Tools — Sachgruppen-Vorhersage",
    description="ML-gestützte Vorhersage von Sachgruppen aus Wörterbuchdaten (lemma + bedeutung).",
    version="0.1.0",
)

# CORS nur für den React+Vite-Dev-Server nötig (Produktion liefert das
# Frontend same-origin aus). Weitere Origins kommasepariert via CORS_ORIGINS.
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


_ANLEITUNG = Path(__file__).resolve().parents[2] / "anleitung.md"


@app.get("/api/anleitung", tags=["meta"])
def anleitung() -> dict:
    """Anleitung (Markdown) für das Einführungs-Modul."""
    text = _ANLEITUNG.read_text(encoding="utf-8") if _ANLEITUNG.exists() else ""
    return {"markdown": text}


for module in (config, models, predict, sachgruppen, training):
    app.include_router(module.router, prefix="/api")


# In Produktion das gebaute Frontend ausliefern (nur wenn vorhanden).
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
