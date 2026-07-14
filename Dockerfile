# ---- Stage 1: Frontend bauen (React + Vite) ----
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Backend (FastAPI) + statisches Frontend ----
FROM python:3.14-slim
WORKDIR /app

# uv für Dependency-Management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Python-Dependencies (ohne Dev-Gruppe). --frozen: exakt der committete Lock,
# kein Neu-Auflösen im Container (sonst droht Drift zu inkompatiblen Versionen,
# z. B. numpy 2.5 -> numba 0.53.1 -> Build-Fehler).
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# ML-Kern (Root-Module) und Fachlogik/API
COPY sachgruppen_classifier.py shap_utils.py stopwords_de.txt anleitung.md ./
COPY data/sachgruppen.csv ./data/sachgruppen.csv
COPY assets/dornseiff_gaz_cache.pkl ./assets/dornseiff_gaz_cache.pkl
COPY pdl_lt_sg_predict/ ./pdl_lt_sg_predict/

# Die gepickelten Modelle werden NICHT ins Image kopiert, sondern zur Laufzeit
# als Volume unter /app/models gemountet (per FTP bestückt). Mount-Point anlegen,
# damit der Pfad auch ohne Volume existiert.
RUN mkdir -p models

# Gebautes Frontend an den von main.py erwarteten Ort
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8000

# FastAPI liefert die API unter /api und das Frontend unter / aus.
CMD ["uv", "run", "uvicorn", "pdl_lt_sg_predict.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
