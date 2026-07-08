# ---- Stage 1: Frontend bauen (React + Vite) ----
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Backend (FastAPI) + statisches Frontend ----
FROM python:3.14-slim
WORKDIR /app

# uv für Dependency-Management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Python-Dependencies (ohne Dev-Gruppe). Ohne --frozen, damit uv den Lock
# bei Bedarf aus pyproject.toml auflöst.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# ML-Kern (Root-Module + gepickelte Modelle) und Fachlogik/API
COPY sachgruppen_classifier.py shap_utils.py sachgruppen.csv stopwords_de.txt anleitung.md ./
COPY assets/ ./assets/
COPY models/ ./models/
COPY pdl_lt_sg_predict/ ./pdl_lt_sg_predict/

# Gebautes Frontend an den von main.py erwarteten Ort
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8000

# FastAPI liefert die API unter /api und das Frontend unter / aus.
CMD ["uv", "run", "uvicorn", "pdl_lt_sg_predict.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
