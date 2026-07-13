# LexoTerm Tools – Sachgruppen-Vorhersage

Automatische Vorhersage von **Sachgruppen** aus Lemma und Bedeutungsdefinition mittels
Machine Learning. Entwickelt für die BDO-Wörterbücher „Fränkisches Wörterbuch" und
„Dialektologisches Informationssystem Bayerisch-Schwaben" (Bayerische Akademie der
Wissenschaften).

Die Oberfläche folgt dem **LexoTerm-Tools-Design** (Workbench-Layout) und teilt sich einen
Technik-Stack mit `pdl-lt-dictconsistency`:

- **Backend:** Python + **FastAPI** (`pdl_lt_sg_predict/`) — dünne API über die bestehende
  ML-Pipeline (`sachgruppen_classifier.py`).
- **Frontend:** **React + Vite + TypeScript** (`frontend/`) — Workbench mit Modul-Rail,
  Konfigurator-Pane, Ergebnis-Panel, Statuszeile, Befehlspalette (⌘K), Hell-/Dunkel-Theme
  und drei Layout-Modi.
- **CLI/ML:** `sachgruppen_classifier.py` bleibt unverändert und dient sowohl als
  Trainings-Worker (Subprozess) als auch als eigenständiges CLI.

---

## Architektur

```
pdl-lt-sg-predict/
├── sachgruppen_classifier.py   # ML-Kernpipeline + CLI + Trainings-Worker (unverändert)
├── shap_utils.py               # SHAP-Erklärbarkeit
├── stopwords_de.txt, anleitung.md, README_CLI.md
├── data/                       # Taxonomie + Trainingsdaten (sachgruppen.csv, woerterbuch_daten_*.csv)
├── assets/                     # Logo, Favicon, Dornseiff-Gazetteer-Cache
├── models/                     # Gespeicherte Modelle (*.pkl + *_metadata.json + *_report.txt)
│
├── pdl_lt_sg_predict/          # Backend
│   ├── core/                   # reine Fachlogik
│   │   ├── bridge.py           #   Pfade, Modell-Cache, Sachgruppen-Mapping, ML-Import
│   │   ├── models.py           #   Modell-Liste, bestes Modell, Reports
│   │   ├── prediction.py       #   Einzel-/Batch-Vorhersage + SHAP
│   │   ├── sachgruppen.py      #   Metriken je Sachgruppe (aus dem Report)
│   │   └── training.py         #   Trainings-Orchestrierung (Subprozess + Fortschritt)
│   └── api/                    # FastAPI
│       ├── main.py             #   App, Router, statisches Frontend
│       └── routers/            #   config · models · predict · sachgruppen · training
│
└── frontend/                   # React + Vite
    └── src/
        ├── design/             # Tokens, Icons, Widgets, UI-Kit, Markdown
        ├── layout/             # Header, Rail, ConfigPane, StatusBar, CommandPalette
        ├── modules/            # Einführung, Einzel-/Batch-Vorhersage, Analyse, Sachgruppen, Training
        └── state/workbench.tsx # globaler Zustand (Theme, Layout, Modelle, aktives Modul)
```

Die API liegt unter `/api`; im Produktionsbetrieb liefert FastAPI zusätzlich das gebaute
Frontend (`frontend/dist`) unter `/` aus.

---

## Entwicklung

Zwei Prozesse (Backend + Vite-Dev-Server mit `/api`-Proxy):

```bash
# 1) Backend (FastAPI, Port 8000)
uv sync
uv run uvicorn pdl_lt_sg_predict.api.main:app --reload

# 2) Frontend (Vite, Port 5173)
cd frontend
npm install
npm run dev
```

Danach: Frontend unter `http://localhost:5173`, API-Docs unter `http://localhost:8000/docs`.

## Produktion (ein Prozess)

```bash
cd frontend && npm install && npm run build && cd ..
uv run uvicorn pdl_lt_sg_predict.api.main:app --host 0.0.0.0 --port 8000
```

App und API unter `http://localhost:8000`.

## Docker

```bash
docker build -t lt-sg-predict .
docker run -p 8000:8000 lt-sg-predict
```

---

## Modelle & Konfiguration

- Vortrainierte Modelle (WBF/DIBS) liegen in `models/`; ohne eigene Modelle können sie
  über <https://lexoterm.de/static/sgpredict-models.zip> nachgeladen werden.
- `.env` (aus Vorlage) steuert `MODELS_DIR`, `SESSIONS_DIR` und `ENABLE_TRAINING` sowie
  optionale SMTP-Werte für CLI-Benachrichtigungen.

---

## Module der Oberfläche

| Modul | Funktion |
|---|---|
| **Einführung** | Anleitung (aus `anleitung.md`) + Umgebungsübersicht |
| **Einzelvorhersage** | Lemma + Bedeutung → Top-3-Sachgruppen inkl. SHAP-Erklärung |
| **Batch-Vorhersage** | CSV → Vorhersagen als Tabelle, CSV-Export |
| **Analyse** | Vergleich aller Modelle, Klassifikations-Report, Modellwahl |
| **Sachgruppen** | Taxonomie (Hallig-Wartburg) mit Precision/Recall/F1 des besten Modells |
| **Training** | Einzel-/Batch-Training, Hyperparameter-Tuning, Live-Fortschritt |

---

## HTTP-API (Auszug)

Grundlage für die geplante öffentliche Vorhersage-API:

| Methode | Pfad | Zweck |
|---|---|---|
| `GET` | `/api/models` | Modell-Liste, bestes Modell |
| `POST` | `/api/predict/single` | `{model_file, lemma, bedeutung}` → Top-k + Wahrscheinlichkeiten |
| `POST` | `/api/predict/shap` | SHAP-Worterklärung zu einer Vorhersage |
| `POST` | `/api/predict/batch` | CSV-Upload → Vorhersagen |
| `GET` | `/api/sachgruppen` | Sachgruppen + Metriken |
| `*` | `/api/training/*` | Upload, Start (Einzel/Batch), Status |

---

## CLI

Das Skript `sachgruppen_classifier.py` ist unverändert nutzbar (Batch-Läufe, Server,
Automatisierung). Vollständige Doku: [README_CLI.md](README_CLI.md).

```bash
python sachgruppen_classifier.py --csv daten.csv --model svm
```

---

## Trainingsdaten-Format

CSV (Semikolon/Komma), Pflichtfelder `bedeutung` und `sachgruppe`, `lemma` optional:

| Spalte | Inhalt | Beispiel |
|---|---|---|
| `lemma` | Stichwort | `Waggala` |
| `bedeutung` | Bedeutungsangabe | `kleines Kind; wackelig auf den Beinen` |
| `sachgruppe` | Sachgruppen-Nummer | `1830` |

---

## Lizenz

Daten & Code: CC-BY-SA 4.0 (Bayerische Akademie der Wissenschaften)
