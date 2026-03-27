# LexoTerm – Sachgruppen-Klassifikation

Automatische Vorhersage von Sachgruppen aus Lemma und Bedeutungsdefinition mittels Machine Learning. Entwickelt für die BDO-Wörterbücher "Bayerisches Wörterbuch", "Fränkisches Wörterbuch" und "Dialektologisches Informationssystem Bayrisch-Schwaben". (Bayerische Akademie der Wissenschaften).

---

## Verwendung

Das Projekt bietet zwei Nutzungswege:

| Weg | Geeignet für |
|---|---|
| **Web-App** (Reflex) | Interaktives Training, Vergleich, Vorhersage im Browser |
| **CLI** | Serverbetrieb, Batch-Training, Automatisierung |

---

## Web-App starten

```bash
uv sync
reflex run
```

Die App ist danach unter `http://localhost:3000` erreichbar.

### Funktionsumfang der Web-App

**Training**
- CSV-Upload mit Vorschau (Spaltenprüfung, Sample-Anzahl, Klassenverteilung)
- Modellauswahl: Linear SVM, Logistische Regression, Random Forest, XGBoost, Neural Network
- Feature-Engineering: Analyzer (`char_wb` / `word`), N-Gram-Bereiche, Mindestlänge, Stopwort-Filterung
- Hyperparameter-Tuning:
  - *Standard*: vordefinierte Defaults
  - *Auto-Tune*: RandomizedSearchCV mit einstellbarer Kombinations- und Fold-Anzahl
  - *Manuell*: direkte Eingabe von SVM-C, XGBoost-Parametern u. a.
- Batch-Training: kartesisches Produkt aus Modell × Analyzer × Stopwörter × Mindestlänge
- Echtzeit-Fortschrittsanzeige mit Zeitschätzung

**Analyse**
- Vergleich aller trainierten Modelle (Accuracy, Trainingszeit, Parameter)
- SHAP-basierte Erklärbarkeit: welche Wörter/N-Gramme tragen zur Vorhersage bei

**Vorhersage**
- Eingabe von Lemma + Bedeutung → Sachgruppen-Vorhersage mit Konfidenzwerten
- Auswahl des Modells aus allen gespeicherten Versionen

Eine ausführliche Bedienungsanleitung für die Web-App findet sich in [anleitung.md](anleitung.md).

---

## CLI

Das Skript `sachgruppen_classifier.py` kann ohne Web-App direkt verwendet werden –
ideal für Remote-Server (z. B. LRZ-VM), Batch-Läufe und Automatisierung.

```bash
# Einzeltraining
python sachgruppen_classifier.py --csv daten.csv --model svm

# Batch-Training: alle Kombinationen
python sachgruppen_classifier.py --csv daten.csv \
  --model svm logistic xgboost \
  --analyzer char_wb word \
  --stopwords false true

# Auto-Tune
python sachgruppen_classifier.py --csv daten.csv --model svm \
  --tune --tune-n-iter 20 --tune-cv 5
```

Pro Lauf werden automatisch gespeichert:
- `*.pkl` – trainiertes Modell
- `*_metadata.json` – Accuracy, Trainingszeit, alle Parameter, beste Tune-Werte
- `*_report.txt` – vollständiger Klassifikationsbericht

Die vollständige CLI-Dokumentation mit allen Parametern und Beispielen: [README_CLI.md](README_CLI.md).

---

## Modelle im Vergleich

| Modell | Geschwindigkeit | Accuracy | Hinweis |
|---|---|---|---|
| **Linear SVM** | schnell | sehr gut | Empfehlung für Produktion |
| Logistische Regression | sehr schnell | gut | gute Baseline |
| Random Forest | mittel | gut | |
| XGBoost | langsam | sehr gut | GPU-beschleunigt (NVIDIA) |
| Neural Network (MLP) | mittel | gut | |

---

## Feature-Engineering

Alle Modelle verwenden eine **TF-IDF-Pipeline** aus zwei kombinierten Vektoren:

- **Lemma-Vektorisierer**: Character-N-Gramme `(2,5)`, bis 10.000 Features
- **Bedeutungs-Vektorisierer**: Character-N-Gramme `(2,4)`, bis 20.000 Features
- **Optional**: zusätzlicher Wort-N-Gramm-Vektorisierer

Vorverarbeitungsschritte (konfigurierbar):
- Interpunktionsbereinigung
- Stopwort-Filterung (263 deutsche Stopwörter)
- Mindestlängenfilter

---

## Projektstruktur

```
pdl-lt-sg-predict/
├── sachgruppen_classifier.py       # ML-Kernpipeline + CLI
├── shap_utils.py                   # SHAP-Erklärbarkeit
├── stopwords_de.txt                # Deutsche Stopwörter
├── anleitung.md                    # Bedienungsanleitung Web-App
├── README_CLI.md                   # CLI-Dokumentation
│
├── pdl_lt_sg_predict_app/
│   ├── pdl_lt_sg_predict_app.py    # Reflex-App, Seitenregistrierung
│   ├── training.py                 # Trainings-Seite + State
│   ├── vorhersage.py               # Vorhersage-Seite
│   ├── analyse.py                  # Analyse/Vergleich-Seite
│   ├── anleitung.py                # Anleitungs-Seite
│   ├── state.py                    # Gemeinsamer State, Konstanten
│   └── components.py               # UI-Komponenten
│
├── scripts/
│   ├── check_gpu.py                # GPU- und ROCm-Verfügbarkeit prüfen
│   ├── extract_data.py             # XML → CSV Extraktion
│   ├── analyze_performance.py      # Performance-Analyse
│   └── model_comparison.py         # Modellvergleich-Skript
│
├── models/                         # Gespeicherte Modelle (*.pkl + Metadaten)
└── pyproject.toml
```

---

## Abhängigkeiten

```bash
uv sync          # alle Abhängigkeiten inkl. Web-App

# oder nur für CLI (ohne Reflex):
pip install scikit-learn pandas numpy tqdm xgboost shap
```

---

## Trainingsdaten-Format

CSV mit Semikolon- oder Kommatrennung, Pflichtfelder:

| Spalte | Inhalt | Beispiel |
|---|---|---|
| `lemma` | Stichwort | `Waggala` |
| `bedeutung` | Bedeutungsangabe | `kleines Kind; wackelig auf den Beinen` |
| `sachgruppe` | Sachgruppen-Nummer | `1830` |

---

## Lizenz

Daten: CC-BY-SA 4.0 (Bayerische Akademie der Wissenschaften)
Code: MIT License
