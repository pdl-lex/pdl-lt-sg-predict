# Sachgruppen-Klassifikation Web-App

Eine Web-Oberfläche für Machine Learning Modelltraining, Analyse und Vorhersage von Sachgruppen in Wörterbuch-Daten.

## Features

### 1. **Training-Seite** (optional aktivierbar)
- CSV-Upload mit Lemma, Bedeutung, Sachgruppe
- Auswahl des Modell-Typs (SVM, Logistic Regression, Random Forest, Neural Network, XGBoost)
- Konfiguration von Parametern (Test-Size, GPU-Nutzung)
- Live-Training mit Progress-Anzeige
- Automatisches Speichern von Modellen und Metadaten

**Wichtig**: Training ist standardmäßig **deaktiviert** für schwache VMs. Zum Aktivieren:
```python
# In lt_sg_predict_app.py, Zeile ~20:
ENABLE_TRAINING = True
```

### 2. **Analyse-Seite**
- Übersicht aller trainierten Modelle
- Vergleich von Accuracy, Trainingszeit, etc.
- Sortierbare Tabelle mit Metadaten

### 3. **Vorhersage-Seite**
- **Einzelvorhersage**: Eingabe von Lemma + Bedeutung
- **Batch-Vorhersage**: CSV-Upload mit Lemma + Bedeutung
- Wahrscheinlichkeitsangabe (wenn vom Modell unterstützt)
- Modell-Auswahl aus allen gespeicherten Modellen

## Installation

### Voraussetzungen
Die App nutzt die bestehenden Python-Scripts (`sachgruppen_classifier.py`, `model_comparison.py`). Diese müssen im Parent-Verzeichnis liegen.

### Setup

1. **Virtual Environment erstellen** (falls noch nicht vorhanden):
```bash
cd /home/wolfgang/Nextcloud/BAdW/Code/Repos/lt_sg_predict
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

2. **Reflex installieren**:
```bash
pip install reflex
```

3. **App initialisieren**:
```bash
cd lt_sg_predict_app
reflex init
```

## Verwendung

### App starten
```bash
cd lt_sg_predict_app
reflex run
```

Die App läuft dann auf `http://localhost:3001`

### Produktions-Modus
```bash
reflex run --env prod
```

## Datei-Struktur

```
lt_sg_predict/
├── sachgruppen_classifier.py    # Core ML-Code (unverändert)
├── model_comparison.py           # Model-Vergleich (unverändert)
├── analyze_performance.py        # Performance-Analyse (unverändert)
├── woerterbuch_daten.csv        # Trainingsdaten
├── model_*.pkl                   # Gespeicherte Modelle
├── model_metadata_*.json        # Model-Metadaten
│
└── lt_sg_predict_app/           # Web-App
    ├── lt_sg_predict_app.py     # Haupt-App Code
    ├── rxconfig.py              # Reflex Config
    ├── pyproject.toml           # Dependencies
    └── README.md                # Diese Datei
```

## CSV-Format

### Training
```csv
lemma,bedeutung,sachgruppe
#Kreuz,"Kreuz (in Redensart)",4114
#spanen,"säugen (Ferkel)",4113
```

### Batch-Vorhersage
```csv
lemma,bedeutung
#Hund,"Hund (Tier)"
#Katze,"Katze (Tier)"
```

## Standalone-Scripts

Die bestehenden Python-Scripts funktionieren **weiterhin unverändert**:

```bash
# Training
python sachgruppen_classifier.py --csv woerterbuch_daten.csv --model xgboost --save model_xgb.pkl

# Model-Vergleich
python model_comparison.py --csv woerterbuch_daten.csv

# Vorhersage
python sachgruppen_classifier.py --predict --load model_xgb.pkl

# Analyse
python analyze_performance.py --csv woerterbuch_daten.csv
```

Die Web-App ist **nur eine GUI** für diese Scripts!

## Konfiguration

### Training aktivieren/deaktivieren
In `lt_sg_predict_app.py`:
```python
ENABLE_TRAINING = False  # False für schwache VMs, True für lokale Maschinen
```

### Port ändern
In `rxconfig.py`:
```python
config = rx.Config(
    app_name="lt_sg_predict_app",
    port=3001,  # Hier Port ändern
)
```

### Max. Upload-Größe
In `lt_sg_predict_app.py`:
```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'sachgruppen_classifier'"
→ Die App muss `sachgruppen_classifier.py` im Parent-Verzeichnis finden. Prüfen Sie die Verzeichnisstruktur.

### Training ist langsam
→ Verwenden Sie schnellere Modelle (SVM, Logistic Regression) oder aktivieren Sie GPU-Training.

### Modelle werden nicht angezeigt
→ Stellen Sie sicher, dass `.pkl` Modell-Dateien im Parent-Verzeichnis liegen.

## Entwicklung

### Live-Reload aktivieren
```bash
reflex run --loglevel debug
```

### Code-Änderungen
Die App lädt automatisch neu bei Änderungen an `lt_sg_predict_app.py`.

## Performance

- **SVM/Logistic**: ~1-2 Minuten (empfohlen für Web-UI)
- **Random Forest**: ~3-5 Minuten
- **Neural Network**: ~5-10 Minuten
- **XGBoost**: ~15-20 Minuten (beste Accuracy: 0.81)

## Lizenz

Siehe Hauptprojekt.
