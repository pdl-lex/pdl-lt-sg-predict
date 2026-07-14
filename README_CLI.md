# Sachgruppen-Klassifikation – CLI-Referenz

Das Skript `sachgruppen_classifier.py` kann direkt über die Kommandozeile genutzt werden,
ohne das Web-Backend (FastAPI) zu starten. Das ist besonders praktisch für das Training
auf Remote-Servern (z. B. LRZ-VM).

## Voraussetzungen

```bash
# Abhängigkeiten installieren (spaCy + Sprachmodell nur bei --use-spacy nötig)
pip install scikit-learn pandas numpy tqdm xgboost shap spacy
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl

# oder via uv (aus pyproject.toml, installiert alles inkl. spaCy-Modell)
uv sync
```

---

## 1. Einzeltraining

Das einfachste Training mit Standardwerten:

```bash
python sachgruppen_classifier.py --csv daten.csv
```

Mit expliziten Parametern:

```bash
python sachgruppen_classifier.py \
  --csv daten.csv \
  --model svm \
  --output-dir models/
```

Ausgabe pro Lauf (immer automatisch gespeichert):

```
models/svm_char_wb_ml1_sw0_20260327_142301.pkl
models/svm_char_wb_ml1_sw0_20260327_142301_metadata.json
models/svm_char_wb_ml1_sw0_20260327_142301_report.txt
```

---

## 2. Parameter-Referenz

### Daten

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--csv PATH` | `test_output.csv` | CSV-Datei mit Spalten `lemma`, `bedeutung`, `sachgruppe` |
| `--test-size FLOAT` | `0.2` | Anteil der Testdaten, z. B. `0.2` = 20 % |
| `--output-dir DIR` | `models` | Zielverzeichnis für Modell, Metadaten, Bericht |

### Modell

| Parameter | Standard | Optionen |
|---|---|---|
| `--model` | `svm` | `svm` `logistic` `rf` `xgboost` `nn` |

### Feature-Engineering

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--analyzer` | `char_wb` | `char_wb` (Zeichen-N-Gramme) oder `word` (Wort-N-Gramme) |
| `--word-ngram-max N` | `1` | Maximale N-Gram-Länge bei `--analyzer word` (z. B. `2` = Unigramme + Bigramme) |
| `--min-length N` | `1` | Wörter kürzer als N werden gefiltert |
| `--stopwords true\|false` | `false` | Deutsche Stopwörter entfernen |

### Semantische Anreicherung

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--use-spacy` | – | spaCy-Wortvektoren (de_core_news_lg, 300 dim) zusätzlich zu TF-IDF (+0,6 pp Top-1) |
| `--use-dornseiff` | – | Dornseiff-Gazetteer-Features zusätzlich zu TF-IDF (+0,3 pp Top-1); lädt `assets/dornseiff_gaz_cache.pkl` |
| `--use-svd` | – | TruncatedSVD-Dimensionsreduktion (NN und XGBoost) |
| `--svd-components N` | `500` | Anzahl SVD-Komponenten bei `--use-svd` |

### Auto-Tune (RandomizedSearchCV)

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--tune` | – | Auto-Tune aktivieren |
| `--tune-n-iter N` | `20` | Anzahl zufälliger Parameterkombinationen |
| `--tune-cv N` | `3` | Anzahl Kreuzvalidierungs-Folds (mind. 2) |

**Trainingszeit mit Auto-Tune:** ca. `n-iter × cv × Einzeltraining`.
Auf einer VM mit 20 vCPUs laufen die Fits parallel (`n_jobs=-1`),
was den Faktor effektiv auf `ceil(n-iter × cv / 20)` reduziert.

### Modell-spezifische Parameter (manuelle Werte ohne Auto-Tune)

| Parameter | Standard | Modell |
|---|---|---|
| `--svm-c FLOAT` | `1.0` | SVM: Regularisierung (kleiner = robuster, größer = enger an Trainingsdaten) |
| `--xgb-n-estimators N` | `300` | XGBoost: Anzahl Bäume |
| `--xgb-max-depth N` | `6` | XGBoost: maximale Baumtiefe |
| `--xgb-learning-rate FLOAT` | `0.05` | XGBoost: Lernrate |
| `--xgb-subsample FLOAT` | `0.8` | XGBoost: Anteil der Trainingsdaten pro Baum |
| `--gpu` | – | XGBoost GPU-Beschleunigung (erfordert CUDA/ROCm) |
| `--nn-hidden-layers STR` | `100` | NN: Hidden-Layer-Größen, kommasepariert (z. B. `200,100,50`) |
| `--nn-alpha FLOAT` | `0.0001` | NN: L2-Regularisierung |
| `--nn-learning-rate-init FLOAT` | `0.0005` | NN: initiale Lernrate |
| `--nn-n-iter-no-change N` | `5` | NN: Early-Stopping-Geduld (Epochen ohne Verbesserung) |

### Benachrichtigung

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--notify EMAIL` | – | E-Mail-Adresse für Benachrichtigung nach Abschluss |

SMTP-Konfiguration über `.env` (siehe Abschnitt 8).

### Vorhersage

| Parameter | Beschreibung |
|---|---|
| `--predict` | Interaktiven Vorhersage-Modus starten |
| `--load PATH` | Zu ladendes Modell (`.pkl`-Datei) |

---

## 3. Beispiele: Einzeltraining mit Parametervarianten

```bash
# SVM mit Stopwort-Filterung
python sachgruppen_classifier.py --csv daten.csv --model svm --stopwords true

# Wort-N-Gramme statt Zeichen-N-Gramme
python sachgruppen_classifier.py --csv daten.csv --model svm --analyzer word --word-ngram-max 2

# Logistische Regression mit Mindestlänge 2
python sachgruppen_classifier.py --csv daten.csv --model logistic --min-length 2

# XGBoost mit GPU (NVIDIA-Server)
python sachgruppen_classifier.py --csv daten.csv --model xgboost --gpu

# SVM mit manuell gesetztem C-Wert
python sachgruppen_classifier.py --csv daten.csv --model svm --svm-c 10.0
```

---

## 4. Auto-Tune

Auto-Tune probiert zufällige Kombinationen aus dem vordefinierten Suchraum
(Feature-Größen, Modell-Hyperparameter) und wählt automatisch die beste Konfiguration.

```bash
# Auto-Tune mit Standardwerten (20 Kombinationen, 3-fach CV)
python sachgruppen_classifier.py --csv daten.csv --model svm --tune

# Schneller Testlauf
python sachgruppen_classifier.py --csv daten.csv --model svm --tune --tune-n-iter 5 --tune-cv 2

# Gründliche Suche (empfohlen auf VM mit vielen Kernen)
python sachgruppen_classifier.py --csv daten.csv --model svm --tune --tune-n-iter 30 --tune-cv 5
```

Die besten gefundenen Parameter werden in den Metadaten unter `best_params` gespeichert.

---

## 5. Batch-Training

Batch-Training trainiert alle Kombinationen der angegebenen Werte nacheinander
und gibt am Ende eine nach Accuracy sortierte Übersicht aus.

**Syntax:** Mehrere Werte pro Parameter durch Leerzeichen trennen.

```bash
# Alle Modelle vergleichen
python sachgruppen_classifier.py --csv daten.csv \
  --model svm logistic rf xgboost

# Zwei Analyzer vergleichen
python sachgruppen_classifier.py --csv daten.csv \
  --model svm \
  --analyzer char_wb word

# Stopwörter: mit und ohne vergleichen
python sachgruppen_classifier.py --csv daten.csv \
  --model svm \
  --stopwords false true

# Mehrere Mindestlängen vergleichen
python sachgruppen_classifier.py --csv daten.csv \
  --model svm \
  --min-length 1 2 3
```

### Vollständiger Vergleich (alle Dimensionen kombiniert)

```bash
python sachgruppen_classifier.py --csv daten.csv \
  --model svm logistic xgboost \
  --analyzer char_wb word \
  --stopwords false true \
  --min-length 1 2
```

Das erzeugt 3 × 2 × 2 × 2 = **24 Konfigurationen**.

Beispiel-Ausgabe am Ende:

```
============================================================
BATCH-ERGEBNIS
============================================================
   1. 0.8742  svm      char_wb  ml=1    (312s)  svm_char_wb_ml1_sw0_20260327_...
   2. 0.8701  svm      char_wb  ml=2    (298s)  svm_char_wb_ml2_sw0_20260327_...
   3. 0.8655  logistic char_wb  ml=1    (87s)   logistic_char_wb_ml1_sw0_...
   ...
```

---

## 6. Batch-Training + Auto-Tune auf der VM

Das ist die stärkste Kombination: **jede Konfiguration wird zusätzlich intern
per RandomizedSearchCV optimiert**.

```bash
# Empfehlung für LRZ-VM (20 vCPUs, V100)
python sachgruppen_classifier.py --csv daten.csv \
  --model svm logistic xgboost \
  --analyzer char_wb word \
  --stopwords false true \
  --tune \
  --tune-n-iter 20 \
  --tune-cv 5 \
  --output-dir models/lrz_batch_run1
```

### Warum das auf der VM Sinn ergibt

`RandomizedSearchCV` nutzt `n_jobs=-1`, d. h. alle verfügbaren Kerne laufen parallel.
Bei 20 vCPUs und `n-iter=20`, `cv=5` = 100 Fits laufen alle 100 gleichzeitig verteilt,
statt sequenziell. Das reduziert die Zeit pro Konfiguration um **Faktor ~5–10** gegenüber
einem 8-Kern-Laptop.

**Grobe Zeitschätzung pro Konfiguration auf der VM:**

| Modell | Einzeltraining | + Auto-Tune (n=20, cv=5) |
|---|---|---|
| SVM | ~1–2 min | ~3–5 min |
| Logistic | ~0.5 min | ~1–2 min |
| XGBoost | ~3–5 min | ~8–12 min |

Bei 12 Batch-Konfigurationen à SVM: ca. **45–60 Minuten** – gut geeignet für einen
über Nacht laufenden Job.

### Tipp: Im Hintergrund laufen lassen

```bash
nohup python sachgruppen_classifier.py --csv daten.csv \
  --model svm logistic xgboost \
  --analyzer char_wb word \
  --stopwords false true \
  --tune --tune-n-iter 20 --tune-cv 5 \
  --output-dir models/run1 \
  --notify deine@email.de \
  > run1.log 2>&1 &

echo "PID: $!"
tail -f run1.log   # optional: Fortschritt live verfolgen
```

---

## 7. Vorhersage mit trainiertem Modell

```bash
python sachgruppen_classifier.py --predict --load models/svm_char_wb_ml1_sw0_....pkl
```

---

## 8. E-Mail-Benachrichtigung

Mit `--notify EMAIL` wird nach Abschluss (Einzel- oder Batch-Lauf) automatisch eine
Zusammenfassung per E-Mail verschickt.

```bash
python sachgruppen_classifier.py --csv daten.csv --model svm logistic \
  --tune --notify deine@email.de
```

Die Mail enthält Accuracy, Trainingszeit und bei Batch-Läufen alle Ergebnisse
sortiert nach Accuracy.

### Einrichtung (einmalig)

SMTP-Zugangsdaten in die `.env`-Datei eintragen:

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=deine@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
SMTP_FROM=deine@gmail.com        # optional, Fallback auf SMTP_USER
```

**Gmail:** Das normale Passwort funktioniert nicht – ein App-Passwort ist nötig.
Erstellen unter: Konto → Sicherheit → 2-Schritt-Verifizierung → App-Passwörter.

**LRZ/Institutsmailserver:** `SMTP_HOST` und `SMTP_PORT` entsprechend anpassen,
ggf. `SMTP_PORT=25` ohne STARTTLS (dann ggf. Anpassung im Code nötig).

Ist kein SMTP konfiguriert, wird die Benachrichtigung still übersprungen –
das Training läuft normal weiter.

---

## 9. Alle Parameter auf einen Blick

```
python sachgruppen_classifier.py --help
```
