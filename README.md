# Sachgruppen-Klassifikation für Fränkisches Wörterbuch

Automatische Vorhersage von Sachgruppen aus Bedeutungsdefinitionen mittels Machine Learning.

## 📁 Projektstruktur

```
.
├── extract_data.py              # Extraktion aus XML-Dateien
├── sachgruppen_classifier.py    # Haupt-ML-Pipeline
├── model_comparison.py          # Vergleich verschiedener Modelle
├── PROJEKTPLAN.md              # Detaillierte Dokumentation
└── README.md                   # Diese Datei
```

## 🚀 Quick Start

### 1. Daten extrahieren

```bash
# Einzelne Datei (Test)
python extract_data.py

# Alle XML-Dateien in einem Verzeichnis
# Editiere extract_data.py und ändere die letzte Zeile zu:
# process_directory('/pfad/zu/xml/dateien', 'woerterbuch_daten.csv')
python extract_data.py
```

**Output**: CSV-Datei mit Spalten: `lemma`, `bedeutung`, `sachgruppe`

### 2. Modell trainieren

```bash
# Einfaches Training mit SVM (empfohlen)
python sachgruppen_classifier.py --csv woerterbuch_daten.csv --model svm

# Mit Hyperparameter-Tuning (dauert länger, bessere Performance)
python sachgruppen_classifier.py --csv woerterbuch_daten.csv --model svm --tune

# Andere Modelle testen
python sachgruppen_classifier.py --csv woerterbuch_daten.csv --model logistic
python sachgruppen_classifier.py --csv woerterbuch_daten.csv --model rf
python sachgruppen_classifier.py --csv woerterbuch_daten.csv --model xgboost  # Benötigt xgboost
```

### 3. Modell verwenden

```bash
# Interaktiver Modus
python sachgruppen_classifier.py --predict --load sachgruppen_model.pkl
```

### 4. In Python verwenden

```python
from sachgruppen_classifier import SachgruppenClassifier

# Modell laden
clf = SachgruppenClassifier.load('sachgruppen_model.pkl')

# Vorhersagen
bedeutungen = [
    "spröder Bursche",
    "Schiedsrichter beim Fußball",
    "ungewolltes Kind"
]

sachgruppen = clf.predict(bedeutungen)
for bedeutung, sg in zip(bedeutungen, sachgruppen):
    print(f"{bedeutung} → Sachgruppe {sg}")
```

## 📊 Modellvergleich

| Modell | Geschwindigkeit | Accuracy | Empfohlen für |
|--------|----------------|----------|---------------|
| **Linear SVM** ⭐ | ⚡⚡⚡ | 🎯🎯🎯 | **Produktions-System** |
| Logistic Regression | ⚡⚡⚡ | 🎯🎯 | Baseline, Prototyping |
| Random Forest | ⚡⚡ | 🎯🎯 | Feature-Analyse |
| XGBoost | ⚡ | 🎯🎯🎯🎯 | Maximale Performance |

**Empfehlung für dein Projekt**: Start mit **Linear SVM**

## 🔧 Abhängigkeiten

### Minimal (für SVM/Logistic/RF)
```bash
pip install pandas numpy scikit-learn
```

### Optional (für XGBoost)
```bash
pip install xgboost
```

### Für Visualisierung
```bash
pip install matplotlib seaborn
```

## 📈 Erwartete Performance

Bei 70.000 Wörterbuch-Einträgen:

- **Baseline (Logistic Regression)**: 60-75% Accuracy
- **Optimiert (Linear SVM)**: 75-85% Accuracy
- **Best Case (XGBoost mit Tuning)**: 85-90% Accuracy

Performance hängt ab von:
- Anzahl der Sachgruppen (mehr = schwieriger)
- Class Balance (unbalanciert = schwieriger)
- Qualität der Definitionen (kurz/lang, eindeutig/mehrdeutig)

## 🎓 Workflow-Beispiel

### Vollständiger Ablauf für 70k Dateien

```bash
# 1. Alle XML-Dateien verarbeiten
python extract_data.py  # Editiere vorher den Pfad!

# 2. Baseline-Modell trainieren
python sachgruppen_classifier.py \
    --csv woerterbuch_daten.csv \
    --model logistic \
    --save baseline_model.pkl

# 3. SVM-Modell mit Tuning trainieren
python sachgruppen_classifier.py \
    --csv woerterbuch_daten.csv \
    --model svm \
    --tune \
    --save svm_tuned_model.pkl

# 4. XGBoost für maximale Performance
python sachgruppen_classifier.py \
    --csv woerterbuch_daten.csv \
    --model xgboost \
    --tune \
    --save xgboost_model.pkl

# 5. Bestes Modell verwenden
python sachgruppen_classifier.py \
    --predict \
    --load svm_tuned_model.pkl
```

## 🔍 Feature Engineering

Das Modell verwendet **TF-IDF** mit Character-level n-grams:

```python
TfidfVectorizer(
    ngram_range=(1, 3),      # Uni-, Bi-, Trigrams
    analyzer='char_wb',       # Character n-grams (erfasst Morphologie)
    max_features=10000,       # Top-10k Features
    min_df=2,                 # Mindestens 2x vorkommen
    sublinear_tf=True         # Log-scaling
)
```

**Warum Character n-grams?**
- Erfasst deutsche Morphologie (z.B. Komposita)
- Robust gegen Schreibvarianten
- Funktioniert mit OOV-Wörtern

## 📝 Beispiel-Output

```
Dataset Info:
  Anzahl Einträge: 68523
  Anzahl Sachgruppen: 342
  Durchschn. Bedeutungslänge: 28.4 Zeichen

Top-10 Sachgruppen:
6121    4521
7010    3892
6114    3245
...

Trainiere SVM-Modell...
Trainingsbeispiele: 54818
Anzahl Klassen: 342
Training abgeschlossen!

EVALUATION
============================================================
Accuracy: 0.8234

Klassifikations-Report:
              precision    recall  f1-score   support
        6121       0.85      0.87      0.86       905
        7010       0.82      0.80      0.81       778
        6114       0.79      0.83      0.81       649
        ...
```

## 🎯 Nächste Schritte

1. **Fehleranalyse**: Welche Bedeutungen werden falsch klassifiziert?
2. **Feature Engineering**: 
   - Lemma als zusätzliches Feature
   - Textlänge
   - Spezielle Keywords
3. **Ensemble**: Kombination mehrerer Modelle
4. **BERT**: Falls klassische Modelle nicht gut genug

## 🐛 Troubleshooting

### "Memory Error" beim Training
→ Reduziere `max_features` in der TfidfVectorizer-Konfiguration

### Niedriger Accuracy
→ Überprüfe Class Balance mit `df['sachgruppe'].value_counts()`
→ Verwende Hyperparameter-Tuning (`--tune` Flag)

### XGBoost nicht gefunden
→ `pip install xgboost`

## 📚 Weitere Ressourcen

- **PROJEKTPLAN.md**: Detaillierte Erklärungen und Theorie
- **model_comparison.py**: Benchmark verschiedener Algorithmen
- Scikit-learn Docs: https://scikit-learn.org/

## 🤝 Contributing

Vorschläge für Verbesserungen:
1. Zusätzliche Features (z.B. POS-Tags)
2. Weitere Modelle (z.B. BERT)
3. Bessere Hyperparameter
4. Visualisierungen

## 📄 Lizenz

Daten: CC-BY-SA 4.0 (Bayerische Akademie der Wissenschaften)
Code: MIT License
