# Anleitung: LexoTerm Sachgruppen-Klassifikation

Dieses Werkzeug trainiert Machine-Learning-Modelle zur automatischen Klassifikation von Wörterbuch-Einträgen in Sachgruppen. Die Eingabe besteht aus Lemma und Bedeutung, die Ausgabe ist eine Sachgruppen-Nummer.

Für den programmatischen Zugriff steht eine REST-API bereit (Modul **API-Referenz**,
erreichbar über den Rail-Button unten links) mit Beispielabfragen sowie Links zur
interaktiven Swagger-UI (`/docs`) und zum OpenAPI-Schema.

---

## 1. Trainingsdaten (CSV)

Die Trainingsdaten müssen als CSV-Datei mit **Semikolon- oder Komma-Trennung** vorliegen und mindestens drei Spalten enthalten:

| Spalte | Inhalt | Beispiel |
|:---|:---|:---|
| `lemma` | Stichwort / Headword | `Waggala` |
| `bedeutung` | Bedeutungsangabe | `kleines Kind; wackelig auf den Beinen` |
| `sachgruppe` | Sachgruppen-Nummer | `1830` |

**Hinweise:**
- Leere Zeilen und NaN-Werte in den drei Pflichtspalten werden automatisch entfernt.
- Klassen mit nur einem einzigen Sample werden ins Trainingsset verschoben und nicht evaluiert.
- Je mehr Samples pro Klasse, desto besser die Erkennungsrate für diese Klasse.


---


## 2. ML-Modelle

Es stehen fünf Algorithmen zur Verfügung. Alle nutzen intern eine **TF-IDF-Vektorisierung** der Texte.

### Warum TF-IDF?

ML-Modelle können nicht direkt mit Text arbeiten — sie brauchen Zahlen. TF-IDF wandelt jeden Eintrag (Lemma + Bedeutung) in einen numerischen Vektor um. Jede Dimension dieses Vektors entspricht einem Wort oder n-Gramm aus dem gesamten Trainingsvokabular; der Wert in dieser Dimension ist das **TF-IDF-Gewicht**:

- **TF** (Term Frequency): Wie oft kommt das Wort in *diesem* Eintrag vor?
- **IDF** (Inverse Document Frequency): Wie selten ist das Wort *über alle* Einträge? Seltene, charakteristische Wörter (z. B. „Pflug") erhalten ein hohes Gewicht; häufige Funktionswörter (z. B. „der", „zum") ein niedriges.

Das Produkt TF × IDF ist direkt der Wert in der jeweiligen Dimension — es gibt keine separate Dimension für das Gewicht. Der Vektorisierer wird einmalig auf den Trainingsdaten berechnet und zusammen mit dem Modell in der `.pkl`-Datei gespeichert, damit er bei der Vorhersage identisch angewendet werden kann.

### Linear SVM (`svm`)
Klassischer Support Vector Machine-Klassifikator (liblinear, One-vs-Rest). Guter Standard für Textklassifikation: schnell, robust, meist gute Accuracy.

- **Stärken:** Schnell, gut bei hochdimensionalen Sparse-Features (TF-IDF), bewährt für Texte.
- **Schwächen:** Ohne Zusatzschritt keine Wahrscheinlichkeitsschätzungen (Vorhersage-Konfidenz ist 0) – per „Kalibrieren" nachrüstbar (→ Abschnitt 5).
- **Trainingsdauer** (~113k Samples): ca. 2 Minuten (mit Kalibrierung ca. 3×).

### Logistic Regression (`logistic`)
Lineares Modell mit Softmax-Ausgabe (SAGA-Solver). Ähnlich wie SVM, gibt aber echte Wahrscheinlichkeiten aus.

- **Stärken:** Liefert kalibrierte Wahrscheinlichkeiten pro Klasse.
- **Schwächen:** Deutlich langsamer als SVM bei großen Datensätzen.
- **Trainingsdauer** (~113k Samples): ca. 70 Minuten.

### Random Forest (`rf`)
Ensemble aus 100 Entscheidungsbäumen (max. Tiefe 20). Parallelisiert über alle CPU-Kerne.

- **Stärken:** Schnell, robust gegenüber Ausreißern, liefert Wahrscheinlichkeiten.
- **Schwächen:** Etwas schwächer als SVM/XGBoost bei reiner Textklassifikation.
- **Trainingsdauer** (~113k Samples): ca. 30 Sekunden.

### Neural Network / MLP (`nn`)
Mehrschichtiges Perceptron (Standard: eine Schicht mit 100 Neuronen, ReLU, Adam; weitere Schichten über die Hyperparameter konfigurierbar). Stoppt automatisch bei Stagnation (Early Stopping).

- **Stärken:** Kann nicht-lineare Muster lernen.
- **Schwächen:** Braucht viele Daten, Training dauert länger, schwerer zu interpretieren.
- **Trainingsdauer** (~113k Samples): ca. 2 Minuten.

### XGBoost (`xgboost`)
Gradient-Boosted Trees (Standard: 300 Bäume, max. Tiefe 6). Oft die höchste Accuracy, aber sehr langsam.

- **Stärken:** Häufig beste Accuracy aller Modelle.
- **Schwächen:** Sehr langsames Training, hoher Speicherbedarf.
- **Trainingsdauer** (~113k Samples): ca. 100 Minuten.

---

## 3. Parameter

### Test-Anteil
Der Anteil der Daten, der **nicht** zum Training genutzt wird, sondern ausschließlich zur Evaluation (Accuracy, Klassifikationsreport). Standard: 20 %. Kleinere Werte geben dem Modell mehr Trainingsdaten, aber die Evaluationszahlen werden weniger belastbar.

### Stoppwörter entfernen
Entfernt häufige Funktionswörter (Artikel, Präpositionen, Hilfsverben) aus der Datei `stopwords_de.txt` vor der Vektorisierung. Sinnvoll, wenn man den Effekt von Inhaltswörtern isolieren möchte.

### Min. Wortlänge
Wörter mit weniger Zeichen als dieser Schwellwert werden vor der Vektorisierung entfernt. Bei `1` (Standard) bleiben alle Wörter erhalten. Ab `2` fallen Einzelbuchstaben weg, ab `3` auch zweistellige Abkürzungen.

### Analyzer
Bestimmt, welche Texteinheiten als Features verwendet werden:

- **`char_wb`** (Standard): Zeichenfolgen innerhalb von Wortgrenzen. Robust gegenüber Tippfehlern, Flexionsformen und Komposita. Empfohlen für morphologisch reiche Sprachen wie Deutsch.
- **`word`**: Ganze Wörter als Features. Intuitiver, aber anfälliger für unbekannte Wortformen.

### N-Gramm (nur bei `word`-Analyzer)
- **1**: Nur einzelne Wörter als Features.
- **2**: Einzelwörter und Wortpaare (z. B. `"kleines Kind"` als ein Feature).

---

## 4. Cross-Validation (optional)

Ergänzt den normalen Train/Test-Split um eine **k-fache Kreuzvalidierung** auf dem
Gesamtdatensatz — eine vom Zufalls-Split unabhängige Genauigkeitsschätzung. Das Modell wird
dafür zusätzlich `k`-mal trainiert (Dauer ≈ `(1 + k) ×` Einzeltraining). Nur beim
Einzeltraining verfügbar, nicht im Batch.

**Fold-Strategie:**
- **Stratified** (Standard): Erhält die Klassenverteilung je Fold.
- **Group**: `GroupKFold` nach der `bedeutung`-Zeichenkette — identische Glossen landen immer
  im selben Fold, nie gleichzeitig in Training und Test. Deckt auf, ob das Modell nur
  Dubletten memoriert statt echt zu generalisieren; ergibt eine realistischere, meist etwas
  niedrigere Schätzung als Stratified.

Ergebnis: Mittelwert ± Standardabweichung der Accuracy über alle Folds, zum Abgleich mit der
Split-Accuracy des regulären Trainings.

---

## 5. Konfidenz-Kalibrierung (`svm`)

Eine lineare SVM liefert standardmäßig keine echten Wahrscheinlichkeiten, nur eine Distanz
zur Trennebene. Der Schalter **„Kalibrieren"** wrappt sie in ein Platt-Scaling
(`CalibratedClassifierCV`, Sigmoid) und macht die Konfidenz dadurch als Prozentwert
interpretierbar — Trainingsdauer steigt dabei auf etwa das Dreifache.

Wirkt nur bei der SVM: Logistic Regression, Random Forest, Neural Network und XGBoost liefern
über ihre eigene Modellstruktur bereits Wahrscheinlichkeiten; der Schalter ist dort nicht
sichtbar bzw. wirkungslos.

---

## 6. Batch-Training

Das Batch-Training trainiert automatisch **alle Kombinationen** der gewählten Optionen in einem Durchlauf. Man wählt jeweils mehrere Werte für:

- Modelltypen
- Stoppwörter (entfernen / behalten)
- Min. Wortlänge
- Analyzer

Aus dem **kartesischen Produkt** aller Auswahlmöglichkeiten entstehen alle Konfigurationen, die nacheinander trainiert werden. Die geschätzte Gesamtdauer wird vor dem Start angezeigt. Jedes trainierte Modell wird einzeln gespeichert und erscheint anschließend in der Analyse-Seite.

---

## 7. Analyse-Seite

Die Analyse-Seite zeigt alle gespeicherten Modelle in einer Tabelle. Die Tabelle lädt automatisch beim Seitenaufruf; über das Pfeil-Icon kann sie manuell neu geladen werden.

**Spalten:**

| Spalte | Bedeutung |
|:---|:---|
| Datei | Dateiname des gespeicherten Modells (`.pkl`) |
| Modell | Algorithmus-Name |
| Accuracy | Anteil korrekt klassifizierter Test-Samples |
| Zeit | Dauer des Trainings (HH:MM:SS) |
| Datum | Datum des Trainings |
| Samples | Anzahl der Trainingssamples |
| Klassen | Anzahl der Sachgruppen im Trainingsset |
| Test | Verwendeter Test-Anteil |
| Lemma | Wurde das Lemma als Feature genutzt? |
| spaCy | Semantische Anreicherung: spaCy-Wortvektoren aktiv? |
| Dornseiff | Semantische Anreicherung: Dornseiff-Gazetteer aktiv? |
| Min-Länge | Verwendete Mindest-Wortlänge |
| Analyzer | Verwendeter Analyzer |
| Stoppw. | Wurden Stoppwörter entfernt? |

**Modell auswählen:** Durch Klick auf eine Zeile (oder die Checkbox) wird das Modell markiert. Es erscheinen zwei Buttons:
- **Klassifikations-Report**: Öffnet den detaillierten Report (→ Abschnitt 8).
- **Modell für Vorhersage auswählen**: Wechselt zur Vorhersage-Seite mit diesem Modell vorausgewählt.

---

## 8. Klassifikations-Report

Nach jedem Training wird ein Klassifikations-Report gespeichert. Er zeigt pro Sachgruppe:

| Spalte | Bedeutung |
|:---|:---|
| `precision` | Anteil korrekt vorhergesagter Treffer dieser Klasse (Genauigkeit) |
| `recall` | Anteil aller echten Instanzen dieser Klasse, die erkannt wurden (Trefferquote) |
| `f1-score` | Harmonisches Mittel aus Precision und Recall |
| `support` | Anzahl Test-Samples dieser Klasse |

**Zusammenfassungszeilen:**

- **accuracy**: Gesamtanteil korrekt klassifizierter Samples.
- **macro avg**: Ungewichteter Durchschnitt über alle Klassen – jede Klasse zählt gleich viel, unabhängig von der Häufigkeit.
- **weighted avg**: Gewichteter Durchschnitt – häufige Klassen haben mehr Einfluss.

**Typische Muster:**
- Klassen mit wenigen Samples (`support` < 5) haben oft schlechte Werte – zu wenig Trainingsdaten.
- Große Differenz zwischen `macro avg` und `weighted avg` deutet auf Klassenimbalance hin: häufige Klassen werden gut erkannt, seltene schlecht.

---

## 9. Vorhersage

Auf der Vorhersage-Seite kann ein gespeichertes Modell geladen und für Klassifikationen verwendet werden.

### Einzelvorhersage
Eingabe von Lemma und Bedeutung → das Modell gibt die wahrscheinlichste Sachgruppe zurück. Bei Modellen, die Wahrscheinlichkeiten liefern (Logistic Regression, Random Forest, Neural Network sowie SVM mit aktivierter Kalibrierung, s. Abschnitt 5), wird zusätzlich die Konfidenz angezeigt; eine unkalibrierte SVM zeigt keine Konfidenz.

### Batch-Vorhersage
CSV-Datei mit den Spalten `lemma` und `bedeutung` hochladen → alle Einträge werden klassifiziert und das Ergebnis als Download-CSV bereitgestellt.

---

## 10. SHAP-Analyse

Nach einer Einzelvorhersage wird automatisch eine **SHAP-Erklärung** berechnet. SHAP (SHapley Additive exPlanations) zeigt, welche Wörter die Vorhersage beeinflusst haben:

- **Grün**: Das Wort unterstützt die vorhergesagte Sachgruppe.
- **Rot**: Das Wort widerspricht der Vorhersage (spricht für eine andere Sachgruppe).
- **Grau**: Das Wort hat kaum Einfluss.

Das Balkendiagramm zeigt die 10 einflussreichsten Wörter nach absolutem SHAP-Wert.

**Hinweis für Neural Networks:** Die SHAP-Berechnung für neuronale Netze ist rechenintensiv (ca. 30–60 Sekunden) und wird erst nach einem manuellen Klick auf „Erklärung anzeigen" gestartet.

Der Schalter **„Stoppwörter ausblenden"** filtert Funktionswörter aus der Darstellung heraus, sodass nur inhaltlich bedeutsame Wörter angezeigt werden.

---

## 11. Konfiguration (.env)

Die Datei `.env` im Projektordner enthält die Laufzeitkonfiguration:

| Variable | Bedeutung | Beispiel |
|:---|:---|:---|
| `MODELS_DIR` | Verzeichnis zur Speicherung der trainierten Modelle (im Entwicklungsbetrieb außerhalb des Projektordners sinnvoll, damit `uvicorn --reload` bei neuen Modellen nicht neu startet). Bei relativem Pfad: relativ zum Projektordner. | `models` oder `/data/sg-models` |
| `ENABLE_TRAINING` | Aktiviert oder deaktiviert die Training-Funktionalität (UI **und** `/api/training/*`-Endpunkte, letztere antworten dann mit 403). Auf öffentlich erreichbaren Instanzen **immer** auf `False` setzen — die Trainings-Endpunkte haben keine Authentifizierung. Im Docker-Image ist `False` voreingestellt. | `True` / `False` |
| `SESSIONS_DIR` | Verzeichnis für Uploads und Trainings-Fortschrittsdateien. Bei relativem Pfad: relativ zum Projektordner. | `.sessions` |
| `CORS_ORIGINS` | Erlaubte CORS-Origins (kommasepariert); nur für den Vite-Dev-Server nötig, in Produktion liefert das Backend das Frontend same-origin aus. | `http://localhost:5173` |

Nach Änderungen an `.env` muss die Anwendung neu gestartet werden.
