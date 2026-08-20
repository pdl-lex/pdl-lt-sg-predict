# Datenextraktion — `scripts/extract_data.py`

Das Skript erzeugt die **Trainingsgrundlage** der Sachgruppen-Vorhersage: Es liest
TEI-Lex-0-Artikel, zieht daraus je Bedeutung ein Tripel *Lemma · Bedeutung · Sachgruppe*
und schreibt es als CSV nach `data/`.

```bash
uv run python scripts/extract_data.py --quelle db          # Regelfall
```

Der Zugang zur Datenbank, die Sollzahlen des Bestands und die DB-seitigen Fallstricke
stehen im Schwesterdokument [Readme Access WBDB.md](Readme%20Access%20WBDB.md). Dieses
Dokument beschreibt, **was das Skript daraus macht**.

---

## 1 · Was passiert, der Reihe nach

1. **Systematik laden** — `data/sachgruppen.csv` liefert die Menge gültiger
   Sachgruppennummern (Abschnitt 4). Mit `--ohne-pruefung` entfällt dieser Schritt.
2. **Quelle öffnen** — entweder ein serverseitiger Cursor über die Wörterbuch-Datenbank
   oder ein rekursiver Durchlauf durch ein Verzeichnis mit XML-Dateien (Abschnitt 2).
3. **Je Artikel parsen** — `extract_from_xml()` zieht Lemma und alle Bedeutungen mit
   Sachgruppencode heraus (Abschnitt 3). Ein Artikel ohne Lemma oder ohne codierte
   Bedeutung liefert schlicht keine Zeile.
4. **Prüfen** — unbekannte Codes fallen weg; bleibt für eine Bedeutung kein gültiger Code
   übrig, entfällt die Zeile ganz (Abschnitt 4).
5. **Schreiben** — CSV, im DB-Modus zusätzlich eine Herkunfts-Begleitdatei; danach eine
   Statistik auf die Konsole (Abschnitt 5).

Alles wird im Arbeitsspeicher als Liste von Tupeln gesammelt und erst am Ende geschrieben —
die 125.719 Zeilen sind dafür klein genug, der gut 1 GB große TL0-Bestand wäre es nicht.
Deshalb wird der blockweise gelesen und nicht am Stück geholt.

---

## 2 · Zwei Quellen, dieselbe Parselogik

| | `--quelle db` | `--quelle dateien` |
|---|---|---|
| Herkunft | View `source.article` + `source.derivative` (`form = 'tl0'`) | Verzeichnis, rekursiv `**/*.xml` |
| Auswahl | `--ressourcen` (Standard `wbf dibs`) | alles unterhalb des Verzeichnisses |
| Korpus-Spalte | aus `resource_id` der Abfrage | aus dem Pfad (`bdo-tl0/wbf/A/…`) |
| Herkunftsdatei | ja | **nein** (die Angaben stehen nur in der DB) |
| Abhängigkeit | `psycopg`, `.env` im Repo-Wurzelverzeichnis | keine |

**Der DB-Modus ist der Regelfall.** Das TL0-Derivat hängt am Inhaltshash des Artikels und
kann deshalb nicht gegenüber dem Artikel veralten. Der Dateiordner `bdo-tl0/` kann das sehr
wohl und tut es auch: 145.362 Dateien bei 151.758 Artikeln, zusätzlich beschädigt durch
einen Windows-Case-Clash. Der Dateimodus bleibt nur erhalten, um **alte Extraktionen zu
reproduzieren** — nicht, um neue zu erzeugen.

`bwb` steht nicht im Standard, weil dort **0 von 66.776** Bedeutungen eine Sachgruppe
tragen. Ein Drittel des Bestands muss so gar nicht erst durch den Parser.

Gelesen wird über einen **benannten** Cursor (`conn.cursor(name="tl0")`, `itersize = 200`),
also serverseitig und blockweise. Er lebt in einer Transaktion: Wer währenddessen auf
derselben Verbindung schreibt und committet, schließt ihn sich selbst.

---

## 3 · Was aus dem TL0 gezogen wird

```xml
<form type="lemma"><orth>Katze</orth></form>
…
<sense>
  <def>weibliche Hauskatze</def>
  <usg type="domain" ana="#sg_6121"/>
  <usg type="domain" ana="#sg_5142"/>
</sense>
```

wird zu

```csv
lemma,bedeutung,sachgruppe,korpus,sachgruppen_alle
Katze,weibliche Hauskatze,6121,wbf,6121;5142
```

| Feld | Herkunft |
|---|---|
| `lemma` | erstes `form[@type="lemma"]/orth`, `.text` getrimmt — fehlt es, liefert der Artikel nichts |
| `bedeutung` | `sense/def`, `.text` getrimmt — fehlt es, entfällt die Bedeutung (17 Fälle im Bestand) |
| `sachgruppe` | **erster** `usg[@type="domain"]`, `ana` ohne das Präfix `#sg_` |
| `korpus` | `resource_id` (DB) bzw. Pfadsegment (Dateien) |
| `sachgruppen_alle` | **alle** Codes des `sense`, dedupliziert, in Dokumentreihenfolge, `;`-getrennt |

**Eine Zeile je Bedeutung, nicht je Artikel.** Ein Artikel mit acht codierten Bedeutungen
liefert acht Zeilen mit demselben Lemma.

**`find()` nimmt bewusst nur den ersten Code.** 19.272 Bedeutungen (15,3 %) tragen mehrere;
`sachgruppe` bleibt trotzdem einwertig, weil das Modell einwertig ist. Zwei Punkte, die man
dabei kennen muss:

- Die Reihenfolge im XML ist zu 99,25 % aufsteigend sortiert — der erste Code ist der
  **kleinste**, nicht der wichtigste.
- 468 Sachgruppen haben Belege, aber nur 427 werden je als Label vergeben: **41 Klassen
  unterschlägt `find()` vollständig**, z. B. 5144 „dünn/hager" mit 284 Belegen und null
  Labels. (Gemessen ohne Systematik-Prüfung; mit Prüfung bleiben 391 Label-Klassen.)

Genau dafür gibt es `sachgruppen_alle`. Die Spalte ist **nicht fürs Training** gedacht —
der Trainer liest Spalten über Namen und sieht sie gar nicht —, sondern für eine ehrliche
Bewertung: Eine Vorhersage, die dort drinsteht, ist fachlich richtig, auch wenn sie nicht
`sachgruppe` ist. `scripts/evaluate_setaware.py` misst das (0,8375 strikt → 0,8529
set-aware; 9,5 % der „Fehler" sind gültige Zweitcodes).

Ob stattdessen *alle* Codes trainiert werden sollen, ist eine **fachliche** Entscheidung,
keine technische — und sie ändert die Trainingsbasis, alle Bestwerte wären danach neu zu
messen.

Ein Parse-Fehler bricht den Lauf **nicht** ab: Die Meldung geht auf die Konsole, der
Artikel wird übersprungen, der Rest läuft weiter.

---

## 4 · Systematik-Prüfung

Jeder Code wird gegen `data/sachgruppen.csv` geprüft (Spalte `Nummer`, Semikolon-getrennt,
`utf-8-sig`):

- unbekannte Codes fallen aus `sachgruppen_alle` heraus;
- bleibt **kein** gültiger Code, entfällt die Bedeutung ganz;
- war der erste Code ungültig, rückt der erste **gültige** als Label nach.

Was dabei wegfällt, landet nicht stillschweigend im Nichts, sondern auf der Konsole und —
im DB-Modus — unter `systematik_pruefung` in der Herkunftsdatei:

```
Systematik-Prüfung: 788 Codezuordnungen verworfen (76 verschiedene Codes)
  Bedeutungen ganz entfallen (kein gültiger Code): 206
  Bedeutungen mit anderem Label (erster gültiger): 450
  häufigste unbekannte Codes: 0 (393x), 7721 (157x), 7503 (90x), 5332 (49x), …
```

> **`data/sachgruppen.csv` ist unvollständig — nicht als Wahrheit behandeln.** Die Liste
> hat 394 Nummern, die Daten benutzen 468. Von den 788 verworfenen Zuordnungen entfallen
> 689 auf vier Codes, und drei davon sind keine Tippfehler: `7721` (157x, direkt neben
> `7720 Teile des Hauses`), `7503` (90x, zwischen `7501 Gerät` und `7505 Werkzeug`),
> `5332` (49x, neben `5330 Atmung`). Nur `0` (393x) ist echter Müll. Wer die Systematik
> ergänzt, gewinnt ~300 korrekt erfasste Zuordnungen zurück.

`--ohne-pruefung` stellt das Verhalten vor Einbau der Prüfung her (125.925 statt 125.719
Zeilen) und ist damit das, was man für die **Reproduktion älterer CSVs** braucht.

---

## 5 · Was herauskommt

### Die CSV

Fünf Spalten, `,`-getrennt, UTF-8, Header in Zeile 1:

```
lemma,bedeutung,sachgruppe,korpus,sachgruppen_alle
```

Das Trennzeichen **innerhalb** von `sachgruppen_alle` ist `;` — der Trainer liest mit
`pd.read_csv(sep=None)` und schnüffelt den Spaltentrenner; das `;` verwirrt ihn nachweislich
nicht. Fehlende Elternverzeichnisse der Ausgabe werden angelegt.

`--ohne-korpus` und `--ohne-mengen` schalten die hinteren Spalten ab; zusammen ergeben sie
das alte Drei-Spalten-Format.

> `woerterbuch_daten_*.csv` ist über `.gitignore` (Zeile 49) ausgenommen — die Exporte
> gehören nicht ins Repository.

### Die Herkunftsdatei

Im DB-Modus entsteht neben der CSV eine `<ausgabe>.herkunft.json`:

```json
{
  "erzeugt_am": "2026-08-19T11:11:49+02:00",
  "quelle": "db",
  "ressourcen": ["wbf", "dibs"],
  "import_id": 2,
  "snapshot": "2026-08",
  "source_archive": "bdo_xml_dump_08-2026.zip",
  "imported_at": "2026-08-18T14:43:24.409811+00:00",
  "pipeline_git": "2295c34-dirty+xsl.da076125e321",
  "csv": "data\\woerterbuch_daten_125719.csv",
  "spalten": ["lemma", "bedeutung", "sachgruppe", "korpus", "sachgruppen_alle"],
  "zeilen": 125719,
  "systematik_pruefung": { "…": "…" }
}
```

**Ohne diese beiden Werte ist eine CSV in einem halben Jahr nicht mehr reproduzierbar:**
`import_id` sagt, *welcher Bestand* gelesen wurde, `pipeline_git`, *welche XSLT* ihn
erzeugt hat. `--ohne-herkunft` schaltet die Datei ab — dafür sollte es einen Grund geben.

### Die Statistik auf der Konsole

Einzigartige Lemmata, einzigartige Sachgruppen, Bedeutungen je Lemma, Zeilen je Korpus
sowie — wenn `sachgruppen_alle` geschrieben wird — der Anteil mehrdeutiger Bedeutungen und
die Zahl der Klassen mit Belegen gegenüber denen, die je als Label vorkommen.

---

## 6 · Parameter

| Parameter | Gilt für | Standard | Wirkung |
|---|---|---|---|
| `input_dir` (positional) | dateien | — | Verzeichnis mit den TL0-XML-Dateien, rekursiv durchsucht. Ist es angegeben, ist `--quelle dateien` der Standard; zusammen mit `--quelle db` ist es ein Fehler. |
| `--quelle {db,dateien}` | — | `dateien`, wenn ein Verzeichnis angegeben ist, sonst `db` | Woher gelesen wird. |
| `-o`, `--output PFAD` | beide | `data/woerterbuch_daten.csv` | Ausgabe-CSV. Fehlende Elternverzeichnisse werden angelegt, eine bestehende Datei wird **ohne Rückfrage überschrieben**. |
| `--ressourcen ID […]` | **nur db** | `wbf dibs` | Welche Korpora gelesen werden; möglich sind `wbf`, `dibs`, `bwb`. Unbekannte Kennungen brechen ab. Bei `--quelle dateien` ist der Parameter ein Fehler. |
| `--limit N` | beide | alles | Nur die ersten N **Artikel bzw. Dateien** — nicht Zeilen. Für Schnelltests. |
| `--ohne-korpus` | beide | aus | CSV ohne die Spalte `korpus`. |
| `--ohne-mengen` | beide | aus | CSV ohne die Spalte `sachgruppen_alle`. Mit `--ohne-korpus` zusammen: das alte Drei-Spalten-Format. |
| `--ohne-herkunft` | **nur db** | aus | Keine Begleitdatei `<ausgabe>.herkunft.json`. Im Dateimodus wirkungslos — dort entsteht ohnehin keine. |
| `--systematik CSV` | beide | `data/sachgruppen.csv` | Liste der gültigen Sachgruppen; gelesen wird die Spalte `Nummer`. Fehlt die Datei oder die Spalte, bricht der Lauf ab. |
| `--ohne-pruefung` | beide | aus | Codes **nicht** gegen die Systematik prüfen — übernimmt auch unbekannte, wie vor Einbau der Prüfung. |

**`--limit` im DB-Modus liefert keine stabile Teilmenge.** Die Abfrage hat kein `ORDER BY`;
welche N Artikel kommen, entscheidet der Ausführungsplan. Im Dateimodus ist die Auswahl
dagegen deterministisch (`sorted()`), also für Vergleichsläufe geeignet.

---

## 7 · Rezepte

```bash
# Regelfall: Neuextraktion aus der Datenbank
uv run python scripts/extract_data.py --quelle db -o data/woerterbuch_daten_neu.csv

# Schnelltest, 200 Artikel, ohne Herkunftsdatei
uv run python scripts/extract_data.py --quelle db --limit 200 \
    --ohne-herkunft -o data/probe.csv

# Alles inklusive bwb (bringt keine zusätzlichen Zeilen — bwb trägt keine Sachgruppen)
uv run python scripts/extract_data.py --quelle db --ressourcen wbf dibs bwb -o data/alles.csv

# Eine alte Drei-Spalten-CSV reproduzieren
uv run python scripts/extract_data.py --quelle db \
    --ohne-korpus --ohne-mengen --ohne-pruefung -o data/alt_format.csv

# Altbestand aus Dateien (nur für Reproduktionen)
uv run python scripts/extract_data.py ~/Nextcloud/BAdW/Wörterbuchdaten/bdo-tl0/
```

---

## 8 · Sollzahlen

Gemessen am 2026-08-19, Import 2 (`2026-08`), XSLT `2295c34-dirty+xsl.da076125e321`,
Ressourcen `wbf dibs`:

| | |
|---|---:|
| gelesene Artikel | 117.262 |
| CSV-Zeilen **ohne** Systematik-Prüfung | 125.925 |
| CSV-Zeilen **mit** Systematik-Prüfung | **125.719** |
| Klassen als Label vergeben | 391 |
| Bedeutungen mit mehr als einem Code | 19.272 (15,3 %) |
| Laufzeit inkl. Parsen | ~65 s |

Trifft ein Lauf diese Zahlen, ist er korrekt. Auf derselben 200-Artikel-Teilmenge sind
DB- und Dateimodus zeilengleich.

---

## 9 · Fallstricke

**Die Zeilenzahl ist ein Vertrag.** Wer am Parser schraubt, prüft gegen Abschnitt 8. Eine
stillschweigend geänderte Trainingsbasis macht jeden Vergleich mit früheren Modellwerten
wertlos — der Bestwert 0,8375 hängt an einer bestimmten CSV.

**Lemma-Groß-/Kleinschreibung ist bedeutungstragend** (Verb vs. Substantiv, 86 Paare).
Niemals auf `lower(lemma)` deduplizieren, weder hier noch nachgelagert.

**`.text` statt `itertext()`** ist hier vertretbar, weil die XSLT die Bedeutungen über
`xsl:value-of` plättet. Für jeden **direkten** Zugriff auf das BDO-XML gilt das nicht — dort
schneidet `.text` beim ersten Kindelement ab.

**Bytes bleiben Bytes.** Die DB-Nutzlast geht als `bytes` an `ET.fromstring()`, der Parser
liest die Kodierungsdeklaration selbst. Wer unterwegs dekodiert, kommt mit `xml.etree` noch
durch, mit `lxml` nicht mehr.

**`process_single_file()`** existiert im Modul, hängt aber an keinem CLI-Schalter — sie ist
nur über einen Import erreichbar.

---

## 10 · Wenn nichts geht

| Meldung | Ursache |
|---|---|
| `psycopg fehlt — es wird nur für --quelle db gebraucht` | `uv sync` fehlt; `psycopg[binary]` liegt in der dev-Gruppe von `pyproject.toml` |
| `Zugangsdaten fehlen: POSTGRES_USER, …` | keine `.env` im Repo-Wurzelverzeichnis; Vorlage `.env.example`, Werte aus `pdl-lt-wbdb/.env` |
| `connection refused` auf 5433 | Docker Desktop läuft nicht, oder im DB-Repo fehlt `docker compose up -d` |
| `relation "source.article" does not exist` | Schema nicht geladen — siehe `pdl-lt-wbdb/readme/setup.md` |
| `Systematik nicht gefunden: …` | `--systematik` zeigt ins Leere, oder `data/sachgruppen.csv` fehlt; notfalls `--ohne-pruefung` |
| `Gefunden: 0 Artikel` | kein Import aktiv gesetzt (`SELECT * FROM source.current_import`) oder `--ressourcen` zu eng |
| Cursor bricht mitten im Lauf ab | jemand hat auf derselben Verbindung geschrieben und committet — Lesen und Schreiben gehören auf zwei Verbindungen |

Alles Weitere zur Datenbank: [Readme Access WBDB.md](Readme%20Access%20WBDB.md),
Abschnitt 10.
