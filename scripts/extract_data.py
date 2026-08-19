#!/usr/bin/env python3
"""
Extrahiert Lemma, Bedeutung und Sachgruppe aus TEI-Lex-0-Daten in eine CSV-Datei.

Zwei Quellen, dieselbe Parselogik:

  --quelle db        aus der Wörterbuch-Datenbank (empfohlen, vollständig)
  --quelle dateien   aus einem Verzeichnis mit TL0-XML-Dateien (Altbestand)

Die Datenbank ist die verlässliche Quelle; der Dateiordner `bdo-tl0/` ist
überholt (145.362 Dateien bei 151.758 Artikeln). Der Dateimodus bleibt nur
erhalten, um alte Extraktionen reproduzieren zu können.

Zugang, Sollzahlen und Fallstricke: scripts/Readme Access WBDB.md
"""

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from tqdm import tqdm

# TEI Namespace
TEI_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ressourcen (Korpora) im Bestand.
RESSOURCEN = ('wbf', 'dibs', 'bwb')
# bwb trägt keine Sachgruppen (0 von 66.776 Bedeutungen) und muss deshalb nicht
# durch den Parser — das spart ein Drittel des Bestands.
RESSOURCEN_STANDARD = ('wbf', 'dibs')

# Das TL0-Derivat hängt am Inhaltshash des Artikels; deshalb kann es nicht
# gegenüber dem Artikel veralten.
ABFRAGE_TL0 = """
SELECT a.article_id, a.resource_id, d.content
FROM source.article a
JOIN source.derivative d
  ON d.source_sha256 = a.content_sha256 AND d.form = 'tl0'
WHERE a.resource_id = ANY(%s)
"""

ABFRAGE_ANZAHL = """
SELECT count(*)
FROM source.article a
JOIN source.derivative d
  ON d.source_sha256 = a.content_sha256 AND d.form = 'tl0'
WHERE a.resource_id = ANY(%s)
"""

CSV_SPALTEN = ['lemma', 'bedeutung', 'sachgruppe', 'korpus', 'sachgruppen_alle']

# Trennzeichen innerhalb von `sachgruppen_alle`. Der Trainer liest mit
# pd.read_csv(sep=None) und schnüffelt den Spaltentrenner; ';' verwirrt ihn
# nicht (geprüft), und es passt zu data/sachgruppen.csv.
MENGEN_TRENNER = ';'


# Die gültige Sachgruppen-Systematik. Codes, die hier nicht stehen, kommen in
# den Daten vereinzelt vor (Tippfehler u. Ä.) und werden verworfen.
SYSTEMATIK_STANDARD = REPO_ROOT / 'data' / 'sachgruppen.csv'


def lade_systematik(pfad):
    """Liest die gültigen Sachgruppennummern aus data/sachgruppen.csv."""
    pfad = Path(pfad)
    if not pfad.is_file():
        raise SystemExit(
            f"Systematik nicht gefunden: {pfad}\n"
            "Mit --systematik einen anderen Pfad angeben oder die Prüfung mit "
            "--ohne-pruefung abschalten."
        )
    codes = set()
    with open(pfad, newline='', encoding='utf-8-sig') as f:
        for zeile in csv.DictReader(f, delimiter=';'):
            nummer = (zeile.get('Nummer') or '').strip()
            if nummer:
                codes.add(nummer)
    if not codes:
        raise SystemExit(f"Systematik {pfad} enthält keine Nummern — Spalte 'Nummer' fehlt?")
    return codes


def neuer_bericht():
    """Sammelt, was die Systematik-Prüfung verworfen hat."""
    return {'verworfene_codes': Counter(), 'zeilen_ohne_code': 0, 'label_getauscht': 0}


def berichte_systematik(bericht):
    """Was verworfen wurde, gehört auf den Schirm — nicht stillschweigend weg."""
    if bericht is None:
        return
    verworfen = bericht['verworfene_codes']
    if not verworfen and not bericht['zeilen_ohne_code']:
        print("Systematik-Prüfung: keine unbekannten Sachgruppen.")
        return

    print(f"\nSystematik-Prüfung: {sum(verworfen.values())} Codezuordnungen verworfen "
          f"({len(verworfen)} verschiedene Codes)")
    print(f"  Bedeutungen ganz entfallen (kein gültiger Code): {bericht['zeilen_ohne_code']}")
    print(f"  Bedeutungen mit anderem Label (erster gültiger): {bericht['label_getauscht']}")
    haeufigste = verworfen.most_common(10)
    if haeufigste:
        print("  häufigste unbekannte Codes: "
              + ', '.join(f"{c} ({n}x)" for c, n in haeufigste))
        if len(verworfen) > len(haeufigste):
            print(f"  … und {len(verworfen) - len(haeufigste)} weitere")


def spaltenwahl(mit_korpus=True, mit_mengen=True):
    """Welche Spalten in die CSV geschrieben werden."""
    spalten = ['lemma', 'bedeutung', 'sachgruppe']
    if mit_korpus:
        spalten.append('korpus')
    if mit_mengen:
        spalten.append('sachgruppen_alle')
    return spalten


def extract_from_xml(quelle, korpus='', bezeichnung=None, gueltige=None, bericht=None):
    """
    Extrahiert Daten aus einem TEI-Lex-0-Dokument.

    Args:
        quelle: Pfad zu einer XML-Datei ODER die XML-Bytes selbst (aus der DB).
        korpus: Ressourcenkennung ('wbf', 'dibs', ...), wird durchgereicht.
        bezeichnung: Name für Fehlermeldungen (Standard: die Quelle selbst).
        gueltige: Menge gültiger Sachgruppennummern; None = keine Prüfung.
        bericht: dict aus neuer_bericht(), sammelt das Verworfene.

    Returns:
        Liste von Tupeln (lemma, bedeutung, sachgruppe, korpus, sachgruppen_alle)
    """
    results = []
    if bezeichnung is None:
        bezeichnung = quelle if not isinstance(quelle, (bytes, bytearray, memoryview)) else '<bytes>'

    try:
        # Bytes bleiben Bytes: der Parser liest die XML-Deklaration selbst.
        if isinstance(quelle, (bytes, bytearray, memoryview)):
            root = ET.fromstring(bytes(quelle))
        else:
            root = ET.parse(quelle).getroot()

        # Lemma extrahieren
        orth_elem = root.find('.//tei:form[@type="lemma"]/tei:orth', TEI_NS)
        if orth_elem is None or orth_elem.text is None:
            return results

        lemma = orth_elem.text.strip()

        # Alle Bedeutungen (senses) durchgehen
        for sense in root.findall('.//tei:sense', TEI_NS):
            # Bedeutung extrahieren
            def_elem = sense.find('./tei:def', TEI_NS)
            if def_elem is None or def_elem.text is None:
                continue

            bedeutung = def_elem.text.strip()

            # Sachgruppe extrahieren. find() nimmt bewusst nur den ersten Code —
            # 19.275 senses tragen mehrere. Das bleibt das Trainingslabel; ob
            # stattdessen alle Codes trainiert werden sollen, ist eine fachliche
            # Entscheidung und ändert die Trainingsbasis (siehe Readme Access WBDB.md).
            usg_elem = sense.find('./tei:usg[@type="domain"]', TEI_NS)
            if usg_elem is not None:
                ana = usg_elem.get('ana', '')
                # Sachgruppe ist im Format "#sg_6121", extrahiere nur die Nummer
                sachgruppe = ana.replace('#sg_', '')

                if sachgruppe:  # Nur hinzufügen wenn Sachgruppe vorhanden
                    # Zusätzlich ALLE Codes des sense, in Dokumentreihenfolge.
                    # Sie sind nicht fürs Training gedacht, sondern für eine
                    # ehrliche Bewertung: eine Vorhersage, die hier drinsteht,
                    # ist fachlich richtig, auch wenn sie nicht `sachgruppe` ist.
                    alle = []
                    for u in sense.findall('./tei:usg[@type="domain"]', TEI_NS):
                        code = u.get('ana', '').replace('#sg_', '')
                        if code and code not in alle:
                            alle.append(code)

                    # Gegen die Systematik prüfen: unbekannte Codes fallen weg,
                    # bleibt keiner übrig, entfällt die Bedeutung ganz.
                    if gueltige is not None:
                        bekannt = [c for c in alle if c in gueltige]
                        if bericht is not None:
                            for c in alle:
                                if c not in gueltige:
                                    bericht['verworfene_codes'][c] += 1
                        if not bekannt:
                            if bericht is not None:
                                bericht['zeilen_ohne_code'] += 1
                            continue
                        if bekannt[0] != sachgruppe and bericht is not None:
                            bericht['label_getauscht'] += 1
                        sachgruppe = bekannt[0]
                        alle = bekannt

                    results.append((lemma, bedeutung, sachgruppe, korpus,
                                    MENGEN_TRENNER.join(alle)))

    except ET.ParseError as e:
        print(f"Fehler beim Parsen von {bezeichnung}: {e}")
    except Exception as e:
        print(f"Unerwarteter Fehler bei {bezeichnung}: {e}")

    return results


# --------------------------------------------------------------------------- #
# Ausgabe
# --------------------------------------------------------------------------- #

def schreibe_csv(all_data, output_csv, spalten=None):
    """Schreibt die Tupel in die CSV; `spalten` wählt aus CSV_SPALTEN aus."""
    if spalten is None:
        spalten = spaltenwahl()
    ziel = Path(output_csv)
    ziel.parent.mkdir(parents=True, exist_ok=True)

    idx = [CSV_SPALTEN.index(s) for s in spalten]
    with open(ziel, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(spalten)
        writer.writerows([row[i] for i in idx] for row in all_data)

    print(f"Daten gespeichert in: {ziel} ({', '.join(spalten)})")


def statistik(all_data, spalten=None):
    if spalten is None:
        spalten = spaltenwahl()
    if not all_data:
        print("Keine Einträge extrahiert — nichts zu berichten.")
        return

    unique_lemmas = len(set(row[0] for row in all_data))
    unique_sachgruppen = len(set(row[2] for row in all_data))

    print(f"\nStatistiken:")
    print(f"  Einzigartige Lemmata: {unique_lemmas}")
    print(f"  Einzigartige Sachgruppen: {unique_sachgruppen}")
    print(f"  Durchschn. Bedeutungen pro Lemma: {len(all_data)/unique_lemmas:.2f}")

    if 'korpus' in spalten:
        je_korpus = {}
        for row in all_data:
            je_korpus[row[3] or '(unbekannt)'] = je_korpus.get(row[3] or '(unbekannt)', 0) + 1
        verteilung = ', '.join(f"{k}: {n}" for k, n in sorted(je_korpus.items()))
        print(f"  Zeilen je Korpus: {verteilung}")

    if 'sachgruppen_alle' in spalten:
        mehrdeutig = sum(1 for row in all_data if MENGEN_TRENNER in row[4])
        codes_gesamt = sum(row[4].count(MENGEN_TRENNER) + 1 for row in all_data)
        klassen_label = len(set(row[2] for row in all_data))
        klassen_alle = len({c for row in all_data for c in row[4].split(MENGEN_TRENNER)})
        print(f"  Mehrdeutige Bedeutungen: {mehrdeutig} ({mehrdeutig/len(all_data):.1%}), "
              f"{codes_gesamt - len(all_data)} Codes über das Label hinaus")
        print(f"  Sachgruppen mit Belegen: {klassen_alle} "
              f"(davon je als Label vergeben: {klassen_label})")


def schreibe_herkunft(output_csv, herkunft):
    """
    Legt eine Begleitdatei neben die CSV. Eine CSV ohne Herkunftsangabe ist in
    einem halben Jahr nicht mehr reproduzierbar: import_id sagt, welcher Bestand
    gelesen wurde, pipeline_git, welche XSLT ihn erzeugt hat.
    """
    ziel = Path(output_csv).with_suffix('.herkunft.json')
    with open(ziel, 'w', encoding='utf-8') as f:
        json.dump(herkunft, f, ensure_ascii=False, indent=2)
    print(f"Herkunft festgehalten in: {ziel}")


# --------------------------------------------------------------------------- #
# Quelle: Dateisystem
# --------------------------------------------------------------------------- #

def korpus_aus_pfad(xml_file, input_dir):
    """Ermittelt die Ressource aus dem Pfad (bdo-tl0/wbf/A/...); sonst ''."""
    try:
        teile = Path(xml_file).resolve().relative_to(Path(input_dir).resolve()).parts
    except ValueError:
        return ''
    for teil in teile:
        if teil in RESSOURCEN:
            return teil
    return ''


def process_directory(input_dir, output_csv, spalten=None, limit=None, gueltige=None):
    """
    Verarbeitet alle XML-Dateien in einem Verzeichnis.
    """
    if spalten is None:
        spalten = spaltenwahl()
    mit_korpus = 'korpus' in spalten
    bericht = neuer_bericht() if gueltige is not None else None
    all_data = []
    xml_files = sorted(Path(input_dir).rglob('*.xml'))
    if limit is not None:
        xml_files = xml_files[:limit]

    print(f"Gefunden: {len(xml_files)} XML-Dateien")

    # tqdm progress bar
    for xml_file in tqdm(xml_files, desc="Verarbeite XML-Dateien", unit="Datei"):
        korpus = korpus_aus_pfad(xml_file, input_dir) if mit_korpus else ''
        all_data.extend(extract_from_xml(xml_file, korpus,
                                         gueltige=gueltige, bericht=bericht))

    print(f"\nInsgesamt {len(all_data)} Einträge extrahiert")
    berichte_systematik(bericht)

    schreibe_csv(all_data, output_csv, spalten)
    statistik(all_data, spalten)
    return all_data


def process_single_file(xml_file, output_csv, spalten=None, gueltige=None):
    """
    Verarbeitet eine einzelne XML-Datei (für Tests).
    """
    data = extract_from_xml(xml_file, gueltige=gueltige)
    schreibe_csv(data, output_csv, spalten)
    print(f"{len(data)} Einträge aus {xml_file} extrahiert")
    return data


# --------------------------------------------------------------------------- #
# Quelle: Datenbank
# --------------------------------------------------------------------------- #

def verbindung():
    """
    Öffnet eine Verbindung zur Wörterbuch-Datenbank (`wbdb`, Docker, Port 5433).

    Zugangsdaten kommen aus der `.env` im Repo-Wurzelverzeichnis; das Passwort
    reist nicht mit dem Repository (siehe Readme Access WBDB.md, Abschnitt 2).
    """
    try:
        import psycopg
    except ModuleNotFoundError:
        raise SystemExit(
            "psycopg fehlt — es wird nur für --quelle db gebraucht.\n"
            "  uv sync    (psycopg[binary] steht in der dev-Gruppe von pyproject.toml)"
        )

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        pass
    else:
        load_dotenv(REPO_ROOT / '.env')
        load_dotenv()  # zusätzlich das übliche Suchverfahren (cwd aufwärts)

    fehlend = [k for k in ('POSTGRES_USER', 'POSTGRES_PASSWORD') if not os.getenv(k)]
    if fehlend:
        raise SystemExit(
            f"Zugangsdaten fehlen: {', '.join(fehlend)}.\n"
            f"Eine .env im Repo-Wurzelverzeichnis anlegen ({REPO_ROOT / '.env'}),\n"
            "Vorlage: .env.example. Die Werte stehen in pdl-lt-wbdb/.env."
        )

    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "wbdb"),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def _json_wert(wert):
    if isinstance(wert, (date, datetime)):
        return wert.isoformat()
    return wert


def lies_herkunft(conn, ressourcen):
    """Import und XSLT-Stand, aus denen die Ausgabe stammt."""
    herkunft = {
        'erzeugt_am': datetime.now().astimezone().isoformat(timespec='seconds'),
        'quelle': 'db',
        'ressourcen': list(ressourcen),
    }
    with conn.cursor() as cur:
        cur.execute(
            "SELECT i.import_id, i.snapshot, i.source_archive, i.imported_at "
            "FROM source.import i JOIN source.current_import c USING (import_id)"
        )
        zeile = cur.fetchone()
        if zeile:
            herkunft.update(zip(
                ('import_id', 'snapshot', 'source_archive', 'imported_at'),
                (_json_wert(w) for w in zeile),
            ))

        cur.execute("SELECT DISTINCT pipeline_git FROM source.derivative WHERE form = 'tl0'")
        pipelines = [r[0] for r in cur.fetchall()]
        herkunft['pipeline_git'] = pipelines[0] if len(pipelines) == 1 else pipelines

    return herkunft


def process_database(output_csv, ressourcen=RESSOURCEN_STANDARD, limit=None,
                     spalten=None, mit_herkunft=True, gueltige=None):
    """
    Liest die TL0-Derivate über einen serverseitigen Cursor.

    `fetchall()` wäre hier falsch: der Bestand ist gut 1 GB TL0. Ein benannter
    Cursor lässt Postgres blockweise liefern, der Speicherbedarf bleibt bei
    wenigen MB.
    """
    if spalten is None:
        spalten = spaltenwahl()
    mit_korpus = 'korpus' in spalten
    bericht = neuer_bericht() if gueltige is not None else None
    ressourcen = list(ressourcen)
    all_data = []

    with verbindung() as conn:
        herkunft = lies_herkunft(conn, ressourcen) if mit_herkunft else None
        if herkunft:
            print(f"Bestand: Import {herkunft.get('import_id')} "
                  f"({herkunft.get('snapshot')}, {herkunft.get('source_archive')}), "
                  f"XSLT {herkunft.get('pipeline_git')}")

        with conn.cursor() as cur:
            cur.execute(ABFRAGE_ANZAHL, (ressourcen,))
            gesamt = cur.fetchone()[0]
        if limit is not None:
            gesamt = min(gesamt, limit)
        print(f"Gefunden: {gesamt} Artikel mit TL0-Derivat "
              f"({', '.join(ressourcen)})")

        abfrage = ABFRAGE_TL0 + ("\nLIMIT %s" if limit is not None else "")
        parameter = (ressourcen, limit) if limit is not None else (ressourcen,)

        # name= ⇒ serverseitiger Cursor. Er lebt in einer Transaktion: wer
        # währenddessen schreibt und committet, schließt ihn sich selbst.
        with conn.cursor(name="tl0") as cur:
            cur.itersize = 200
            cur.execute(abfrage, parameter)
            for article_id, resource_id, content in tqdm(
                cur, total=gesamt, desc="Verarbeite Artikel", unit="Artikel"
            ):
                all_data.extend(extract_from_xml(
                    bytes(content),
                    resource_id if mit_korpus else '',
                    bezeichnung=article_id,
                    gueltige=gueltige,
                    bericht=bericht,
                ))

    print(f"\nInsgesamt {len(all_data)} Einträge extrahiert")
    berichte_systematik(bericht)

    schreibe_csv(all_data, output_csv, spalten)
    if herkunft:
        herkunft['csv'] = str(Path(output_csv))
        herkunft['spalten'] = list(spalten)
        herkunft['zeilen'] = len(all_data)
        if bericht:
            herkunft['systematik_pruefung'] = {
                'verworfene_codes': dict(bericht['verworfene_codes'].most_common()),
                'bedeutungen_entfallen': bericht['zeilen_ohne_code'],
                'label_getauscht': bericht['label_getauscht'],
            }
        schreibe_herkunft(output_csv, herkunft)
    statistik(all_data, spalten)
    return all_data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description='Extrahiert Lemma, Bedeutung und Sachgruppe aus TEI-Lex-0-Daten '
                    'in eine CSV-Datei — aus der Wörterbuch-Datenbank oder aus '
                    'einem Verzeichnis mit XML-Dateien.',
        epilog='Beispiele:\n'
               '  extract_data.py --quelle db\n'
               '  extract_data.py --quelle db --ressourcen wbf dibs bwb -o data/alles.csv\n'
               '  extract_data.py ~/Nextcloud/BAdW/Wörterbuchdaten/bdo-tl0/\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input_dir',
        nargs='?',
        help='Nur für --quelle dateien: Verzeichnis mit den TEI-Lex-0-XML-Dateien '
             '(wird rekursiv durchsucht), z. B. ~/Nextcloud/BAdW/Wörterbuchdaten/bdo-tl0/. '
             'Wird ein Verzeichnis angegeben, ist --quelle dateien der Standard.'
    )
    parser.add_argument(
        '--quelle',
        choices=('db', 'dateien'),
        default=None,
        help='Woher die Daten kommen. Standard: dateien, wenn ein Verzeichnis '
             'angegeben ist, sonst db.'
    )
    parser.add_argument(
        '-o', '--output',
        default='data/woerterbuch_daten.csv',
        help='Pfad zur Ausgabe-CSV (Standard: %(default)s)'
    )
    parser.add_argument(
        '--ressourcen',
        nargs='+',
        metavar='ID',
        default=None,
        help='Nur für --quelle db: welche Korpora gelesen werden. '
             f'Möglich: {", ".join(RESSOURCEN)}. '
             f'Standard: {" ".join(RESSOURCEN_STANDARD)} (bwb trägt keine Sachgruppen).'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Nur die ersten N Artikel bzw. Dateien verarbeiten (Schnelltest).'
    )
    parser.add_argument(
        '--ohne-korpus',
        action='store_true',
        help='CSV ohne die Spalte "korpus" schreiben.'
    )
    parser.add_argument(
        '--ohne-mengen',
        action='store_true',
        help='CSV ohne die Spalte "sachgruppen_alle" schreiben. Mit --ohne-korpus '
             'zusammen ergibt das das alte Drei-Spalten-Format.'
    )
    parser.add_argument(
        '--ohne-herkunft',
        action='store_true',
        help='Nur für --quelle db: keine Begleitdatei <ausgabe>.herkunft.json schreiben.'
    )
    parser.add_argument(
        '--systematik',
        default=str(SYSTEMATIK_STANDARD),
        metavar='CSV',
        help='Liste der gültigen Sachgruppen (Spalte "Nummer", %(default)s).'
    )
    parser.add_argument(
        '--ohne-pruefung',
        action='store_true',
        help='Sachgruppen NICHT gegen die Systematik prüfen (übernimmt auch '
             'unbekannte Codes, wie vor dem Einbau der Prüfung).'
    )
    args = parser.parse_args()

    quelle = args.quelle or ('dateien' if args.input_dir else 'db')
    spalten = spaltenwahl(mit_korpus=not args.ohne_korpus,
                          mit_mengen=not args.ohne_mengen)

    if args.ohne_pruefung:
        gueltige = None
        print("Systematik-Prüfung abgeschaltet (--ohne-pruefung).")
    else:
        gueltige = lade_systematik(args.systematik)
        print(f"Systematik: {len(gueltige)} gültige Sachgruppen aus {args.systematik}")

    if quelle == 'dateien':
        if not args.input_dir:
            parser.error("--quelle dateien braucht ein Eingabeverzeichnis als Argument.")
        if args.ressourcen is not None:
            parser.error("--ressourcen gilt nur für --quelle db.")

        input_dir = os.path.expanduser(args.input_dir)
        if not os.path.isdir(input_dir):
            parser.error(f"Eingabeverzeichnis existiert nicht: {input_dir}")

        process_directory(input_dir, args.output, spalten, args.limit, gueltige)
        return

    if args.input_dir:
        parser.error("Ein Eingabeverzeichnis gehört zu --quelle dateien, nicht zu --quelle db.")

    ressourcen = args.ressourcen if args.ressourcen is not None else list(RESSOURCEN_STANDARD)
    unbekannt = [r for r in ressourcen if r not in RESSOURCEN]
    if unbekannt:
        parser.error(f"Unbekannte Ressource(n): {', '.join(unbekannt)}. "
                     f"Möglich: {', '.join(RESSOURCEN)}.")

    process_database(args.output, ressourcen, args.limit, spalten,
                     mit_herkunft=not args.ohne_herkunft, gueltige=gueltige)


if __name__ == '__main__':
    main()
