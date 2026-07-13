#!/usr/bin/env python3
"""
Extrahiert Lemma, Bedeutung und Sachgruppe aus TEI Lex-0 XML-Dateien
und speichert sie in einer CSV-Datei.
"""

import argparse
import xml.etree.ElementTree as ET
import csv
import os
from pathlib import Path
from tqdm import tqdm

# TEI Namespace
TEI_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

def extract_from_xml(xml_file):
    """
    Extrahiert Daten aus einer einzelnen XML-Datei.
    
    Returns:
        Liste von Tupeln (lemma, bedeutung, sachgruppe)
    """
    results = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
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
            
            # Sachgruppe extrahieren
            usg_elem = sense.find('./tei:usg[@type="domain"]', TEI_NS)
            if usg_elem is not None:
                ana = usg_elem.get('ana', '')
                # Sachgruppe ist im Format "#sg_6121", extrahiere nur die Nummer
                sachgruppe = ana.replace('#sg_', '')
                
                if sachgruppe:  # Nur hinzufügen wenn Sachgruppe vorhanden
                    results.append((lemma, bedeutung, sachgruppe))
    
    except ET.ParseError as e:
        print(f"Fehler beim Parsen von {xml_file}: {e}")
    except Exception as e:
        print(f"Unerwarteter Fehler bei {xml_file}: {e}")
    
    return results

def process_directory(input_dir, output_csv):
    """
    Verarbeitet alle XML-Dateien in einem Verzeichnis.
    """
    all_data = []
    xml_files = list(Path(input_dir).rglob('*.xml'))
    
    print(f"Gefunden: {len(xml_files)} XML-Dateien")
    
    # tqdm progress bar
    for xml_file in tqdm(xml_files, desc="Verarbeite XML-Dateien", unit="Datei"):
        data = extract_from_xml(xml_file)
        all_data.extend(data)
    
    print(f"\nInsgesamt {len(all_data)} Einträge extrahiert")
    
    # In CSV schreiben
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lemma', 'bedeutung', 'sachgruppe'])
        writer.writerows(all_data)
    
    print(f"Daten gespeichert in: {output_csv}")
    
    # Statistiken
    unique_lemmas = len(set(row[0] for row in all_data))
    unique_sachgruppen = len(set(row[2] for row in all_data))
    
    print(f"\nStatistiken:")
    print(f"  Einzigartige Lemmata: {unique_lemmas}")
    print(f"  Einzigartige Sachgruppen: {unique_sachgruppen}")
    print(f"  Durchschn. Bedeutungen pro Lemma: {len(all_data)/unique_lemmas:.2f}")

def process_single_file(xml_file, output_csv):
    """
    Verarbeitet eine einzelne XML-Datei (für Tests).
    """
    data = extract_from_xml(xml_file)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lemma', 'bedeutung', 'sachgruppe'])
        writer.writerows(data)
    
    print(f"{len(data)} Einträge aus {xml_file} extrahiert")
    print(f"Daten gespeichert in: {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description='Extrahiert Lemma, Bedeutung und Sachgruppe aus TEI-Lex-0-XML-Dateien '
                    'in eine CSV-Datei.'
    )
    parser.add_argument(
        'input_dir',
        help='Verzeichnis mit den TEI-Lex-0-XML-Dateien (wird rekursiv durchsucht), '
             'z. B. ~/Nextcloud/BAdW/Wörterbuchdaten/bdo-tl0/'
    )
    parser.add_argument(
        '-o', '--output',
        default='data/woerterbuch_daten.csv',
        help='Pfad zur Ausgabe-CSV (Standard: %(default)s)'
    )
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    if not os.path.isdir(input_dir):
        parser.error(f"Eingabeverzeichnis existiert nicht: {input_dir}")

    process_directory(input_dir, args.output)


if __name__ == '__main__':
    main()
