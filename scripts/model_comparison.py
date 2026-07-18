#!/usr/bin/env python3
"""
Umfassender ML-Modell-Vergleich für Sachgruppen-Klassifikation.

Testet alle verfügbaren Modelle:
1. Support Vector Machine (LinearSVC)
2. Logistische Regression
3. Random Forest
4. XGBoost
5. Neural Network (MLP)

Mit Lemma + Bedeutung als Features.
"""

import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path
import argparse
from datetime import datetime

from sachgruppen_classifier import SachgruppenClassifier, train_and_evaluate
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_naming import generate_model_name

def compare_models(csv_file, save_models=True, test_size=0.2):
    """
    Vergleicht alle Modelle auf demselben Datensatz.
    """
    print("="*70)
    print("ML-MODELL-VERGLEICH")
    print("="*70)
    print(f"CSV-Datei: {csv_file}")
    print(f"Test-Size: {test_size}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Daten laden
    print("Lade Daten...")
    df = pd.read_csv(csv_file)

    print(f"\nDataset Info:")
    print(f"  Anzahl Einträge: {len(df):,}")
    print(f"  Anzahl Sachgruppen: {df['sachgruppe'].nunique()}")
    print(f"  Durchschn. Lemma-Länge: {df['lemma'].str.len().mean():.1f} Zeichen")
    print(f"  Durchschn. Bedeutungslänge: {df['bedeutung'].str.len().mean():.1f} Zeichen")

    # Klassenverteilung
    class_counts = df['sachgruppe'].value_counts()
    print(f"\nTop-10 häufigste Sachgruppen:")
    for sg, count in class_counts.head(10).items():
        print(f"  {sg}: {count} ({count/len(df)*100:.1f}%)")

    # Problematische Klassen
    small_classes = class_counts[class_counts < 10]
    if len(small_classes) > 0:
        print(f"\n⚠ Warnung: {len(small_classes)} Sachgruppen haben <10 Beispiele")
        print(f"  (macht {small_classes.sum()/len(df)*100:.1f}% der Daten aus)")

    # Modelle definieren (schnellste zuerst für schnelles Feedback)
    models = [
        ('svm', 'Linear SVM', False),           # Schnell, gute Baseline
        ('logistic', 'Logistische Regression', False),  # Sehr schnell
        ('rf', 'Random Forest', False),         # Mittel
        ('nn', 'Neural Network', False),        # Mittel-langsam
        ('xgboost', 'XGBoost', False),          # Langsam
    ]

    results = []

    print("\n" + "="*70)
    print("TRAINING & EVALUATION")
    print("="*70)

    for model_type, model_name, use_gpu in models:
        print(f"\n{'─'*70}")
        print(f"🤖 Modell: {model_name}")
        print(f"{'─'*70}")

        try:
            start_time = time.time()

            # Training & Evaluation
            save_path = (
                f"models/{generate_model_name(model_type, Path('models'))}.pkl"
                if save_models else None
            )
            clf, accuracy, _report = train_and_evaluate(
                csv_file,
                model_type=model_type,
                test_size=test_size,
                tune=False,
                save_path=save_path,
                use_gpu=use_gpu
            )

            total_time = time.time() - start_time

            result = {
                'Modell': model_name,
                'Typ': model_type,
                'Accuracy': accuracy,
                'Zeit (s)': total_time,
                'Zeit (min)': total_time / 60,
                'Gespeichert': save_path if save_models else 'Nein'
            }
            results.append(result)

            print(f"\n✓ {model_name} abgeschlossen in {total_time:.1f}s ({total_time/60:.1f} min)")

        except Exception as e:
            print(f"\n✗ Fehler bei {model_name}: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'Modell': model_name,
                'Typ': model_type,
                'Accuracy': np.nan,
                'Zeit (s)': np.nan,
                'Zeit (min)': np.nan,
                'Gespeichert': 'Fehler'
            }
            results.append(result)

    # Zusammenfassung
    print("\n" + "="*70)
    print("📊 ERGEBNISSE")
    print("="*70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    # Formatierte Ausgabe
    print("\nRangliste nach Accuracy:")
    print()
    for idx, row in results_df.iterrows():
        if pd.notna(row['Accuracy']):
            print(f"{row['Modell']:25} | Accuracy: {row['Accuracy']:.4f} | "
                  f"Zeit: {row['Zeit (s)']:6.1f}s ({row['Zeit (min)']:4.1f} min)")
        else:
            print(f"{row['Modell']:25} | FEHLER")

    # Beste Modelle
    print("\n" + "="*70)
    print("🏆 EMPFEHLUNG")
    print("="*70)

    valid_results = results_df[results_df['Accuracy'].notna()]

    if len(valid_results) > 0:
        best_model = valid_results.iloc[0]
        fastest_model = valid_results.loc[valid_results['Zeit (s)'].idxmin()]

        print(f"\n🥇 Beste Accuracy: {best_model['Modell']}")
        print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        print(f"   Trainingszeit: {best_model['Zeit (min)']:.1f} Minuten")

        print(f"\n⚡ Schnellstes Modell: {fastest_model['Modell']}")
        print(f"   Accuracy: {fastest_model['Accuracy']:.4f}")
        print(f"   Trainingszeit: {fastest_model['Zeit (s)']:.1f} Sekunden")

        # Speedup vs. Accuracy Trade-off
        if best_model['Modell'] != fastest_model['Modell']:
            acc_diff = best_model['Accuracy'] - fastest_model['Accuracy']
            time_ratio = best_model['Zeit (s)'] / fastest_model['Zeit (s)']

            print(f"\n📈 Trade-off:")
            print(f"   {best_model['Modell']} ist {acc_diff:.4f} ({acc_diff*100:.2f}%) besser,")
            print(f"   aber {time_ratio:.1f}x langsamer als {fastest_model['Modell']}")

        print("\n💡 Empfehlung für Ihre Anwendung:")

        # Heuristik für Empfehlung
        if best_model['Zeit (min)'] < 5:
            print(f"   → Verwenden Sie {best_model['Modell']} (beste Accuracy, akzeptable Zeit)")
        elif fastest_model['Accuracy'] >= best_model['Accuracy'] - 0.02:
            print(f"   → Verwenden Sie {fastest_model['Modell']} (nur minimal schlechter, viel schneller)")
        else:
            print(f"   → Entwicklung: {fastest_model['Modell']} (schnelle Iteration)")
            print(f"   → Produktion: {best_model['Modell']} (beste Performance)")

    # CSV exportieren
    output_file = 'model_comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Ergebnisse gespeichert: {output_file}")

    return results_df

def quick_comparison(csv_file):
    """
    Schneller Vergleich: Nur SVM, Logistic Regression und XGBoost.
    """
    print("="*70)
    print("SCHNELL-VERGLEICH (3 Modelle)")
    print("="*70)

    models = [
        ('svm', 'Linear SVM', False),
        ('logistic', 'Logistische Regression', False),
        ('xgboost', 'XGBoost', False),
    ]

    results = []

    for model_type, model_name, use_gpu in models:
        print(f"\n{'─'*70}")
        print(f"🤖 {model_name}")

        try:
            start_time = time.time()
            clf, accuracy, _report = train_and_evaluate(
                csv_file,
                model_type=model_type,
                test_size=0.2,
                save_path=None,  # Nicht speichern
                use_gpu=use_gpu
            )
            total_time = time.time() - start_time

            results.append({
                'Modell': model_name,
                'Accuracy': accuracy,
                'Zeit (min)': total_time / 60
            })

            print(f"✓ Fertig in {total_time/60:.1f} min")

        except Exception as e:
            print(f"✗ Fehler: {e}")

    # Ergebnisse
    print("\n" + "="*70)
    print("ERGEBNISSE")
    print("="*70)
    for r in results:
        print(f"{r['Modell']:25} | Accuracy: {r['Accuracy']:.4f} | Zeit: {r['Zeit (min)']:.1f} min")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vergleiche verschiedene ML-Modelle für Sachgruppen-Klassifikation'
    )

    parser.add_argument('--csv', type=str, default='data/woerterbuch_daten.csv',
                       help='Pfad zur CSV-Datei mit Trainingsdaten')
    parser.add_argument('--quick', action='store_true',
                       help='Schnell-Vergleich (nur 3 Modelle)')
    parser.add_argument('--no-save', action='store_true',
                       help='Trainierte Modelle nicht speichern')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Anteil der Test-Daten (default: 0.2)')

    args = parser.parse_args()

    if args.quick:
        quick_comparison(args.csv)
    else:
        compare_models(
            args.csv,
            save_models=not args.no_save,
            test_size=args.test_size
        )
