#!/usr/bin/env python3
"""
Detaillierte Analyse der Klassifikations-Performance
Hilft zu verstehen, welche Sachgruppen gut/schlecht funktionieren
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def analyze_results(csv_file, model_path=None):
    """
    Analysiert die Performance nach Klassengröße.
    """
    # Daten laden
    df = pd.read_csv(csv_file)
    
    # Klassenverteilung
    class_counts = df['sachgruppe'].value_counts()
    
    print("="*70)
    print("KLASSENVERTEILUNG")
    print("="*70)
    
    # Statistiken nach Größe
    bins = [0, 5, 10, 20, 50, 100, 500, float('inf')]
    labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '101-500', '>500']
    
    class_counts_df = pd.DataFrame({
        'sachgruppe': class_counts.index,
        'count': class_counts.values
    })
    class_counts_df['bin'] = pd.cut(class_counts_df['count'], bins=bins, labels=labels)
    
    print("\nAnzahl Sachgruppen nach Größe:")
    print(class_counts_df['bin'].value_counts().sort_index())
    
    print("\n" + "="*70)
    print("PROBLEMATISCHE KLASSEN (wenige Beispiele)")
    print("="*70)
    
    # Sehr kleine Klassen (<10 Beispiele)
    small_classes = class_counts[class_counts < 10]
    print(f"\nSachgruppen mit <10 Beispielen: {len(small_classes)}")
    print(f"Diese machen {small_classes.sum()} von {len(df)} Beispielen aus ({small_classes.sum()/len(df)*100:.1f}%)")
    
    if len(small_classes) > 0:
        print("\nKleinste Sachgruppen:")
        print(small_classes.head(20))
    
    # Mittlere Klassen (10-50 Beispiele)
    medium_classes = class_counts[(class_counts >= 10) & (class_counts < 50)]
    print(f"\nSachgruppen mit 10-50 Beispielen: {len(medium_classes)}")
    
    # Große Klassen (>50 Beispiele)
    large_classes = class_counts[class_counts >= 50]
    print(f"\nSachgruppen mit >=50 Beispielen: {len(large_classes)}")
    print(f"Diese machen {large_classes.sum()} von {len(df)} Beispielen aus ({large_classes.sum()/len(df)*100:.1f}%)")
    
    print("\nGrößte Sachgruppen:")
    print(class_counts.head(20))
    
    print("\n" + "="*70)
    print("EMPFEHLUNGEN")
    print("="*70)
    
    if len(small_classes) > 0:
        print(f"\n⚠️  {len(small_classes)} Sachgruppen haben <10 Beispiele")
        print("   → Diese sind schwer zu lernen und senken die Macro-Avg Performance")
        print("   → Optionen:")
        print("     1. Mehr Daten sammeln für diese Klassen")
        print("     2. Diese Klassen zusammenlegen mit ähnlichen Sachgruppen")
        print("     3. Diese Klassen aus dem Training entfernen")
        print("     4. Oversampling (SMOTE) verwenden")
    
    if len(medium_classes) > len(large_classes):
        print(f"\n⚡ Viele mittlere Klassen ({len(medium_classes)}) vorhanden")
        print("   → Mit mehr Daten könnte Performance weiter steigen")
    
    # Erwartete Performance basierend auf Klassengrößen
    expected_performance = estimate_expected_performance(class_counts)
    print(f"\n📊 Erwartete Performance basierend auf Klassenverteilung:")
    print(f"   Weighted Avg F1: {expected_performance['weighted']:.2f} (tatsächlich: wahrscheinlich ähnlich)")
    print(f"   Macro Avg F1: {expected_performance['macro']:.2f}")
    
    # Visualisierung
    plot_class_distribution(class_counts)
    
    return class_counts_df

def estimate_expected_performance(class_counts):
    """
    Schätzt erwartete Performance basierend auf Klassengrößen.
    Heuristik: F1 ~= min(0.9, 0.3 + 0.05 * log10(n))
    """
    performances = []
    weights = []
    
    for count in class_counts.values:
        # Heuristische Formel
        if count < 5:
            f1 = 0.1
        elif count < 10:
            f1 = 0.3
        elif count < 20:
            f1 = 0.5
        elif count < 50:
            f1 = 0.7
        else:
            f1 = min(0.9, 0.6 + 0.03 * np.log10(count))
        
        performances.append(f1)
        weights.append(count)
    
    performances = np.array(performances)
    weights = np.array(weights)
    
    macro_avg = performances.mean()
    weighted_avg = (performances * weights).sum() / weights.sum()
    
    return {
        'macro': macro_avg,
        'weighted': weighted_avg
    }

def plot_class_distribution(class_counts, top_n=30):
    """
    Visualisiert die Klassenverteilung.
    """
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Top-N Klassen
    plt.subplot(1, 2, 1)
    top_classes = class_counts.head(top_n)
    plt.barh(range(len(top_classes)), top_classes.values)
    plt.yticks(range(len(top_classes)), top_classes.index)
    plt.xlabel('Anzahl Beispiele')
    plt.ylabel('Sachgruppe')
    plt.title(f'Top-{top_n} Sachgruppen')
    plt.gca().invert_yaxis()
    
    # Plot 2: Verteilung nach Größe
    plt.subplot(1, 2, 2)
    bins = [0, 5, 10, 20, 50, 100, 500, max(class_counts.values)]
    plt.hist(class_counts.values, bins=bins, edgecolor='black')
    plt.xlabel('Anzahl Beispiele pro Klasse')
    plt.ylabel('Anzahl Sachgruppen')
    plt.title('Verteilung der Klassengrößen')
    plt.yscale('log')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/class_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualisierung gespeichert: class_distribution.png")

def identify_confused_classes(y_true, y_pred, top_n=10):
    """
    Identifiziert die am häufigsten verwechselten Klassenpaare.
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    
    # Finde Verwechslungen (außerhalb der Diagonale)
    confusion_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_class': classes[i],
                    'predicted_class': classes[j],
                    'count': cm[i, j]
                })
    
    confusion_df = pd.DataFrame(confusion_pairs)
    confusion_df = confusion_df.sort_values('count', ascending=False)
    
    print("\n" + "="*70)
    print("HÄUFIGSTE VERWECHSLUNGEN")
    print("="*70)
    print(f"\nTop-{top_n} verwechselte Klassenpaare:")
    print(confusion_df.head(top_n).to_string(index=False))
    
    return confusion_df

def suggest_improvements(class_counts):
    """
    Gibt konkrete Verbesserungsvorschläge.
    """
    print("\n" + "="*70)
    print("KONKRETE VERBESSERUNGSVORSCHLÄGE")
    print("="*70)
    
    small_classes = class_counts[class_counts < 10]
    total_examples = class_counts.sum()
    
    print("\n1️⃣  KURZFRISTIG (sofort umsetzbar):")
    print("   ✓ Dein Modell ist bereits gut (80% Accuracy)")
    print("   ✓ Für produktive Nutzung: Filtern nach Konfidenz")
    print("     → Nur Vorhersagen über einem Schwellwert verwenden")
    
    print("\n2️⃣  MITTELFRISTIG (mit etwas Aufwand):")
    
    if len(small_classes) > 10:
        print(f"   → Entferne die {len(small_classes)} kleinsten Klassen (<10 Beispiele)")
        print(f"     Das würde nur {small_classes.sum()/total_examples*100:.1f}% der Daten entfernen")
        print(f"     Aber Macro-Avg F1 würde vermutlich auf ~0.75 steigen")
    
    print("   → Hyperparameter-Tuning:")
    print("     python sachgruppen_classifier.py --csv daten.csv --model svm --tune")
    
    print("   → XGBoost probieren (oft 2-5% besser):")
    print("     python sachgruppen_classifier.py --csv daten.csv --model xgboost")
    
    print("\n3️⃣  LANGFRISTIG (mehr Entwicklung):")
    print("   → Feature Engineering:")
    print("     • Lemma als zusätzliches Feature")
    print("     • Textlänge als Feature")
    print("     • Word2Vec/FastText Embeddings")
    
    print("   → Hierarchische Klassifikation:")
    print("     • Erst Haupt-Sachgruppe (z.B. 6000er vs 7000er)")
    print("     • Dann Unter-Sachgruppe (z.B. 6121 vs 6120)")
    
    print("   → Deep Learning (wenn andere Methoden ausgeschöpft):")
    print("     • German BERT Fine-tuning")
    print("     • Nur bei >10k Trainingsbeispielen sinnvoll")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analysiere Klassifikations-Performance')
    parser.add_argument('--csv', type=str, required=True, help='CSV-Datei mit Daten')
    parser.add_argument('--model', type=str, help='Trainiertes Modell (optional)')
    
    args = parser.parse_args()
    
    analyze_results(args.csv, args.model)
    suggest_improvements(pd.read_csv(args.csv)['sachgruppe'].value_counts())
