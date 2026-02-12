#!/usr/bin/env python3
"""
Komplette ML-Pipeline für Sachgruppen-Klassifikation

Features:
- Automatische Datenaufteilung
- Feature Engineering (TF-IDF)
- Model Training mit mehreren Algorithmen
- Hyperparameter-Tuning
- Cross-Validation
- Model Persistence
- Vorhersage-Interface
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost nicht installiert. Überspringe XGBoost-Modell.")


class SachgruppenClassifier:
    """
    Hauptklasse für Sachgruppen-Klassifikation.
    """
    
    def __init__(self, model_type='svm', random_state=42, use_gpu=False, use_lemma=True):
        """
        Args:
            model_type: 'svm', 'logistic', 'rf', 'xgboost', oder 'nn' (neural network)
            random_state: Für Reproduzierbarkeit
            use_gpu: GPU-Beschleunigung nutzen (nur für XGBoost)
            use_lemma: Lemma-Spalte als zusätzliche Features verwenden
        """
        self.model_type = model_type
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.use_lemma = use_lemma
        self.pipeline = None
        self.classes_ = None
        self.label_encoder = None  # Für XGBoost: String → Integer
        
    def create_pipeline(self):
        """Erstellt die ML-Pipeline."""

        # Feature Extraction
        # Separate Vectorizer für Lemma und Bedeutung
        if self.use_lemma:
            # Lemma: Kürzere n-grams, da Lemmata oft kürzer sind
            lemma_vectorizer = TfidfVectorizer(
                ngram_range=(1, 4),  # Längere n-grams für Lemma
                analyzer='char_wb',
                max_features=5000,
                min_df=2,
                sublinear_tf=True,
                strip_accents='unicode'
            )

            # Bedeutung: Standard character n-grams
            bedeutung_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                analyzer='char_wb',
                max_features=10000,
                min_df=2,
                sublinear_tf=True,
                strip_accents='unicode'
            )

            # ColumnTransformer: Kombiniert Features von beiden Spalten
            vectorizer = ColumnTransformer([
                ('lemma', lemma_vectorizer, 'lemma'),
                ('bedeutung', bedeutung_vectorizer, 'bedeutung')
            ])
        else:
            # Nur Bedeutung (alte Methode, für Vergleich)
            bedeutung_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                analyzer='char_wb',
                max_features=10000,
                min_df=2,
                sublinear_tf=True,
                strip_accents='unicode'
            )
            vectorizer = ColumnTransformer([
                ('bedeutung', bedeutung_vectorizer, 'bedeutung')
            ])
        
        # Model Selection
        if self.model_type == 'svm':
            classifier = LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=2000,
                random_state=self.random_state,
                dual=True,  # Besser für n_samples < n_features
                verbose=1   # Zeigt Fortschritt
            )
        
        elif self.model_type == 'logistic':
            classifier = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state,
                solver='saga',
                verbose=1   # Zeigt Fortschritt
            )
        
        elif self.model_type == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1   # Zeigt Fortschritt
            )
        
        elif self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ValueError("XGBoost nicht installiert!")

            # GPU-Parameter wenn verfügbar
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'verbosity': 2  # Zeigt Fortschritt (0=silent, 1=warning, 2=info, 3=debug)
            }

            if self.use_gpu:
                print("🚀 GPU-Modus aktiviert!")
                xgb_params.update({
                    'device': 'cuda',  # Funktioniert auch mit ROCm
                    'tree_method': 'hist',  # GPU-optimierter Algorithmus
                })
            else:
                xgb_params['n_jobs'] = -1

            classifier = xgb.XGBClassifier(**xgb_params)

        elif self.model_type == 'nn':
            # Neural Network (Multi-layer Perceptron)
            classifier = MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),  # 3 versteckte Schichten
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 Regularisierung
                batch_size=256,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=50,  # Epochen
                random_state=self.random_state,
                verbose=True,  # Zeigt Fortschritt
                early_stopping=True,  # Stoppt bei Stagnation
                validation_fraction=0.1,
                n_iter_no_change=5
            )

        else:
            raise ValueError(f"Unbekannter model_type: {self.model_type}")
        
        # Pipeline zusammensetzen
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        return self.pipeline
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """
        Trainiert das Modell.
        
        Args:
            X_train: Trainings-Texte
            y_train: Trainings-Labels
            tune_hyperparameters: Ob Hyperparameter-Tuning durchgeführt werden soll
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        print(f"\nTrainiere {self.model_type.upper()}-Modell...")
        print(f"Trainingsbeispiele: {len(X_train)}")
        print(f"Anzahl Klassen: {len(np.unique(y_train))}")
        
        # Für XGBoost und Neural Network: String-Labels in Integers umwandeln
        if self.model_type in ['xgboost', 'nn']:
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            print(f"Labels für {self.model_type.upper()} encodiert (String → Integer)")
        else:
            y_train_encoded = y_train
        
        if tune_hyperparameters:
            self._tune_hyperparameters(X_train, y_train_encoded)
        else:
            print("Starte Training...")
            self.pipeline.fit(X_train, y_train_encoded)
        
        self.classes_ = self.pipeline.classes_
        print("✓ Training abgeschlossen!")
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Hyperparameter-Tuning mit Grid Search."""
        print("\nStarte Hyperparameter-Tuning (dauert länger)...")
        
        if self.model_type == 'svm':
            param_grid = {
                'vectorizer__max_features': [5000, 10000, 15000],
                'classifier__C': [0.1, 1.0, 10.0]
            }
        elif self.model_type == 'logistic':
            param_grid = {
                'vectorizer__max_features': [5000, 10000],
                'classifier__C': [0.1, 1.0, 10.0]
            }
        elif self.model_type == 'rf':
            param_grid = {
                'vectorizer__max_features': [5000, 10000],
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [10, 20]
            }
        else:
            param_grid = {}
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.pipeline = grid_search.best_estimator_
        
        print(f"\nBeste Parameter: {grid_search.best_params_}")
        print(f"Bester CV-Score: {grid_search.best_score_:.4f}")
    
    def evaluate(self, X_test, y_test, verbose=True):
        """Evaluiert das Modell auf Test-Daten."""
        if self.pipeline is None:
            raise ValueError("Modell muss erst trainiert werden!")

        # Für XGBoost und Neural Network: Labels encodieren und unbekannte Labels filtern
        if self.model_type in ['xgboost', 'nn'] and self.label_encoder is not None:
            # Finde Labels die im Training vorkamen
            known_labels = set(self.label_encoder.classes_)
            mask = y_test.isin(known_labels)

            if not mask.all():
                unknown_count = (~mask).sum()
                unknown_labels = set(y_test[~mask].unique())
                print(f"\nWarnung: {unknown_count} Test-Samples mit unbekannten Labels gefunden: {unknown_labels}")
                print(f"Diese werden für die Evaluation übersprungen.")

                # Filtere Test-Set
                X_test = X_test[mask]
                y_test = y_test[mask]

            y_test_encoded = self.label_encoder.transform(y_test)
            y_pred_encoded = self.pipeline.predict(X_test)
            # Zurück zu Original-Labels für Anzeige
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = self.pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        if verbose:
            print("\n" + "="*60)
            print("EVALUATION")
            print("="*60)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nKlassifikations-Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

        return accuracy, y_pred
    
    def cross_validate(self, X, y, cv=5):
        """Führt Cross-Validation durch."""
        if self.pipeline is None:
            self.create_pipeline()
        
        print(f"\nFühre {cv}-fold Cross-Validation durch...")
        scores = cross_val_score(
            self.pipeline, X, y, 
            cv=cv, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        print(f"F1-Scores: {scores}")
        print(f"Mittelwert: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
    def predict(self, texts):
        """Vorhersage für neue Texte."""
        if self.pipeline is None:
            raise ValueError("Modell muss erst trainiert werden!")
        
        predictions = self.pipeline.predict(texts)

        # Für XGBoost und Neural Network: Integer-Vorhersagen zurück zu String-Labels
        if self.model_type in ['xgboost', 'nn'] and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions
    
    def predict_proba(self, texts):
        """Vorhersage-Wahrscheinlichkeiten (wenn verfügbar)."""
        if self.pipeline is None:
            raise ValueError("Modell muss erst trainiert werden!")
        
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(texts)
        else:
            raise ValueError(f"{self.model_type} unterstützt keine Wahrscheinlichkeiten")
    
    def save(self, filepath):
        """Speichert das trainierte Modell."""
        if self.pipeline is None:
            raise ValueError("Modell muss erst trainiert werden!")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'model_type': self.model_type,
                'classes': self.classes_,
                'label_encoder': self.label_encoder,  # Für XGBoost
                'use_gpu': self.use_gpu,
                'use_lemma': self.use_lemma
            }, f)

        print(f"\nModell gespeichert: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Lädt ein gespeichertes Modell."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            model_type=data['model_type'],
            use_gpu=data.get('use_gpu', False),
            use_lemma=data.get('use_lemma', True)  # Default: True für neue Modelle
        )
        instance.pipeline = data['pipeline']
        instance.classes_ = data['classes']
        instance.label_encoder = data.get('label_encoder', None)  # Für XGBoost

        print(f"Modell geladen: {filepath}")
        return instance


def train_and_evaluate(csv_file, model_type='svm', test_size=0.2,
                       tune=False, save_path=None, use_gpu=False):
    """
    Hauptfunktion zum Trainieren und Evaluieren.

    Args:
        use_gpu: GPU-Beschleunigung nutzen (nur für XGBoost)
    """
    # Daten laden
    print(f"Lade Daten aus {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"\nDataset Info:")
    print(f"  Anzahl Einträge: {len(df)}")
    print(f"  Anzahl Sachgruppen: {df['sachgruppe'].nunique()}")
    print(f"  Durchschn. Bedeutungslänge: {df['bedeutung'].str.len().mean():.1f} Zeichen")
    
    # Class Distribution
    print(f"\nTop-10 Sachgruppen:")
    print(df['sachgruppe'].value_counts().head(10))

    # Datenbereinigung: NaN-Werte behandeln
    print("\nBereinigende Daten...")

    # Zeige NaN-Statistik
    nan_counts = df[['lemma', 'bedeutung', 'sachgruppe']].isna().sum()
    if nan_counts.any():
        print("Gefundene NaN-Werte:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")

    # Entferne Zeilen mit NaN in wichtigen Spalten
    df_clean = df.dropna(subset=['lemma', 'bedeutung', 'sachgruppe'])

    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"⚠ {removed} Zeilen mit fehlenden Werten entfernt")
        print(f"Verbleibend: {len(df_clean)} Einträge")

    # Ersetze leere Strings mit Platzhalter
    df_clean['lemma'] = df_clean['lemma'].astype(str).replace('', 'LEER')
    df_clean['bedeutung'] = df_clean['bedeutung'].astype(str).replace('', 'LEER')

    # Train-Test Split
    # X ist jetzt ein DataFrame mit beiden Spalten (lemma + bedeutung)
    X = df_clean[['lemma', 'bedeutung']]
    y = df_clean['sachgruppe'].astype(str)  # Sicherstellen dass es Strings sind

    # Stratified split mit besserer Behandlung von seltenen Klassen
    # Finde Klassen mit nur 1 Sample
    class_counts = y.value_counts()
    single_sample_classes = class_counts[class_counts == 1].index

    if len(single_sample_classes) > 0:
        print(f"\nWarnung: {len(single_sample_classes)} Klassen mit nur 1 Sample gefunden.")
        print(f"Diese werden ins Trainingsset verschoben.")

        # Separiere Single-Sample Klassen
        mask_single = y.isin(single_sample_classes)
        X_single = X[mask_single]
        y_single = y[mask_single]
        X_multi = X[~mask_single]
        y_multi = y[~mask_single]

        # Stratified split für Multi-Sample Klassen
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi, y_multi,
                test_size=test_size,
                random_state=42,
                stratify=y_multi
            )
        except ValueError:
            # Falls immer noch Probleme
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi, y_multi,
                test_size=test_size,
                random_state=42
            )

        # Füge Single-Sample Klassen zum Trainingsset hinzu
        X_train = pd.concat([X_train, X_single])
        y_train = pd.concat([y_train, y_single])
    else:
        # Normale stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
        except ValueError:
            # Falls Klassen zu selten für stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )
    
    print(f"\nSplit:")
    print(f"  Training: {len(X_train)} Beispiele")
    print(f"  Test: {len(X_test)} Beispiele")
    
    # Modell erstellen und trainieren
    clf = SachgruppenClassifier(model_type=model_type, use_gpu=use_gpu)
    clf.train(X_train, y_train, tune_hyperparameters=tune)
    
    # Evaluieren
    accuracy, y_pred = clf.evaluate(X_test, y_test)
    
    # Speichern
    if save_path:
        clf.save(save_path)
    
    return clf, accuracy


def predict_interactive(model_path):
    """Interaktive Vorhersage-Loop."""
    clf = SachgruppenClassifier.load(model_path)

    print("\n" + "="*60)
    print("INTERAKTIVER VORHERSAGE-MODUS")
    print("="*60)

    if clf.use_lemma:
        print("Gib Lemma und Bedeutung ein, um Sachgruppen vorherzusagen.")
        print("Beende mit 'quit' oder Strg+C\n")

        while True:
            try:
                lemma = input("Lemma: ").strip()
                if lemma.lower() in ['quit', 'exit', 'q']:
                    break

                if not lemma:
                    continue

                bedeutung = input("Bedeutung: ").strip()
                if not bedeutung:
                    continue

                # Erstelle DataFrame für Vorhersage
                X_pred = pd.DataFrame({
                    'lemma': [lemma],
                    'bedeutung': [bedeutung]
                })

                sachgruppe = clf.predict(X_pred)[0]
                print(f"→ Sachgruppe: {sachgruppe}\n")

            except KeyboardInterrupt:
                print("\n\nBeendet.")
                break
    else:
        print("Gib Bedeutungen ein, um Sachgruppen vorherzusagen.")
        print("Beende mit 'quit' oder Strg+C\n")

        while True:
            try:
                bedeutung = input("Bedeutung: ").strip()
                if bedeutung.lower() in ['quit', 'exit', 'q']:
                    break

                if not bedeutung:
                    continue

                # Erstelle DataFrame für Vorhersage
                X_pred = pd.DataFrame({
                    'lemma': [''],
                    'bedeutung': [bedeutung]
                })

                sachgruppe = clf.predict(X_pred)[0]
                print(f"→ Sachgruppe: {sachgruppe}\n")

            except KeyboardInterrupt:
                print("\n\nBeendet.")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sachgruppen-Klassifikation für Wörterbuch-Daten'
    )
    
    parser.add_argument('--csv', type=str, default='test_output.csv',
                       help='Pfad zur CSV-Datei mit Trainingsdaten')
    parser.add_argument('--model', type=str, default='svm',
                       choices=['svm', 'logistic', 'rf', 'xgboost', 'nn'],
                       help='Modell-Typ (nn = Neural Network)')
    parser.add_argument('--tune', action='store_true',
                       help='Hyperparameter-Tuning durchführen')
    parser.add_argument('--save', type=str, default='sachgruppen_model.pkl',
                       help='Pfad zum Speichern des Modells')
    parser.add_argument('--predict', action='store_true',
                       help='Interaktiver Vorhersage-Modus')
    parser.add_argument('--load', type=str,
                       help='Gespeichertes Modell laden')
    parser.add_argument('--gpu', action='store_true',
                       help='GPU-Beschleunigung nutzen (erfordert ROCm/CUDA)')
    
    args = parser.parse_args()
    
    if args.predict:
        if not args.load:
            print("Fehler: --load muss angegeben werden für --predict")
        else:
            predict_interactive(args.load)
    else:
        train_and_evaluate(
            args.csv,
            model_type=args.model,
            tune=args.tune,
            save_path=args.save,
            use_gpu=args.gpu
        )
