#!/usr/bin/env python3
"""
Complete ML pipeline for Sachgruppen classification.

Features:
- Automatic train/test split
- Feature engineering (TF-IDF)
- Model training with multiple algorithms
- Hyperparameter tuning
- Cross-validation
- Model persistence
- Prediction interface
"""

import collections
import re

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
import os
import sys

# Force UTF-8 on stdout/stderr so Unicode (e.g. "→") prints work even when the
# subprocess inherits a legacy code page (cp1252 on Windows).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score


class StopwordRemover(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that removes stopwords from DataFrame text columns.

    Defined as a top-level class (not a local function) so it is fully picklable
    and can be serialized inside sklearn pipelines.
    """

    def __init__(self, stopwords_path: str | None = None):
        self.stopwords_path = stopwords_path  # None = default path (stopwords_de.txt)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from shap_utils import load_stopwords
        sw = load_stopwords(self.stopwords_path)
        result = X.copy()
        for col in ['lemma', 'bedeutung']:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda t: " ".join(w for w in str(t).split() if w.lower() not in sw)
                )
        return result


class PunctuationStripper(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that strips punctuation from words.

    Normalizes e.g. "Kind," and "Kind;" to "Kind" so punctuation does not
    generate distinct TF-IDF features (relevant for char_wb analyzer).
    Fully picklable as a top-level class.
    """

    import re
    _pattern = re.compile(r"[^\w\s]", re.UNICODE)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        for col in ['lemma', 'bedeutung']:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda t: self._pattern.sub("", str(t))
                )
        return result


class MinLengthFilter(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that removes words below a minimum length.

    Filters single characters and very short tokens before TF-IDF vectorization.
    Fully picklable as a top-level class.
    """

    def __init__(self, min_length: int = 1):
        self.min_length = min_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        for col in ['lemma', 'bedeutung']:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda t: " ".join(w for w in str(t).split() if len(w) >= self.min_length)
                )
        return result

_DORNSEIFF_GAZ = Path(__file__).resolve().parent / "assets" / "dornseiff_gaz_cache.pkl"


def _load_dornseiff_gazetteer() -> dict:
    """Lädt den vorberechneten Dornseiff-Gazetteer (assets/dornseiff_gaz_cache.pkl)."""
    if not _DORNSEIFF_GAZ.exists():
        raise FileNotFoundError(
            f"Dornseiff-Gazetteer nicht gefunden: {_DORNSEIFF_GAZ}\n"
            f"Datei aus dem Repo-Assets abrufen oder mit _extract_dornseiff.py neu erzeugen."
        )
    print(f"  Dornseiff-Gazetteer geladen ({_DORNSEIFF_GAZ.name})")
    with open(_DORNSEIFF_GAZ, "rb") as f:
        return pickle.load(f)


class SpacyVectorizer(BaseEstimator, TransformerMixin):
    """Dense 300-dim spaCy word vectors (de_core_news_lg), L2-normalised.

    Loaded lazily so the nlp object is never pickled — only the class name
    is needed to reload it on the next transform() call.
    """

    SPACY_MODEL = "de_core_news_lg"

    def __init__(self):
        self._nlp = None

    def fit(self, X, y=None):
        return self

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(
                self.SPACY_MODEL,
                disable=["tagger", "parser", "ner", "lemmatizer",
                         "attribute_ruler", "morphologizer", "tok2vec"],
            )
        return self._nlp

    def transform(self, X):
        from scipy.sparse import csr_matrix
        nlp = self._get_nlp()
        texts = [str(x) for x in X]
        vecs = np.zeros((len(texts), nlp.vocab.vectors_length), dtype=np.float32)
        for i, doc in enumerate(nlp.pipe(texts, batch_size=512)):
            v = doc.vector
            n = np.linalg.norm(v)
            if n > 0:
                vecs[i] = v / n
        return csr_matrix(vecs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_nlp"] = None
        return state


class DornseiffVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF over Dornseiff group codes present in each text.

    Gazetteer (word → groups) loaded from assets/dornseiff_gaz_cache.pkl.
    Texts are lemmatised with de_core_news_lg before lookup so inflected
    forms (e.g. "kocht" → "kochen", "warmen" → "warm") are matched.
    The fitted gazetteer and internal TfidfVectorizer are fully picklable;
    the nlp object is excluded from pickling and reloaded lazily.
    """

    SPACY_MODEL = "de_core_news_lg"
    # keep tok2vec + morphologizer + attribute_ruler + lemmatizer
    _SPACY_DISABLE = ["parser", "senter", "ner"]

    def __init__(self):
        self._gazetteer: dict | None = None
        self._tfidf = None
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.SPACY_MODEL, disable=self._SPACY_DISABLE)
        return self._nlp

    def _lemmatize_batch(self, texts: list[str]) -> list[list[str]]:
        nlp = self._get_nlp()
        result = []
        for doc in nlp.pipe(texts, batch_size=512):
            result.append([
                t.lemma_.lower() for t in doc
                if t.is_alpha and len(t.lemma_) >= 3
            ])
        return result

    def _codes_for_lemmas(self, lemmas: list[str]) -> list[str]:
        out = []
        for lemma in lemmas:
            for g in self._gazetteer.get(lemma, ()):
                out.append(g)
        return out

    def fit(self, X, y=None):
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        self._gazetteer = _load_dornseiff_gazetteer()
        lemma_lists = self._lemmatize_batch([str(x) for x in X])
        self._tfidf = _TV(analyzer=self._codes_for_lemmas, sublinear_tf=True)
        self._tfidf.fit(lemma_lists)
        return self

    def transform(self, X):
        lemma_lists = self._lemmatize_batch([str(x) for x in X])
        return self._tfidf.transform(lemma_lists)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_nlp"] = None
        return state


try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Skipping XGBoost model.")


class SachgruppenClassifier:
    """
    Main class for Sachgruppen classification.
    """

    def __init__(self, model_type='svm', random_state=42, use_gpu=False,
                 use_lemma=True, remove_stopwords=False,
                 min_word_length: int = 1,
                 analyzer: str = 'char_wb',
                 word_ngram_max: int = 1,
                 use_word_features: bool = True,
                 use_svd: bool = False,
                 svd_components: int = 500,
                 use_spacy: bool = False,
                 use_dornseiff: bool = False,
                 svm_c: float = 1.0,
                 xgb_n_estimators: int = 300,
                 xgb_max_depth: int = 6,
                 xgb_learning_rate: float = 0.05,
                 xgb_subsample: float = 0.8,
                 nn_hidden_layer_sizes: tuple = (100,),
                 nn_alpha: float = 0.0001,
                 nn_learning_rate_init: float = 0.0005,
                 nn_n_iter_no_change: int = 5):
        """
        Args:
            model_type: 'svm', 'logistic', 'rf', 'xgboost', or 'nn' (neural network)
            random_state: For reproducibility
            use_gpu: Enable GPU acceleration (XGBoost only)
            use_lemma: Include lemma column as additional features
            remove_stopwords: Remove stopwords from text before TF-IDF
            min_word_length: Minimum word length (1–5); shorter words removed before TF-IDF
            analyzer: TF-IDF analyzer: 'char_wb' (character n-grams) or 'word' (word-level)
            word_ngram_max: Max n for word analyzer (1=(1,1), 2=(1,2)); ignored for char_wb
            use_word_features: Extra word-level branch for bedeutung (char_wb mode only)
            use_svd: Enable TruncatedSVD (LSA) after vectorization (XGBoost: right after
                vectorizer; NN: after the MaxAbsScaler, to compress the ~46k-dim sparse
                input before the MLP)
            svd_components: Number of SVD dimensions (only relevant when use_svd=True)
            use_spacy: Add de_core_news_lg 300-dim word vectors on top of TF-IDF (+0.6 pp Top-1)
            use_dornseiff: Add Dornseiff gazetteer TF-IDF features on top of TF-IDF (+0.3 pp Top-1); loads from assets/dornseiff_gaz_cache.pkl
            svm_c: Regularization parameter C for LinearSVC (default: 1.0)
            xgb_n_estimators: Number of trees for XGBoost (default: 300)
            xgb_max_depth: Maximum tree depth for XGBoost (default: 6)
            xgb_learning_rate: Learning rate for XGBoost (default: 0.05)
            xgb_subsample: Row subsampling rate for XGBoost (default: 0.8)
            nn_hidden_layer_sizes: MLP hidden layer sizes as tuple (default: (100,))
            nn_alpha: L2 regularization strength for MLP (default: 0.0001)
            nn_learning_rate_init: Initial learning rate for MLP adam solver (default: 0.0005)
            nn_n_iter_no_change: Early-stopping patience for MLP, in epochs (default: 5)
        """
        self.model_type = model_type
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.use_lemma = use_lemma
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.analyzer = analyzer
        self.word_ngram_max = word_ngram_max
        self.use_word_features = use_word_features
        self.use_svd = use_svd
        self.svd_components = svd_components
        self.use_spacy = use_spacy
        self.use_dornseiff = use_dornseiff
        self.svm_c = svm_c
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_subsample = xgb_subsample
        self.nn_hidden_layer_sizes = nn_hidden_layer_sizes
        self.nn_alpha = nn_alpha
        self.nn_learning_rate_init = nn_learning_rate_init
        self.nn_n_iter_no_change = nn_n_iter_no_change
        self.pipeline = None
        self.classes_ = None
        self.label_encoder = None  # XGBoost/NN/RF: string → integer encoding
        self.best_params_: dict = {}   # Best params after auto-tune (empty if not tuned)
        self.best_cv_score_: float = 0.0
        self.eval_metrics_: dict = {}  # Top-k / hierarchy / confidence (set by evaluate())

    def create_pipeline(self):
        """Build the ML pipeline."""

        # Separate vectorizers for lemma and bedeutung
        if self.analyzer == 'word':
            # Word-level: configurable n-gram size
            common_word_params = dict(
                ngram_range=(1, self.word_ngram_max),
                analyzer='word',
                min_df=2,
                sublinear_tf=True,
            )
            lemma_vectorizer = TfidfVectorizer(max_features=5000, **common_word_params)
            bedeutung_vectorizer = TfidfVectorizer(max_features=10000, **common_word_params)
        else:
            # Character n-grams (default)
            # Umlauts (ä, ö, ü) are intentionally NOT normalized (no strip_accents)
            # because they carry morphological information in German.
            lemma_vectorizer = TfidfVectorizer(
                ngram_range=(2, 5),
                analyzer='char_wb',
                max_features=10000,
                min_df=2,
                sublinear_tf=True,
            )
            bedeutung_vectorizer = TfidfVectorizer(
                ngram_range=(2, 4),
                analyzer='char_wb',
                max_features=20000,
                min_df=2,
                sublinear_tf=True,
            )

        if self.use_lemma:
            transformers = [
                ('lemma', lemma_vectorizer, 'lemma'),
                ('bedeutung', bedeutung_vectorizer, 'bedeutung'),
            ]
            # Additional word-level branch for bedeutung in char_wb mode
            if self.use_word_features and self.analyzer == 'char_wb':
                bedeutung_word_vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    analyzer='word',
                    max_features=15000,
                    min_df=2,
                    sublinear_tf=True,
                )
                transformers.append(('bedeutung_word', bedeutung_word_vectorizer, 'bedeutung'))
        else:
            # Bedeutung only (legacy mode)
            transformers = [('bedeutung', bedeutung_vectorizer, 'bedeutung')]

        # Semantic enrichment add-ons (both append to transformers regardless of use_lemma)
        if self.use_spacy:
            transformers.append(('spacy_bed', SpacyVectorizer(), 'bedeutung'))
        if self.use_dornseiff:
            transformers.append(('dornseiff', DornseiffVectorizer(), 'bedeutung'))

        vectorizer = ColumnTransformer(transformers)
        
        # Model selection
        if self.model_type == 'svm':
            classifier = LinearSVC(
                C=self.svm_c,
                class_weight='balanced',
                max_iter=5000,
                random_state=self.random_state,
                dual=False,  # More efficient when n_samples > n_features
                verbose=0
            )
        
        elif self.model_type == 'logistic':
            classifier = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state,
                solver='saga',
                n_jobs=-1,
                verbose=1
            )

        elif self.model_type == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        
        elif self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ValueError("XGBoost not installed!")

            # GPU parameters when available
            xgb_params = {
                'n_estimators': self.xgb_n_estimators,
                'max_depth': self.xgb_max_depth,
                'learning_rate': self.xgb_learning_rate,
                'subsample': self.xgb_subsample,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_state,
                'verbosity': 1,
            }

            if self.use_gpu:
                print("GPU mode enabled!")
                xgb_params.update({
                    'device': 'cuda',  # also works with ROCm
                    'tree_method': 'hist',  # GPU-optimized algorithm
                })
            else:
                xgb_params['n_jobs'] = -1

            classifier = xgb.XGBClassifier(**xgb_params)

        elif self.model_type == 'nn':
            # Neural network (multi-layer perceptron)
            classifier = MLPClassifier(
                hidden_layer_sizes=self.nn_hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=self.nn_alpha,
                batch_size=256,
                learning_rate='adaptive',
                learning_rate_init=self.nn_learning_rate_init,
                max_iter=50,  # epochs
                random_state=self.random_state,
                verbose=True,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=self.nn_n_iter_no_change
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Pipeline assembly: Punctuation → MinLength → Stopwords → Vectorizer → Classifier
        steps = [('punctuation_stripper', PunctuationStripper())]
        if self.min_word_length > 1:
            steps.append(('min_length_filter', MinLengthFilter(min_length=self.min_word_length)))
        if self.remove_stopwords:
            steps.append(('stopword_remover', StopwordRemover()))
        steps.append(('vectorizer', vectorizer))
        if self.use_svd and self.model_type == 'xgboost':
            steps.append(('svd', TruncatedSVD(n_components=self.svd_components,
                                               random_state=self.random_state)))
        if self.model_type == 'nn':
            # MLP is scale-sensitive, unlike LinearSVC/LogisticRegression/trees. Without
            # this, the dense spaCy/Dornseiff add-on blocks sit on a different scale than
            # the sparse char_wb-TF-IDF blocks and destabilise gradient training.
            # MaxAbsScaler keeps sparsity (unlike StandardScaler's mean-centering).
            steps.append(('scaler', MaxAbsScaler()))
            if self.use_svd:
                # After scaling, not before: SVD is itself scale-sensitive, so it must see
                # the aligned blocks, not the raw TF-IDF/spaCy/Dornseiff scale mismatch.
                steps.append(('svd', TruncatedSVD(n_components=self.svd_components,
                                                   random_state=self.random_state)))
        steps.append(('classifier', classifier))
        self.pipeline = Pipeline(steps)
        
        return self.pipeline
    
    def train(self, X_train, y_train, tune_hyperparameters=False,
              tune_n_iter=20, tune_cv=3, progress_callback=None):
        """
        Train the model.

        Args:
            X_train: Training texts
            y_train: Training labels
            tune_hyperparameters: Whether to run hyperparameter tuning
            tune_n_iter: Number of random parameter combinations for auto-tune
            tune_cv: Number of cross-validation folds for auto-tune
            progress_callback: Optional function(pct, msg) for progress reporting
        """
        if self.pipeline is None:
            self.create_pipeline()

        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Number of classes: {len(np.unique(y_train))}")

        # XGBoost and Neural Network require integer labels; RandomForest needs it too
        # because sklearn's class_weight='balanced' errors on all-digit string labels
        # (e.g. "1000") -- see RandomForestClassifier._validate_y_class_weight.
        if self.model_type in ['xgboost', 'nn', 'rf']:
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            print(f"Labels encoded for {self.model_type.upper()} (string → integer)")
        else:
            y_train_encoded = y_train

        if tune_hyperparameters:
            self._tune_hyperparameters(X_train, y_train_encoded,
                                       n_iter=tune_n_iter, cv=tune_cv)
        else:
            # Report real progress via callback for XGBoost
            if progress_callback and self.model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    n_rounds = self.xgb_n_estimators
                    _cb = progress_callback

                    class _RoundProgress(xgb.callback.TrainingCallback):
                        def after_iteration(self, model, epoch, evals_log):
                            pct = int(35 + (epoch + 1) / n_rounds * 50)
                            _cb(pct, f"XGBoost: round {epoch + 1}/{n_rounds}")
                            return False

                    self.pipeline.named_steps['classifier'].set_params(
                        callbacks=[_RoundProgress()]
                    )
                except Exception:
                    pass  # do not crash on callback errors

            print("Starting training...")
            self.pipeline.fit(X_train, y_train_encoded)

        self.classes_ = self.pipeline.classes_
        print("Training complete!")
    
    def _tune_hyperparameters(self, X_train, y_train, n_iter=20, cv=3):
        """Hyperparameter tuning via RandomizedSearchCV."""
        print(f"\nStarting hyperparameter tuning (n_iter={n_iter}, cv={cv}) ...")

        if self.model_type == 'svm':
            param_distributions = {
                'vectorizer__lemma__max_features': [5000, 10000, 20000],
                'vectorizer__bedeutung__max_features': [10000, 20000, 30000],
                'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            }
        elif self.model_type == 'logistic':
            param_distributions = {
                'vectorizer__lemma__max_features': [5000, 10000, 20000],
                'vectorizer__bedeutung__max_features': [10000, 20000],
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
            }
        elif self.model_type == 'rf':
            param_distributions = {
                'vectorizer__lemma__max_features': [5000, 10000],
                'vectorizer__bedeutung__max_features': [10000, 20000],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
            }
        elif self.model_type == 'xgboost':
            param_distributions = {
                'classifier__n_estimators': [200, 300, 500],
                'classifier__max_depth': [4, 6, 8],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__subsample': [0.7, 0.8, 0.9],
            }
        elif self.model_type == 'nn':
            param_distributions = {
                'classifier__hidden_layer_sizes': [(100,), (200, 100), (200, 100, 50), (300, 150, 75)],
                'classifier__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                'classifier__learning_rate_init': [1e-4, 5e-4, 1e-3, 5e-3],
            }
        else:
            param_distributions = {}

        if not param_distributions:
            print("No tuning defined for this model type.")
            return

        # XGBoost requires contiguous integer classes 0..N-1 in every CV fold.
        # Classes with fewer than cv samples are temporarily excluded from the search;
        # the final model is then retrained with best params on the full dataset.
        X_search, y_search = X_train, y_train
        needs_retrain = False
        if self.model_type in ['xgboost', 'nn']:
            counts = np.bincount(y_train)
            keep_classes = counts >= cv
            mask = keep_classes[y_train]
            if not mask.all():
                n_removed_samples = (~mask).sum()
                n_removed_classes = (~keep_classes).sum()
                print(f"  Auto-tune: {n_removed_samples} samples from {n_removed_classes} "
                      f"rare classes (< {cv} samples) temporarily excluded.")
                X_search = X_train[mask]
                y_search = y_train[mask]
                # Re-encode to contiguous integers 0..M-1 for the filtered subset
                from sklearn.preprocessing import LabelEncoder as _TmpLE
                _tmp_enc = _TmpLE()
                y_search = _tmp_enc.fit_transform(y_search)
                needs_retrain = True

        search = RandomizedSearchCV(
            self.pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state,
        )

        search.fit(X_search, y_search)
        self.best_params_ = search.best_params_
        self.best_cv_score_ = search.best_score_

        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.4f}")

        if needs_retrain:
            # Retrain final pipeline with best params on the full dataset
            print("  Training final model with best parameters on full data...")
            self.pipeline.set_params(**search.best_params_)
            self.pipeline.fit(X_train, y_train)
        else:
            self.pipeline = search.best_estimator_
    
    def evaluate(self, X_test, y_test, verbose=True):
        """Evaluate the model on test data."""
        if self.pipeline is None:
            raise ValueError("Model must be trained first!")

        # XGBoost, Neural Network and RandomForest: encode labels and filter unknown ones
        if self.model_type in ['xgboost', 'nn', 'rf'] and self.label_encoder is not None:
            known_labels = set(self.label_encoder.classes_)
            mask = y_test.isin(known_labels)

            if not mask.all():
                unknown_count = (~mask).sum()
                unknown_labels = set(y_test[~mask].unique())
                print(f"\nWarning: {unknown_count} test samples with unknown labels found: {unknown_labels}")
                print(f"These will be skipped during evaluation.")

                X_test = X_test[mask]
                y_test = y_test[mask]

            y_pred_encoded = self.pipeline.predict(X_test)
            # Decode back to original string labels
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = self.pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, zero_division=0)

        # Top-k, hierarchical and confidence/coverage metrics.
        # Computed on the (possibly filtered) X_test/y_test so they align with y_pred.
        try:
            self.eval_metrics_ = self._extended_metrics(X_test, y_test)
            report_str = report_str + "\n\n" + self._format_extended_metrics(self.eval_metrics_)
        except Exception as e:
            print(f"Warning: extended metrics skipped ({e})")
            self.eval_metrics_ = {}

        if verbose:
            print("\n" + "="*60)
            print("EVALUATION")
            print("="*60)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification report:")
            print(report_str)

        return accuracy, y_pred, report_str

    def _extended_metrics(self, X_test, y_test, ks=(1, 3, 5), group_len=2):
        """Top-k accuracy, hierarchical (group) accuracy and a confidence/coverage curve.

        Uses predict_proba when available, otherwise decision_function (SVM), so
        it works for every model type. The group accuracy compares the first
        ``group_len`` digits of the Sachgruppen code (e.g. 2 = Zweisteller-Gruppe).
        """
        y_true = np.asarray(y_test).astype(str)
        classifier = self.pipeline.named_steps['classifier']
        classes = classifier.classes_
        if self.label_encoder is not None:  # XGBoost/NN: integer → original string
            classes = self.label_encoder.inverse_transform(classes)
        classes = np.asarray(classes).astype(str)

        scores = None
        if self.model_type != 'svm' and hasattr(self.pipeline, 'predict_proba'):
            try:
                scores = np.asarray(self.pipeline.predict_proba(X_test), dtype=float)
            except Exception:
                scores = None
        if scores is None:
            scores = np.asarray(self.pipeline.decision_function(X_test), dtype=float)
        if scores.ndim == 1:  # binary decision_function safety
            scores = np.column_stack([-scores, scores])

        order = np.argsort(scores, axis=1)[:, ::-1]
        ranked = classes[order]
        pred1 = ranked[:, 0]

        metrics = {}
        for k in ks:
            metrics[f'top{k}'] = float((ranked[:, :k] == y_true[:, None]).any(axis=1).mean())

        grp_pred = np.array([s[:group_len] for s in pred1])
        grp_true = np.array([s[:group_len] for s in y_true])
        metrics['group_acc'] = float((grp_pred == grp_true).mean())
        metrics['group_len'] = group_len

        # Confidence = margin between best and second-best score; accept the most
        # confident predictions first and report Top-1 accuracy on that subset.
        sorted_scores = np.sort(scores, axis=1)[:, ::-1]
        margin = sorted_scores[:, 0] - sorted_scores[:, 1]
        correct1 = (pred1 == y_true)
        conf_order = np.argsort(margin)[::-1]
        correct_sorted = correct1[conf_order]
        n = len(correct_sorted)
        metrics['coverage'] = {
            f'{int(c * 100)}%': float(correct_sorted[:max(1, int(n * c))].mean())
            for c in (0.5, 0.7, 0.9)
        }
        return metrics

    @staticmethod
    def _format_extended_metrics(m: dict) -> str:
        """Render the extended metrics dict as a text block for the report file."""
        if not m:
            return ""
        lines = [
            "=" * 60,
            "TOP-k / HIERARCHIE / KONFIDENZ",
            "=" * 60,
            f"Top-1: {m['top1']:.4f}   Top-3: {m['top3']:.4f}   Top-5: {m['top5']:.4f}",
            f"Richtige {m['group_len']}-stellige Gruppe (Top-1): {m['group_acc']:.4f}",
            "Konfidenz/Abdeckung (Top-1 automatisch akzeptieren):",
        ]
        for cov, acc in m['coverage'].items():
            lines.append(f"  {cov:>4} Abdeckung  ->  {acc:.4f} Accuracy")
        return "\n".join(lines)
    
    def cross_validate(self, X, y, cv=5):
        """Run cross-validation."""
        if self.pipeline is None:
            self.create_pipeline()

        print(f"\nRunning {cv}-fold cross-validation...")
        scores = cross_val_score(
            self.pipeline, X, y,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )

        print(f"F1 scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
    def predict(self, texts):
        """Predict labels for new texts."""
        if self.pipeline is None:
            raise ValueError("Model must be trained first!")

        predictions = self.pipeline.predict(texts)

        # Decode integer predictions back to string labels for XGBoost/NN/RF
        if self.model_type in ['xgboost', 'nn', 'rf'] and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions
    
    def predict_proba(self, texts):
        """Prediction probabilities (if available)."""
        if self.pipeline is None:
            raise ValueError("Model must be trained first!")

        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(texts)
        else:
            raise ValueError(f"{self.model_type} does not support probabilities")
    
    def explain(self, X_pred, predicted_label: str, model_path: str = "",
                filter_stopwords: bool = True) -> dict:
        """
        Compute word-level SHAP scores for a single prediction.

        Args:
            X_pred: DataFrame with 'lemma' and/or 'bedeutung' columns (1 row)
            predicted_label: Predicted Sachgruppe as string
            model_path: Path to the model file for explainer caching

        Returns:
            {"lemma": [(word, score), ...], "bedeutung": [(word, score), ...]}
            score normalized to [-1, 1]; positive = supports the prediction
        """
        try:
            import shap_utils
        except ImportError as e:
            raise ImportError(
                "shap package not installed. Run 'pip install shap'."
            ) from e
        return shap_utils.get_word_shap_scores(
            self, X_pred, predicted_label, model_path, filter_stopwords=filter_stopwords
        )

    def save(self, filepath):
        """Save the trained model."""
        if self.pipeline is None:
            raise ValueError("Model must be trained first!")

        # Remove XGBoost callbacks before saving: local callback classes
        # (defined inside train()) cannot be pickled.
        if self.model_type == 'xgboost':
            try:
                self.pipeline.named_steps['classifier'].set_params(callbacks=None)
            except Exception:
                pass

        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'model_type': self.model_type,
                'classes': self.classes_,
                'label_encoder': self.label_encoder,  # XGBoost/NN/RF
                'use_gpu': self.use_gpu,
                'use_lemma': self.use_lemma,
                'remove_stopwords': self.remove_stopwords,
                'min_word_length': self.min_word_length,
                'analyzer': self.analyzer,
                'word_ngram_max': self.word_ngram_max,
                'use_word_features': self.use_word_features,
                'use_spacy': self.use_spacy,
                'use_dornseiff': self.use_dornseiff,
                'use_svd': self.use_svd,
                'svd_components': self.svd_components,
                'svm_c': self.svm_c,
                'xgb_n_estimators': self.xgb_n_estimators,
                'xgb_max_depth': self.xgb_max_depth,
                'xgb_learning_rate': self.xgb_learning_rate,
                'xgb_subsample': self.xgb_subsample,
                'nn_hidden_layer_sizes': self.nn_hidden_layer_sizes,
                'nn_alpha': self.nn_alpha,
                'nn_learning_rate_init': self.nn_learning_rate_init,
                'nn_n_iter_no_change': self.nn_n_iter_no_change,
            }, f)

        print(f"\nModel saved: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model."""
        import sachgruppen_classifier as _sc_mod

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Classes pickled as __main__ or __mp_main__ (multiprocessing)
                # must resolve to their canonical module.
                if module in ('__main__', '__mp_main__'):
                    if hasattr(_sc_mod, name):
                        return getattr(_sc_mod, name)
                return super().find_class(module, name)

        with open(filepath, 'rb') as f:
            data = _Unpickler(f).load()

        instance = cls(
            model_type=data['model_type'],
            use_gpu=data.get('use_gpu', False),
            use_lemma=data.get('use_lemma', True),
            remove_stopwords=data.get('remove_stopwords', False),
            min_word_length=data.get('min_word_length', 1),
            analyzer=data.get('analyzer', 'char_wb'),
            word_ngram_max=data.get('word_ngram_max', 1),
            use_word_features=data.get('use_word_features', False),  # False for backward compatibility
            use_spacy=data.get('use_spacy', False),
            use_dornseiff=data.get('use_dornseiff', False),
            use_svd=data.get('use_svd', False),
            svd_components=data.get('svd_components', 500),
            svm_c=data.get('svm_c', 1.0),
            xgb_n_estimators=data.get('xgb_n_estimators', 300),
            xgb_max_depth=data.get('xgb_max_depth', 6),
            xgb_learning_rate=data.get('xgb_learning_rate', 0.05),
            xgb_subsample=data.get('xgb_subsample', 0.8),
            nn_hidden_layer_sizes=data.get('nn_hidden_layer_sizes', (200, 100, 50)),
            nn_alpha=data.get('nn_alpha', 0.0001),
            nn_learning_rate_init=data.get('nn_learning_rate_init', 0.001),
            nn_n_iter_no_change=data.get('nn_n_iter_no_change', 5),
        )
        instance.pipeline = data['pipeline']
        instance.classes_ = data['classes']
        instance.label_encoder = data.get('label_encoder', None)  # XGBoost/NN/RF

        print(f"Model loaded: {filepath}")
        return instance


def train_and_evaluate(csv_file, model_type='svm', test_size=0.2,
                       tune=False, save_path=None, use_gpu=False,
                       use_lemma=True,
                       remove_stopwords=False, min_word_length=1,
                       analyzer='char_wb', word_ngram_max=1,
                       use_word_features=True, use_svd=False, svd_components=500,
                       use_spacy=False, use_dornseiff=False,
                       svm_c=1.0, xgb_n_estimators=300, xgb_max_depth=6,
                       xgb_learning_rate=0.05, xgb_subsample=0.8,
                       nn_hidden_layer_sizes=(100,), nn_alpha=0.0001,
                       nn_learning_rate_init=0.0005, nn_n_iter_no_change=5,
                       tune_n_iter=20, tune_cv=3,
                       progress_callback=None):
    """
    Main function for training and evaluation.

    Args:
        use_gpu: Enable GPU acceleration (XGBoost only)
        progress_callback: Optional function(pct: int, msg: str) for progress reporting
    """
    def _cb(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    # Load data
    _cb(5, "Lade Daten…")
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file, sep=None, engine="python")
    df.columns = [c.lstrip('﻿').strip() for c in df.columns]

    print(f"\nDataset info:")
    print(f"  Entries: {len(df)}")
    print(f"  Sachgruppen: {df['sachgruppe'].nunique()}")
    print(f"  Avg. bedeutung length: {df['bedeutung'].str.len().mean():.1f} chars")
    
    # Class distribution
    print(f"\nTop-10 Sachgruppen:")
    print(df['sachgruppe'].value_counts().head(10))

    # Data cleaning: handle NaN values
    _cb(15, "Bereinige Daten…")
    print("\nCleaning data...")

    # Auto-detect: disable use_lemma if the lemma column is missing or entirely empty
    if use_lemma:
        if 'lemma' not in df.columns:
            print("Info: Spalte 'lemma' nicht vorhanden – use_lemma automatisch deaktiviert.")
            use_lemma = False
        else:
            real_lemmata = df['lemma'].dropna().astype(str).str.strip()
            real_lemmata = real_lemmata[real_lemmata.str.len() > 0]
            if real_lemmata.empty:
                print("Info: Lemma-Spalte leer – use_lemma automatisch deaktiviert.")
                use_lemma = False

    required_cols = (['lemma', 'bedeutung', 'sachgruppe'] if use_lemma
                     else ['bedeutung', 'sachgruppe'])
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        print("NaN values found:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")

    df_clean = df.dropna(subset=required_cols)

    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"Warning: {removed} rows with missing values removed")
        print(f"Remaining: {len(df_clean)} entries")

    # Replace empty strings with placeholder
    if use_lemma:
        df_clean = df_clean.copy()
        df_clean['lemma'] = df_clean['lemma'].astype(str).replace('', 'LEER')
    df_clean['bedeutung'] = df_clean['bedeutung'].astype(str).replace('', 'LEER')

    # Train/test split; X is a DataFrame with the relevant columns
    x_cols = ['lemma', 'bedeutung'] if use_lemma else ['bedeutung']
    X = df_clean[x_cols]
    y = df_clean['sachgruppe'].astype(str)

    # Stratified split with special handling for rare classes.
    #
    # Known caveat (documented, deliberately not changed): ~75% of the corpus
    # rows share their 'bedeutung' string with at least one other row (e.g.
    # "Familienname" 1400+ times). The random split puts identical glosses in
    # both train and test, so the reported test accuracy is inflated by
    # memorization. For an honest generalization estimate use the goldstandard
    # evaluation (scripts/evaluate_goldstandard_nn.py); the gap is quantified
    # in scripts/diagnose_goldstandard_gap.py.
    class_counts = y.value_counts()
    single_sample_classes = class_counts[class_counts == 1].index

    if len(single_sample_classes) > 0:
        print(f"\nWarning: {len(single_sample_classes)} classes with only 1 sample found.")
        print(f"These will be moved to the training set.")

        # Separate single-sample classes
        mask_single = y.isin(single_sample_classes)
        X_single = X[mask_single]
        y_single = y[mask_single]
        X_multi = X[~mask_single]
        y_multi = y[~mask_single]

        # Stratified split for multi-sample classes
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi, y_multi,
                test_size=test_size,
                random_state=42,
                stratify=y_multi
            )
        except ValueError:
            # Fallback if stratification still fails
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi, y_multi,
                test_size=test_size,
                random_state=42
            )

        # Append single-sample classes to training set
        X_train = pd.concat([X_train, X_single])
        y_train = pd.concat([y_train, y_single])
    else:
        # Standard stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
        except ValueError:
            # Fallback if classes are too rare for stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )
    
    _cb(25, "Teile Daten auf…")
    print(f"\nSplit:")
    print(f"  Training: {len(X_train)} examples")
    print(f"  Test: {len(X_test)} examples")

    # Create and train model
    _cb(35, f"Trainiere {model_type.upper()}-Modell ({len(y_train)} Samples, {y.nunique()} Klassen)…")
    clf = SachgruppenClassifier(
        model_type=model_type, use_gpu=use_gpu,
        use_lemma=use_lemma,
        remove_stopwords=remove_stopwords,
        min_word_length=min_word_length,
        analyzer=analyzer,
        word_ngram_max=word_ngram_max,
        use_word_features=use_word_features,
        use_svd=use_svd,
        svd_components=svd_components,
        use_spacy=use_spacy,
        use_dornseiff=use_dornseiff,
        svm_c=svm_c,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_subsample=xgb_subsample,
        nn_hidden_layer_sizes=nn_hidden_layer_sizes,
        nn_alpha=nn_alpha,
        nn_learning_rate_init=nn_learning_rate_init,
        nn_n_iter_no_change=nn_n_iter_no_change,
    )
    clf.train(X_train, y_train, tune_hyperparameters=tune,
              tune_n_iter=tune_n_iter, tune_cv=tune_cv,
              progress_callback=progress_callback)

    # Evaluate
    _cb(85, "Evaluiere Modell…")
    accuracy, y_pred, report_str = clf.evaluate(X_test, y_test)

    # Save
    _cb(95, "Speichere Modell…")
    if save_path:
        clf.save(save_path)

    _cb(100, "Fertig!")
    return clf, accuracy, report_str


def predict_interactive(model_path):
    """Interactive prediction loop."""
    clf = SachgruppenClassifier.load(model_path)

    print("\n" + "="*60)
    print("INTERAKTIVER VORHERSAGE-MODUS")
    print("="*60)

    if clf.use_lemma:
        print("Enter lemma and bedeutung to predict Sachgruppen.")
        print("Exit with 'quit' or Ctrl+C\n")

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

                X_pred = pd.DataFrame({
                    'lemma': [lemma],
                    'bedeutung': [bedeutung]
                })

                sachgruppe = clf.predict(X_pred)[0]
                print(f"→ Sachgruppe: {sachgruppe}\n")

            except KeyboardInterrupt:
                print("\n\nExiting.")
                break
    else:
        print("Enter bedeutung to predict Sachgruppen.")
        print("Exit with 'quit' or Ctrl+C\n")

        while True:
            try:
                bedeutung = input("Bedeutung: ").strip()
                if bedeutung.lower() in ['quit', 'exit', 'q']:
                    break

                if not bedeutung:
                    continue

                X_pred = pd.DataFrame({
                    'lemma': [''],
                    'bedeutung': [bedeutung]
                })

                sachgruppe = clf.predict(X_pred)[0]
                print(f"→ Sachgruppe: {sachgruppe}\n")

            except KeyboardInterrupt:
                print("\n\nExiting.")
                break


def send_notification(to_addr: str, subject: str, body: str) -> bool:
    """Send an e-mail notification via SMTP.

    Configuration via environment variables / .env:
        SMTP_HOST      – SMTP server (required)
        SMTP_PORT      – Port, default 587
        SMTP_USER      – Username / sender address (required)
        SMTP_PASSWORD  – Password (required)
        SMTP_FROM      – Sender address (optional, falls back to SMTP_USER)

    Returns True on success, False on error.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    host = os.environ.get('SMTP_HOST', '')
    port = int(os.environ.get('SMTP_PORT', 587))
    user = os.environ.get('SMTP_USER', '')
    password = os.environ.get('SMTP_PASSWORD', '')
    from_addr = os.environ.get('SMTP_FROM', user)

    if not (host and user and password):
        print("Notification skipped: SMTP_HOST, SMTP_USER or SMTP_PASSWORD not configured.")
        print("Set the values in the .env file.")
        return False

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        print(f"Notification sent to {to_addr}")
        return True
    except Exception as e:
        print(f"Notification failed: {e}")
        return False


if __name__ == '__main__':
    import itertools
    import json
    import time as _time

    parser = argparse.ArgumentParser(
        description='Sachgruppen classification for dictionary data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    parser.add_argument('--csv', type=str, default='test_output.csv',
                        help='Path to CSV file with training data')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set fraction (0.0–1.0)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for model, metadata and report')

    # --- Model (one or more values → batch) ---
    parser.add_argument('--model', type=str, nargs='+', default=['svm'],
                        choices=['svm', 'logistic', 'rf', 'xgboost', 'nn'],
                        metavar='MODEL',
                        help='Model type(s): svm logistic rf xgboost nn')

    # --- Feature engineering (one or more values → batch) ---
    parser.add_argument('--analyzer', type=str, nargs='+', default=['char_wb'],
                        choices=['char_wb', 'word'],
                        metavar='ANALYZER',
                        help='Vectorizer mode: char_wb word')
    parser.add_argument('--word-ngram-max', type=int, default=1,
                        help='Maximum n-gram size for analyzer=word')
    parser.add_argument('--min-length', type=int, nargs='+', default=[1],
                        metavar='N',
                        help='Minimum word length(s): 1 2 3')
    parser.add_argument('--stopwords', type=str, nargs='+', default=['false'],
                        choices=['true', 'false'],
                        metavar='BOOL',
                        help='Remove stopwords: true false (both → batch)')

    # --- Hyperparameter tuning ---
    parser.add_argument('--tune', action='store_true',
                        help='Auto-tune: run RandomizedSearchCV')
    parser.add_argument('--tune-n-iter', type=int, default=20,
                        help='Number of random combinations for auto-tune')
    parser.add_argument('--tune-cv', type=int, default=3,
                        help='Number of CV folds for auto-tune (min. 2)')

    # --- Model-specific parameters ---
    parser.add_argument('--svm-c', type=float, default=1.0,
                        help='SVM regularization parameter C')
    parser.add_argument('--xgb-n-estimators', type=int, default=300,
                        help='XGBoost: number of trees')
    parser.add_argument('--xgb-max-depth', type=int, default=6,
                        help='XGBoost: maximum tree depth')
    parser.add_argument('--xgb-learning-rate', type=float, default=0.05,
                        help='XGBoost: learning rate')
    parser.add_argument('--xgb-subsample', type=float, default=0.8,
                        help='XGBoost: row subsampling rate')
    parser.add_argument('--nn-hidden-layers', type=str, default='100',
                        help='NN: hidden layer sizes as comma-separated integers (default: 100)')
    parser.add_argument('--nn-alpha', type=float, default=0.0001,
                        help='NN: L2 regularization strength (default: 0.0001)')
    parser.add_argument('--nn-learning-rate-init', type=float, default=0.0005,
                        help='NN: initial learning rate for adam solver (default: 0.0005)')
    parser.add_argument('--nn-n-iter-no-change', type=int, default=5,
                        help='NN: early-stopping patience in epochs (default: 5)')
    parser.add_argument('--use-svd', action='store_true',
                        help='Enable TruncatedSVD (LSA) dimensionality reduction (XGBoost: right '
                             'after the vectorizer; NN: after the MaxAbsScaler)')
    parser.add_argument('--svd-components', type=int, default=500,
                        help='Number of SVD dimensions when --use-svd is set (default: 500)')
    parser.add_argument('--gpu', action='store_true',
                        help='GPU acceleration for XGBoost (requires CUDA/ROCm)')

    # --- Semantic enrichment ---
    parser.add_argument('--use-spacy', action='store_true',
                        help='Add de_core_news_lg 300-dim word vectors on top of TF-IDF')
    parser.add_argument('--use-dornseiff', action='store_true',
                        help='Add Dornseiff gazetteer TF-IDF features on top of TF-IDF (loads assets/dornseiff_gaz_cache.pkl)')

    # --- Internal options (web app) ---
    parser.add_argument('--progress-file', type=str, default='',
                        metavar='PATH',
                        help='JSON file for progress updates (set by the web app)')

    # --- Notification ---
    parser.add_argument('--notify', type=str, default='',
                        metavar='EMAIL',
                        help='E-mail address for notification after completion')

    # --- Prediction ---
    parser.add_argument('--predict', action='store_true',
                        help='Interactive prediction mode')
    parser.add_argument('--load', type=str,
                        help='Load a saved model (for --predict)')

    args = parser.parse_args()

    # Parse --nn-hidden-layers string "200,100,50" → tuple (200, 100, 50)
    args.nn_hidden_layer_sizes = tuple(int(x) for x in args.nn_hidden_layers.split(','))

    if args.predict:
        if not args.load:
            print("Error: --load is required for --predict")
        else:
            predict_interactive(args.load)
    else:
        # Cartesian product of all batch dimensions
        stopwords_vals = [s == 'true' for s in args.stopwords]
        configs = list(itertools.product(
            args.model,
            args.analyzer,
            args.min_length,
            stopwords_vals,
        ))

        batch_mode = len(configs) > 1
        total = len(configs)
        if batch_mode:
            print(f"\n{'='*60}")
            print(f"BATCH TRAINING: {total} configurations")
            print(f"{'='*60}")

        import os
        os.makedirs(args.output_dir, exist_ok=True)
        tune_cv = max(2, args.tune_cv)
        batch_results = []

        # Determine sample count once (for metadata and time estimation)
        try:
            _df_tmp = pd.read_csv(args.csv, sep=None, engine='python')
            num_samples = int(len(_df_tmp.dropna(subset=['lemma', 'bedeutung', 'sachgruppe'])))
            del _df_tmp
        except Exception:
            num_samples = 0

        # Progress file: written by this subprocess, read by the web app
        _pf = args.progress_file
        def _write_progress(pct: int, msg: str, done: bool = False,
                            accuracy: float = 0.0, model_file: str = '',
                            training_time: float = 0.0, error: str = '',
                            config_idx: int = 0, config_total: int = 0):
            if not _pf:
                return
            try:
                with open(_pf, 'w') as _f:
                    json.dump({
                        'pct': pct, 'msg': msg, 'done': done,
                        'accuracy': accuracy, 'model_file': model_file,
                        'training_time': training_time, 'error': error,
                        'config_idx': config_idx, 'config_total': config_total,
                    }, _f)
            except OSError:
                pass

        _write_progress(0, 'Starte…', config_total=total)

        for i, (model, analyzer, min_len, sw) in enumerate(configs, 1):
            if batch_mode:
                print(f"\n[{i}/{total}] model={model}  analyzer={analyzer}  "
                      f"min_length={min_len}  stopwords={sw}")
                print("-" * 60)

            # Derive filename from configuration
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sw_tag = "sw1" if sw else "sw0"
            stem = f"{model}_{analyzer}_ml{min_len}_{sw_tag}_{ts}"
            save_path = os.path.join(args.output_dir, f"{stem}.pkl")

            _start = _time.time()
            try:
                clf, accuracy, report_str = train_and_evaluate(
                    args.csv,
                    model_type=model,
                    test_size=args.test_size,
                    tune=args.tune,
                    save_path=save_path,
                    use_gpu=args.gpu,
                    remove_stopwords=sw,
                    min_word_length=min_len,
                    analyzer=analyzer,
                    word_ngram_max=args.word_ngram_max,
                    use_spacy=args.use_spacy,
                    use_dornseiff=args.use_dornseiff,
                    svm_c=args.svm_c,
                    xgb_n_estimators=args.xgb_n_estimators,
                    xgb_max_depth=args.xgb_max_depth,
                    xgb_learning_rate=args.xgb_learning_rate,
                    xgb_subsample=args.xgb_subsample,
                    nn_hidden_layer_sizes=args.nn_hidden_layer_sizes,
                    nn_alpha=args.nn_alpha,
                    nn_learning_rate_init=args.nn_learning_rate_init,
                    nn_n_iter_no_change=args.nn_n_iter_no_change,
                    use_svd=args.use_svd,
                    svd_components=args.svd_components,
                    tune_n_iter=args.tune_n_iter,
                    tune_cv=tune_cv,
                    progress_callback=lambda pct, msg: _write_progress(
                        pct, msg, config_idx=i, config_total=total),
                )
            except Exception as exc:
                _write_progress(0, str(exc), done=True, error=str(exc),
                                config_idx=i, config_total=total)
                raise
            training_time = _time.time() - _start

            # Save metadata and classification report
            save_base = save_path.replace('.pkl', '')
            metadata = {
                "model_type": model,
                "accuracy": accuracy,
                "training_time": training_time,
                "num_samples": num_samples,
                "num_classes": len(clf.classes_) if clf.classes_ is not None else 0,
                "topk_metrics": getattr(clf, 'eval_metrics_', {}),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "csv_file": args.csv,
                "test_size": args.test_size,
                "use_lemma": clf.use_lemma,
                "use_spacy": args.use_spacy,
                "use_dornseiff": args.use_dornseiff,
                "remove_stopwords": sw,
                "min_word_length": min_len,
                "analyzer": analyzer,
                "word_ngram_max": args.word_ngram_max,
                "tune": args.tune,
                "tune_n_iter": args.tune_n_iter if args.tune else None,
                "tune_cv": tune_cv if args.tune else None,
                "best_params": clf.best_params_ if args.tune else {},
                "best_cv_score": getattr(clf, 'best_cv_score_', 0.0) if args.tune else 0.0,
                "svm_c": args.svm_c if model == 'svm' else None,
                "xgb_n_estimators": args.xgb_n_estimators if model == 'xgboost' else None,
                "xgb_max_depth": args.xgb_max_depth if model == 'xgboost' else None,
                "xgb_learning_rate": args.xgb_learning_rate if model == 'xgboost' else None,
                "xgb_subsample": args.xgb_subsample if model == 'xgboost' else None,
                "nn_hidden_layer_sizes": list(args.nn_hidden_layer_sizes) if model == 'nn' else None,
                "nn_alpha": args.nn_alpha if model == 'nn' else None,
                "nn_learning_rate_init": args.nn_learning_rate_init if model == 'nn' else None,
                "nn_n_iter_no_change": args.nn_n_iter_no_change if model == 'nn' else None,
                "use_svd": args.use_svd,
                "svd_components": args.svd_components if args.use_svd else None,
            }
            with open(f"{save_base}_metadata.json", "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, ensure_ascii=False)
            with open(f"{save_base}_report.txt", "w", encoding="utf-8") as fh:
                fh.write(report_str)

            print(f"Accuracy: {accuracy:.4f}  |  Time: {training_time:.1f}s")
            print(f"Model:    {save_path}")
            print(f"Metadata: {save_base}_metadata.json")
            print(f"Report:   {save_base}_report.txt")

            batch_results.append((stem, model, analyzer, min_len, sw, accuracy, training_time))

            # Update progress after each run
            _write_progress(100 if not batch_mode else int(i / total * 100),
                            'Fertig!' if not batch_mode else f'Konfiguration {i}/{total} abgeschlossen',
                            done=not batch_mode,
                            accuracy=accuracy, model_file=save_path,
                            training_time=training_time,
                            config_idx=i, config_total=total)

        if batch_mode:
            print(f"\n{'='*60}")
            print("BATCH RESULTS")
            print(f"{'='*60}")
            batch_results.sort(key=lambda x: x[5], reverse=True)
            for rank, (stem, model, analyzer, ml, sw, acc, t) in enumerate(batch_results, 1):
                sw_str = "sw" if sw else "  "
                print(f"  {rank:2}. {acc:.4f}  {model:<8} {analyzer:<8} ml={ml} {sw_str}  "
                      f"({t:.0f}s)  {stem}")
            if batch_results:
                best = batch_results[0]
                _write_progress(100, 'Fertig!', done=True,
                                accuracy=best[5],
                                model_file=os.path.join(args.output_dir, f"{best[0]}.pkl"),
                                training_time=sum(r[6] for r in batch_results))

        if args.notify:
            total_time = sum(r[6] for r in batch_results)
            best = batch_results[0] if batch_results else None

            if batch_mode:
                lines = [
                    f"Batch-Training abgeschlossen: {total} Konfigurationen in {total_time/60:.1f} min",
                    f"CSV: {args.csv}",
                    f"Auto-Tune: {'ja' if args.tune else 'nein'}",
                    "",
                    "Ergebnisse (sortiert nach Accuracy):",
                ]
                for rank, (stem, model, analyzer, ml, sw, acc, t) in enumerate(batch_results, 1):
                    sw_str = "+sw" if sw else ""
                    lines.append(f"  {rank:2}. {acc:.4f}  {model} {analyzer} ml={ml}{sw_str}  ({t:.0f}s)")
                if best:
                    lines += ["", f"Bestes Modell: {best[0]}.pkl"]
            else:
                r = batch_results[0]
                lines = [
                    f"Training abgeschlossen in {r[6]/60:.1f} min",
                    f"Modell: {r[1]}  |  Accuracy: {r[5]:.4f}",
                    f"CSV: {args.csv}",
                    f"Gespeichert: {r[0]}.pkl",
                ]

            send_notification(
                to_addr=args.notify,
                subject=f"LexoTerm Training abgeschlossen – beste Accuracy: {batch_results[0][5]:.4f}" if batch_results else "LexoTerm Training abgeschlossen",
                body="\n".join(lines),
            )
