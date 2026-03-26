"""
SHAP-basierte Erklärung für SachgruppenClassifier.

Aggregiert Char-N-Gram SHAP-Werte auf Wort-Ebene, damit der Nutzer sehen kann,
welche Wörter des Inputs die Vorhersage wie beeinflusst haben.
"""

import unicodedata
import numpy as np
from pathlib import Path
from typing import Any

# Modul-level Explainer-Cache: keyed by model_path (oder id(clf) als Fallback)
_EXPLAINER_CACHE: dict[str, Any] = {}

# Modul-level Stopwort-Cache
_STOPWORDS_CACHE: frozenset | None = None
_STOPWORDS_PATH = Path(__file__).parent / "stopwords_de.txt"


def load_stopwords(path: str | Path | None = None) -> frozenset[str]:
    """
    Lädt Stoppwörter aus Datei (case-insensitive).

    Format: ein Wort pro Zeile, Zeilen mit # werden ignoriert.
    Fallback auf leeres Set wenn Datei nicht gefunden.
    """
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None and path is None:
        return _STOPWORDS_CACHE

    target = Path(path) if path else _STOPWORDS_PATH
    try:
        with open(target, encoding="utf-8") as f:
            words = frozenset(
                line.strip().lower() for line in f
                if line.strip() and not line.startswith("#")
            )
    except FileNotFoundError:
        words = frozenset()

    if path is None:
        _STOPWORDS_CACHE = words
    return words


def strip_accents_unicode(s: str) -> str:
    """Reproduziert scikit-learns strip_accents='unicode' Logik."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def char_wb_ngrams(word: str, min_n: int, max_n: int) -> set[str]:
    """
    Reproduziert scikit-learns char_wb Tokenisierung exakt.

    char_wb: Wort wird mit Leerzeichen gepaddet, dann werden N-Gramme extrahiert.
    Beispiel: "Kind" → " Kind " → {" K", "Ki", "in", "nd", "d ", " Ki", ...}
    """
    processed = strip_accents_unicode(word)
    padded = " " + processed + " "
    ngrams = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(padded) - n + 1):
            ngrams.add(padded[i:i + n])
    return ngrams


def _aggregate_to_words_word_level(
    text: str,
    vectorizer,
    shap_vals_slice: np.ndarray,
    stopwords: frozenset[str] = frozenset(),
) -> list[tuple[str, float]]:
    """
    Aggregiert word-level TF-IDF SHAP-Werte auf Wort-Ebene.

    Für Unigrams: SHAP-Wert des Worts direkt.
    Für Bigrams (ngram_range=(1,2)): SHAP-Wert aller Bigrams, die das Wort enthalten,
    wird zum Score des Worts addiert.

    Returns:
        Liste von (originalwort, raw_score) Paaren (noch NICHT normalisiert)
    """
    feature_names = vectorizer.get_feature_names_out()
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    original_words = text.split()
    if not original_words:
        return []

    # Normalisierte Version für Feature-Lookup
    normalized_words = [strip_accents_unicode(w.lower()) for w in original_words]

    result = []
    for i, (orig_word, norm_word) in enumerate(zip(original_words, normalized_words)):
        if norm_word in stopwords:
            continue
        # Unigram + angrenzende Bigrams
        candidates = [norm_word]
        if i > 0:
            candidates.append(f"{normalized_words[i - 1]} {norm_word}")
        if i < len(normalized_words) - 1:
            candidates.append(f"{norm_word} {normalized_words[i + 1]}")
        scores = [
            float(shap_vals_slice[feature_index[c]])
            for c in candidates
            if c in feature_index
        ]
        result.append((orig_word, sum(scores)))

    return result


def _aggregate_to_words(
    text: str,
    vectorizer,
    shap_vals_slice: np.ndarray,
    stopwords: frozenset[str] = frozenset(),
) -> list[tuple[str, float]]:
    """
    Aggregiert char-n-gram SHAP-Werte auf Wort-Ebene.

    Für jedes Wort: Summiere die SHAP-Werte aller N-Gramme, die aus diesem Wort
    stammen. Summe statt Mittelwert, damit längere Inhaltswörter (mehr N-Gramme,
    mehr spezifisches Signal) stärker gewichtet werden als kurze Stoppwörter.

    Stoppwörter werden vollständig übersprungen wenn `stopwords` nicht leer ist.

    Hinweis: Normalisierung erfolgt NICHT hier, sondern global über alle Felder
    zusammen in get_word_shap_scores(), damit Lemma- und Bedeutungs-Scores
    vergleichbar bleiben.

    Returns:
        Liste von (wort, raw_score) Paaren (noch NICHT normalisiert)
    """
    min_n, max_n = vectorizer.ngram_range
    feature_names = vectorizer.get_feature_names_out()
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    words = text.split()
    if not words:
        return []

    result = []
    for word in words:
        if word.lower() in stopwords:
            continue  # Stoppwort überspringen
        ngrams = char_wb_ngrams(word, min_n, max_n)
        scores = [
            float(shap_vals_slice[feature_index[ng]])
            for ng in ngrams
            if ng in feature_index
        ]
        # Summe: längere, spezifischere Wörter akkumulieren mehr Signal
        result.append((word, sum(scores)))

    return result


def _build_explainer(model_type: str, classifier, X_transformed):
    """
    Erstellt den passenden SHAP-Explainer für den Modelltyp.

    - SVM / Logistic: LinearExplainer (schnell, exakt)
    - RF / XGBoost: TreeExplainer (schnell, exakt)
    - NN (MLP): PermutationExplainer (langsam, approximiert)
    """
    import shap

    if model_type in ("svm", "logistic"):
        # Null-Vektor als Hintergrund: SHAP-Wert = Koeffizient × Feature-Wert
        # (= Abweichung vom "kein Feature vorhanden"-Baseline)
        import scipy.sparse as sp
        background = sp.csr_matrix((1, X_transformed.shape[1]))
        masker = shap.maskers.Independent(background, max_samples=1)
        return shap.LinearExplainer(
            classifier, masker=masker, feature_perturbation="interventional"
        )
    elif model_type in ("rf", "xgboost"):
        return shap.TreeExplainer(classifier)
    elif model_type == "nn":
        # PermutationExplainer: Null-Vektor als dichtes Hintergrund-Array.
        # Sparse-Matrizen führen zu Shape-Fehlern weil SHAP intern squeeze() aufruft
        # (Background (1,N) → (N,)) und beim Aufruf dann (1,N) erwartet.
        n_features = X_transformed.shape[1]
        background = np.zeros((1, n_features))
        return shap.PermutationExplainer(
            classifier.predict_proba,
            background,
            max_evals=2 * n_features + 1,
        )
    else:
        raise ValueError(f"Kein SHAP-Explainer für Modelltyp '{model_type}' bekannt.")


def get_word_shap_scores(
    clf,
    X_pred_df,
    predicted_label: str,
    model_path: str = "",
    filter_stopwords: bool = True,
) -> dict:
    """
    Berechnet wort-level SHAP-Scores für eine einzelne Vorhersage.

    Args:
        clf: SachgruppenClassifier-Instanz (geladen)
        X_pred_df: DataFrame mit 'lemma' und/oder 'bedeutung' Spalten (1 Zeile)
        predicted_label: Vorhergesagte Sachgruppe als String (z.B. "6000")
        model_path: Pfad zur Modelldatei, für Explainer-Caching
        filter_stopwords: Wenn True, werden Stoppwörter aus stopwords_de.txt
                          aus der Anzeige herausgefiltert

    Returns:
        {"lemma": [(wort, score), ...], "bedeutung": [(wort, score), ...]}
        score ist normiert auf [-1, 1]; positiv = stärkt die Vorhersage
    """
    import shap

    # Stoppwörter laden (aus Cache oder Datei)
    stopwords = load_stopwords() if filter_stopwords else frozenset()

    # Pipeline-Schritte extrahieren
    # Wenn Stoppwort-Removal in Pipeline enthalten: vectorizer ist nicht der erste Schritt
    named_steps = clf.pipeline.named_steps
    vectorizer_step = named_steps["vectorizer"]
    classifier_step = named_steps["classifier"]

    # Eingabe in Feature-Raum transformieren (alle Schritte bis zum Classifier)
    # Preprocessing-Schritte in Pipeline-Reihenfolge anwenden
    X_preprocessed = X_pred_df
    for step_name in ("min_length_filter", "stopword_remover"):
        if step_name in named_steps:
            X_preprocessed = named_steps[step_name].transform(X_preprocessed)
    X_transformed = vectorizer_step.transform(X_preprocessed)

    # Explainer aus Cache laden oder neu erstellen
    cache_key = str(model_path) if model_path else str(id(clf))
    if cache_key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[cache_key] = _build_explainer(
            clf.model_type, classifier_step, X_transformed
        )
    explainer = _EXPLAINER_CACHE[cache_key]

    # SHAP-Werte berechnen
    # PermutationExplainer (NN) erwartet dichtes Array – sparse → dense konvertieren
    X_for_shap = X_transformed.toarray() if (
        clf.model_type == "nn" and hasattr(X_transformed, "toarray")
    ) else X_transformed
    shap_explanation = explainer(X_for_shap)
    shap_vals = shap_explanation.values  # (1, n_features) oder (1, n_features, n_classes)

    # Klassen-Index für die vorhergesagte Klasse ermitteln
    if clf.model_type in ("xgboost", "nn") and clf.label_encoder is not None:
        # XGBoost/NN nutzen Integer-Labels; classes_ = [0, 1, ..., n-1]
        pred_int = clf.label_encoder.transform([predicted_label])[0]
        pred_class_idx = int(pred_int)
    else:
        # SVM/Logistic: classes_ enthält die Original-String-Labels
        classes_list = list(classifier_step.classes_)
        pred_class_idx = classes_list.index(predicted_label)

    # SHAP-Werte für die vorhergesagte Klasse extrahieren
    if shap_vals.ndim == 3:
        # Multiclass: (1, n_features, n_classes) → (n_features,)
        flat_shap = np.array(shap_vals[0, :, pred_class_idx])
    else:
        # Binary oder lineare Ausgabe: (1, n_features) → (n_features,)
        flat_shap = np.array(shap_vals[0, :])

    # SHAP-Werte aufteilen: jeder Transformer (lemma, bedeutung) hat einen eigenen Slice
    # Originaltext für Stoppwort-Matching (nicht den vorverarbeiteten!)
    result = {}
    offset = 0
    for name, transformer, col in vectorizer_step.transformers_:
        if transformer == "drop" or not hasattr(transformer, "get_feature_names_out"):
            continue
        n_features = len(transformer.get_feature_names_out())
        slice_vals = flat_shap[offset:offset + n_features]

        # Originaltext (vor Stoppwort-Removal) für konsistente Wort-Anzeige
        text = str(X_pred_df[col].iloc[0]) if col in X_pred_df.columns else ""
        # Dispatch: word-level oder char-level Aggregation
        if getattr(transformer, "analyzer", None) == "word":
            result[name] = _aggregate_to_words_word_level(text, transformer, slice_vals, stopwords)
        else:
            result[name] = _aggregate_to_words(text, transformer, slice_vals, stopwords)
        offset += n_features

    # Sicherstellen, dass beide Schlüssel immer vorhanden sind
    result.setdefault("lemma", [])
    result.setdefault("bedeutung", [])

    # Globale Normalisierung auf [-1, 1] über BEIDE Felder zusammen,
    # damit Lemma- und Bedeutungs-Scores vergleichbar bleiben.
    all_scores = [s for pairs in result.values() for _, s in pairs]
    max_abs = max(abs(s) for s in all_scores) if all_scores else 1.0
    if max_abs == 0:
        max_abs = 1.0

    return {
        field: [(w, s / max_abs) for w, s in pairs]
        for field, pairs in result.items()
    }
