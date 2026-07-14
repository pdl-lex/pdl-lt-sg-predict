"""
SHAP-based explanation for SachgruppenClassifier.

Aggregates char-n-gram SHAP values to word level so the user can see
which input words influenced the prediction and in which direction.
"""

import re
import unicodedata
import numpy as np
from pathlib import Path
from typing import Any

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Module-level explainer cache: keyed by model_path (or id(clf) as fallback).
# Capped: each explainer holds a reference to its classifier, so an unbounded
# cache would keep evicted models (100-330 MB each) alive in memory.
_EXPLAINER_CACHE: dict[str, Any] = {}
_EXPLAINER_CACHE_MAX = 2

# Module-level stopwords cache
_STOPWORDS_CACHE: frozenset | None = None
_STOPWORDS_PATH = Path(__file__).parent / "stopwords_de.txt"


def load_stopwords(path: str | Path | None = None) -> frozenset[str]:
    """
    Load stopwords from file (case-insensitive).

    Format: one word per line; lines starting with # are ignored.
    Falls back to empty set if file not found.
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
    """Reproduces scikit-learn's strip_accents='unicode' logic."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def char_wb_ngrams(word: str, min_n: int, max_n: int) -> set[str]:
    """
    Reproduces scikit-learn's char_wb tokenization exactly.

    char_wb: word is padded with spaces, then n-grams are extracted.
    Example: "Kind" → " Kind " → {" K", "Ki", "in", "nd", "d ", " Ki", ...}
    """
    padded = " " + word + " "
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
    Aggregate word-level TF-IDF SHAP values to word level.

    For unigrams: SHAP value of the word directly.
    For bigrams (ngram_range=(1,2)): SHAP values of all bigrams containing
    the word are added to the word's score.

    Returns:
        List of (original_word, raw_score) pairs (NOT yet normalized)
    """
    feature_names = vectorizer.get_feature_names_out()
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    original_words = text.split()
    if not original_words:
        return []

    # Lowercase for feature lookup
    normalized_words = [w.lower() for w in original_words]

    result = []
    for i, (orig_word, norm_word) in enumerate(zip(original_words, normalized_words)):
        clean_orig = _PUNCT_RE.sub("", orig_word)
        clean_norm = _PUNCT_RE.sub("", norm_word)
        if not clean_norm:
            continue
        if clean_norm in stopwords:
            continue
        # Unigram + adjacent bigrams
        candidates = [clean_norm]
        if i > 0:
            candidates.append(f"{normalized_words[i - 1]} {norm_word}")
        if i < len(normalized_words) - 1:
            candidates.append(f"{norm_word} {normalized_words[i + 1]}")
        scores = [
            float(shap_vals_slice[feature_index[c]])
            for c in candidates
            if c in feature_index
        ]
        result.append((clean_orig, sum(scores)))

    return result


def _aggregate_to_words(
    text: str,
    vectorizer,
    shap_vals_slice: np.ndarray,
    stopwords: frozenset[str] = frozenset(),
) -> list[tuple[str, float]]:
    """
    Aggregate char-n-gram SHAP values to word level.

    For each word: sum the SHAP values of all n-grams originating from that word.
    Sum instead of mean so that longer content words (more n-grams, more specific
    signal) are weighted more strongly than short stopwords.

    Stopwords are skipped entirely when `stopwords` is non-empty.

    Note: normalization is NOT done here but globally across all fields in
    get_word_shap_scores(), so lemma and bedeutung scores remain comparable.

    Returns:
        List of (word, raw_score) pairs (NOT yet normalized)
    """
    min_n, max_n = vectorizer.ngram_range
    feature_names = vectorizer.get_feature_names_out()
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    words = text.split()
    if not words:
        return []

    result = []
    for word in words:
        clean_word = _PUNCT_RE.sub("", word)
        if not clean_word:
            continue
        if clean_word.lower() in stopwords:
            continue  # skip stopword
        ngrams = char_wb_ngrams(clean_word, min_n, max_n)
        scores = [
            float(shap_vals_slice[feature_index[ng]])
            for ng in ngrams
            if ng in feature_index
        ]
        # Sum: longer, more specific words accumulate more signal
        result.append((clean_word, sum(scores)))

    return result


def _build_explainer(model_type: str, classifier, X_transformed):
    """
    Create the appropriate SHAP explainer for the model type.

    - SVM / Logistic: LinearExplainer (fast, exact)
    - RF / XGBoost: TreeExplainer (fast, exact)
    - NN (MLP): PermutationExplainer (slow, approximate)
    """
    import shap

    if model_type in ("svm", "logistic"):
        # Zero vector as background: SHAP value = coefficient × feature value
        # (= deviation from "no feature present" baseline)
        import scipy.sparse as sp
        background = sp.csr_matrix((1, X_transformed.shape[1]))
        masker = shap.maskers.Independent(background, max_samples=1)
        return shap.LinearExplainer(
            classifier, masker=masker, feature_perturbation="interventional"
        )
    elif model_type in ("rf", "xgboost"):
        return shap.TreeExplainer(classifier)
    elif model_type == "nn":
        # PermutationExplainer: zero vector as dense background array.
        # Sparse matrices cause shape errors because SHAP internally calls squeeze()
        # (background (1,N) → (N,)) but then expects (1,N) on invocation.
        n_features = X_transformed.shape[1]
        background = np.zeros((1, n_features))
        return shap.PermutationExplainer(
            classifier.predict_proba,
            background,
            max_evals=2 * n_features + 1,
        )
    else:
        raise ValueError(f"No SHAP explainer known for model type '{model_type}'.")


def get_word_shap_scores(
    clf,
    X_pred_df,
    predicted_label: str,
    model_path: str = "",
    filter_stopwords: bool = True,
) -> dict:
    """
    Compute word-level SHAP scores for a single prediction.

    Args:
        clf: SachgruppenClassifier instance (loaded)
        X_pred_df: DataFrame with 'lemma' and/or 'bedeutung' columns (1 row)
        predicted_label: Predicted Sachgruppe as string (e.g. "6000")
        model_path: Path to model file for explainer caching
        filter_stopwords: If True, stopwords from stopwords_de.txt are hidden

    Returns:
        {"lemma": [(word, score), ...], "bedeutung": [(word, score), ...]}
        score normalized to [-1, 1]; positive = supports the prediction
    """
    import shap

    # Load stopwords (from cache or file)
    stopwords = load_stopwords() if filter_stopwords else frozenset()

    # Extract pipeline steps
    named_steps = clf.pipeline.named_steps
    if "svd" in named_steps:
        raise ValueError(
            "SHAP explanations are not available when TruncatedSVD is enabled (use_svd=True)."
        )
    vectorizer_step = named_steps["vectorizer"]
    classifier_step = named_steps["classifier"]

    # Transform input to feature space (all steps before the classifier)
    X_preprocessed = X_pred_df
    for step_name in ("punctuation_stripper", "min_length_filter", "stopword_remover"):
        if step_name in named_steps:
            X_preprocessed = named_steps[step_name].transform(X_preprocessed)
    X_transformed = vectorizer_step.transform(X_preprocessed)
    # Apply any steps between vectorizer and classifier (e.g. the MaxAbsScaler
    # in NN pipelines): the classifier expects the feature scale it was trained
    # on. Element-wise scaling keeps feature count and order unchanged, so the
    # per-transformer n-gram slicing below stays valid.
    step_names = [name for name, _ in clf.pipeline.steps]
    for step_name in step_names[step_names.index("vectorizer") + 1:-1]:
        X_transformed = named_steps[step_name].transform(X_transformed)

    # Load explainer from cache or build a new one
    cache_key = str(model_path) if model_path else str(id(clf))
    if cache_key not in _EXPLAINER_CACHE:
        while len(_EXPLAINER_CACHE) >= _EXPLAINER_CACHE_MAX:
            _EXPLAINER_CACHE.pop(next(iter(_EXPLAINER_CACHE)))
        _EXPLAINER_CACHE[cache_key] = _build_explainer(
            clf.model_type, classifier_step, X_transformed
        )
    explainer = _EXPLAINER_CACHE[cache_key]

    # Compute SHAP values
    # NN and RF/XGBoost need a dense float array; SVM/Logistic can use sparse
    if clf.model_type in ("nn", "rf", "xgboost") and hasattr(X_transformed, "toarray"):
        X_for_shap = X_transformed.toarray().astype(float)
    else:
        X_for_shap = X_transformed
    shap_explanation = explainer(X_for_shap)
    shap_vals = shap_explanation.values  # (1, n_features) or (1, n_features, n_classes)

    # Determine class index for the predicted class
    if clf.model_type in ("xgboost", "nn") and clf.label_encoder is not None:
        # XGBoost/NN use integer labels; classes_ = [0, 1, ..., n-1]
        pred_int = clf.label_encoder.transform([predicted_label])[0]
        pred_class_idx = int(pred_int)
    else:
        # SVM/Logistic: classes_ contains original string labels
        classes_list = list(classifier_step.classes_)
        pred_class_idx = classes_list.index(predicted_label)

    # Extract SHAP values for the predicted class
    if shap_vals.ndim == 3:
        # Multiclass: (1, n_features, n_classes) → (n_features,)
        flat_shap = np.array(shap_vals[0, :, pred_class_idx])
    else:
        # Binary or linear output: (1, n_features) → (n_features,)
        flat_shap = np.array(shap_vals[0, :])

    # Split SHAP values: each transformer (lemma, bedeutung) has its own slice
    result = {}
    offset = 0
    for name, transformer, col in vectorizer_step.transformers_:
        if transformer == "drop" or not hasattr(transformer, "get_feature_names_out"):
            continue
        n_features = len(transformer.get_feature_names_out())
        slice_vals = flat_shap[offset:offset + n_features]

        # Preprocessed text (after punctuation stripping) for consistent word display
        text = str(X_preprocessed[col].iloc[0]) if col in X_preprocessed.columns else ""
        # Dispatch: word-level or char-level aggregation
        if getattr(transformer, "analyzer", None) == "word":
            result[name] = _aggregate_to_words_word_level(text, transformer, slice_vals, stopwords)
        else:
            result[name] = _aggregate_to_words(text, transformer, slice_vals, stopwords)
        offset += n_features

    # Merge the word-level bedeutung branch into the main key.
    # char_wb and word process the same text → add scores positionally,
    # do not concatenate (which would show each word twice).
    if "bedeutung_word" in result:
        char_pairs = result.get("bedeutung", [])
        word_pairs = result.pop("bedeutung_word")
        if len(char_pairs) == len(word_pairs):
            result["bedeutung"] = [
                (w, sc + sw)
                for (w, sc), (_, sw) in zip(char_pairs, word_pairs)
            ]
        else:
            # Lengths differ (should not happen): merge by word text
            from collections import defaultdict
            scores_by_word: dict[str, list[float]] = defaultdict(list)
            order: list[str] = []
            for w, s in char_pairs + word_pairs:
                if w not in scores_by_word:
                    order.append(w)
                scores_by_word[w].append(s)
            result["bedeutung"] = [(w, sum(scores_by_word[w])) for w in order]
    # Ensure both keys are always present
    result.setdefault("lemma", [])
    result.setdefault("bedeutung", [])

    # Global normalization to [-1, 1] across BOTH fields so lemma and
    # bedeutung scores remain comparable.
    all_scores = [s for pairs in result.values() for _, s in pairs]
    max_abs = max(abs(s) for s in all_scores) if all_scores else 1.0
    if max_abs == 0:
        max_abs = 1.0

    return {
        field: [(w, s / max_abs) for w, s in pairs]
        for field, pairs in result.items()
    }
