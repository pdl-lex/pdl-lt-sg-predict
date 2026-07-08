"""LexoTerm Tools — Sachgruppen-Vorhersage (Backend-Paket).

Enthält die reine Fachlogik (``core``) und die FastAPI-Schicht (``api``).
Die eigentliche ML-Pipeline liegt weiterhin im Projekt-Root
(``sachgruppen_classifier.py`` / ``shap_utils.py``), damit bestehende, gepickelte
Modelle unverändert geladen werden können.
"""

__version__ = "0.1.0"
