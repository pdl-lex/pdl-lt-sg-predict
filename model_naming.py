"""Adjektiv-Frucht-Namensschema fuer trainierte Modelle.

Modelldateien hiessen bisher z. B. ``nn_char_wb_ml1_sw0_20260709_230256.pkl`` —
mit ueber 20 moeglichen Trainingsparametern laesst sich das ohnehin nicht mehr
sinnvoll in einem Dateinamen unterbringen (die vollstaendige Konfiguration steht
schon in der ``_metadata.json``-Sidecar-Datei, siehe ``sachgruppen_classifier.py``).
Neue Namen bestehen stattdessen nur noch aus Modelltyp + Adjektiv + Frucht,
z. B. ``nn_sunny_orange.pkl`` oder ``svm_blue_banana.pkl``.

Jede Adjektiv-Frucht-Kombination wird nur einmal vergeben — auch dann, wenn das
zugehoerige Modell spaeter geloescht wird. Dazu wird bei jeder Vergabe eine
Registry-Datei (``used_model_names.json``) im Modellverzeichnis gepflegt.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Nur positive, harmlose Adjektive.
ADJECTIVES = [
    "sunny", "happy", "gentle", "bright", "calm", "brave", "kind", "swift",
    "lucky", "jolly", "breezy", "cheerful", "clever", "cozy", "dazzling",
    "eager", "fuzzy", "graceful", "humble", "lively",
]

FRUITS = [
    "orange", "banana", "mango", "kiwi", "papaya", "lemon", "cherry", "peach",
    "plum", "grape", "melon", "lychee", "guava", "apricot", "fig", "pear",
    "lime", "coconut", "pineapple", "blueberry",
]

REGISTRY_FILENAME = "used_model_names.json"


def _registry_path(models_dir: Path) -> Path:
    return models_dir / REGISTRY_FILENAME


def _load_used_combos(models_dir: Path) -> set[str]:
    """Bereits vergebene Adjektiv-Frucht-Kombinationen (Registry + vorhandene Dateien)."""
    used: set[str] = set()

    reg = _registry_path(models_dir)
    if reg.exists():
        try:
            used.update(json.loads(reg.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass

    # Zusaetzlich vorhandene .pkl-Dateien scannen, falls die Registry fehlt oder
    # veraltet ist (z. B. nach manuellem Kopieren von Modellen). Nur echte
    # Adjektiv-Frucht-Endungen zaehlen, damit Reste alter Namensschemata
    # (Parameter/Zeitstempel) die Registry nicht verunreinigen.
    if models_dir.exists():
        for pkl in models_dir.glob("*.pkl"):
            parts = pkl.stem.split("_")
            if len(parts) >= 3 and parts[-2] in ADJECTIVES and parts[-1] in FRUITS:
                used.add("_".join(parts[-2:]))

    return used


def generate_model_name(model_type: str, models_dir: Path) -> str:
    """Neuen eindeutigen Modellnamen erzeugen, z. B. 'nn_sunny_orange'.

    Registriert die vergebene Kombination sofort in ``used_model_names.json``,
    damit sie nie wieder vergeben wird — auch nicht nach dem Loeschen des
    Modells.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    used = _load_used_combos(models_dir)
    available = [
        (adj, fruit)
        for adj in ADJECTIVES
        for fruit in FRUITS
        if f"{adj}_{fruit}" not in used
    ]
    if not available:
        raise RuntimeError(
            f"Alle {len(ADJECTIVES)}x{len(FRUITS)}={len(ADJECTIVES) * len(FRUITS)} "
            "Adjektiv-Frucht-Kombinationen sind bereits vergeben. Bitte die Listen "
            "in model_naming.py erweitern."
        )

    adj, fruit = random.choice(available)
    combo = f"{adj}_{fruit}"

    reg = _registry_path(models_dir)
    all_used = sorted(used | {combo})
    reg.write_text(json.dumps(all_used, indent=2, ensure_ascii=False), encoding="utf-8")

    return f"{model_type}_{combo}"
