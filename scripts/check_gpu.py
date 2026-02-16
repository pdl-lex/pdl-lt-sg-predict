#!/usr/bin/env python3
"""
GPU-Verfügbarkeit und XGBoost GPU-Support prüfen
"""

import sys
import subprocess

def check_rocm():
    """Prüft ob ROCm installiert ist."""
    print("🔍 ROCm Installation...")
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ROCm ist installiert")
            print(result.stdout)
            return True
        else:
            print("✗ ROCm nicht gefunden oder nicht funktionsfähig")
            return False
    except FileNotFoundError:
        print("✗ ROCm nicht installiert (rocm-smi nicht gefunden)")
        print("\nInstallations-Anleitung: siehe GPU_SETUP.md")
        return False

def check_gpu_devices():
    """Prüft verfügbare GPU-Devices."""
    print("\n🔍 GPU-Geräte...")
    try:
        result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
        amd_gpus = [line for line in result.stdout.split('\n') if 'AMD' in line and 'Display' in line]
        if amd_gpus:
            print(f"✓ {len(amd_gpus)} AMD GPU(s) gefunden:")
            for gpu in amd_gpus:
                print(f"  - {gpu}")
            return True
        else:
            print("✗ Keine AMD GPU gefunden")
            return False
    except Exception as e:
        print(f"✗ Fehler beim Prüfen: {e}")
        return False

def check_xgboost():
    """Prüft XGBoost GPU-Support."""
    print("\n🔍 XGBoost GPU-Support...")
    try:
        import xgboost as xgb
        import warnings
        print(f"✓ XGBoost Version: {xgb.__version__}")

        # Versuche GPU-Training
        import numpy as np
        print("\n  Teste GPU-Training...")

        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        dtrain = xgb.DMatrix(X, label=y)

        params = {
            'device': 'cuda',
            'tree_method': 'hist',
            'verbosity': 1  # Aktiviere Warnungen
        }

        # Fange Warnungen ab
        gpu_fallback = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                model = xgb.train(params, dtrain, num_boost_round=5)

                # Prüfe ob GPU→CPU Fallback stattgefunden hat
                for warning in w:
                    if "No visible GPU" in str(warning.message) or "changed from GPU to CPU" in str(warning.message):
                        gpu_fallback = True
                        break

                if gpu_fallback:
                    print("  ⚠ GPU-Training fällt auf CPU zurück!")
                    print("  → XGBoost PyPI-Paket unterstützt nur NVIDIA GPUs (CUDA)")
                    print("  → Für AMD GPUs: XGBoost mit HIP-Support kompilieren")
                    print("  → Siehe XGBOOST_ROCM_BUILD.md für Details")
                    print("  → Alternative: LightGBM hat besseren ROCm-Support")
                    return False
                else:
                    print("  ✓ GPU-Training funktioniert!")
                    return True

            except Exception as e:
                print(f"  ✗ GPU-Training fehlgeschlagen: {e}")
                print("  → XGBoost wurde ohne GPU-Support kompiliert")
                return False

    except ImportError:
        print("✗ XGBoost nicht installiert")
        print("  Installation: pip install xgboost")
        return False

def check_permissions():
    """Prüft GPU-Berechtigungen."""
    print("\n🔍 GPU-Berechtigungen...")
    try:
        import os
        import grp

        # Prüfe Gruppen-Mitgliedschaft
        groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]

        has_render = 'render' in groups
        has_video = 'video' in groups

        if has_render and has_video:
            print("✓ Benutzer ist in 'render' und 'video' Gruppen")
            return True
        else:
            print(f"✗ Fehlende Gruppen-Mitgliedschaften:")
            if not has_render:
                print("  - 'render' Gruppe fehlt")
            if not has_video:
                print("  - 'video' Gruppe fehlt")
            print("\n  Fix:")
            print("  sudo usermod -a -G render $USER")
            print("  sudo usermod -a -G video $USER")
            print("  # Dann neu anmelden oder System neu starten")
            return False

    except Exception as e:
        print(f"⚠ Warnung: Konnte Berechtigungen nicht prüfen: {e}")
        return None

def main():
    print("="*60)
    print("GPU-SUPPORT CHECK")
    print("="*60)

    results = {
        'GPU Hardware': check_gpu_devices(),
        'ROCm': check_rocm(),
        'Berechtigungen': check_permissions(),
        'XGBoost GPU': check_xgboost()
    }

    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)

    for check, status in results.items():
        if status is True:
            symbol = "✓"
        elif status is False:
            symbol = "✗"
        else:
            symbol = "⚠"
        print(f"{symbol} {check}")

    print()

    if all(v is True for v in results.values()):
        print("🎉 Alles bereit für GPU-beschleunigtes Training!")
        print("\nVerwendung:")
        print("  python sachgruppen_classifier.py --csv data.csv --model xgboost --gpu")
    else:
        print("⚠ Setup noch nicht vollständig.")
        print("   Siehe GPU_SETUP.md für Details.")

    return 0 if all(v is not False for v in results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
