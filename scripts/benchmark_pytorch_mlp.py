#!/usr/bin/env python3
"""
Testet, ob eine klassengewichtete PyTorch-MLP das sklearn-NN schlaegt.

sklearns MLPClassifier unterstuetzt kein sample_weight/class_weight -- anders
als SVM/LogisticRegression/RandomForest (dort class_weight='balanced'). Das
ist eine echte Faehigkeitsluecke, kein Hyperparameter-Tuning-Ziel; siehe
.claude/memory/nn-retrain-mit-scaler-geplant.md. Dieses Skript prueft isoliert,
ob genau dieser eine Hebel (inverse-Frequenz-gewichteter CrossEntropyLoss,
gleiche Formel wie sklearns 'balanced') etwas bringt -- sonst nichts anderes.

Nutzt ein bereits trainiertes sklearn-NN-Pickle NUR zur Feature-Extraktion
(Vectorizer + MaxAbsScaler sind dort schon gefittet) -- spart das erneute
Fitten von TF-IDF/spaCy/Dornseiff und garantiert exakt dieselben Features wie
im sklearn-Vergleichslauf. Reproduziert den Split exakt wie
sachgruppen_classifier.py (random_state=42).

Eigenstaendiges Experiment-Skript: aendert nichts am Hauptcode und nichts an
den Projekt-Dependencies. PyTorch wird ephemer via `uv run --with torch`
geladen, landet nicht in pyproject.toml/uv.lock.

Beispiel:
    uv run --with torch python scripts/benchmark_pytorch_mlp.py \
        --model models/nn_char_wb_ml1_sw0_20260709_230256.pkl \
        --csv woerterbuch_daten_124217.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sachgruppen_classifier import SachgruppenClassifier  # noqa: E402


def load_split(csv_file: str, test_size: float, use_lemma: bool):
    """Reproduziert Datenaufbereitung + Split 1:1 wie
    sachgruppen_classifier.train_and_evaluate() (random_state=42)."""
    df = pd.read_csv(csv_file, sep=None, engine="python")
    df.columns = [c.lstrip('﻿').strip() for c in df.columns]

    required_cols = ['lemma', 'bedeutung', 'sachgruppe'] if use_lemma else ['bedeutung', 'sachgruppe']
    df_clean = df.dropna(subset=required_cols).copy()
    if use_lemma:
        df_clean['lemma'] = df_clean['lemma'].astype(str).replace('', 'LEER')
    df_clean['bedeutung'] = df_clean['bedeutung'].astype(str).replace('', 'LEER')

    x_cols = ['lemma', 'bedeutung'] if use_lemma else ['bedeutung']
    X = df_clean[x_cols]
    y = df_clean['sachgruppe'].astype(str)

    class_counts = y.value_counts()
    single_sample_classes = class_counts[class_counts == 1].index

    if len(single_sample_classes) > 0:
        mask_single = y.isin(single_sample_classes)
        X_single, y_single = X[mask_single], y[mask_single]
        X_multi, y_multi = X[~mask_single], y[~mask_single]
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi, y_multi, test_size=test_size, random_state=42, stratify=y_multi)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_multi, y_multi, test_size=test_size, random_state=42)
        X_train = pd.concat([X_train, X_single])
        y_train = pd.concat([y_train, y_single])
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def iterate_batches(X_sparse, y_arr, batch_size, shuffle, rng):
    n = X_sparse.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        xb = torch.from_numpy(X_sparse[batch_idx].toarray().astype(np.float32))
        yb = torch.from_numpy(y_arr[batch_idx].astype(np.int64))
        yield xb, yb


def train_mlp(X_train, y_train, X_val, y_val, n_classes, class_weights,
              hidden, lr, batch_size, max_epochs, patience, device, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = MLP(X_train.shape[1], hidden, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight_t = None
    if class_weights is not None:
        weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_t)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in iterate_batches(X_train, y_train, batch_size, True, rng):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in iterate_batches(X_val, y_val, 1024, False, rng):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        val_acc = correct / total
        print(f"  Epoch {epoch:2d}: val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping nach Epoche {epoch} (bestes val_acc={best_val_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model


def topk_eval(model, device, X_test, y_test_arr, ks=(1, 3, 5)):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, X_test.shape[0], 1024):
            xb = torch.from_numpy(X_test[start:start + 1024].toarray().astype(np.float32)).to(device)
            all_logits.append(model(xb).cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    order = np.argsort(logits, axis=1)[:, ::-1]
    return {k: (order[:, :k] == y_test_arr[:, None]).any(axis=1).mean() for k in ks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='models/nn_char_wb_ml1_sw0_20260709_230256.pkl',
                     help='sklearn-NN-Pickle, dient nur der Feature-Extraktion (Vectorizer+Scaler)')
    ap.add_argument('--csv', default='data/woerterbuch_daten_124217.csv')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--hidden', type=int, default=100, help='Hidden-Layer-Groesse (wie sklearn-Basis: 100)')
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLade Feature-Pipeline aus {args.model} (nur Vectorizer+Scaler, kein Classifier) ...")
    clf = SachgruppenClassifier.load(args.model)
    feature_pipe = clf.pipeline[:-1]  # alles ausser dem letzten Schritt ('classifier')

    print(f"Lade + splitte {args.csv} (random_state=42, wie sachgruppen_classifier.py) ...")
    X_train_df, X_test_df, y_train_s, y_test_s = load_split(args.csv, args.test_size, clf.use_lemma)
    print(f"  Train: {len(X_train_df)}  Test: {len(X_test_df)}")

    le = LabelEncoder()
    le.fit(pd.concat([y_train_s, y_test_s]))
    y_train = le.transform(y_train_s)
    y_test = le.transform(y_test_s)
    n_classes = len(le.classes_)

    print("Transformiere Features (nutzt die bereits gefitteten Schritte, kein Refit) ...")
    t0 = time.time()
    X_train = sparse.csr_matrix(feature_pipe.transform(X_train_df))
    X_test = sparse.csr_matrix(feature_pipe.transform(X_test_df))
    print(f"  {X_train.shape[1]} Dimensionen, {n_classes} Klassen, {time.time() - t0:.1f}s")

    # Validation-Split aus den Trainingsdaten (10 %, stratifiziert -- wie sklearns
    # MLPClassifier(early_stopping=True, validation_fraction=0.1) intern verfaehrt).
    idx_all = np.arange(X_train.shape[0])
    try:
        tr_idx, val_idx = train_test_split(idx_all, test_size=0.1, random_state=42, stratify=y_train)
    except ValueError:
        tr_idx, val_idx = train_test_split(idx_all, test_size=0.1, random_state=42)
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # sklearns class_weight='balanced'-Formel: n_samples / (n_classes * bincount)
    balanced_weights = X_tr.shape[0] / (n_classes * np.bincount(y_tr, minlength=n_classes).clip(min=1))

    print(f"\nVergleichswert: sklearn-Basis siehe {args.model.replace('.pkl', '_metadata.json')}")

    for name, weights in [
        ("unweighted (Sanity-Check ggue. sklearn-Basis)", None),
        ("class-weighted (balanced, wie SVM/LogReg/RF)", balanced_weights),
    ]:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        t0 = time.time()
        model = train_mlp(
            X_tr, y_tr, X_val, y_val, n_classes, weights,
            hidden=args.hidden, lr=args.lr, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=args.patience, device=device,
        )
        scores = topk_eval(model, device, X_test, y_test)
        print(f"  Trainingszeit: {time.time() - t0:.1f}s")
        for k, v in scores.items():
            print(f"  Top-{k}: {v:.4f}")


if __name__ == '__main__':
    main()
