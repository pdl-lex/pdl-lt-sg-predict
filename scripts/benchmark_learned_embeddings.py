#!/usr/bin/env python3
"""
Experiment 3 der Verfahrensfrage: GELERNTE EMBEDDINGS statt TF-IDF.

Test 1 variiert die Gewichtung innerhalb von TF-IDF, Test 2 die Tokenisierung
(Subword statt Zeichenfenster) -- beide bleiben bei "eine Spalte pro Merkmal,
Gewicht per idf". Dieses Skript verlaesst das Verfahren ganz: jede Subword-
Einheit bekommt einen dichten Vektor, der AM KLASSIFIKATIONSZIEL mitgelernt
wird. Damit kann das Modell Einheiten als aehnlich behandeln, die in TF-IDF
voellig unabhaengige Spalten waeren.

Architektur (fastText-artig, bewusst klein gehalten):

    lemma     --tok--> ids --EmbeddingBag(mean)--> [dim] --.
                                                           +--> [2*dim] --MLP--> 428
    bedeutung --tok--> ids --EmbeddingBag(mean)--> [dim] --'

TOKENISIERUNG (--tokenizer):
  charngram (Default)  dieselben char_wb-n-Gramme wie die TF-IDF-Baseline
  subword              SentencePiece

Der Default ist charngram, damit gegenueber der Baseline NUR die Kodierung
variiert (fest+duennbesetzt -> gelernt+dicht). Test 2 hat gezeigt, dass
Subword auf diesem Dialektkorpus fuer sich genommen 0,6-2,0 pp kostet, weil
Schreibvarianten das Wort neu segmentieren; mit --tokenizer subword wuerde
dieses Skript diesen Nachteil erben und zwei Aenderungen vermengen.

Getrennte Vokabulare und Embedding-Tabellen je Feld: lemma ist bairischer
Dialekt, bedeutung hochdeutsch -- zwei verschiedene Varietaeten, die sich kein
Vokabular teilen sollten. Das spiegelt die getrennten Vektorisierer der
TF-IDF-Pipeline.

Der wichtigste Unterschied zur Baseline ist NICHT die Modellklasse, sondern
die Repraesentation: TF-IDF ist eine feste, ungelernte Kodierung, hier wird
sie mitgelernt. Deshalb ist der Vergleichswert das SVM/NN auf TF-IDF, und
class_weight wird zugeschaltet, damit der Unterschied nicht bloss an der
Klassenungleichverteilung haengt (sklearns MLPClassifier kann das nicht --
siehe benchmark_pytorch_mlp.py).

Das Vokabular (beide Varianten) wird NUR auf dem Trainingssplit gelernt; der
Testsplit beeinflusst es nicht.

Eigenstaendiges Experiment-Skript: aendert nichts am Hauptcode und nichts an
den Projekt-Dependencies. torch/sentencepiece werden ephemer geladen.

Beispiel:
    uv run --with torch --with sentencepiece python \
        scripts/benchmark_learned_embeddings.py \
        --csv data/woerterbuch_daten_124217.csv
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASELINE_SVM = 0.8346954386247434   # svm, char_wb, ohne Add-ons/Kalibrierung
BASELINE_NN = 0.837514              # nn_lively_kiwi, mit Add-ons + Scaler


def load_split(csv_file: str, test_size: float, random_state: int):
    """Reproduziert Datenaufbereitung + Split 1:1 wie
    sachgruppen_classifier.train_and_evaluate()."""
    df = pd.read_csv(csv_file, sep=None, engine="python")
    df.columns = [c.lstrip('﻿').strip() for c in df.columns]

    required = ['lemma', 'bedeutung', 'sachgruppe']
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Fehlende Spalten: {missing}; vorhanden: {list(df.columns)}")

    df = df.dropna(subset=required)
    for c in required:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df['bedeutung'] != '') & (df['sachgruppe'] != '')]

    y = df['sachgruppe'].values
    # stratify nur, wenn jede Klasse >=2 Vertreter hat (wie im Hauptcode)
    counts = pd.Series(y).value_counts()
    strat = y if counts.min() >= 2 else None
    return train_test_split(
        df['lemma'].values, df['bedeutung'].values, y,
        test_size=test_size, random_state=random_state, stratify=strat,
    )


def train_sp(texts, vocab_size: int) -> bytes:
    """SentencePiece-Unigram-Modell auf den uebergebenen Texten."""
    import sentencepiece as spm
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(str(t) for t in texts),
        model_writer=buf,
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=1.0,   # deutsches Alphabet inkl. Umlaute ist klein
        bos_id=-1, eos_id=-1,
        minloglevel=2,
    )
    return buf.getvalue()


def build_charngram(texts, ngram_range, max_features):
    """Zeichen-n-Gramm-Vokabular wie in der TF-IDF-Pipeline.

    Wird gebraucht, weil Test 2 gezeigt hat, dass Subword-Tokenisierung auf
    diesem Dialektkorpus sproede gegen Schreibvarianten ist ('Bauernhaus' ->
    ['_Bauern','haus'] aber 'Baurnhaus' -> ['_Bau','r','n','haus']). Wuerde
    Test 3 ebenfalls Subword nutzen, vermengte er zwei Aenderungen: gelernte
    Repraesentation UND schlechtere Tokenisierung. Mit Zeichen-n-Grammen ist
    die Tokenisierung identisch zur Baseline und nur die Kodierung variiert.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer='char_wb', ngram_range=ngram_range,
                         max_features=max_features, min_df=2)
    cv.fit([str(t) for t in texts])
    return cv


def encode_charngram(cv, texts):
    """-> Liste von id-Tensoren ueber das Zeichen-n-Gramm-Vokabular."""
    analyze = cv.build_analyzer()
    vocab = cv.vocabulary_
    out = []
    for t in texts:
        ids = [vocab[g] for g in analyze(str(t)) if g in vocab]
        if not ids:
            ids = [0]
        out.append(torch.tensor(ids, dtype=torch.long))
    return out


def encode(sp, texts):
    """-> Liste von id-Tensoren, einer je Dokument.

    Pro Dokument statt einem flachen Array, weil die Batches beliebige Zeilen
    auswaehlen: so kostet das Zusammensetzen eines Batches nur ein torch.cat
    ueber fertige Tensoren statt Index-Arithmetik pro Zeile.
    """
    out = []
    for t in texts:
        ids = sp.encode(str(t), out_type=int)
        if not ids:            # leerer Text -> ein Padding-Token, sonst
            ids = [0]          # bekaeme EmbeddingBag ein leeres Segment
        out.append(torch.tensor(ids, dtype=torch.long))
    return out


class FieldEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_l, vocab_b, dim, hidden, n_classes, dropout):
        super().__init__()
        self.emb_l = nn.EmbeddingBag(vocab_l, dim, mode='mean')
        self.emb_b = nn.EmbeddingBag(vocab_b, dim, mode='mean')
        self.head = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, l_ids, l_off, b_ids, b_off):
        return self.head(torch.cat([self.emb_l(l_ids, l_off),
                                    self.emb_b(b_ids, b_off)], dim=1))


def batches(n, batch_size, shuffle, generator=None):
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def make_bag(docs, sel):
    """Setzt aus den Dokument-Tensoren die (flat_ids, offsets) eines Batches."""
    parts = [docs[j] for j in sel.tolist()]
    lengths = torch.tensor([p.numel() for p in parts], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    return torch.cat(parts), offsets


def topk_accuracy(logits, y_true, ks=(1, 3, 5)):
    out = {}
    maxk = max(ks)
    top = logits.topk(maxk, dim=1).indices
    for k in ks:
        hit = (top[:, :k] == y_true.unsqueeze(1)).any(dim=1).float().mean().item()
        out[k] = hit
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--tokenizer', choices=['charngram', 'subword'], default='charngram',
                    help='charngram: dieselbe Zerlegung wie die TF-IDF-Baseline, so dass '
                         'NUR die Kodierung variiert (Default). subword: SentencePiece '
                         '-- vergleichbar mit Test 2, aber mit gelernten Embeddings')
    ap.add_argument('--vocab-lemma', type=int, default=8000)
    ap.add_argument('--vocab-bedeutung', type=int, default=16000)
    ap.add_argument('--embed-dim', type=int, default=300)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--patience', type=int, default=4,
                    help='Early-Stopping-Geduld in Epochen (Val-Accuracy)')
    ap.add_argument('--val-size', type=float, default=0.1,
                    help='Anteil des TRAININGSsplits als Validierung fuers Early Stopping')
    ap.add_argument('--weight-decay', type=float, default=0.0,
                    help='L2-Regularisierung im Adam-Optimierer')
    ap.add_argument('--no-class-weight', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    print("Lade Daten …")
    l_tr, l_te, b_tr, b_te, y_tr, y_te = load_split(
        args.csv, args.test_size, args.random_state)
    print(f"  Train {len(y_tr)}  Test {len(y_te)}")

    le = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
    y_tr_i = torch.tensor(le.transform(y_tr), dtype=torch.long)
    y_te_i = torch.tensor(le.transform(y_te), dtype=torch.long)
    n_classes = len(le.classes_)
    print(f"  Klassen {n_classes}")

    # Tokenisierung: in beiden Faellen NUR auf dem Trainingssplit gelernt
    if args.tokenizer == 'subword':
        print("Trainiere SentencePiece (nur auf dem Trainingssplit) …")
        import sentencepiece as spm
        sp_l = spm.SentencePieceProcessor(model_proto=train_sp(l_tr, args.vocab_lemma))
        sp_b = spm.SentencePieceProcessor(model_proto=train_sp(b_tr, args.vocab_bedeutung))
        n_vocab_l, n_vocab_b = sp_l.get_piece_size(), sp_b.get_piece_size()
        docs_l_tr, docs_b_tr = encode(sp_l, l_tr), encode(sp_b, b_tr)
        docs_l_te, docs_b_te = encode(sp_l, l_te), encode(sp_b, b_te)
    else:
        print("Baue Zeichen-n-Gramm-Vokabular (nur auf dem Trainingssplit) …")
        cv_l = build_charngram(l_tr, (2, 5), args.vocab_lemma)
        cv_b = build_charngram(b_tr, (2, 4), args.vocab_bedeutung)
        n_vocab_l, n_vocab_b = len(cv_l.vocabulary_), len(cv_b.vocabulary_)
        docs_l_tr, docs_b_tr = encode_charngram(cv_l, l_tr), encode_charngram(cv_b, b_tr)
        docs_l_te, docs_b_te = encode_charngram(cv_l, l_te), encode_charngram(cv_b, b_te)
    print(f"  lemma vocab {n_vocab_l}  bedeutung vocab {n_vocab_b}")
    print(f"  Tokens/Zeile: lemma {sum(len(d) for d in docs_l_tr)/len(docs_l_tr):.1f}  "
          f"bedeutung {sum(len(d) for d in docs_b_tr)/len(docs_b_tr):.1f}")

    # Validierungssplit aus dem TRAININGSsplit -- der Testsplit bleibt unberuehrt
    n_tr = len(y_tr_i)
    rng = np.random.RandomState(args.random_state)
    perm = rng.permutation(n_tr)
    n_val = int(n_tr * args.val_size)
    val_idx = torch.tensor(perm[:n_val], dtype=torch.long)
    fit_idx = torch.tensor(perm[n_val:], dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = FieldEmbeddingClassifier(
        n_vocab_l, n_vocab_b,
        args.embed_dim, args.hidden, n_classes, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter: {n_params:,}")

    if args.no_class_weight:
        weight = None
    else:
        # identische Formel wie sklearns class_weight='balanced'
        counts = np.bincount(y_tr_i.numpy(), minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weight = torch.tensor(len(y_tr_i) / (n_classes * counts),
                              dtype=torch.float32, device=device)
    lossfn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    def forward_batch(docs_l, docs_b, sel):
        li, lo = make_bag(docs_l, sel)
        bi, bo = make_bag(docs_b, sel)
        return model(li.to(device), lo.to(device), bi.to(device), bo.to(device))

    def forward_idx(sel):
        return forward_batch(docs_l_tr, docs_b_tr, sel)

    best_val, best_state, bad = -1.0, None, 0
    gen = torch.Generator().manual_seed(args.random_state)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for sel_pos in batches(len(fit_idx), args.batch_size, True, gen):
            sel = fit_idx[sel_pos]
            opt.zero_grad()
            loss = lossfn(forward_idx(sel), y_tr_i[sel].to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)

        model.eval()
        with torch.no_grad():
            logits = torch.cat([forward_idx(val_idx[s]).cpu()
                                for s in batches(len(val_idx), 1024, False)])
        val_acc = (logits.argmax(1) == y_tr_i[val_idx]).float().mean().item()
        print(f"  Epoche {epoch:2d}  loss {tot/len(fit_idx):.4f}  "
              f"val_acc {val_acc:.4f}  ({time.time()-t0:.0f}s)")

        if val_acc > best_val:
            best_val, bad = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  Early Stopping nach Epoche {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        te_logits = torch.cat([forward_batch(docs_l_te, docs_b_te, sel).cpu()
                               for sel in batches(len(y_te_i), 1024, False)])

    acc = topk_accuracy(te_logits, y_te_i)
    elapsed = time.time() - t0

    print("\n" + "=" * 62)
    print("GELERNTE EMBEDDINGS STATT TF-IDF")
    print("=" * 62)
    print(f"  tokenizer {args.tokenizer}  vocab {n_vocab_l}/{n_vocab_b}  dim {args.embed_dim}  "
          f"hidden {args.hidden}  dropout {args.dropout}")
    print(f"  class_weight: {'aus' if args.no_class_weight else 'balanced'}")
    print(f"  weight_decay {args.weight_decay}")
    print(f"  Parameter {n_params:,}   Zeit {elapsed:.0f}s")
    print()
    print(f"  {'':<26}{'Top-1':<10}{'Top-3':<10}{'Top-5'}")
    print(f"  {'gelernte Embeddings':<26}{acc[1]:<10.4f}{acc[3]:<10.4f}{acc[5]:.4f}")
    print(f"  {'TF-IDF + SVM (Basis)':<26}{BASELINE_SVM:<10.4f}{'0.8931':<10}{'0.9076'}")
    print(f"  {'TF-IDF + NN (bestes)':<26}{BASELINE_NN:<10.4f}{'0.9111':<10}{'0.9300'}")
    print()
    print(f"  Delta zu SVM-Basis: {100*(acc[1]-BASELINE_SVM):+.2f} pp Top-1")
    print(f"  Delta zu bestem NN: {100*(acc[1]-BASELINE_NN):+.2f} pp Top-1")


if __name__ == '__main__':
    main()
