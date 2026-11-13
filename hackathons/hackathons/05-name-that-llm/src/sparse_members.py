"""Track 05 - cheap decorrelated members (stylometry, char-only, word-only, kNN).

Members that make *different kinds* of mistakes are worth more to the stack than a marginally
better transformer: the linear TF-IDF member alone scores only 0.36 yet lifted the cross-fitted
blend from 0.5557 to 0.5957 because its errors are uncorrelated with the language models'.

This script trains four such members under the shared protocol
(StratifiedGroupKFold(5, shuffle, seed 42, groups=prompt_id) over the 2,699 real rows; the
held-out fold's real rows are excluded and the mirror corpus - disjoint prompt namespace
`g###` vs `p###`, verified zero text overlap - is always available for training).

Outputs (auto-discovered by `src/stack_v2.py`)
  outputs/oof_<member>_fold{0..4}.npy + oof_<member>_idx_fold{0..4}.npy
  outputs/sparse_members_status.json
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

D = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{D}/outputs"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
C = len(LABELS)
lab2id = {l: i for i, l in enumerate(LABELS)}

real = pd.read_csv(f"{D}/data/competition/train.csv")
real["y"] = real["model"].map(lab2id)
real = real.dropna(subset=["y"]).reset_index(drop=True)
real["y"] = real["y"].astype(int)
real["text"] = real["text"].astype(str).str.slice(0, 6000)

mirror = pd.read_parquet(f"{D}/data/compact/passages.parquet")
mirror = mirror[mirror.model.isin(LABELS)].dropna(subset=["text"]).sample(n=9000, random_state=42)
mirror = mirror.reset_index(drop=True)
mirror["y"] = mirror["model"].map(lab2id).astype(int)
mirror["text"] = mirror["text"].astype(str).str.slice(0, 6000)
log("real", real.shape, "mirror", mirror.shape)

pool_text = pd.concat([real.text, mirror.text], ignore_index=True)
pool_y = np.concatenate([real.y.to_numpy(), mirror.y.to_numpy()])
n_real = len(real)

folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
             .split(np.zeros(n_real), real.y, groups=real.prompt_id))
log("folds built", [len(v) for _, v in folds])


def save_member(name, oof, fold_scores):
    for fold in range(len(folds)):
        vi = folds[fold][1]
        np.save(f"{OUT}/oof_{name}_fold{fold}.npy", oof[vi].astype(np.float32))
        np.save(f"{OUT}/oof_{name}_idx_fold{fold}.npy", vi)
    log(f"{name}: OOF bal_acc {balanced_accuracy_score(real.y, oof.argmax(1)):.4f} "
        f"folds {[round(s, 4) for s in fold_scores]}")
    return {"oof_bal_acc": float(balanced_accuracy_score(real.y, oof.argmax(1))),
            "fold_bal_acc": [float(s) for s in fold_scores]}


# --------------------------------------------------------------- stylometric view
PATTERNS = {
    "comma": r",", "period": r"\.", "semicolon": r";", "colon": r":", "emdash": "\u2014",
    "endash": "\u2013", "exclam": r"!", "question": r"\?", "paren": r"\(", "bracket": r"\[",
    "quot": r'"', "apos": r"'", "bullet": r"(?m)^\s*[-*+]\s", "numlist": r"(?m)^\s*\d+[.)]\s",
    "header": r"(?m)^#{1,6}\s", "bold": r"\*\*", "italic": r"(?<!\*)\*(?!\*)", "code": r"`",
    "table": r"(?m)^\s*\|", "arrow": "\u2192", "slash": r"/", "tilde": r"~", "at": r"@",
}
RE = {k: re.compile(v) for k, v in PATTERNS.items()}
SENT = re.compile(r"[.!?]+\s")


def style_matrix(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        lines = t.split("\n")
        line_lens = [len(x) for x in lines] or [0]
        sents = [s for s in SENT.split(t) if s.strip()]
        sent_lens = [len(s) for s in sents] or [0]
        words = t.split()
        wl = [len(w) for w in words] or [0]
        feats = [
            np.log1p(n), np.log1p(len(lines)), np.log1p(len(words)), np.log1p(len(sents)),
            float(np.mean(line_lens)), float(np.max(line_lens)), float(np.std(line_lens)),
            float(np.mean(sent_lens)), float(np.max(sent_lens)), float(np.std(sent_lens)),
            float(np.mean(wl)), float(np.max(wl)), float(np.std(wl)),
            sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
            sum(c.isspace() for c in t) / n,
            sum(not c.isascii() for c in t) / n,
            t.count("\n\n") / max(len(lines), 1),
            sum(1 for x in line_lens if x < 40) / len(line_lens),
            sum(1 for x in line_lens if x > 300) / len(line_lens),
        ]
        for rx in RE.values():
            feats.append(len(rx.findall(t)) / n * 1000.0)
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)


log("building stylometric matrix")
S_all = style_matrix(pool_text.tolist())
log("style features", S_all.shape)


def run_sgd(X, name, alpha=3e-6, loss="log_loss"):
    oof = np.zeros((n_real, C), dtype=np.float64)
    scores = []
    for fold, (ti, vi) in enumerate(folds):
        val_rows = vi  # real rows occupy the first n_real rows of the pool
        keep = np.setdiff1d(np.arange(X.shape[0]), val_rows)
        clf = SGDClassifier(loss=loss, alpha=alpha, max_iter=60, tol=1e-4, random_state=42,
                            n_jobs=-1, average=True)
        clf.fit(X[keep], pool_y[keep])
        p = clf.predict_proba(X[val_rows])
        full = np.zeros((len(val_rows), C))
        for j, c in enumerate(clf.classes_):
            full[:, c] = p[:, j]
        oof[vi] = full
        scores.append(balanced_accuracy_score(real.y.to_numpy()[vi], full.argmax(1)))
        log(f"  {name} fold {fold} {scores[-1]:.4f}")
    return save_member(name, oof, scores)


results = {}
scaler = StandardScaler()
S_scaled = scaler.fit_transform(S_all)
results["style"] = run_sgd(csr_matrix(S_scaled), "style", alpha=1e-4)

# --------------------------------------------------------------- char / word views
log("fit char tf-idf")
char = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                       max_features=250_000, lowercase=True)
Xc = char.fit_transform(pool_text)
log("char", Xc.shape)
results["char"] = run_sgd(Xc, "char", alpha=3e-6)

log("fit word tf-idf")
word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), min_df=2, max_features=200_000,
                       lowercase=True, strip_accents="unicode")
Xw = word.fit_transform(pool_text)
log("word", Xw.shape)
results["word"] = run_sgd(Xw, "word", alpha=3e-6)

# --------------------------------------------------------------- cosine kNN on char tf-idf
log("kNN member")
Xn = Xc.copy()
Xn = Xn.multiply(1.0 / np.maximum(np.sqrt(Xn.multiply(Xn).sum(axis=1)), 1e-9)).tocsr()
K = 30
oof = np.zeros((n_real, C))
scores = []
for fold, (ti, vi) in enumerate(folds):
    keep = np.setdiff1d(np.arange(Xn.shape[0]), vi)
    sims = (Xn[vi] @ Xn[keep].T).toarray()
    yk = pool_y[keep]
    idx = np.argpartition(-sims, K, axis=1)[:, :K]
    probs = np.full((len(vi), C), 1e-3)
    for r in range(len(vi)):
        cols = idx[r]
        np.add.at(probs[r], yk[cols], np.clip(sims[r, cols], 0, None) ** 3)
    probs /= probs.sum(axis=1, keepdims=True)
    oof[vi] = probs
    scores.append(balanced_accuracy_score(real.y.to_numpy()[vi], probs.argmax(1)))
    log(f"  knn fold {fold} {scores[-1]:.4f}")
results["knn"] = save_member("knn", oof, scores)

json.dump(results, open(f"{OUT}/sparse_members_status.json", "w"), indent=1)
log("DONE", json.dumps({k: round(v["oof_bal_acc"], 4) for k, v in results.items()}))
