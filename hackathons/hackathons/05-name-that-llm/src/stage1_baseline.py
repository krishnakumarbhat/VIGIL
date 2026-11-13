"""Track 05 Stage 1 - fast stylometric + n-gram baseline with StratifiedGroupKFold.

Grouped by prompt_id (never random KFold). Sublinear TF-IDF char(2-6) + word(1-3),
plus structural/stylometric features; LogisticRegression / Ridge / LinearSVC ensemble.
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

T0 = time.time()
OUT = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm/outputs"
os.makedirs(OUT, exist_ok=True)
LOG = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm/logs/experiments.tsv"


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}

BASE = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm/data/compact"
df = pd.read_parquet(f"{BASE}/passages.parquet")
df = df[df.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
df["text"] = df["text"].astype(str)
df["y"] = df.model.map(lab2id)
log("rows", len(df), "prompts", df.prompt_id.nunique(), "classes", df.y.nunique())


def stylometric(texts):
    rows = []
    for t in texts:
        n_chars = max(len(t), 1)
        words = t.split()
        nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        slens = [len(s.split()) for s in sents] or [0]
        mean_w = sum(slens) / len(slens)
        var_w = sum((x - mean_w) ** 2 for x in slens) / len(slens)
        uniq = len(set(w.lower() for w in words))
        rows.append([
            n_chars / 1000, nw / 100, mean_w / 50, (var_w ** 0.5) / 20,
            uniq / nw,                                   # type-token ratio
            t.count("—") / n_chars * 1000,               # em-dash density
            t.count("–") / n_chars * 1000,
            t.count('"') / n_chars * 1000,
            t.count("'") / n_chars * 1000,
            t.count("(") / n_chars * 1000,
            t.count("!") / n_chars * 1000,
            t.count("?") / n_chars * 1000,
            t.count(",") / n_chars * 1000,
            t.count(";") / n_chars * 1000,
            t.count(":") / n_chars * 1000,
            sum(1 for c in t if c.isupper()) / n_chars,
            sum(1 for c in t if c.isdigit()) / n_chars,
            sum(1 for c in t if c.islower()) / n_chars,
            len(re.findall(r"^\s*[-*+]\s", t, re.M)) / nw * 100,   # bullet density
            len(re.findall(r"^#{1,6}\s", t, re.M)) / nw * 100,      # header density
            len(re.findall(r"```", t)) / nw * 100,                  # code fence
            len(re.findall(r"`[^`]+`", t)) / nw * 100,              # inline code
            len(re.findall(r"\*\*", t)) / nw * 100,                # bold
            len(re.findall(r"\[.*?\]\(.*?\)", t)) / nw * 100,        # markdown links
            len(re.findall(r"\b[A-Z]{2,}\b", t)) / nw * 100,          # SHOUTED acronyms
            len(re.findall(r"\d", t)) / nw * 100,
            len(re.findall(r"[A-Za-z]+", t)) / nw,
            (t.count("\n") + 1) / nw,
            len(t.split("\n\n")) / max(len(sents), 1),
            len(set(re.findall(r"[a-z]{3,}", t.lower()))) / nw,
            sum(1 for w in words if len(w) > 6) / nw,
            sum(1 for w in words if w.isupper() and len(w) > 1) / nw,
            (max(slens) - min(slens)) / max(mean_w, 1),
        ])
    return np.asarray(rows, dtype=np.float32)


texts = df.text.tolist()
log("extracting stylometry")
sty = stylometric(texts)
log("char tfidf")
cv_char = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), sublinear_tf=True, lowercase=False,
                           min_df=2, max_features=300000, dtype=np.float32)
Xc = cv_char.fit_transform(texts)
log("word tfidf")
cv_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), sublinear_tf=True, lowercase=False,
                           min_df=2, max_features=200000, dtype=np.float32)
Xw = cv_word.fit_transform(texts)
X = hstack([Xc, Xw, csr_matrix(sty)]).tocsr()
log("X", X.shape)

groups = df.prompt_id.to_numpy()
y = df.y.to_numpy()
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X, y, groups=groups))

oofs = {k: np.zeros((len(df), len(LABELS)), dtype=np.float32) for k in ["lr", "ridge"]}
for f, (ti, vi) in enumerate(folds):
    for name, clf in [
        ("lr", LogisticRegression(max_iter=2000, C=4.0, n_jobs=8)),
        ("ridge", RidgeClassifier(alpha=1.0)),
    ]:
        clf.fit(X[ti], y[ti])
        p = clf.predict_proba(X[vi]) if hasattr(clf, "predict_proba") else None
        if p is None:
            d = clf.decision_function(X[vi])
            e = np.exp(d - d.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
        oofs[name][vi] = p
    acc = balanced_accuracy_score(y[vi], oofs["lr"][vi].argmax(1))
    log(f"fold {f} lr bal_acc {acc:.4f}")

blend = 0.5 * oofs["lr"] + 0.5 * oofs["ridge"]
res = {
    "n_rows": int(len(df)), "n_prompts": int(df.prompt_id.nunique()),
    "bal_acc_lr": float(balanced_accuracy_score(y, oofs["lr"].argmax(1))),
    "bal_acc_ridge": float(balanced_accuracy_score(y, oofs["ridge"].argmax(1))),
    "bal_acc_blend": float(balanced_accuracy_score(y, blend.argmax(1))),
    "acc_blend": float(accuracy_score(y, blend.argmax(1))),
}
log(json.dumps(res, indent=1))

# length-bin accuracy
bins = [0, 60, 120, 180, 1000]
nwords = df.text.str.split().str.len().to_numpy()
pred = blend.argmax(1)
len_acc = {}
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (nwords > lo) & (nwords <= hi)
    if m.sum():
        len_acc[f"({lo},{hi}]"] = {"n": int(m.sum()), "bal_acc": float(balanced_accuracy_score(y[m], pred[m]))}
res["length_bin_accuracy"] = len_acc

from sklearn.metrics import classification_report
rep = classification_report(y, pred, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0, output_dict=True)
res["per_class_recall"] = {k: float(v["recall"]) for k, v in rep.items() if isinstance(v, dict) and "recall" in v}

np.save(f"{OUT}/oof_stage1.npy", blend)
pd.DataFrame({"prompt_id": df.prompt_id, "model": df.model, "y": y, "pred": [LABELS[i] for i in pred]}
             ).to_parquet(f"{OUT}/oof_stage1.parquet", index=False)
json.dump(res, open(f"{OUT}/cv_report_stage1.json", "w"), indent=1)
with open(LOG, "a") as fh:
    fh.write(f"05-name-that-llm\tbal_acc_stage1\t{res['bal_acc_blend']:.5f}\tchar+word tfidf + stylometry, StratifiedGroupKFold(5, group=prompt_id), n={len(df)}\n")
log("saved")
