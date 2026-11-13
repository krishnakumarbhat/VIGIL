"""Track 05 - does the 133k-row external corpus help the sparse members?

The A100 curriculum only had budget for 8k external rows, but the linear members are cheap, so
they can absorb the whole corpus. This experiment trains the best-performing sparse recipe
(word 1-2 + char_wb 3-5 TF-IDF -> SGD log-loss) with a balanced external sample added to the
mirror + non-fold real training pool, and writes `oof_svmext_*` so the stacker can A/B it against
`oof_svm_*` without clobbering the incumbent.

Leakage guards: the external corpus uses a disjoint prompt namespace (verified: 0 of 63,262
prompts overlap the 150 real prompts, 0 identical text prefixes), and the held-out fold's real
rows are always removed from training.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

D = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{D}/outputs"
EXT_ROWS = int(os.environ.get("EXT_ROWS", "40000"))
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

mirror = pd.read_parquet(f"{D}/data/compact/passages.parquet")
mirror = mirror[mirror.model.isin(LABELS)].dropna(subset=["text"]).sample(n=9000, random_state=42)
mirror = mirror.reset_index(drop=True)
mirror["y"] = mirror["model"].map(lab2id).astype(int)

ext = pd.read_parquet(f"{D}/data/compact/external.parquet")
ext = ext[ext.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
ext["y"] = ext["model"].map(lab2id).astype(int)
per = max(1, EXT_ROWS // C)
ext = (ext.groupby("y", group_keys=False)
          .apply(lambda g: g.sample(n=min(per, len(g)), random_state=42))
          .reset_index(drop=True))
log("real", len(real), "mirror", len(mirror), "external", len(ext))

texts = (real.text.astype(str).str.slice(0, 6000).tolist()
         + mirror.text.astype(str).str.slice(0, 6000).tolist()
         + ext.text.astype(str).str.slice(0, 6000).tolist())
ys = np.concatenate([real.y.to_numpy(), mirror.y.to_numpy(), ext.y.to_numpy()])

word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=120_000,
                       strip_accents="unicode", lowercase=True)
char = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                       max_features=250_000, lowercase=True)
X = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
log("features", X.shape)

folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
             .split(np.zeros(len(real)), real.y, groups=real.prompt_id))
oof = np.zeros((len(real), C), dtype=np.float32)
scores = []
for fold, (ti, vi) in enumerate(folds):
    keep = np.setdiff1d(np.arange(X.shape[0]), vi)
    clf = SGDClassifier(loss="log_loss", alpha=3e-6, max_iter=40, tol=1e-4, random_state=42,
                        n_jobs=-1, average=True)
    clf.fit(X[keep], ys[keep])
    p = clf.predict_proba(X[vi])
    full = np.zeros((len(vi), C))
    for j, c in enumerate(clf.classes_):
        full[:, c] = p[:, j]
    oof[vi] = full
    scores.append(float(balanced_accuracy_score(real.y.to_numpy()[vi], full.argmax(1))))
    log(f"fold {fold} bal_acc {scores[-1]:.4f} acc {accuracy_score(real.y.to_numpy()[vi], full.argmax(1)):.4f}")
    np.save(f"{OUT}/oof_svmext_fold{fold}.npy", full.astype(np.float32))
    np.save(f"{OUT}/oof_svmext_idx_fold{fold}.npy", vi)

bal = float(balanced_accuracy_score(real.y, oof.argmax(1)))
prev = 0.0
if os.path.exists(f"{OUT}/svm_ext_status.json"):
    prev = float(json.load(open(f"{OUT}/svm_ext_status.json"))["oof_bal_acc"])
json.dump({"oof_bal_acc": bal, "fold_bal_acc": scores, "ext_rows": int(len(ext)),
           "features": int(X.shape[1]), "previous": prev},
          open(f"{OUT}/svm_ext_status.json", "w"), indent=1)
log(f"OOF bal_acc {bal:.4f} (incumbent mirror-only svm 0.3616)")
log("DONE")
