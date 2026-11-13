"""Track 05 - sparse TF-IDF linear member (CPU, decorrelated from the transformer members).

Character n-grams capture the punctuation/markdown habits that separate model families far
better than word tokens (a strong signal for `glm-*` vs `gpt-*` vs `claude-*`). The member is
trained with the same 5-fold StratifiedGroupKFold(groups=prompt_id) protocol used by every other
Track 05 member, so its OOF matrix can be stacked directly.

Outputs
  outputs/oof_svm_fold{0..4}.npy / oof_svm_idx_fold{i}.npy   (out-of-fold probabilities)
  outputs/test_svm.npy                                        (visible test rows)
  outputs/svm_status.json
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
lab2id = {l: i for i, l in enumerate(LABELS)}

real = pd.read_csv(f"{D}/data/competition/train.csv")
real["y"] = real["model"].map(lab2id)
real = real.dropna(subset=["y"]).reset_index(drop=True)
real["y"] = real["y"].astype(int)

mirror = pd.read_parquet(f"{D}/data/compact/passages.parquet")
mirror = mirror[mirror.model.isin(LABELS)].dropna(subset=["text"]).copy()
mirror = mirror.sample(n=min(9000, len(mirror)), random_state=42)
mirror["y"] = mirror["model"].map(lab2id)
mirror = mirror.dropna(subset=["y"]).reset_index(drop=True)
mirror["y"] = mirror["y"].astype(int)
log("real", real.shape, "mirror", mirror.shape)

text_col_real, text_col_mir = "text", "text"
pool = pd.concat([
    real[["prompt_id", text_col_real, "y"]].assign(src="real"),
    mirror[["prompt_id", text_col_mir, "y"]].assign(src="mirror"),
], ignore_index=True)
pool = pool.rename(columns={text_col_real: "text"})
pool["text"] = pool["text"].astype(str).str.slice(0, 4000)

word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=120_000,
                       strip_accents="unicode", lowercase=True)
char = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                       max_features=200_000, lowercase=True)

Xw = word.fit_transform(pool.text)
Xc = char.fit_transform(pool.text)
X = hstack([Xw, Xc]).tocsr()
log("features", X.shape)

is_real = (pool.src == "real").to_numpy()
real_pos = np.where(is_real)[0]
y_real = pool.y.to_numpy()[real_pos]
groups = pool.prompt_id.to_numpy()[real_pos]

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(real_pos)), y_real, groups=groups))

oof = np.zeros((len(real_pos), len(LABELS)), dtype=np.float32)
accs = []
rng = np.random.default_rng(0)

for fold, (ti, vi) in enumerate(folds):
    val_rows = real_pos[vi]
    train_rows = np.setdiff1d(np.arange(X.shape[0]), val_rows)
    clf = SGDClassifier(loss="log_loss", alpha=3e-6, max_iter=40, tol=1e-4,
                        random_state=42, n_jobs=-1, average=True)
    clf.fit(X[train_rows], pool.y.to_numpy()[train_rows])
    p = clf.predict_proba(X[val_rows])
    # align to the full label space (SGD may miss a class in a training split)
    full = np.zeros((len(val_rows), len(LABELS)), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        full[:, c] = p[:, j]
    oof[vi] = full
    acc = float(balanced_accuracy_score(y_real[vi], full.argmax(1)))
    accs.append(acc)
    log(f"fold {fold} bal_acc {acc:.4f} acc {accuracy_score(y_real[vi], full.argmax(1)):.4f}")
    np.save(f"{OUT}/oof_svm_fold{fold}.npy", full)
    np.save(f"{OUT}/oof_svm_idx_fold{fold}.npy", vi)

log("OOF bal_acc", round(float(balanced_accuracy_score(y_real, oof.argmax(1))), 4),
    "mean fold", round(float(np.mean(accs)), 4))

# test predictions (visible rows only)
try:
    te = pd.read_csv(f"{D}/data/competition/test.csv")
    Xt = hstack([word.transform(te.text.astype(str)), char.transform(te.text.astype(str))]).tocsr()
    pt = clf.predict_proba(Xt)
    full = np.zeros((len(te), len(LABELS)), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        full[:, c] = pt[:, j]
    np.save(f"{OUT}/test_svm.npy", full)
    log("test", full.shape)
except Exception as e:
    log("test skipped", repr(e)[:100])

json.dump({"oof_bal_acc": float(balanced_accuracy_score(y_real, oof.argmax(1))),
           "fold_bal_acc": [float(a) for a in accs],
           "features": int(X.shape[1]), "rows_real": int(len(real_pos)),
           "rows_mirror": int((~is_real).sum())},
          open(f"{OUT}/svm_status.json", "w"), indent=1)
log("DONE")
