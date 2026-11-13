"""Track 05 - fit and persist every non-transformer member for hidden-test inference.

The OOF matrices prove the members work, but a probability matrix is not a deployable artefact:
to score the private test set we need the fitted vectorisers and classifiers on disk. This script
refits each sparse member on the full training pool (real + mirror, same recipes that produced the
OOF matrices the blend recipe was fitted on) and writes them to `outputs/deploy/`.

Artefacts
  outputs/deploy/svm.joblib    word(1-2)+char_wb(3-5) TF-IDF -> SGD log-loss
  outputs/deploy/char.joblib   char_wb(2-5) TF-IDF -> SGD log-loss
  outputs/deploy/word.joblib   word(1-3) TF-IDF -> SGD log-loss
  outputs/deploy/style.joblib  stylometric features + scaler -> SGD log-loss
  outputs/deploy/knn.joblib    char_wb TF-IDF matrix + labels for cosine kNN
  outputs/deploy/manifest.json
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ntl_style import style_matrix  # exact features used by the OOF member

D = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{D}/outputs"
DEP = f"{OUT}/deploy"
os.makedirs(DEP, exist_ok=True)
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
real["text"] = real["text"].astype(str).str.slice(0, 6000)

mirror = pd.read_parquet(f"{D}/data/compact/passages.parquet")
mirror = mirror[mirror.model.isin(LABELS)].dropna(subset=["text"]).sample(n=9000, random_state=42)
mirror = mirror.reset_index(drop=True)
mirror["y"] = mirror["model"].map(lab2id).astype(int)
mirror["text"] = mirror["text"].astype(str).str.slice(0, 6000)

texts = real.text.tolist() + mirror.text.tolist()
y = np.concatenate([real.y.to_numpy(), mirror.y.to_numpy()])
log("pool", len(texts), "rows")

manifest = {}

# ---------------------------------------------------------------- svm / char / word
char = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                       max_features=250_000, lowercase=True)
Xc = char.fit_transform(texts)
word12 = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=120_000,
                         strip_accents="unicode", lowercase=True)
Xw12 = word12.fit_transform(texts)
word13 = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), min_df=2, max_features=200_000,
                         strip_accents="unicode", lowercase=True)
Xw13 = word13.fit_transform(texts)
log("tf-idf built", Xc.shape, Xw12.shape, Xw13.shape)


def fit_sgd(X, name, alpha=3e-6):
    clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=60, tol=1e-4, random_state=42,
                        average=True)
    clf.fit(X, y)
    joblib.dump({"model": clf, "classes": clf.classes_.tolist(), "labels": LABELS},
                f"{DEP}/{name}.joblib", compress=3)
    manifest[name] = {"kind": "sgd", "features": int(X.shape[1]), "alpha": alpha,
                      "train_rows": int(X.shape[0])}
    log(name, "fitted")


fit_sgd(hstack([Xw12, Xc]).tocsr(), "svm")
fit_sgd(Xc, "char")
fit_sgd(Xw13, "word")
joblib.dump({"char": char, "word12": word12, "word13": word13}, f"{DEP}/vectorisers.joblib", compress=3)

# ---------------------------------------------------------------- style
S = style_matrix(texts)
scaler = StandardScaler().fit(S)
clf_style = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=80, tol=1e-4, random_state=42,
                          average=True)
clf_style.fit(csr_matrix(scaler.transform(S)), y)
joblib.dump({"model": clf_style, "scaler": scaler, "classes": clf_style.classes_.tolist(),
             "labels": LABELS}, f"{DEP}/style.joblib", compress=3)
manifest["style"] = {"kind": "sgd", "features": int(S.shape[1]), "alpha": 1e-4,
                     "train_rows": int(S.shape[0])}
log("style fitted")

# ---------------------------------------------------------------- kNN index
norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(axis=1))).ravel()
Xn = Xc.multiply(1.0 / np.maximum(norms, 1e-9)[:, None]).tocsr()
joblib.dump({"matrix": Xn, "y": y, "labels": LABELS, "k": 30, "vectoriser": char},
            f"{DEP}/knn.joblib", compress=3)
manifest["knn"] = {"kind": "cosine_knn", "k": 30, "train_rows": int(Xn.shape[0])}
log("knn index built")

json.dump({"members": manifest, "labels": LABELS,
           "note": "fit on real + mirror(9k); the same recipes produced the OOF matrices used "
                   "to fit outputs/ensemble_recipe.json"},
          open(f"{DEP}/manifest.json", "w"), indent=1)
log("DONE", list(manifest))
