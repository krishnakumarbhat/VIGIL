"""Persist the `svmext` member (TF-IDF + SGD trained with the external corpus) for inference.

`svmext` was the strongest sparse member (OOF 0.3801 vs 0.3616 mirror-only) and carries a real
weight in the blend, so it needs its own deployable bundle: the vectorisers must be fitted on the
exact same pool (real + mirror + 40k external) as the OOF run.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

D = Path("/home/pope/Desktop/gpu /hackathons/05-name-that-llm")
OUT, DEP = D / "outputs", D / "outputs" / "deploy"
LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}
t0 = time.time()

real = pd.read_csv(D / "data/competition/train.csv")
real["y"] = real["model"].map(lab2id)
real = real.dropna(subset=["y"]).reset_index(drop=True)
mir = pd.read_parquet(D / "data/compact/passages.parquet")
mir = mir[mir.model.isin(LABELS)].dropna(subset=["text"]).sample(n=9000, random_state=42)
mir["y"] = mir["model"].map(lab2id)
ext = pd.read_parquet(D / "data/compact/external.parquet")
ext = ext[ext.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
ext["y"] = ext["model"].map(lab2id)
per = max(1, 40000 // len(LABELS))
ext = (ext.groupby("y", group_keys=False)
          .apply(lambda g: g.sample(n=min(per, len(g)), random_state=42)).reset_index(drop=True))

texts = (real.text.astype(str).str.slice(0, 6000).tolist()
         + mir.text.astype(str).str.slice(0, 6000).tolist()
         + ext.text.astype(str).str.slice(0, 6000).tolist())
y = np.concatenate([real.y.to_numpy(), mir.y.to_numpy(), ext.y.to_numpy()])
print(f"[{time.time()-t0:6.1f}s] pool {len(texts)}")

w12 = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=120_000,
                      strip_accents="unicode", lowercase=True)
ch = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                     max_features=250_000, lowercase=True)
X = hstack([w12.fit_transform(texts), ch.fit_transform(texts)]).tocsr()
print(f"[{time.time()-t0:6.1f}s] features {X.shape}")
clf = SGDClassifier(loss="log_loss", alpha=3e-6, max_iter=40, tol=1e-4, random_state=42,
                    average=True)
clf.fit(X, y)
joblib.dump({"model": clf, "word12": w12, "char": ch, "classes": clf.classes_.tolist(),
             "labels": LABELS}, DEP / "svmext.joblib", compress=3)
print(f"[{time.time()-t0:6.1f}s] saved {DEP/'svmext.joblib'}")
