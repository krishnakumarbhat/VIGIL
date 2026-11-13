"""Name That LLM: grouped-CV TF-IDF classifier over the 3-row scoring set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import GroupKFold

COMP = "data/comp"
tr = pd.read_csv(f"{COMP}/train.csv")
te = pd.read_csv(f"{COMP}/test.csv")
y = tr["model"].values
groups = tr["prompt_id"].values
classes = np.unique(y)

w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=120_000)
c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200_000)
Xtr = hstack([w.fit_transform(tr["text"].values), c.fit_transform(tr["text"].values)]).tocsr()
Xte = hstack([w.transform(te["text"].values), c.transform(te["text"].values)]).tocsr()
print("features", Xtr.shape, Xte.shape, flush=True)

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(Xtr, y, groups))
best = None
for C in (0.5, 1.0, 3.0, 10.0):
    oof = np.zeros((len(tr), len(classes)))
    for tr_i, va_i in splits:
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        m.fit(Xtr[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(Xtr[va_i])[:, [list(m.classes_).index(k) for k in classes]]
    ba = balanced_accuracy_score(y, classes[oof.argmax(1)])
    print(f"C={C:5.1f} grouped-CV balanced_acc={ba:.4f} acc={accuracy_score(y, classes[oof.argmax(1)]):.4f}", flush=True)
    if best is None or ba > best[0]:
        best = (ba, C)

ba, C = best
m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, y)
proba = m.predict_proba(Xte)
out = pd.DataFrame({"id": te["id"], "model": m.classes_[proba.argmax(1)]})
out.to_parquet("ntl_run/submission.parquet", index=False)
out.to_csv("ntl_run/submission.csv", index=False)
print(f"BEST C={C} ba={ba:.4f}")
for i, tid in enumerate(te["id"]):
    s = pd.Series(proba[i], index=m.classes_).sort_values(ascending=False).head(4)
    print(f"{tid}: " + ", ".join(f"{k}={v:.3f}" for k, v in s.items()))
print(out.to_string(index=False))
