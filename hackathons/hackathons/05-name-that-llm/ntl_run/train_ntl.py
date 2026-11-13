"""Name That LLM: grouped-CV TF-IDF classifier for the 3-row scoring set.

Metric is balanced accuracy over 18 model classes.  Train has 2,699 rows grouped by
prompt_id, so CV must be grouped by prompt to avoid leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import GroupKFold

COMP = "data/comp"


def build_feats(train_text, other_text):
    """Fit word and character TF-IDF blocks on train and transform both."""
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        strip_accents="unicode", max_features=120_000)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        sublinear_tf=True, max_features=200_000)
    Aw, Bw = w.fit_transform(train_text), w.transform(other_text)
    Ac, Bc = c.fit_transform(train_text), c.transform(other_text)
    return hstack([Aw, Ac]).tocsr(), hstack([Bw, Bc]).tocsr()


def main() -> None:
    tr = pd.read_csv(f"{COMP}/train.csv")
    te = pd.read_csv(f"{COMP}/test.csv")
    y = tr["model"].values
    groups = tr["prompt_id"].values

    best = None
    for C in (1.0, 3.0, 10.0, 30.0):
        Xtr, Xte = build_feats(tr["text"].values, te["text"].values)
        oof = np.zeros((len(tr), len(np.unique(y))), dtype=np.float64)
        classes = np.unique(y)
        gkf = GroupKFold(n_splits=5)
        for tr_i, va_i in gkf.split(Xtr, y, groups):
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            m.fit(Xtr[tr_i], y[tr_i])
            oof[va_i] = m.predict_proba(Xtr[va_i])[:, [list(m.classes_).index(c) for c in classes]]
        ba = balanced_accuracy_score(y, classes[oof.argmax(1)])
        acc = accuracy_score(y, classes[oof.argmax(1)])
        print(f"C={C:5.1f}  grouped-CV balanced_acc={ba:.4f}  acc={acc:.4f}")
        if best is None or ba > best[0]:
            best = (ba, C, oof, Xtr, Xte, classes)

    ba, C, oof, Xtr, Xte, classes = best
    model = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, y)
    proba = model.predict_proba(Xte)
    print(f"\nbest C={C} grouped-CV balanced_acc={ba:.4f}")
    out = pd.DataFrame({"id": te["id"], "model": model.classes_[proba.argmax(1)]})
    out.to_parquet("ntl_run/submission.parquet", index=False)
    out.to_csv("ntl_run/submission.csv", index=False)
    top = pd.DataFrame(proba, columns=model.classes_)
    for i, tid in enumerate(te["id"]):
        s = top.iloc[i].sort_values(ascending=False).head(4)
        print(f"  {tid}: " + ", ".join(f"{k}={v:.3f}" for k, v in s.items()))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
