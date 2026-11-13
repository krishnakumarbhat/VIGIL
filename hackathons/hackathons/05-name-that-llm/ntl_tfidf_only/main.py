"""Name That LLM - fast CPU baseline kernel (word+char TF-IDF -> logistic regression)."""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

OUT = "/kaggle/working"


def find(fn):
    for root, _d, files in os.walk("/kaggle/input"):
        if fn in files:
            return os.path.join(root, fn)
    raise FileNotFoundError(fn)


tr = pd.read_csv(find("train.csv"))
te = pd.read_csv(find("test.csv"))
print("train", tr.shape, "test", te.shape, flush=True)

labels = sorted(tr["model"].unique())
y = tr["model"].map({l: i for i, l in enumerate(labels)}).values

w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=120_000)
c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                    max_features=200_000)
Xtr = hstack([w.fit_transform(tr["text"]), c.fit_transform(tr["text"])]).tocsr()
Xte = hstack([w.transform(te["text"]), c.transform(te["text"])]).tocsr()

m = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced").fit(Xtr, y)
proba = np.zeros((len(te), len(labels)))
proba[:, m.classes_.astype(int)] = m.predict_proba(Xte)

sub = pd.DataFrame({"id": te["id"], "model": [labels[i] for i in proba.argmax(1)]})
sub.to_csv(f"{OUT}/submission.csv", index=False)
sub.to_parquet(f"{OUT}/submission.parquet", index=False)
print(sub.head(), sub.shape, flush=True)
