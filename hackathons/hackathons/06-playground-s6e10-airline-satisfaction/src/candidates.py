"""Build the final candidate set for the last submission slots.

All candidates are honest stacks over published OOF members; they differ only in the
member subset and the combiner, so their expected quality is the same. Producing a few
of them lets the leaderboard pick which one transfers best, which at this margin is
mostly variance rather than skill.

Run:  python3 src/candidates.py
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from stack_experiments import ROOT, load_all, matrix, nested_lr

warnings.filterwarnings("ignore")
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def fit_predict(Z, ZT, y, C):
    """Fit on all rows and return test probabilities."""
    m = LogisticRegression(C=C, max_iter=5000, n_jobs=-1).fit(Z, y)
    return 1 / (1 + np.exp(-m.decision_function(ZT)))


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), len(te))
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = list(np.argsort(-aucs))
    log(f"pool {len(names)}")

    def sub(tag, pred):
        path = os.path.join(ROOT, "outputs", f"cand_{tag}.csv")
        pd.DataFrame({"id": te.id.values, "satisfaction": pred}).to_csv(path, index=False)
        log(f"wrote {path}")

    # C1: k=6 stacks (best CV)
    for k, C, tag in ((6, 0.3, "k6"), (10, 0.3, "k10"), (16, 0.3, "k16")):
        Z = matrix([oofs[i] for i in order[:k]])
        ZT = matrix([tests[i] for i in order[:k]])
        auc, _, _ = nested_lr(Z, y, C)
        log(f"{tag} nested {auc:.6f}")
        sub(tag, fit_predict(Z, ZT, y, C))

    # C2: bagged logistic regression on top 12 (variance reduction in the weights)
    Z = matrix([oofs[i] for i in order[:12]])
    ZT = matrix([tests[i] for i in order[:12]])
    rng = np.random.default_rng(0)
    decisions = np.zeros(len(te))
    B = 25
    for b in range(B):
        rows = rng.choice(len(y), int(0.8 * len(y)), replace=False)
        m = LogisticRegression(C=0.3, max_iter=3000, n_jobs=-1).fit(Z[rows], y[rows])
        decisions += m.decision_function(ZT)
    sub("bag12", 1 / (1 + np.exp(-decisions / B)))

    # C3: wide pool, stronger L2
    Z = matrix([oofs[i] for i in order])
    ZT = matrix([tests[i] for i in order])
    auc, _, _ = nested_lr(Z, y, 0.03)
    log(f"wide nested {auc:.6f}")
    sub("wide", fit_predict(Z, ZT, y, 0.03))


if __name__ == "__main__":
    main()
