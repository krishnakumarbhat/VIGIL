"""Transductive smoothing: average each prediction with its nearest same-route neighbours.

The generator keeps synthetic rows near other rows, and near rows agree on the label
~87% of the time. Averaging predictions over a row's neighbourhood therefore removes
independent model noise while keeping the shared signal. Neighbours never contribute
labels, so this is a legal post-processing step for the test set too.

Measured on OOF first: if smoothing helps there, it is applied to the submission.

Run:  python3 src/transductive_smooth.py
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from stack_experiments import ROOT, load_all, matrix

warnings.filterwarnings("ignore")
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def neighbour_mean(df: pd.DataFrame, kcols: list[str], values: np.ndarray, k: int = 5,
                   exclude_self: bool = False) -> np.ndarray:
    """Mean of ``values`` over the k nearest same-route rows (Hamming on kcols)."""
    from knn_signal_test import encode, build_cats

    cats = build_cats(df.copy(), kcols)
    X = encode(df, kcols, cats)
    route = df["Flight Distance"].values
    out = np.zeros(len(df), dtype=np.float64)
    order = np.argsort(route, kind="stable")
    rs = route[order]
    for r in np.unique(route):
        lo, hi = np.searchsorted(rs, r, "left"), np.searchsorted(rs, r, "right")
        idx = order[lo:hi]
        rows = np.where(route == r)[0]
        if len(idx) < 2:
            out[rows] = values[rows]
            continue
        D = (len(kcols) - X[rows] @ X[idx].T).astype(np.float32)
        if exclude_self and len(rows) == len(idx):
            D[np.arange(len(rows)), np.arange(len(idx))] = 1e6
        kk = min(k, len(idx) - 1)
        near = np.argsort(D, axis=1)[:, :kk]
        out[rows] = values[idx][near].mean(1)
    return out


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y = tr.satisfaction.astype(int).values
    names, oofs, _ = load_all(len(y), 299844)
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = list(np.argsort(-aucs))[:10]
    Z = matrix([oofs[i] for i in order])

    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(y, y))
    oof = np.zeros(len(y))
    for a, b in folds:
        oof[b] = LogisticRegression(C=0.3, max_iter=4000, n_jobs=-1).fit(Z[a], y[a]).decision_function(Z[b])
    base = roc_auc_score(y, oof)
    log(f"base nested AUC {base:.6f}")

    p = 1 / (1 + np.exp(-oof))
    route_cols = [c for c in tr.columns if c not in ("id", "satisfaction")]
    for k in (3, 5, 10):
        nb = neighbour_mean(tr, route_cols, p, k=k, exclude_self=True)
        for alpha in (0.1, 0.2, 0.3, 0.5):
            sm = np.log(np.clip((1 - alpha) * p + alpha * nb, 1e-9, 1 - 1e-9) /
                        (1 - np.clip((1 - alpha) * p + alpha * nb, 1e-9, 1 - 1e-9)))
            a_ = roc_auc_score(y, sm)
            log(f"k={k:2d} alpha={alpha:.1f} AUC {a_:.6f}  delta {a_ - base:+.6f}")


if __name__ == "__main__":
    main()
