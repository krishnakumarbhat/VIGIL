"""Quick test: do train-to-train nearest-neighbour labels add signal on top of the raw columns?

Setup: split train into A (fit) and B (held out). Neighbour features for B are computed
only from A, so the comparison is clean. Neighbours are restricted to the same
Flight Distance (route) because that is where the generator keeps near copies.

Run:  python3 src/knn_signal_test.py
"""
from __future__ import annotations

import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()
CATS = {
    "Gender": None, "Customer Type": None, "Type of Travel": None, "Class": None,
}


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def encode(df: pd.DataFrame, cols: list[str], cats: dict[str, np.ndarray]) -> np.ndarray:
    """One-hot each column with a shared category map so rows stay comparable."""
    blocks = []
    for c in cols:
        v = df[c].values
        if c in cats:
            vv = np.searchsorted(cats[c], v)
            vv = np.clip(vv, 0, len(cats[c]) - 1)
        else:
            # numeric column treated as equality: use its own value index
            vv = np.searchsorted(cats[c], v)
            vv = np.clip(vv, 0, len(cats[c]) - 1)
        m = np.zeros((len(df), len(cats[c])), dtype=np.float32)
        m[np.arange(len(df)), vv] = 1.0
        blocks.append(m)
    return np.hstack(blocks)


def build_cats(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for c in cols:
        vals = df[c].values
        if c in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
            bins = np.digitize(np.nan_to_num(vals, nan=-1), [0.5, 15, 60, 180])
            out[c] = np.array([-1, 0, 1, 2, 3, 4], dtype=np.float64)
            df[c] = bins
        else:
            out[c] = np.unique(vals)
    return out


def knn_feats(A: pd.DataFrame, B: pd.DataFrame, cols: list[str], cats: dict[str, np.ndarray],
              exclude_self: bool = False) -> pd.DataFrame:
    """For every B row, label statistics of its nearest same-route neighbours in A.

    Distance is a plain Hamming count over the 20 non-route columns; the route itself is
    the grouping key, so it must not enter the one-hot (it would add thousands of columns
    and swamp the distance with a constant).
    """
    kcols = [c for c in cols if c != "Flight Distance"]
    XA = encode(A, kcols, cats)
    XB = encode(B, kcols, cats)
    F = len(kcols)
    a_route = A["Flight Distance"].values
    b_route = B["Flight Distance"].values
    yA = A["satisfaction"].values.astype(np.float32)

    out_d = np.full(len(B), 99, dtype=np.float32)
    out_lab = np.full(len(B), 0.5, dtype=np.float32)
    out_mean3 = np.full(len(B), 0.5, dtype=np.float32)
    out_w = np.full(len(B), 0.5, dtype=np.float32)

    order = np.argsort(a_route, kind="stable")
    a_sorted = a_route[order]
    for r in np.unique(b_route):
        lo, hi = np.searchsorted(a_sorted, r, "left"), np.searchsorted(a_sorted, r, "right")
        if hi <= lo:
            continue
        idx = order[lo:hi]
        rowsel = np.where(b_route == r)[0]
        D = (F - XB[rowsel] @ XA[idx].T)  # Hamming distance
        if exclude_self and len(rowsel) == len(idx):
            D[np.arange(len(rowsel)), np.arange(len(idx))] = 1e6
        nn = np.argmin(D, axis=1)
        d = D[np.arange(len(rowsel)), nn]
        out_d[rowsel] = d
        out_lab[rowsel] = yA[idx][nn]
        # mean of the 3 nearest
        k3 = np.argsort(D, axis=1)[:, :3]
        out_mean3[rowsel] = yA[idx][k3].mean(1)
        out_w[rowsel] = (yA[idx][k3] * (1.0 / (1.0 + D[np.arange(len(rowsel))[:, None], k3]))).sum(1) / \
                        (1.0 / (1.0 + D[np.arange(len(rowsel))[:, None], k3])).sum(1)
    return pd.DataFrame({"nn_dist": out_d, "nn_lab": out_lab, "nn3_lab": out_mean3, "nn_w": out_w})


def main() -> None:
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y = df.satisfaction.astype(int).values
    cols = [c for c in df.columns if c not in ("id", "satisfaction")]
    A, B = train_test_split(df, test_size=0.28, random_state=0, stratify=y)
    log(f"A {len(A)} B {len(B)}")

    A2 = A.copy()
    cats = build_cats(A2, cols)
    B2 = B.copy()
    for c in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
        B2[c] = np.digitize(np.nan_to_num(B2[c].values, nan=-1), [0.5, 15, 60, 180])
    feats = knn_feats(A2, B2, cols, cats)
    log("neighbour features done")
    log(f"nn_dist quantiles {np.percentile(feats.nn_dist, [10, 50, 90])}")

    # numeric copies for the GBM (route features are already computed on the raw frame)
    A2 = A2.copy()
    B2 = B2.copy()
    for c in cols:
        if A2[c].dtype == object:
            codes = {v: i for i, v in enumerate(cats[c])}
            A2[c] = A2[c].map(codes).astype(np.int16)
            B2[c] = B2[c].map(codes).astype(np.int16)

    base = [c for c in cols]
    yA, yB = A2.satisfaction.astype(int).values, B2.satisfaction.astype(int).values
    params = dict(n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    m0 = lgb.LGBMClassifier(**params).fit(A2[base], yA)
    a0 = roc_auc_score(yB, m0.predict_proba(B2[base])[:, 1])
    log(f"base AUC {a0:.6f}")

    featsA = knn_feats(A.copy(), A.copy(), cols, cats, exclude_self=True)
    m1 = lgb.LGBMClassifier(**params).fit(pd.concat([A2[base].reset_index(drop=True), featsA], axis=1), yA)
    XB = pd.concat([B2[base].reset_index(drop=True), feats], axis=1)
    a1 = roc_auc_score(yB, m1.predict_proba(XB)[:, 1])
    log(f"base+knn AUC {a1:.6f}   delta {a1 - a0:+.6f}")


if __name__ == "__main__":
    main()
