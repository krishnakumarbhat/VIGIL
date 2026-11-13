"""Build a kNN-augmented member: nearest same-route passengers in the training fold carry labels.

The generator keeps synthetic rows close to real passengers, so for most rows a handful of
other rows in the same Flight Distance agree on nearly every column. Those neighbours'
labels are legitimate features as long as they come from the training part of the fold.

Outputs pool-format artefacts: data/own_pool/oof_knn_lgb.npy and test_knn_lgb.npy.

Run:  python3 src/build_knn_member.py
"""
from __future__ import annotations

import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from knn_signal_test import build_cats, encode

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def neighbour_block(A: pd.DataFrame, B: pd.DataFrame, kcols: list[str], cats, exclude_self=False):
    """Return per-B-row neighbour statistics computed from A only."""
    XA = encode(A, kcols, cats)
    XB = encode(B, kcols, cats)
    F = len(kcols)
    a_route = A["Flight Distance"].values
    b_route = B["Flight Distance"].values
    yA = A["satisfaction"].values.astype(np.float32)
    n = len(B)
    out = np.zeros((n, 7), dtype=np.float32)
    order = np.argsort(a_route, kind="stable")
    a_sorted = a_route[order]

    for r in np.unique(b_route):
        lo, hi = np.searchsorted(a_sorted, r, "left"), np.searchsorted(a_sorted, r, "right")
        rowsel = np.where(b_route == r)[0]
        if hi <= lo:
            out[rowsel] = [F, 0.5, 0.5, 0.5, 0, 0, 0.5]
            continue
        idx = order[lo:hi]
        D = F - XB[rowsel] @ XA[idx].T
        if exclude_self and len(rowsel) == len(idx):
            D[np.arange(len(rowsel)), np.arange(len(idx))] = 1e6
        srt = np.argsort(D, axis=1)
        d1 = D[np.arange(len(rowsel)), srt[:, 0]]
        lab1 = yA[idx][srt[:, 0]]
        k3 = srt[:, :3]
        lab3 = yA[idx][k3].mean(1)
        wts = 1.0 / (1.0 + D[np.arange(len(rowsel))[:, None], k3])
        labw = (yA[idx][k3] * wts).sum(1) / wts.sum(1)
        out[rowsel, 0] = d1
        out[rowsel, 1] = lab1
        out[rowsel, 2] = lab3
        out[rowsel, 3] = labw
        out[rowsel, 4] = (D <= 3).sum(1)
        out[rowsel, 5] = (D <= 5).sum(1)
        out[rowsel, 6] = yA[idx].mean()
    return pd.DataFrame(out, columns=["nn_dist", "nn_lab", "nn3_lab", "nn_w", "cnt3", "cnt5", "route_lab"])


def base_features(df: pd.DataFrame, cats) -> pd.DataFrame:
    """Raw columns plus cheap aggregates; categoricals become integer codes."""
    out = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    rating = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
              "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
              "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
    out["rating_sum"] = out[rating].sum(1)
    out["rating_zero"] = (out[rating] == 0).sum(1)
    out["rating_mean_nz"] = out[rating].replace(0, np.nan).mean(1)
    out["delay_diff"] = out["Arrival Delay in Minutes"] - out["Departure Delay in Minutes"]
    out["arr_nan"] = out["Arrival Delay in Minutes"].isna().astype(int)
    for c in ("Gender", "Customer Type", "Type of Travel", "Class"):
        m = {v: i for i, v in enumerate(cats[c])}
        out[c] = out[c].map(m).astype(np.int16)
    return out


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = tr.satisfaction.astype(int).values
    cols = [c for c in te.columns if c != "id"]
    kcols = [c for c in cols if c != "Flight Distance"]
    cats = build_cats(tr.copy(), cols)

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    Ftr = base_features(tr, cats)
    Fte = base_features(te, cats)
    nn_tr = pd.DataFrame(0.0, index=range(len(tr)), columns=["nn_dist", "nn_lab", "nn3_lab", "nn_w", "cnt3", "cnt5", "route_lab"])
    for fi, (a, b) in enumerate(folds):
        nn_tr.iloc[b] = neighbour_block(tr.iloc[a], tr.iloc[b], kcols, cats).values
        log(f"fold {fi} neighbour features done")
    nn_te = neighbour_block(tr, te, kcols, cats)
    log("test neighbour features done")

    Xtr = pd.concat([Ftr.reset_index(drop=True), nn_tr.reset_index(drop=True)], axis=1)
    Xte = pd.concat([Fte.reset_index(drop=True), nn_te.reset_index(drop=True)], axis=1)
    log(f"features {Xtr.shape}")

    params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    for fi, (a, b) in enumerate(folds):
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr.iloc[a], y[a], eval_set=[(Xtr.iloc[b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
        test_pred += m.predict_proba(Xte)[:, 1] / len(folds)
        log(f"fold {fi} auc {roc_auc_score(y[b], oof[b]):.6f} best_iter {m.best_iteration_}")
    log(f"OOF AUC {roc_auc_score(y, oof):.6f}")

    os.makedirs(os.path.join(ROOT, "data", "own_pool"), exist_ok=True)
    np.save(os.path.join(ROOT, "data", "own_pool", "oof_knn_lgb.npy"), oof)
    np.save(os.path.join(ROOT, "data", "own_pool", "test_knn_lgb.npy"), test_pred)
    log("saved oof_knn_lgb.npy / test_knn_lgb.npy")


if __name__ == "__main__":
    main()
