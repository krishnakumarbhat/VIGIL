"""Combine several meta-learners (lean, mid, wide) and pick the blend by nested OOF.

Different member subsets make different mistakes, so averaging their OOF decision
functions usually beats any single one. Weights come from a coarse simplex grid,
never from the leaderboard.

Run:  python3 src/meta_blend.py --out outputs/submission_meta_blend.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from stack_experiments import ROOT, load_all, matrix

warnings.filterwarnings("ignore")
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def fit_config(Z: np.ndarray, y: np.ndarray, folds, C: float) -> tuple[np.ndarray, np.ndarray]:
    """Return nested OOF decisions and the full-fit decision function on XT."""
    oof = np.zeros(len(y))
    for a, b in folds:
        oof[b] = LogisticRegression(C=C, max_iter=4000, n_jobs=-1).fit(Z[a], y[a]).decision_function(Z[b])
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission_meta_blend.csv"))
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), len(te))
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = np.argsort(-aucs)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(y, y))

    configs = {"lean10": (order[:10], 0.3), "mid40": (order[:40], 0.3), "wide": (order[:212], 0.3)}
    parts, tests_parts = {}, {}
    for tag, (idx, C) in configs.items():
        Z = matrix([oofs[i] for i in idx])
        ZT = matrix([tests[i] for i in idx])
        parts[tag] = fit_config(Z, y, folds, C)
        tests_parts[tag] = ZT
        log(f"{tag:7s} n={len(idx):3d} nested {roc_auc_score(y, parts[tag]):.6f}")

    # weights on the 3-way simplex (0.05 grid)
    best = (0.0, None)
    grid = np.arange(0, 1.0001, 0.05)
    for w1, w2 in itertools.product(grid, grid):
        w3 = 1 - w1 - w2
        if w3 < -1e-9:
            continue
        o = w1 * parts["lean10"] + w2 * parts["mid40"] + w3 * parts["wide"]
        a = roc_auc_score(y, o)
        if a > best[0]:
            best = (a, (w1, w2, round(float(w3), 2)))
    log(f"best blend {best[1]} nested {best[0]:.6f}")

    w = best[1]
    test_pred = np.zeros(len(te))
    for wt, tag in zip(w, ("lean10", "mid40", "wide")):
        ZT = tests_parts[tag]
        m = LogisticRegression(C=configs[tag][1], max_iter=5000, n_jobs=-1).fit(
            matrix([oofs[i] for i in configs[tag][0]]), y)
        test_pred += wt * m.decision_function(ZT)
    pred = 1 / (1 + np.exp(-test_pred))
    pd.DataFrame({"id": te.id.values, "satisfaction": pred}).to_csv(args.out, index=False)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
