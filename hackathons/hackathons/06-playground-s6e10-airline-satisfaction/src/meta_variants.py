"""Meta-learner variants: subset size, rank transform, segment gating, nonlinear stackers.

Everything is scored with the same nested 5-fold protocol (seed 7) so the numbers are
comparable to ``stack_experiments.py``. Nothing here trains a base model.

Run:  python3 src/meta_variants.py
"""
from __future__ import annotations

import json
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


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def nested(Z: np.ndarray, y: np.ndarray, C: float, seed: int = 7) -> tuple[float, np.ndarray]:
    """Nested logistic regression over fixed folds."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(Z, y):
        m = LogisticRegression(C=C, max_iter=4000, n_jobs=-1).fit(Z[tr], y[tr])
        oof[va] = m.decision_function(Z[va])
    return roc_auc_score(y, oof), oof


def nested_gated(Z: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, seed: int = 7) -> float:
    """One logistic regression per passenger segment, nested over the same folds."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(Z, y):
        for g in np.unique(groups):
            mtr = tr[groups[tr] == g]
            mva = va[groups[va] == g]
            if len(mva) == 0:
                continue
            if len(mtr) < 500 or len(np.unique(y[mtr])) < 2:
                oof[mva] = 0.0
                continue
            m = LogisticRegression(C=C, max_iter=4000, n_jobs=-1).fit(Z[mtr], y[mtr])
            oof[mva] = m.decision_function(Z[mva])
    return roc_auc_score(y, oof)


def rank_transform(Z: np.ndarray) -> np.ndarray:
    """Map every column to normal scores so no member dominates through scale alone."""
    from scipy.stats import norm
    R = np.empty_like(Z)
    n = Z.shape[0]
    for j in range(Z.shape[1]):
        order = np.argsort(Z[:, j], kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1) / (n + 1)
        R[:, j] = norm.ppf(ranks)
    return R.astype(np.float32)


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     usecols=["satisfaction", "Class", "Type of Travel", "Customer Type"])
    y = tr.satisfaction.astype(int).values
    names, oofs, _ = load_all(len(y), 299844)
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = np.argsort(-aucs)
    report: dict[str, float] = {}

    Z_full = matrix(oofs)
    for k in (10, 20, 40, 80, 148):
        idx = order[:k]
        for C in (0.1, 0.3):
            auc, _ = nested(Z_full[:, idx], y, C)
            key = f"top{k}_C{C}"
            report[key] = auc
            log(f"{key:16s} {auc:.6f}")

    Z_rank = rank_transform(Z_full)
    for k in (40, 148):
        idx = order[:k]
        auc, _ = nested(Z_rank[:, idx], y, 0.1)
        report[f"rank_top{k}"] = auc
        log(f"rank_top{k:<9d} {auc:.6f}")

    seg = (tr["Class"].astype(str) + "|" + tr["Type of Travel"].astype(str) + "|"
           + tr["Customer Type"].astype(str)).values
    for k in (40, 148):
        idx = order[:k]
        auc = nested_gated(Z_full[:, idx], y, seg, 0.3)
        report[f"gated_top{k}"] = auc
        log(f"gated_top{k:<8d} {auc:.6f}")

    with open(os.path.join(ROOT, "outputs", "meta_variants.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    best = max(report, key=report.get)
    log(f"BEST {best} = {report[best]:.6f}")


if __name__ == "__main__":
    main()
