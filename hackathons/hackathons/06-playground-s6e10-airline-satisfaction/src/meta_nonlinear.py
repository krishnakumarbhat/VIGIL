"""Nonlinear and greedy meta-learners, scored on the same nested folds as the linear stack.

Competitors on the public plateau all combine the same members; the only remaining
freedom is the combiner. This script measures whether anything beats a lean logistic
regression, using top-40 members and identical folds.

Run:  python3 src/meta_nonlinear.py
"""
from __future__ import annotations

import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from stack_experiments import ROOT, load_all, matrix

warnings.filterwarnings("ignore")
T0 = time.time()
N_TEST = 299844


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def caruana(Z: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, steps: int = 40) -> np.ndarray:
    """Greedy ensemble selection with replacement (Caruana), on rank-normalised members."""
    from scipy.stats import rankdata
    Ztr = np.apply_along_axis(rankdata, 0, Z[tr]) / len(tr)
    Zva = np.apply_along_axis(rankdata, 0, Z[va]) / len(va)
    counts = np.zeros(Z.shape[1])
    cur = np.zeros(len(va))
    best_hist = []
    for step in range(1, steps + 1):
        scores = np.array([roc_auc_score(y[va], (cur * (step - 1) + Zva[:, j]) / step) for j in range(Z.shape[1])])
        j = int(np.argmax(scores))
        counts[j] += 1
        cur = (cur * (step - 1) + Zva[:, j]) / step
        best_hist.append(scores[j])
    return counts / counts.sum(), best_hist[-1]


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), N_TEST)
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = np.argsort(-aucs)[:40]
    sel = [names[i] for i in order]
    X = matrix([oofs[i] for i in order])
    XT = matrix([tests[i] for i in order])
    log(f"members {X.shape}")

    skf = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y))
    report: dict[str, float] = {}

    # 1. linear reference
    oof = np.zeros(len(y))
    for a, b in skf:
        oof[b] = LogisticRegression(C=0.3, max_iter=4000, n_jobs=-1).fit(X[a], y[a]).decision_function(X[b])
    report["lr40"] = roc_auc_score(y, oof)
    log(f"LR-40            {report['lr40']:.6f}")

    # 2. LightGBM stacker
    oof = np.zeros(len(y))
    for a, b in skf:
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                               n_jobs=-1, verbose=-1)
        m.fit(X[a], y[a])
        oof[b] = m.predict_proba(X[b])[:, 1]
    report["lgb40"] = roc_auc_score(y, oof)
    log(f"LGBM-40          {report['lgb40']:.6f}")

    # 3. greedy Caruana
    oof = np.zeros(len(y))
    for a, b in skf:
        w, _ = caruana(X, y, a, b, steps=40)
        oof[b] = X[b] @ w
    report["caruana40"] = roc_auc_score(y, oof)
    log(f"Caruana-40       {report['caruana40']:.6f}")

    # 4. 50/50 logit average of LR and LGBM stackers is often the most stable
    lr_oof = np.zeros(len(y))
    gb_oof = np.zeros(len(y))
    for a, b in skf:
        lr_oof[b] = LogisticRegression(C=0.3, max_iter=4000, n_jobs=-1).fit(X[a], y[a]).decision_function(X[b])
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                               n_jobs=-1, verbose=-1)
        m.fit(X[a], y[a])
        gb_oof[b] = np.log(np.clip(m.predict_proba(X[b])[:, 1], 1e-9, 1 - 1e-9) /
                           (1 - np.clip(m.predict_proba(X[b])[:, 1], 1e-9, 1 - 1e-9)))
    for w in (0.3, 0.5, 0.7):
        z = w * lr_oof + (1 - w) * gb_oof
        report[f"mix{w}"] = roc_auc_score(y, z)
        log(f"LR/LGBM mix {w:.1f}  {report[f'mix{w}']:.6f}")

    with open(os.path.join(ROOT, "outputs", "meta_nonlinear.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
