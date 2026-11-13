"""Lean-stack sweep with correlation de-duplication.

The pool contains many near-copies (same recipe, different seed). A wide logistic
regression spends capacity on those copies, so members are thinned to one representative
per correlation cluster before the k-sweep. Scored with nested 5-fold (seed 7).

Run:  python3 src/lean_sweep.py
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from stack_experiments import ROOT, load_all, matrix, nested_lr

warnings.filterwarnings("ignore")
T0 = time.time()
BLEND_FLAG = ("stack", "blend", "ens_", "meta", "consensus", "climb", "probe", "pool")


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def dedup(names, oofs, aucs, order, thr=0.9995, n_sample=50_000, seed=0):
    """Keep the strongest member of each cluster of |corr| > thr, on a row sample."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(oofs[0]), size=min(n_sample, len(oofs[0])), replace=False)
    S = np.vstack([oofs[i][rows] for i in range(len(oofs))])
    S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)
    kept = []
    for i in order:
        if kept and np.max(np.abs(S[kept] @ S[i]) / len(rows)) > thr:
            continue
        kept.append(i)
    return kept


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), 299844)
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = list(np.argsort(-aucs))
    log(f"pool {len(names)}")

    clean_idx = [i for i in order if not any(f in names[i] for f in BLEND_FLAG)]
    clusters = dedup(names, oofs, aucs, clean_idx)
    log(f"after cluster dedup: {len(clusters)} (from {len(clean_idx)} clean)")

    report = {}
    best = (0.0, None)
    for k in (6, 8, 10, 12, 15, 20, 30):
        sel = clusters[:k]
        Z = matrix([oofs[i] for i in sel])
        for C in (0.1, 0.3, 1.0):
            auc, _, _ = nested_lr(Z, y, C)
            report[f"k{k}_C{C}"] = auc
            log(f"k={k:<3d} C={C:<5} {auc:.6f}")
            if auc > best[0]:
                best = (auc, (k, C))
    log(f"BEST {best[1]} -> {best[0]:.6f}")
    with open(os.path.join(ROOT, "outputs", "lean_sweep.json"), "w") as fh:
        json.dump({"best": report[f"k{best[1][0]}_C{best[1][1]}"], "config": best[1], "all": report}, fh, indent=1)


if __name__ == "__main__":
    main()
