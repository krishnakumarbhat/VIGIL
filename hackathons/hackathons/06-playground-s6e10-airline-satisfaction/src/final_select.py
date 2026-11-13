"""Final lean-stack selection over the full member pool.

Rebuilds the pool from every downloaded dataset (npy and npz), then sweeps the number
of members and the L2 strength under nested 5-fold CV. Writes the winning submission.

Run:  python3 src/final_select.py
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


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), len(te))
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = list(np.argsort(-aucs))
    log(f"pool {len(names)}; top: " + ", ".join(f"{names[i]}={aucs[i]:.5f}" for i in order[:6]))

    report = {}
    best = (0.0, None)
    for k in (6, 8, 10, 12, 15, 20, 25, 35, 60):
        Z = matrix([oofs[i] for i in order[:k]])
        for C in (0.1, 0.3, 1.0):
            auc, _, _ = nested_lr(Z, y, C)
            report[f"k{k}_C{C}"] = auc
            if auc > best[0]:
                best = (auc, (k, C))
    for key in sorted(report, key=report.get, reverse=True)[:10]:
        log(f"{key:12s} {report[key]:.6f}")
    log(f"BEST {best[1]} {best[0]:.6f}")

    k, C = best[1]
    sel = order[:k]
    X = matrix([oofs[i] for i in sel])
    XT = matrix([tests[i] for i in sel])
    model = LogisticRegression(C=C, max_iter=5000, n_jobs=-1).fit(X, y)
    pred = 1 / (1 + np.exp(-model.decision_function(XT)))
    out = os.path.join(ROOT, "outputs", "submission_final_select.csv")
    pd.DataFrame({"id": te.id.values, "satisfaction": pred}).to_csv(out, index=False)
    json.dump({"best": best[0], "k": int(k), "C": float(C),
               "members": [names[i] for i in sel], "sweep": report},
              open(os.path.join(ROOT, "outputs", "final_select.json"), "w"), indent=1)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
