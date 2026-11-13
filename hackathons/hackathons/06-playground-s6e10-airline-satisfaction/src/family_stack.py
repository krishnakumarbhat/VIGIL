"""Two-level stack: level-1 per model family, level-2 over the family experts.

Family experts are fitted inside every CV fold, so the level-2 score is honest.
The bet is that a family's own weighting is more stable than one global weighting
over 200 correlated columns.

Run:  python3 src/family_stack.py
"""
from __future__ import annotations

import json
import os
import re
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
FAMILIES = {
    "realmlp": r"rmlp|realmlp",
    "cat": r"cat",
    "lgb": r"lgb",
    "xgb": r"xgb",
    "tfm": r"tabpfn|pfn|tabicl|tabm|tfm|deepfm|tabfm",
    "route": r"route|_og|ogx",
    "stack": r"stack|blend|ens_v|meta",
}


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), 299844)
    aucs = np.array([roc_auc_score(y, o) for o in oofs])

    fam_idx: dict[str, list[int]] = {}
    for fam, pat in FAMILIES.items():
        idx = [i for i, n in enumerate(names) if re.search(pat, n, re.I) and aucs[i] > 0.952]
        fam_idx[fam] = sorted(idx, key=lambda i: -aucs[i])[:30]
        log(f"{fam:8s} {len(idx):3d} members (using {len(fam_idx[fam])})")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(y, y))
    fams = [f for f in FAMILIES if fam_idx[f]]
    L1_oof = np.zeros((len(y), len(fams)))
    L1_test = np.zeros((299844, len(fams)))
    for fi, fam in enumerate(fams):
        Z = matrix([oofs[i] for i in fam_idx[fam]])
        ZT = matrix([tests[i] for i in fam_idx[fam]])
        for a, b in folds:
            m = LogisticRegression(C=0.3, max_iter=4000, n_jobs=-1).fit(Z[a], y[a])
            L1_oof[b, fi] = m.decision_function(Z[b])
        m = LogisticRegression(C=0.3, max_iter=4000, n_jobs=-1).fit(Z, y)
        L1_test[:, fi] = m.decision_function(ZT)
        log(f"L1 {fam:8s} nest {roc_auc_score(y, L1_oof[:, fi]):.6f}")

    Z2 = (L1_oof - L1_oof.mean(0)) / (L1_oof.std(0) + 1e-12)
    Z2T = (L1_test - L1_test.mean(0)) / (L1_test.std(0) + 1e-12)
    for C in (0.1, 0.3, 1.0, 10.0):
        oof = np.zeros(len(y))
        for a, b in folds:
            m = LogisticRegression(C=C, max_iter=4000, n_jobs=-1).fit(Z2[a], y[a])
            oof[b] = m.decision_function(Z2[b])
        log(f"L2 C={C:<5} nest {roc_auc_score(y, oof):.6f}")

    # equal-weight level-2 as a robust reference
    log(f"L2 equal-weight nest {roc_auc_score(y, Z2.mean(1)):.6f}")


if __name__ == "__main__":
    main()
