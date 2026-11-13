"""Stacking experiments for Playground S6E10.

Loads every published OOF/test pair, then compares meta-learners under one honest
nested protocol. The member pool is the community's, so the only thing measured
here is how much the combiner adds on top of it.

Run:  python3 src/stack_experiments.py
"""
from __future__ import annotations

import glob
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

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()
BLEND = re.compile(r"stack|blend|_ens|ens_v|meta|consensus|climb|probe|pool", re.I)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def to_logit(v: np.ndarray) -> np.ndarray:
    """Map predictions to log-odds; vectors already outside [0, 1] are logits."""
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.min() < 0.0 or v.max() > 1.0:
        return np.clip(v, -30, 30)
    p = np.clip(v, 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -30, 30)


def load_all(n_train: int, n_test: int):
    """Return member names, OOF logits and test logits from every pool directory."""
    names, oofs, tests = [], [], []
    dirs = [os.path.join(ROOT, "data", "oof_pool"), os.path.join(ROOT, "data", "own_pool")]
    seen = set()
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "oof_*.npy"))):
            stem = os.path.basename(f)[4:-4]
            tf = os.path.join(d, f"test_{stem}.npy")
            if stem in seen or not os.path.exists(tf):
                continue
            try:
                o = np.load(f).ravel().astype(np.float64)
                t = np.load(tf).ravel().astype(np.float64)
            except Exception:
                continue
            if len(o) != n_train or len(t) != n_test or not (np.isfinite(o).all() and np.isfinite(t).all()):
                continue
            names.append(stem)
            oofs.append(to_logit(o))
            tests.append(to_logit(t))
            seen.add(stem)
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            try:
                z = np.load(f)
            except Exception:
                continue
            for key in z.files:
                if not key.startswith("oof_"):
                    continue
                stem = os.path.basename(f)[:-4] + ":" + key[4:]
                tkey = "test_" + key[4:]
                if stem in seen or tkey not in z.files:
                    continue
                o = z[key].ravel().astype(np.float64)
                t = z[tkey].ravel().astype(np.float64)
                if len(o) != n_train or len(t) != n_test:
                    continue
                if not (np.isfinite(o).all() and np.isfinite(t).all()):
                    continue
                names.append(stem)
                oofs.append(to_logit(o))
                tests.append(to_logit(t))
                seen.add(stem)
    return names, oofs, tests


def matrix(vecs: list[np.ndarray]) -> np.ndarray:
    """Column-standardise logits so one L2 strength applies uniformly."""
    Z = np.vstack(vecs).T
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return Z.astype(np.float32)


def nested_lr(Z: np.ndarray, y: np.ndarray, C: float, seed: int = 7):
    """Nested 5-fold logistic regression; returns AUC, OOF decisions and fold list."""
    oof = np.zeros(len(y))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y))
    for tr, va in folds:
        m = LogisticRegression(C=C, max_iter=4000, n_jobs=-1).fit(Z[tr], y[tr])
        oof[va] = m.decision_function(Z[va])
    return roc_auc_score(y, oof), oof, folds


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), len(te))
    log(f"members on disk: {len(names)}")

    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = np.argsort(-aucs)
    log("best single members: " + ", ".join(f"{names[i]}={aucs[i]:.5f}" for i in order[:8]))
    log(f"member AUC: max {aucs.max():.5f} median {np.median(aucs):.5f} min {aucs.min():.5f}")

    clean = [i for i, n in enumerate(names) if not BLEND.search(n)]
    withstack = list(range(len(names)))
    report = {}

    X = matrix([oofs[i] for i in clean])
    XT = matrix([tests[i] for i in clean])
    log(f"clean pool {X.shape}")
    best = (0.0, None, None)
    for C in (0.003, 0.01, 0.03, 0.1, 0.3):
        auc, oof, folds = nested_lr(X, y, C)
        report[f"lr_C{C}"] = auc
        log(f"LR C={C:<5} nested {auc:.6f}")
        if auc > best[0]:
            best = (auc, C, "clean")
    log(f"best clean: C={best[1]} {best[0]:.6f}")

    Xs = matrix([oofs[i] for i in withstack])
    XTs = matrix([tests[i] for i in withstack])
    for C in (best[1], 0.1):
        auc, _, _ = nested_lr(Xs, y, C)
        report[f"lr_withstack_C{C}"] = auc
        log(f"LR+stack-of-stacks C={C:<5} nested {auc:.6f}")

    # Keep the artefacts of the best clean configuration.
    model = LogisticRegression(C=best[1], max_iter=5000, n_jobs=-1).fit(X, y)
    pred = 1 / (1 + np.exp(-model.decision_function(XT)))
    out = os.path.join(ROOT, "outputs", "submission_stack_clean.csv")
    pd.DataFrame({"id": te.id.values, "satisfaction": pred}).to_csv(out, index=False)
    log(f"wrote {out}")

    with open(os.path.join(ROOT, "outputs", "stack_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
