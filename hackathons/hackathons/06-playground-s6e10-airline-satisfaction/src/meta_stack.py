"""Stage-1 meta stack for Playground S6E10.

Collects every published OOF/test prediction pair on disk, converts probabilities
to logits, prunes unusable or redundant members, and fits an L2 logistic regression
with nested cross-validation so the reported score is honest.

Run:  python3 src/meta_stack.py --out outputs/submission_meta.csv
"""
from __future__ import annotations

import argparse
import glob
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


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def to_logit(v: np.ndarray) -> np.ndarray:
    """Map a prediction vector to log-odds. Columns already outside [0, 1] are logits."""
    v = np.asarray(v, dtype=np.float64)
    if v.min() < 0.0 or v.max() > 1.0:
        return np.clip(v, -30, 30)
    p = np.clip(v, 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -30, 30)


def load_pool(dirs: list[str], n_train: int, n_test: int) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Scan directories for aligned ``oof_*.npy`` / ``test_*.npy`` pairs."""
    names, oofs, tests = [], [], []
    seen: set[str] = set()
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "oof_*.npy"))):
            stem = os.path.basename(f)[4:-4]
            if stem in seen:
                continue
            tf = os.path.join(d, f"test_{stem}.npy")
            if not os.path.exists(tf):
                continue
            try:
                o = np.load(f, mmap_mode="r")
                t = np.load(tf, mmap_mode="r")
            except Exception:
                continue
            if o.shape[0] != n_train or t.shape[0] != n_test:
                continue
            o = np.asarray(o, dtype=np.float64).ravel()
            t = np.asarray(t, dtype=np.float64).ravel()
            if not (np.isfinite(o).all() and np.isfinite(t).all()):
                continue
            names.append(f"{os.path.basename(d)}:{stem}")
            oofs.append(o)
            tests.append(t)
            seen.add(stem)
    return names, oofs, tests


def build_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Standardise each logit column so a single L2 penalty means the same thing for all."""
    Z = np.vstack([to_logit(v) for v in vectors]).T
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return Z.astype(np.float32)


def nested_auc(Z: np.ndarray, y: np.ndarray, C: float, seed: int = 7) -> tuple[float, np.ndarray]:
    """Nested 5-fold logistic regression; returns AUC and the honest OOF score vector."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(Z, y):
        m = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(Z[tr], y[tr])
        oof[va] = m.decision_function(Z[va])
    return roc_auc_score(y, oof), oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission_meta.csv"))
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--keep-blends", action="store_true", help="keep blend-level members (stacked stacks)")
    ap.add_argument("--min-auc", type=float, default=0.94)
    args = ap.parse_args()
    BLEND = re.compile(r"stack|blend|_ens|ens_v|meta|consensus|climb|probe|pool", re.I)

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["id", "satisfaction"])
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"])
    y = tr.satisfaction.astype(int).values
    n_train, n_test = len(tr), len(te)
    log(f"train {n_train} test {n_test} pos_rate {y.mean():.4f}")

    dirs = [os.path.join(ROOT, "data", "oof_pool")] + args.extra
    names, oofs, tests = load_pool(dirs, n_train, n_test)
    log(f"loaded {len(names)} member pairs")

    # --- prune: weak members and near-duplicate members -------------------------------
    keep_names, keep_o, keep_t, seen_sig = [], [], [], []
    for n, o, t in zip(names, oofs, tests):
        if not args.keep_blends and BLEND.search(n.split(":")[-1]):
            continue
        auc = roc_auc_score(y, o)
        if auc < args.min_auc:
            continue
        sig = np.round(to_logit(o)[:2000], 4)
        if any(np.allclose(sig, s, atol=1e-3) for s in seen_sig):
            continue
        seen_sig.append(sig)
        keep_names.append(n)
        keep_o.append(o)
        keep_t.append(t)
    log(f"kept {len(keep_names)} members after AUC/duplicate pruning")

    X = build_matrix(keep_o)
    XT = build_matrix(keep_t)
    log(f"matrix {X.shape}")

    results = {}
    for C in (0.003, 0.01, 0.03, 0.1):
        auc, _ = nested_auc(X, y, C)
        results[C] = auc
        log(f"C={C:<6} nested AUC {auc:.6f}")
    best_C = max(results, key=results.get)
    log(f"best C={best_C} AUC={results[best_C]:.6f}")

    model = LogisticRegression(C=best_C, max_iter=5000, n_jobs=-1).fit(X, y)
    pred = 1 / (1 + np.exp(-model.decision_function(XT)))
    sub = pd.DataFrame({"id": te.id.values, "satisfaction": pred})
    sub.to_csv(args.out, index=False)
    log(f"wrote {args.out}  ({len(sub)} rows, range {pred.min():.4f}-{pred.max():.4f})")

    coef = pd.Series(np.abs(model.coef_[0]), index=keep_names).sort_values(ascending=False)
    log("top weights: " + ", ".join(f"{k}={v:.3f}" for k, v in coef.head(12).items()))


if __name__ == "__main__":
    main()
