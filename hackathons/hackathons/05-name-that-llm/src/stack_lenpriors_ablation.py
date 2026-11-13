"""Track 05 n9 leg-2 ablation: WHERE does the boost come from?

Isolating blocks:
  A1  S54 + one-hot(BIN real)                 C=32, C=1000
  A2  S54 + r(real)                            C=32
  A3  S54 + one-hot(BIN permuted)              C=32
  A4  S54 + r(permuted)                        C=32
  A5  post-hoc alpha grid {0.2..3.0} on anchor OOF with real train tables
  Plus: max|r| diagnostics real vs permuted, per-fold BA for A1.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stack_lenpriors import (BIN, CACHE, CLEAN, COMP, FOLDS, LABELS, NB, NC,  # noqa: E402
                             S54, Y, build_features, count_table, class_marginal,
                             cross_fit_lr, log, ratio_table, score)
from run_experiments import score as _score  # noqa: E402  (same fn)


def r_max(tag: str, m: float = 10.0) -> float:
    ti, vi = FOLDS[0]
    T = count_table(ti)
    r = ratio_table(T, T.sum(1), class_marginal(ti), m)
    log(f"  {tag}: r_validation scale max|r|={np.abs(r).max():.3f}")
    return float(np.abs(r).max())


def block_only(which: str, real: bool, m: float = 10.0) -> np.ndarray:
    """44/58-dim: S54 plus exactly one feature block."""
    mod = sys.modules["stack_lenpriors"]
    orig = mod.BIN.copy()
    if not real:
        mod.BIN = np.random.default_rng(11).permutation(orig)
    full = build_features(m)
    mod.BIN = orig
    if which == "onehot":
        return np.hstack([S54, full[:, 54:54 + NB]])
    return np.hstack([S54, full[:, 54 + NB:]])


def main() -> dict:
    out: dict = {}
    anchor = score(cross_fit_lr(S54.copy(), 8.0))
    log(f"anchor {anchor:.5f}")
    r_max("real-r", 10.0)
    r_max("permuted-r", 10.0)

    oof_a = cross_fit_lr(S54.copy(), 8.0)
    for alpha in (0.5, 1.0, 1.5, 2.0, 3.0):
        adj = np.zeros_like(oof_a)
        for f, (ti, vi) in enumerate(FOLDS):
            T = count_table(ti)
            r = ratio_table(T, T.sum(1), class_marginal(ti), 10.0)[BIN[vi]]
            adj[vi] = oof_a[vi] * np.clip(np.exp(alpha * r), 1e-9, None)
            adj[vi] /= np.clip(adj[vi].sum(1, keepdims=True), 1e-12, None)
        out[f"posthoc_a{alpha:g}"] = score(adj)
        log(f"  posthoc alpha={alpha:g} -> {out[f'posthoc_a{alpha:g}']:.5f}")

    for name, args in (("A1_onehot_real", ("onehot", True, 10.0, 32.0)),
                       ("A1b_onehot_real", ("onehot", True, 10.0, 1000.0)),
                       ("A2_r_real", ("r", True, 10.0, 32.0)),
                       ("A3_onehot_perm", ("onehot", False, 10.0, 32.0)),
                       ("A3b_onehot_perm", ("onehot", False, 10.0, 1000.0)),
                       ("A5_r_real_m1e6", ("r", True, 1e6, 32.0))):
        which, real, m, C = args
        F = block_only(which, real, m)
        v = score(cross_fit_lr(F, C))
        out[f"{name}_c{C:g}"] = v
        log(f"  {name} C={C:g} m={m:g} -> {v:.5f}")

    # per-fold BA for A1 at C=32
    oof = cross_fit_lr(block_only("onehot", True, 10.0), 32.0)
    folds = [score(oof, np.isin(np.arange(len(COMP)), vi) & CLEAN) for _, vi in FOLDS]
    out["A1_fold_ba"] = [round(b, 4) for b in folds]
    log(f"  A1 fold BA {out['A1_fold_ba']}")

    json.dump(out, open("/tmp/opencode/lenprior_ablation.json", "w"), indent=1)
    log("ablation done")
    return out


if __name__ == "__main__":
    main()
