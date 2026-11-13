"""Track 05 n9 leg-2 drill 2: control-integrity + gain decomposition.

1. Table integrity: max |P(c|b) - P(c)| for fold-0 real vs PERMUTED bins.
   If the permuted table is NOT near-marginal the negative control was void.
2. A4: r(permuted)-only, C=32 -> does the r-block itself produce the gain
   without any real bin information?
3. m-sweep of A2 (r-real-only): {5, 10, 40, 200, 1000} at C=32.
4. Per-class recall: anchor C=8 vs A2 r-real C=32 (where does macro gain sit).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stack_lenpriors import (BIN, CLEAN, COMP, FOLDS, LABELS, NB, NC, S54, Y,  # noqa: E402
                             build_features, count_table, class_marginal,
                             cross_fit_lr, log, score)
from run_experiments import score as _score  # noqa: E402


def table_dev(real: bool) -> float:
    mod = sys.modules["stack_lenpriors"]
    orig = mod.BIN.copy()
    if not real:
        mod.BIN = np.random.default_rng(11).permutation(orig)
    ti, _ = FOLDS[0]
    T = count_table(ti)
    N_b, P_c = T.sum(1), class_marginal(ti)
    dev = float(np.abs(T / N_b[:, None] - P_c[None, :]).max())
    mod.BIN = orig
    log(f"  table {'REAL' if real else 'PERM'}: max|P(c|b)-P(c)| = {dev:.4f}")
    return dev


def block(which: str, real: bool, m: float) -> np.ndarray:
    mod = sys.modules["stack_lenpriors"]
    orig = mod.BIN.copy()
    if not real:
        mod.BIN = np.random.default_rng(11).permutation(orig)
    full = build_features(m)
    mod.BIN = orig
    if which == "r":
        return np.hstack([S54, full[:, 54 + NB:]])
    if which == "onehot":
        return np.hstack([S54, full[:, 54:54 + NB]])
    return full


def recalls(oof: np.ndarray) -> list[float]:
    pred = oof.argmax(1)
    return [float((pred[CLEAN][Y[CLEAN] == c] == c).mean()) for c in range(NC)]


def main() -> dict:
    out: dict = {}
    out["dev_real"] = table_dev(True)
    out["dev_perm"] = table_dev(False)

    anchor_oof = cross_fit_lr(S54.copy(), 8.0)
    rec_a = recalls(anchor_oof)

    oof_a2 = cross_fit_lr(block("r", True, 10.0), 32.0)
    rec_b = recalls(oof_a2)
    out["recall_anchor_c8"] = dict(zip(LABELS, [round(r, 3) for r in rec_a]))
    out["recall_r_real_c32"] = dict(zip(LABELS, [round(r, 3) for r in rec_b]))
    D = np.array(rec_b) - np.array(rec_a)
    log("  recall delta (rreal-anchor): " +
        ", ".join(f"{l.split('-')[0][:8]}={d:+.3f}" for l, d in zip(LABELS, D)))

    out["A4_r_perm_c32"] = score(cross_fit_lr(block("r", False, 10.0), 32.0))
    log(f"  A4 r-PERMUTED-only C=32 -> {out['A4_r_perm_c32']:.5f}")

    out["rreal_m_sweep"] = {}
    for m in (5.0, 10.0, 40.0, 200.0, 1000.0):
        k = f"m{m:g}"
        out["rreal_m_sweep"][k] = score(cross_fit_lr(block("r", True, m), 32.0))
        log(f"  r-real-only m={m:g} C=32 -> {out['rreal_m_sweep'][k]:.5f}")

    json.dump(out, open("/tmp/opencode/lenprior_drill2.json", "w"), indent=1)
    log("drill2 done")
    return out


if __name__ == "__main__":
    main()
