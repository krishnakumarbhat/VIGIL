"""Track 05 n9 leg-2 VERDICT test: fully decoupled prior tables.

Hypothesis for the anomaly: r-features for train rows (LO tables of fold f) and
for validation rows (full fold-f table) derive from OVERLAPPING row sets, so
each fold's stacker can fit fold-level table noise that its own validation
rows reproduce (shared-noise memorisation channel).  Test that channel while
preserving any REAL length-bin signal:

  For each fold: split ti -> A (positioo even) / B (odd).
    - validation rows: r-table from A rows only (NO label-bearing contact
      with the stacker's fit set B).
    - stacker fit rows: B rows only, with LO-within-B tables.
    - anchor: identical budget, S54-only fit on B rows.
  Delta(BA on clean val rows) is compared for REAL vs PERMUTED bins:
    real >> perm  -> length-bin prior signal is real (decoupled transfer)  -> KEEP-path
    real ~= perm  -> shared-noise memorisation channel; signal void        -> DISCARD
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
from stack_lenpriors import (CLEAN, COMP, FOLDS, LABELS, NB, NC, S54, Y,  # noqa: E402
                             count_table, class_marginal, ratio_table, log, score)


def decoupled_run(real: bool, m: float, C: float, seed: int = 11) -> float:
    mod = sys.modules["stack_lenpriors"]
    orig = mod.BIN.copy()
    if not real:
        mod.BIN = np.random.default_rng(seed).permutation(orig)
    try:
        pred_oof = np.zeros((len(COMP), NC), dtype=np.float32)
        for fold_i, (ti, vi) in enumerate(FOLDS):
            rng = np.random.default_rng(100 + fold_i)
            shuf = rng.permutation(ti)
            tiA, tiB = shuf[0::2], shuf[1::2]
            # validation table: A rows only (independent of the fit set B)
            TA = count_table(tiA)
            r_val = ratio_table(TA, TA.sum(1), class_marginal(tiA), m)
            r_map = r_val[mod.BIN[vi]]                       # (len(vi), NC)

            # fit rows: B rows only, LO-within-B tables
            per_prompt = {}
            for p, sub in pd.DataFrame(
                    {"g": COMP["group"].to_numpy()[tiB],
                     "b": mod.BIN[tiB], "y": Y[tiB]}).groupby("g"):
                t = np.zeros((NB, NC))
                np.add.at(t, (sub["b"].to_numpy(), sub["y"].to_numpy()), 1.0)
                per_prompt[p] = t
            TB = count_table(tiB)
            PB = TB.sum(0) / TB.sum()
            Xb = np.hstack([S54[tiB], np.eye(NB)[mod.BIN[tiB]],
                            np.zeros((len(tiB), NC))])
            Rb = np.zeros((len(tiB), NC))
            for pos, p in enumerate(COMP["group"].to_numpy()[tiB]):
                lo = TB - per_prompt[p]
                Rb[pos] = ratio_table(lo, lo.sum(1), lo.sum(0) / lo.sum(), m)[mod.BIN[tiB][pos]]
            Fb = np.hstack([S54[tiB], Xb[:, 54:54 + NB], Rb])
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            clf.fit(Fb, Y[tiB])
            pred_oof[vi] = clf.predict_proba(
                np.hstack([S54[vi], np.eye(NB)[mod.BIN[vi]], r_map]))

        # score on clean validation rows
        return float(balanced_accuracy_score(Y[CLEAN], pred_oof[CLEAN].argmax(1)))
    finally:
        mod.BIN = orig


def decoupled_anchor(C: float) -> float:
    pred_oof = np.zeros((len(COMP), NC), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        rng = np.random.default_rng(100 + f)
        shuf = rng.permutation(ti)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(S54[shuf[1::2]], Y[shuf[1::2]])
        pred_oof[vi] = clf.predict_proba(S54[vi])
    return float(balanced_accuracy_score(Y[CLEAN], pred_oof[CLEAN].argmax(1)))


def main() -> dict:
    out: dict = {}
    for C in (8.0, 32.0):
        out[f"anchor_dec_c{C:g}"] = decoupled_anchor(C)
        log(f"  decoupled anchor C={C:g} -> {out[f'anchor_dec_c{C:g}']:.5f}")
        real = decoupled_run(True, 10.0, C)
        perm = decoupled_run(False, 10.0, C)
        out[f"real_dec_c{C:g}"] = real
        out[f"perm_dec_c{C:g}"] = perm
        log(f"  decoupled REAL C={C:g} -> {real:.5f} "
            f"(delta {real - out[f'anchor_dec_c{C:g}']:+.5f})")
        log(f"  decoupled PERM C={C:g} -> {perm:.5f} "
            f"(delta {perm - out[f'anchor_dec_c{C:g}']:+.5f})")

    json.dump(out, open("/tmp/opencode/lenprior_verdict.json", "w"), indent=1)
    log("verdict test done")
    return out


if __name__ == "__main__":
    main()
