"""Track 05 n9 leg-2 adversarial verification for stack_lenpriors (KEEP due-diligence).

1. Saturation: extend the C grid past the selected edge (C=32).
2. Negative control: row-wise permutation of the length-bin labels (same bin
   multiset) destroys the true P(y|bin) association while keeping every other
   property of the features -> a REAL bin-prior signal must collapse back to
   the 0.40863 anchor; anything else means the number is artifact- or
   leak-driven and the keep is void.
3. Fold stability + all-rows BA at the candidate config.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stack_lenpriors import (BIN, CACHE, CLEAN, COMP, FOLDS, LABELS, NB, NC,  # noqa: E402
                             S54, Y, build_features, cross_fit_lr, log, score)


def permuted_bins(seed: int) -> None:
    """Monke patch-free negative control: rewire module-level BIN."""
    rng = np.random.default_rng(seed)
    mod = sys.modules["stack_lenpriors"]
    orig = mod.BIN.copy()
    mod.BIN = rng.permutation(orig)
    try:
        F = build_features(10.0)
        oof = cross_fit_lr(F, 32.0)
        return score(oof)
    finally:
        mod.BIN = orig


def main() -> None:
    anchor = score(cross_fit_lr(S54.copy(), 8.0))
    log(f"anchor {anchor:.5f}")

    # ---- 1) saturation beyond the selected C=32 edge (m=10) ----
    sat: dict[str, float] = {}
    for C in (32.0, 64.0, 256.0, 1000.0):
        oof = cross_fit_lr(build_features(10.0), C)
        sat[f"c{C:g}_m10"] = score(oof)
        log(f"  saturation C={C:g} -> {sat[f'c{C:g}_m10']:.5f}")

    # ---- fold stability at C=32 (build again for deterministic reuse) ----
    oof32 = cross_fit_lr(build_features(10.0), 32.0)
    fold_ba = [score(oof32, (np.isin(np.arange(len(COMP)), vi)) & CLEAN)
               for _, vi in FOLDS]
    log(f"  fold clean BA at C=32 m=10: {[round(b, 4) for b in fold_ba]}")

    # ---- 2) negative control: bin-label permutation (deterministic seeds) ----
    ctrl = [permuted_bins(s) for s in (11, 12)]
    log(f"  negative control (permuted bins): {[round(c, 5) for c in ctrl]}  "
        f"vs anchor {anchor:.5f}")

    real = sat["c32_m10"]
    voided = min(ctrl) > anchor + 0.002          # control must NOT beat anchor
    out = {
        "anchor": anchor, "saturation": sat,
        "fold_ba_c32m10": [round(b, 4) for b in fold_ba],
        "negative_control": [round(c, 5) for c in ctrl],
        "control_collapses_to_anchor": bool(not voided),
        "real_is_signal": bool(not voided and real > anchor + 0.002)}
    log(f"verdict {out}")
    json.dump(out, open("/tmp/opencode/lenprior_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
