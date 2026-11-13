"""Track 05 n9 leg-2 FINAL disambiguation.

P-A: scale-matched per-row random r-block (breaks the shared-per-bin structure,
     keeps magnitude ~ real tables) at C=32 -> if ~0.50, the gain is a
     weight-geometry channel and isVOID as a length-prior effect.
P-B: prompt-split honesty check: select the recipe on TUNE prompts (255,
     grouped 5-fold restricted eval), then measure ONCE on EST prompts (255).
     Repeated-peeking selection optimism cannot inflate EST rows.
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
from stack_lenpriors import (BIN, CLEAN, COMP, FOLDS, GROUP, LABELS, NB, NC,  # noqa: E402
                             S54, Y, build_features, count_table, class_marginal,
                             cross_fit_lr, log, ratio_table, score)
from run_experiments import score as _score  # noqa: E402


def main() -> dict:
    out: dict = {}
    anchor = score(cross_fit_lr(S54.copy(), 8.0))
    log(f"anchor {anchor:.5f}")

    # ---- P-A: per-row random r-block at matched scale -------------------
    full = build_features(10.0)
    R = full[:, 54 + NB:]
    scale = float(np.abs(R).max())
    rng = np.random.default_rng(0)
    R_rand = rng.uniform(-scale, scale, size=R.shape)
    F_rand = np.hstack([S54, R_rand])
    out["PA_random_rblock_c32"] = score(cross_fit_lr(F_rand, 32.0))
    log(f"  P-A random-rblock C=32 -> {out['PA_random_rblock_c32']:.5f}")
    out["PA_random_rblock_c8"] = score(cross_fit_lr(F_rand, 8.0))
    log(f"  P-A random-rblock C=8 -> {out['PA_random_rblock_c8']:.5f}")

    # ---- P-B: prompt-split tune/est ------------------------------------
    uniq = pd.Series(GROUP).unique()
    rng2 = np.random.default_rng(5)
    perm = rng2.permutation(len(uniq))
    tune_p, est_p = set(uniq[perm[:len(uniq) // 2]]), set(uniq[perm[len(uniq) // 2:]])
    tune_m = np.isin(GROUP, list(tune_p))
    est_m = np.isin(GROUP, list(est_p))
    log(f"  P-B tune rows {tune_m.sum()}, est rows {est_m.sum()}")

    # tune: cross-fit ONLY over folds, scoring restricted to tune rows
    grid = {}
    for tag, m in (("full_m10", 10.0), ("ronly_m10", 11.0)):
        F = build_features(10.0) if tag == "full_m10" else np.hstack(
            [S54, build_features(10.0)[:, 54 + NB:]])
        for C in (8.0, 32.0):
            oof = cross_fit_lr(F, C)
            grid[f"{tag}_c{C:g}"] = score(oof, tune_m & CLEAN)
            log(f"  P-B TUNE {tag} C={C:g} -> {grid[f'{tag}_c{C:g}']:.5f}")
    oof_anchor = cross_fit_lr(S54.copy(), 8.0)
    grid["anchor_c8"] = score(oof_anchor, tune_m & CLEAN)
    log(f"  P-B TUNE anchor C=8 -> {grid['anchor_c8']:.5f}")
    best_tag = max((k for k in grid if k != "anchor_c8"), key=grid.get)
    log(f"  P-B selected on TUNE: {best_tag}")

    F_best = build_features(10.0) if best_tag.startswith("full") else np.hstack(
        [S54, build_features(10.0)[:, 54 + NB:]])
    C_best = 32.0 if "c32" in best_tag else 8.0
    oof_best = cross_fit_lr(F_best, C_best)
    est_best = score(oof_best, est_m & CLEAN)
    est_anchor = score(oof_anchor, est_m & CLEAN)
    out["PB"] = {"selected": best_tag, "est_best": est_best, "est_anchor": est_anchor,
                 "est_delta": est_best - est_anchor,
                 "tune_grid": grid}
    log(f"  P-B EST: best {est_best:.5f} vs anchor {est_anchor:.5f} "
        f"(delta {est_best - est_anchor:+.5f})")

    json.dump(out, open("/tmp/opencode/lenprior_final.json", "w"), indent=1)
    log("final drill done")
    return out


if __name__ == "__main__":
    main()
