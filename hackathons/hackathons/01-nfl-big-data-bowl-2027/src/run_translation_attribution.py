"""Attribution ablation for the translation-gap families (lightgbm-only).

Splits the kept augmentation into its two causal families and re-tests each
against the SAME baseline feature set, folds and harness (``run_cv`` called
directly; mean-dummy bracket included, shuffled-null omitted for speed -- the
null is feature-set independent and was already logged in both prior runs).

    family sx_  : within-athlete combine session variability (repeatability)
    family translate_ : combine <-> game translation gaps on the play level
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kce_model import run_cv, strip_heavy  # noqa: E402
from run_analysis import (  # noqa: E402
    PRE_SNAP_CTX, PLAY_KIN, build_frames, load_tables, log_row,
)
from translation_features import build_translation_frame  # noqa: E402

OUT_JSON = os.path.join("outputs", "translation_attribution.json")
TARGET = "separation_at_pass_forward"


def _numeric_design(frame, cols):
    """Column selection + dummy expansion (same rules as prepare_matrix)."""
    import pandas as pd
    X = frame[cols].copy()
    obj = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]
    if obj:
        X = pd.get_dummies(X, columns=obj, dummy_na=False, dtype=float)
    return X


def oof_score(y: np.ndarray, oof: np.ndarray) -> Dict:
    m = np.isfinite(oof) & np.isfinite(y)
    from sklearn.metrics import r2_score
    return {"r2": float(r2_score(y[m], oof[m]))}


def cv_pair(X, y, groups, label, out) -> None:
    """One feature set: lgbm + mean dummy, by_player and by_game_play.

    ``X``/``y`` are already filtered to target-present rows; ``groups`` maps a
    grouping name to that ALREADY-MASKED group vector.
    """
    res: Dict[str, Dict] = {}
    for gname, g in groups.items():
        r = run_cv(X, y, g, "lightgbm")
        d = strip_heavy({k: r[k] if k != "oof" and k != "y_true" else None
                         for k in ("r2", "mae", "spearman")})
        mean_r = run_cv(X, y, g, "mean")
        d["mean_dummy_r2"] = float(mean_r["r2"])
        res[gname] = d
        print(f"  {label:12s} {gname:13s} lgbm r2={d['r2']:+.4f} "
              f"mae={d['mae']:.4f} rho={d['spearman']:+.4f}", flush=True)
    out[label] = res


def main() -> None:
    t0 = time.time()
    t = load_tables()
    comb, pf, players, pp, pcs = build_frames(t)

    kin_game = [c for c in PLAY_KIN if c in pf.columns]
    ctx = [c for c in PRE_SNAP_CTX if c in pf.columns]
    cmb_cols = [c for c in comb.columns
                if comb[c].dtype.kind in "if" and c not in
                {"nfl_id", "draft_year", "n_sep_obs", "n_yac_obs"}
                and c not in FORBIDDEN] if False else None  # silence lint
    from run_analysis import FORBIDDEN
    cmb_cols = [c for c in comb.columns
                if comb[c].dtype.kind in "if" and c not in
                {"nfl_id", "draft_year", "n_sep_obs", "n_yac_obs"}
                and c not in FORBIDDEN]
    base = kin_game + ctx + [f"cmb_{c}" for c in cmb_cols
                             if f"cmb_{c}" in pf.columns]

    pf_aug, new_cols = build_translation_frame(t["sess"], pf)
    sx = [c for c in new_cols if c.startswith("sx_")]
    tr = [c for c in new_cols if c.startswith("translate_")]

    # target matrix: rows with the target present only (mask applied once,
    # to X/y AND the group vectors, everywhere downstream)
    mask = pf[TARGET].notna().to_numpy()
    y = pf[TARGET].to_numpy(float)[mask]
    groups = {"by_player": pf["nfl_id"].to_numpy()[mask],
              "by_game_play": pf["game_play_id"].to_numpy()[mask]}

    variants: Dict[str, List[str]] = {
        "baseline": base,
        "sx_only": base + [c for c in sx if c in pf_aug.columns and
                           pf_aug[c].dtype.kind in "if"],
        "translate_only": base + tr,
    }
    results: Dict[str, Dict] = {}
    frames: Dict[str, List[str]] = {
        "baseline": [c for c in variants["baseline"] if c in pf.columns],
        "sx_only": [c for c in variants["sx_only"] if c in pf_aug.columns],
        "translate_only": [c for c in variants["translate_only"]
                           if c in pf_aug.columns],
    }
    # NOTE: do NOT filter these lists to numeric dtypes -- the categorical
    # context features (route_ran, alignment, formation, positions) are
    # one-hot-expanded later by _numeric_design and belong in EVERY variant;
    # a numeric-only filter silently removed them, invalidating comparisons.
    for label, _cols in frames.items():
        frame = pf_aug if label != "baseline" else pf
        X = _numeric_design(frame.loc[mask], frames[label])
        print(f"== {label}: {X.shape[1]} features", flush=True)
        cv_pair(X, y, groups, label, results)

    # both families together (fast re-score of the kept variant, lgbm only)
    cols = frames["sx_only"] + [c for c in frames["translate_only"]
                                if c not in frames["sx_only"]]
    X = _numeric_design(pf_aug.loc[mask], cols)
    print(f"== both: {X.shape[1]} features", flush=True)
    res: Dict[str, Dict] = {}
    for gname, g in groups.items():
        r = run_cv(X, y, g, "lightgbm")
        d = {"r2": float(r["r2"]), "mae": float(r["mae"]),
             "spearman": float(r["spearman"])}
        res[gname] = d
        print(f"  {'both':12s} {gname:13s} lgbm r2={d['r2']:+.4f} "
              f"mae={d['mae']:.4f} rho={d['spearman']:+.4f}", flush=True)
    results["both"] = res

    with open(OUT_JSON, "w") as fh:
        json.dump({"variants": results,
                   "runtime_s": time.time() - t0}, fh, indent=2)
    base_hl = results["baseline"]["by_game_play"]["r2"]
    for label in ("sx_only", "translate_only", "both"):
        hl = results[label]["by_game_play"]["r2"]
        bp = results[label]["by_player"]["r2"]
        log_row(f"translation_attrib::{label}", round(hl, 4),
                f"headline_delta={round(hl - base_hl, 4)} "
                f"by_player={round(bp, 4)} (lgbm, identical folds)")
    print(f"attribution done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
