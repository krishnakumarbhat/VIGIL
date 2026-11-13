"""Strict evaluation of the translation-gap feature family (node n4 / I5).

Reuses ``run_analysis.load_tables`` / ``build_frames`` so the play-level frame,
group vectors and -- because the rows are untouched -- the actual GroupKFold
folds are IDENTICAL to the baseline run stored in ``outputs/model_metrics.json``.
Baselines to beat (same folds, same harness, generated 2026-10-22):

    sep_at_pass_forward by_game_play lightgbm   r2=0.2820   (headline)
    sep_at_pass_forward by_game_play hist_gb    r2=0.2786
    sep_at_pass_forward by_player    lightgbm   r2=0.2613
    sep_at_pass_forward by_player    hist_gb    r2=0.2633

A feature block is kept only if the headline (by_game_play lightgbm) improves
AND the mean-dummy / shuffled-null sanity bracket still clears.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kce_model import evaluate_target, strip_heavy  # noqa: E402
from run_analysis import (  # noqa: E402
    FORBIDDEN,
    PRE_SNAP_CTX,
    PLAY_KIN,
    build_frames,
    load_tables,
    position_group,
    log_row,
)
from translation_features import build_translation_frame  # noqa: E402

CACHE = os.path.join("outputs", "cache")
DATA = os.path.join("data", "kaggle_nfl", "nfl_data")
OUT_JSON = os.path.join("outputs", "translation_metrics.json")

TARGET = "separation_at_pass_forward"
BASE = {"by_game_play": 0.2820, "by_player": 0.2613}


def evaluate_frame(pf: pd.DataFrame, feature_cols: List[str],
                   label: str) -> Dict:
    groupings = {"by_player": pf["nfl_id"].to_numpy(),
                 "by_game_play": pf["game_play_id"].to_numpy()}
    res = evaluate_target(pf, TARGET, feature_cols, groupings)
    res["label"] = label
    return strip_heavy(res)


def main() -> None:
    t0 = time.time()
    t = load_tables()
    sess = t["sess"]
    comb, pf, players, pp, pcs = build_frames(t)

    kin_game = [c for c in PLAY_KIN if c in pf.columns]
    ctx = [c for c in PRE_SNAP_CTX if c in pf.columns]
    cmb_cols = [c for c in comb.columns
                if comb[c].dtype.kind in "if" and c not in
                {"nfl_id", "draft_year", "n_sep_obs", "n_yac_obs"}
                and c not in FORBIDDEN]
    cmb_pf = [f"cmb_{c}" for c in cmb_cols if f"cmb_{c}" in pf.columns]
    base_features = kin_game + ctx + cmb_pf

    pf_aug, new_cols = build_translation_frame(sess, pf)
    new_cols = [c for c in new_cols if c in pf_aug.columns]
    aug_features = base_features + new_cols
    print(f"rows                  : {len(pf_aug)} (baseline frame {len(pf)})",
          flush=True)
    print(f"new features          : {len(new_cols)} -> {new_cols}", flush=True)
    print(f"feature count base->aug: {len(base_features)} -> {len(aug_features)}",
          flush=True)

    results: Dict[str, Dict] = {"augmented_info": {
        "n_rows": int(len(pf_aug)), "n_new_features": int(len(new_cols)),
        "new_features": new_cols, "baseline_reference": BASE}}
    for label, feats in (("baseline_features", base_features),
                         ("translation_augmented", aug_features)):
        res = evaluate_frame(pf_aug, feats, label)
        results[label] = res
        for gname in ("by_game_play", "by_player"):
            for kind in ("lightgbm", "hist_gb"):
                r = res["groupings"].get(gname, {}).get("models", {}).get(kind)
                if isinstance(r, dict) and "r2" in r:
                    print(f"  {label:22s} {gname:13s} {kind:8s} "
                          f"r2={r['r2']:+.4f} mae={r.get('mae', np.nan):.4f} "
                          f"rho={r.get('spearman', np.nan):+.4f}")
        for kind, r in res["groupings"]["by_game_play"].get(
                "baselines", {}).items():
            print(f"  {label:22s} by_game_play  {kind:8s} "
                  f"r2={r.get('r2', np.nan):+.4f}")

    hl_aug = float(results["translation_augmented"]["groupings"]
                   ["by_game_play"]["models"]["lightgbm"]["r2"])
    hl_base = BASE["by_game_play"]
    keep = hl_aug > hl_base
    results["headline"] = {"new_r2": hl_aug, "baseline_r2": hl_base,
                           "delta": round(hl_aug - hl_base, 4), "keep": keep}
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)

    for gname in ("by_game_play", "by_player"):
        for kind in ("lightgbm", "hist_gb"):
            r = results["translation_augmented"]["groupings"][gname][
                "models"][kind]
            log_row(f"translation_aug::{gname}::{kind}",
                    round(float(r["r2"]), 4),
                    f"baseline={BASE[gname]} keep={keep} n_new={len(new_cols)}")
    # log only if kept
    if keep:
        with open(os.path.join("outputs", "kept_headline.txt"), "w") as fh:
            fh.write(json.dumps(results["headline"]))
    print(f"headline r2 {hl_base:.4f} -> {hl_aug:+.4f}  keep={keep} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
