"""Translation-gap feature family: combine <- game deltas + session variability.

Hypothesis (node n4 / idea I5 extension)
---------------------------------------
Separation at the pass-forward instant depends on *how a route compares with
the athlete's own trained movement profile*, which no existing feature
expresses.  Two families are added:

1. ``translate_*`` deltas / ratios on the play level:
   ``game kinematics - combine kinematics`` (same athlete), plus
   ``game / combine`` speed ratio -- the *translation gap* between a player's
   in-game route execution and his combine clinic execution.
2. ``sx_*`` session-consistency features from the combine drill sessions:
   within-athlete std, best-vs-mean gap and coefficient of variation of the
   headline kinematics -- a direct measure of movement *repeatability*, the
   property scouts call "consistent release".

Every feature is athlete-pre-snap-information only: they use tracking data
recorded BEFORE the draft and current-play pre-snap kinematics up to the pass
forward instant, never the target or any outcome field.

Sampling caveat, documented also in DATA_SCHEMA.md: ``game_tracking_2023`` is
the sampled public release (1.6 athletes / play), so *defender-relative*
features are not reconstructable; these features are receiver-side only.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EPS = 1e-9

# combine_player.csv column name -> play_features.csv column name.
# Each pair measures the "same" quantity in the two settings; deltas are
# game - combine, ratios are game / (combine + eps).
DELTA_PAIRS: Dict[str, str] = {
    "mean_speed": "mean_speed",
    "max_speed": "max_speed",
    "p95_speed": "p95_speed",
    "path_efficiency": "path_efficiency",
    "net_displacement": "net_displacement",
    "lateral_displacement": "lateral_displacement",
    "peak_decel": "peak_decel",
    "mean_turn_rate": "mean_turn_rate",
    "path_length": "path_length",
}
# special-purpose pairs with different names on each side
SPECIAL_PAIRS: Dict[str, Tuple[str, str, str]] = {
    # name: (cmb_col, game_col, kind)  kind in {"delta", "ratio"}
    "kce_route_vs_drill": ("mean_kce", "mean_pre_pass_kce", "delta"),
    "dbr_route_vs_drill": ("mean_dbr", "mean_pre_pass_dbr", "delta"),
    "burst_route_vs_combine": ("mean_burst_peak", "release_burst", "delta"),
}
# combine_sessions columns turned into within-athlete variability features
SESSION_STATS = (
    "mean_speed", "max_speed", "path_efficiency", "mean_kce", "mean_dbr",
    "mean_burst_peak", "mean_decel_peak", "mean_cut_turn_deg", "duration_s",
)


def combine_session_variability(sess: pd.DataFrame) -> pd.DataFrame:
    """Within-athlete session variability from the combine drill sessions.

    Per athlete: std, (max - mean) best-gap, and coeff. of variation
    (std / (mean + eps)) for each headline kinematic, plus the session count
    distribution across drill bases.  One row per athlete.
    """
    rows = []
    for nfl_id, g in sess.groupby("nfl_id"):
        row: Dict[str, object] = {"nfl_id": nfl_id}
        for c in SESSION_STATS:
            if c not in g.columns:
                continue
            v = pd.to_numeric(g[c], errors="coerce")
            mu, sd, mx = float(v.mean()), float(v.std(ddof=1)), float(v.max())
            row[f"sx_{c}_std"] = sd
            row[f"sx_{c}_gap"] = mx - mu
            row[f"sx_{c}_cv"] = sd / (abs(mu) + EPS)
        row["sx_drill_bases"] = int(g["drill_base"].nunique()) \
            if "drill_base" in g.columns else 0
        row["sx_n_sessions"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def add_translation_deltas(pf: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add combine<->game translation-gap columns to the play-level frame.

    Returns the frame and the list of newly created columns (in deterministic
    order).  Missing combine values propagate as NaN and are handled by the
    harness's train-fold median imputation, exactly like every other
    combine feature.
    """
    out = pf.copy()
    new_cols: List[str] = []
    for cmb_col, game_col in DELTA_PAIRS.items():
        if cmb_col not in out.columns or game_col not in out.columns:
            continue
        name = f"translate_{game_col}_minus_cmb"
        out[name] = out[game_col].astype(float) - out[cmb_col].astype(float)
        new_cols.append(name)
    for name, (cmb_col, game_col, kind) in SPECIAL_PAIRS.items():
        if cmb_col not in out.columns or game_col not in out.columns:
            continue
        if kind == "delta":
            out[f"translate_{name}"] = (out[game_col].astype(float)
                                        - out[cmb_col].astype(float))
        else:  # ratio
            out[f"translate_{name}"] = (out[game_col].astype(float)
                                        / (out[cmb_col].astype(float).abs() + EPS))
        new_cols.append(f"translate_{name}")
    if "max_speed" in out.columns and "cmb_max_speed" in out.columns:
        name = "translate_speed_retained_ratio"
        out[name] = (out["max_speed"].astype(float)
                     / (out["cmb_max_speed"].astype(float).abs() + EPS))
        new_cols.append(name)
    return out, new_cols


def build_translation_frame(sess: pd.DataFrame, pf: pd.DataFrame
                            ) -> Tuple[pd.DataFrame, List[str]]:
    """Full new-feature block: merge session variability + translation deltas.

    Rows of ``pf`` are untouched and the merge is 1:1 on ``nfl_id`` (each
    athlete appears exactly once in the session-variability table, validated),
    so downstream GroupKFold folds -- which depend only on the group vectors --
    stay identical to the baseline evaluation.
    """
    var = combine_session_variability(sess)
    if "nfl_id" in var.columns and var["nfl_id"].duplicated().any():
        var = var.drop_duplicates("nfl_id")
    var_pref = var.add_prefix("sx_")
    var_pref = var_pref.rename(columns={"sx_nfl_id": "nfl_id"})
    sx_cols = [c for c in var_pref.columns if c not in {"nfl_id"}]
    out = pf.merge(var_pref, on="nfl_id", how="left", validate="many_to_one")
    out, delta_cols = add_translation_deltas(out)
    return out, ["nx_" + c for c in []] + sx_cols + delta_cols
