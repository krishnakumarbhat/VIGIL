"""Stage 2: targets, models, metrics, figures and the experiment log.

Everything here runs on the REAL Kaggle tables (cached by ``src/build_tables.py``).
Eight target formulations are evaluated with identical machinery and all of them
are reported in ``outputs/model_metrics.json``; the headline is chosen by
measured OOF R^2 and documented honestly.

Targets
-------
1. ``sep_at_pass_forward``  -- official player_play field, play level
   (REAL target; predicts separation at the pass-forward instant from
   pre-pass route kinematics + combine athleticism).
2. ``player_mean_sep``      -- same field averaged per athlete (REAL target,
   cross-domain: combine kinematics -> in-game separation, one row per athlete).
3. ``sep_gain``             -- separation_at_pass_forward - cushion (REAL fields,
   "how much separation the route created after the snap").  Because ``cushion``
   is algebraically inside this target, it is always reported together with
   ``sep_gain_no_cushion`` (same rows/folds, ``cushion`` removed) so the size of
   the algebraic contribution is measured, not hidden.
4. ``derived_sep_pf``       -- PROXY derived from game_tracking: distance to the
   nearest tracked opponent at the pass-forward frame, validated against (1)
   before use (Pearson/Spearman reported).
5. ``career_snaps_log``     -- log(1 + career_offensive_snaps), cross-domain
   sanity target, one row per athlete.
6. ``derived_sep_catch``    -- PROXY: distance to the nearest *tracked* opponent
   at the catch frame (same construction as (4)); validated against (1) and
   reported whatever the outcome.
7. ``yards_after_catch``    -- official player_play field, play level on
   targeted+caught plays (REAL target, outcome after the catch).
0. ``legacy_player_features_on_play_target`` -- reproduces the earlier
   formulation (player-level features against a play-level target) so the
   improvement is measured, not asserted.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kce_features import add_normalised_features  # noqa: E402
from kce_model import (  # noqa: E402
    evaluate_target,
    prepare_matrix,
    run_cv,
    run_mixedlm,
    run_mixedlm_cv,
    strip_heavy,
)

DATA = os.path.join("data", "kaggle_nfl", "nfl_data")
CACHE = os.path.join("outputs", "cache")
FIG_DIR = os.path.join("outputs", "figures")
METRICS_OUT = os.path.join("outputs", "model_metrics.json")
LOG_TSV = os.path.join("logs", "experiments.tsv")
CHECKPOINT = os.path.join(CACHE, "analysis_checkpoint.json")
OOF_NPZ = os.path.join(CACHE, "oof_store.npz")
TRACK = "01_nfl_big_data_bowl"
# every target result is appended here so a run killed by a wall-clock limit
# can resume instead of starting over (see _load_checkpoint / _save_checkpoint)
OOF_STORE: Dict[str, dict] = {}


def _save_checkpoint(metrics: dict) -> None:
    """Atomically persist partial results (metrics JSON is always usable)."""
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(strip_heavy(metrics), fh, default=float)
    os.replace(tmp, CHECKPOINT)
    with open(METRICS_OUT, "w") as fh:
        json.dump(strip_heavy(metrics), fh, indent=2, default=float)
    try:
        save_oof(OOF_STORE, OOF_NPZ)
    except Exception as exc:
        print(f"  oof persist skipped: {exc}", flush=True)


def _load_checkpoint(metrics: dict) -> None:
    """Merge results from a previous (possibly killed) run and restore OOF."""
    if not os.path.exists(CHECKPOINT):
        return
    try:
        with open(CHECKPOINT) as fh:
            ck = json.load(fh)
    except Exception as exc:
        print(f"checkpoint unreadable: {exc}", flush=True)
        return
    for k, v in ck.items():
        metrics.setdefault(k, v)
    if os.path.exists(OOF_NPZ):
        try:
            reinject_oof_arrays(metrics, OOF_NPZ)
        except Exception as exc:
            print(f"oof reinject skipped: {exc}", flush=True)
    done = [k for k in metrics if isinstance(metrics[k], dict)
            and "groupings" in metrics[k]]
    if done:
        print(f"resuming: {len(done)} target(s) already measured "
              f"({', '.join(done)})", flush=True)


def reinject_oof_arrays(metrics: dict, path: str) -> int:
    """Put stored OOF / y_true arrays back into an unstripped metrics dict."""
    if not os.path.exists(path):
        return 0
    z = np.load(path)
    n = 0
    for key in z.files:
        parts = key.split("__")
        if len(parts) != 3:
            continue
        name, gname, field = parts
        if field == "y_true" or name not in metrics:
            continue
        bucket = "models" if field in ("lightgbm", "hist_gb") else "baselines"
        g = metrics[name].get("groupings", {}).get(gname)
        if not g:
            continue
        slot = (g.get(bucket) or {}).get(field)
        if isinstance(slot, dict):
            slot["oof"] = np.asarray(z[key], dtype=float).tolist()
            if field == "lightgbm":
                yk = f"{name}__{gname}__y_true"
                if yk in z.files:
                    slot["y_true"] = np.asarray(z[yk], dtype=float).tolist()
            n += 1
    return n

# ---------------------------------------------------------------- feature sets --
PLAY_KIN = [
    "mean_speed", "max_speed", "p95_speed", "time_to_peak_speed",
    "frac_above_80pct_peak", "peak_accel", "peak_decel", "mean_jerk", "p95_jerk",
    "mean_turn_rate", "max_turn_rate", "path_length", "net_displacement",
    "path_efficiency", "lateral_displacement", "stop_count", "start_count",
    "moving_frac", "release_time_s", "release_burst", "speed_at_pf",
    "a_long_at_pf", "total_turn_deg", "snap_to_pf_s", "n_pre_pass_cuts",
    "has_pre_pass_cut", "mean_cut_turn_deg", "mean_pre_pass_kce",
    "mean_pre_pass_dbr",
]
PRE_SNAP_CTX = [
    "cushion", "route_ran", "receiver_alignment", "in_motion_at_ball_snap",
    "down", "yards_to_go", "quarter", "play_direction", "offense_formation",
    "lined_up_position", "position_group",
]
MEASURABLES = [
    "forty", "three_cone", "short_shuttle", "ten_yd_split", "vertical",
    "broad_jump", "bench_reps", "combine_height", "combine_weight", "hand_size",
    "arm_length", "wing_span", "ngs_athleticism_score",
]
# outcome fields that must never be used as features
FORBIDDEN = {
    "separation_at_pass_forward", "cushion_target", "rec_yards", "yards_after_catch",
    "expected_yards_after_catch", "pass_result", "pass_length", "yards_gained",
    "expected_points_added", "target", "cushion_gain", "derived_sep_pf",
    "player_mean_sep", "career_offensive_snaps", "sep_gain",
    "player_mean_yac", "player_mean_cushion", "player_mean_gain",
    "derived_sep_catch",
}
# pre-registered small feature set for the n=145 athlete-level separation
# target (fixed before looking at any score, to keep p >> n in check)
PREREG_PLAYER_SEP = [
    "forty", "three_cone", "short_shuttle", "ten_yd_split", "vertical",
    "broad_jump", "bench_reps", "ngs_athleticism_score", "combine_height",
    "combine_weight", "mean_speed", "max_speed", "p95_speed",
    "path_efficiency", "peak_accel", "mean_turn_rate", "n_cuts",
]


def log_row(metric: str, value, notes: str = "") -> None:
    """Append one measured experiment row to logs/experiments.tsv."""
    os.makedirs(os.path.dirname(LOG_TSV), exist_ok=True)
    new = not os.path.exists(LOG_TSV)
    with open(LOG_TSV, "a") as fh:
        if new:
            fh.write("track\tmetric\tvalue\tnotes\n")
        fh.write(f"{TRACK}\t{metric}\t{value}\t{str(notes).replace(chr(9), ' ')}\n")


# ------------------------------------------------------------------- loading --
def position_group(pos: str) -> str:
    """Coarse roster group used for position normalisation (measured counts in
    DATA_SCHEMA.md): SKILL / DB / LINE / EDGE-LB."""
    p = str(pos).upper()
    if p in {"WR", "TE", "RB", "FB"}:
        return "SKILL"
    if p in {"CB", "SS", "FS", "DB"}:
        return "DB"
    if p in {"C", "G", "T", "NT", "DT"}:
        return "LINE"
    return "EDGE_LB"


def load_tables() -> Dict[str, pd.DataFrame]:
    """Load the cached derived tables plus the raw lookup tables."""
    sess = pd.read_csv(os.path.join(CACHE, "combine_sessions.csv"))
    cuts = pd.read_csv(os.path.join(CACHE, "combine_cuts.csv"))
    comb = pd.read_csv(os.path.join(CACHE, "combine_player.csv"))
    pf = pd.read_csv(os.path.join(CACHE, "play_features.csv"), low_memory=False)
    dv = pd.read_csv(os.path.join(CACHE, "derived_sep.csv"))
    cr = pd.read_csv(os.path.join(DATA, "combine_results.csv"))
    players = pd.read_csv(os.path.join(DATA, "players.csv"))
    pcs = pd.read_csv(os.path.join(DATA, "player_career_successes.csv"))
    pp = pd.read_csv(os.path.join(DATA, "player_play.csv"), low_memory=False,
                     dtype={"game_id": "int32", "play_id": "int32", "nfl_id": "int32"})
    return dict(sess=sess, cuts=cuts, comb=comb, pf=pf, dv=dv, cr=cr,
                players=players, pcs=pcs, pp=pp)


def build_frames(t: Dict[str, pd.DataFrame]):
    """Merge athlete-level combine features into every downstream frame.

    Combine columns are prefixed ``cmb_`` before they meet the play-level
    frame: both tables carry generic names such as ``max_speed`` and
    ``path_efficiency``, and silently letting the play-level column shadow the
    combine one would (a) drop the cross-domain signal and (b) duplicate
    features in the design matrix.
    """
    players = t["players"].copy()
    players["position_group"] = players["nfl_position"].map(position_group)
    comb = t["comb"].merge(t["cr"], on="nfl_id", how="left")
    comb = comb.merge(players[["nfl_id", "position_group", "nfl_position"]],
                      on="nfl_id", how="left")
    # body-size normalisation: speeds/accelerations in body-lengths per second
    kin_cols = [c for c in comb.columns
                if comb[c].dtype.kind == "if" and ("speed" in c or "accel" in c)]
    comb = add_normalised_features(comb, kin_cols)
    comb.to_csv(os.path.join("outputs", "kce_agg.csv"), index=False)
    t["cuts"].to_csv(os.path.join("outputs", "kce_agg_cuts.csv"), index=False)

    keep_plain = {"nfl_id", "position_group"}
    comb_pref = comb.rename(columns={c: f"cmb_{c}" for c in comb.columns
                                     if c not in keep_plain})
    pf = t["pf"].merge(comb_pref, on="nfl_id", how="left")
    pf["sep_gain"] = pf["separation_at_pass_forward"] - pf["cushion"]
    pf["game_play_id"] = pf.groupby(["game_id", "play_id"]).ngroup()
    # tracking-derived proxy target (validated against the official field later)
    dv_pf = t["dv"][t["dv"]["moment"] == "pf"].copy()
    pf = pf.merge(dv_pf[["game_id", "play_id", "nfl_id", "derived_sep",
                         "n_opponents_tracked"]].rename(
        columns={"derived_sep": "derived_sep_pf"}),
        on=["game_id", "play_id", "nfl_id"], how="left")
    # catch-frame proxy (explicitly labelled; validated before any use)
    dv_c = t["dv"][t["dv"]["moment"] == "catch"].copy()
    pf = pf.merge(dv_c[["game_id", "play_id", "nfl_id", "derived_sep"]].rename(
        columns={"derived_sep": "derived_sep_catch"}),
        on=["game_id", "play_id", "nfl_id"], how="left")
    return comb, pf, players, t["pp"], t["pcs"]


# ------------------------------------------------------------------- targets --
def player_level_frame(pp: pd.DataFrame, comb: pd.DataFrame, pcs: pd.DataFrame,
                       min_obs: int = 5) -> pd.DataFrame:
    """Athlete-level targets: mean in-game separation, YAC and career snaps."""
    sep = pp[pp["separation_at_pass_forward"].notna()]
    g = sep.groupby("nfl_id").agg(
        player_mean_sep=("separation_at_pass_forward", "mean"),
        n_sep_obs=("separation_at_pass_forward", "size"),
        player_mean_cushion=("cushion", "mean"),
    ).reset_index()
    gain = (sep.assign(g=sep["separation_at_pass_forward"] - sep["cushion"])
            .groupby("nfl_id")["g"].mean().rename("player_mean_gain").reset_index())
    g = g.merge(gain, on="nfl_id", how="left")
    g = g[g["n_sep_obs"] >= min_obs]
    # yards after catch on targeted+caught plays (official field, athlete level)
    yac = pp[(pp["target"] == True) & pp["yards_after_catch"].notna()]  # noqa: E712
    yac_g = yac.groupby("nfl_id").agg(
        player_mean_yac=("yards_after_catch", "mean"),
        n_yac_obs=("yards_after_catch", "size")).reset_index()
    yac_g = yac_g[yac_g["n_yac_obs"] >= min_obs]
    out = g.merge(comb, on="nfl_id", how="inner")
    out = out.merge(yac_g, on="nfl_id", how="left")
    out = out.merge(pcs, on="nfl_id", how="left")
    out["career_snaps_log"] = np.log1p(out["career_offensive_snaps"].fillna(0))
    return out


# ---------------------------------------------------------------- evaluation --
def _evaluate(name: str, df: pd.DataFrame, target: str, features: Sequence[str],
              groupings: Dict[str, np.ndarray], metrics: dict) -> dict:
    """Evaluate one target (or reuse a checkpointed result), log progress."""
    prev = metrics.get(name)
    if isinstance(prev, dict) and "groupings" in prev:
        print(f"  -> {name}: already measured ({prev.get('rows')} rows, "
              f"checkpoint)", flush=True)
        return prev
    print(f"  -> {name}: rows={df[target].notna().sum()} features={len(features)}", flush=True)
    t0 = time.time()
    res = evaluate_target(df, target, features, groupings)
    res["name"] = name
    res["seconds"] = round(time.time() - t0, 1)
    metrics[name] = res
    _save_checkpoint(metrics)          # hard wall-clock safety: persist now
    return res


def headline_scores(res: dict) -> Dict[str, Dict[str, float]]:
    """Flatten {grouping: model: {r2, mae, spearman}} for quick comparison."""
    out = {}
    for gname, g in res.get("groupings", {}).items():
        row = {}
        for kind, r in g.get("models", {}).items():
            if isinstance(r, dict) and "r2" in r:
                row[kind] = {"r2": r["r2"], "mae": r["mae"], "spearman": r["spearman"]}
        for kind, r in g.get("baselines", {}).items():
            if isinstance(r, dict) and "r2" in r:
                row[kind] = {"r2": r["r2"], "mae": r["mae"], "spearman": r["spearman"]}
        out[gname] = row
    return out


def learning_curve(df: pd.DataFrame, target: str, features: Sequence[str],
                   groups: np.ndarray, fracs: Sequence[float] = (0.2, 0.4, 0.6,
                                                                 0.8, 1.0)) -> List[dict]:
    """OOF R2 / MAE vs training size: subsample whole groups, re-run GroupKFold.

    Group-level subsampling keeps the curve leak-free (an athlete / play never
    appears in two folds) and each point is a genuine 5-fold OOF score.
    """
    mask = df[target].notna().to_numpy()
    d = df[mask]
    X, used = prepare_matrix(d, features, drop=[target])
    y = d[target].to_numpy(float)
    g = np.asarray(groups)[mask]
    uniq = np.unique(g)
    rng = np.random.default_rng(42)
    out: List[dict] = []
    for f in fracs:
        k = int(max(3, round(len(uniq) * f)))
        chosen = rng.choice(uniq, size=k, replace=False)
        keep = np.isin(g, chosen)
        if keep.sum() < 200:
            continue
        res = run_cv(X[keep], y[keep], g[keep], "lightgbm", n_splits=5)
        out.append(dict(frac=round(float(f), 2), groups=k, rows=int(keep.sum()),
                        r2=round(float(res["r2"]), 4),
                        mae=round(float(res["mae"]), 4),
                        spearman=round(float(res["spearman"]), 4)))
    return out


def per_position_scores(df: pd.DataFrame, target: str, model_res: dict,
                        pos_col: str = "position_group") -> Dict[str, dict]:
    """OOF R2 / MAE / Spearman per roster position group for one model run."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    y = np.asarray(model_res.get("y_true") if model_res.get("y_true") is not None
                   else [], dtype=float)
    oof = np.asarray(model_res.get("oof") if model_res.get("oof") is not None
                     else [], dtype=float)
    if len(y) != len(oof) or not len(y):
        return {}
    d = df[df[target].notna()].reset_index(drop=True)
    if len(d) != len(y) or pos_col not in d.columns:
        return {}
    out: Dict[str, dict] = {}
    for pos, idx in d.groupby(pos_col).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        m = np.isfinite(oof[idx]) & np.isfinite(y[idx])
        if m.sum() < 20:
            continue
        yy, pp = y[idx][m], oof[idx][m]
        rho = float(spearmanr(yy, pp).statistic)
        out[str(pos)] = dict(n=int(m.sum()),
                             r2=round(float(r2_score(yy, pp)), 4),
                             mae=round(float(mean_absolute_error(yy, pp)), 4),
                             spearman=round(rho if rho == rho else float("nan"), 4))
    return out


def save_oof(oof_store: Dict[str, dict], path: str) -> None:
    """Persist every OOF / y_true array so figures can be rebuilt without CV."""
    arrays: Dict[str, np.ndarray] = {}
    for name, res in oof_store.items():
        if not isinstance(res, dict):
            continue
        for gname, g in (res.get("groupings") or {}).items():
            for kind, r in {**(g.get("models") or {}), **(g.get("baselines") or {})}.items():
                if isinstance(r, dict) and r.get("oof") is not None:
                    arrays[f"{name}__{gname}__{kind}"] = np.asarray(r["oof"], dtype=float)
                    if r.get("y_true") is not None:
                        arrays[f"{name}__{gname}__y_true"] = np.asarray(r["y_true"],
                                                                       dtype=float)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"wrote {path} ({len(arrays)} arrays)", flush=True)


def main() -> None:  # noqa: C901
    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)
    t = load_tables()
    comb, pf, players, pp, pcs = build_frames(t)
    print(f"tables loaded: combine={len(comb)} cuts={len(t['cuts'])} plays={len(pf)}",
          flush=True)

    features_all = [c for c in comb.columns
                    if comb[c].dtype.kind in "ifu" and c not in
                    {"nfl_id", "draft_year", "n_sep_obs", "n_yac_obs"} and
                    c not in FORBIDDEN]
    play_kin = [c for c in PLAY_KIN if c in pf.columns]
    ctx = [c for c in PRE_SNAP_CTX if c in pf.columns]
    cross_domain = features_all                    # combine-only, athlete level
    cross_domain_pf = [f"cmb_{c}" for c in features_all if f"cmb_{c}" in pf.columns]
    kin_model = play_kin + cross_domain_pf         # route kinematics + combine
    full_model = kin_model + ctx                   # + pre-snap context

    metrics: dict = {
        "track": TRACK,
        "data_source": "REAL kaggle_nfl/nfl_data (no synthetic rows anywhere)",
        "model_notes": ("Savitzky-Golay position differentiation (window=11, poly=3, "
                        "10 Hz) for all kinematics; `a` column is unsigned and unused"),
        "proxy_explicit": True,
        "cv": "GroupKFold 5 splits; no athlete / game_play leaks across folds",
        "boosters": "lightgbm 4.7.0 (primary) + sklearn HistGradientBoostingRegressor",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    oof_store = OOF_STORE
    _load_checkpoint(metrics)

    # ---- T1: official separation at pass forward, play level -----------------
    print("T1 sep_at_pass_forward", flush=True)
    res1 = _evaluate("sep_at_pass_forward", pf, "separation_at_pass_forward",
                     full_model,
                     {"by_player": pf["nfl_id"].to_numpy(),
                      "by_game_play": pf["game_play_id"].to_numpy()}, metrics)
    oof_store["sep_at_pass_forward"] = res1
    # information-horizon ablations on the same rows and folds
    res1k = _evaluate("sep_at_pass_forward_kinematics_only", pf,
                      "separation_at_pass_forward", kin_model,
                      {"by_player": pf["nfl_id"].to_numpy()}, metrics)
    oof_store["sep_at_pass_forward_kinematics_only"] = res1k
    presnap_cols = cross_domain_pf + [c for c in ctx if c != "route_ran"]
    res1p = _evaluate("sep_at_pass_forward_presnap_only", pf,
                      "separation_at_pass_forward", presnap_cols,
                      {"by_player": pf["nfl_id"].to_numpy()}, metrics)
    oof_store["sep_at_pass_forward_presnap_only"] = res1p

    # ---- T2: athlete-level mean separation from combine kinematics -----------
    print("T2 player_mean_sep", flush=True)
    pl = player_level_frame(pp, comb, pcs)
    res2 = _evaluate("player_mean_sep", pl, "player_mean_sep", cross_domain,
                     {"by_player": pl["nfl_id"].to_numpy()}, metrics)
    oof_store["player_mean_sep"] = res2
    # same target with the pre-registered 17-feature set (n=145 vs ~50 columns)
    res2p = _evaluate("player_mean_sep_preregistered", pl, "player_mean_sep",
                      [c for c in PREREG_PLAYER_SEP if c in pl.columns],
                      {"by_player": pl["nfl_id"].to_numpy()}, metrics)
    oof_store["player_mean_sep_preregistered"] = res2p

    # ---- T3: separation created after the snap ------------------------------
    print("T3 sep_gain", flush=True)
    res3 = _evaluate("sep_gain", pf, "sep_gain", kin_model + ctx,
                     {"by_player": pf["nfl_id"].to_numpy(),
                      "by_game_play": pf["game_play_id"].to_numpy()}, metrics)
    oof_store["sep_gain"] = res3
    # honesty ablation: `cushion` appears algebraically inside sep_gain, so a
    # model that sees it can reconstruct part of the target without kinematics
    print("T3b sep_gain_no_cushion", flush=True)
    no_cushion = [c for c in (kin_model + ctx) if c != "cushion"]
    res3b = _evaluate("sep_gain_no_cushion", pf, "sep_gain", no_cushion,
                      {"by_player": pf["nfl_id"].to_numpy(),
                       "by_game_play": pf["game_play_id"].to_numpy()}, metrics)
    oof_store["sep_gain_no_cushion"] = res3b
    res3["cushion_ablation"] = "sep_gain_no_cushion"
    res3["algebraic_note"] = (
        "cushion is a term of this target (sep_gain = separation_at_pass_forward "
        "- cushion); read the headline R2 together with sep_gain_no_cushion")
    metrics["sep_gain_cushion_ablation"] = {
        "with_cushion_by_game_play":
            headline_scores(res3).get("by_game_play", {}).get("lightgbm", {}),
        "without_cushion_by_game_play":
            headline_scores(res3b).get("by_game_play", {}).get("lightgbm", {}),
        "with_cushion_by_player":
            headline_scores(res3).get("by_player", {}).get("lightgbm", {}),
        "without_cushion_by_player":
            headline_scores(res3b).get("by_player", {}).get("lightgbm", {}),
    }

    # ---- T4: tracking-derived separation (explicit PROXY) -------------------
    print("T4 derived_sep_pf", flush=True)
    res4 = _evaluate("derived_sep_pf", pf, "derived_sep_pf", kin_model + ctx,
                     {"by_player": pf["nfl_id"].to_numpy()}, metrics)
    oof_store["derived_sep_pf"] = res4
    # proxy validation against the official field -- decides if it may be used
    v = pf[pf["derived_sep_pf"].notna() & pf["separation_at_pass_forward"].notna()]
    if len(v) > 10:
        from scipy.stats import spearmanr
        val = dict(rows=int(len(v)),
                   pearson=float(v["derived_sep_pf"].corr(v["separation_at_pass_forward"])),
                   spearman=float(spearmanr(v["derived_sep_pf"],
                                            v["separation_at_pass_forward"]).statistic),
                   derived_mean_yd=float(v["derived_sep_pf"].mean()),
                   official_mean_yd=float(v["separation_at_pass_forward"].mean()))
        val["status"] = ("ACCEPTED as target" if abs(val["pearson"]) >= 0.4 else
                         "REJECTED as target: fails validation against the official "
                         "field (game_tracking tracks only 1.6 athletes/play, so the "
                         "nearest *tracked* opponent is usually not the covering "
                         "defender)")
        metrics["derived_sep_pf_validation_vs_official"] = val
        res4["validation_vs_official"] = val
        res4["eligible_for_headline"] = abs(val["pearson"]) >= 0.4
        log_row("derived_proxy_validation_pearson", round(val["pearson"], 4),
                f"rows={val['rows']} derived vs official sep at pass_forward; "
                f"{'accepted' if abs(val['pearson']) >= 0.4 else 'REJECTED'}")

    # ---- T5: career opportunity from the combine (all 510 athletes) ---------
    print("T5 career_snaps_log", flush=True)
    career = comb.merge(pcs, on="nfl_id", how="left")
    career["career_snaps_log"] = np.log1p(career["career_offensive_snaps"].fillna(0))
    career["career_games_log"] = np.log1p(career["career_games_active"].fillna(0))
    res5 = _evaluate("career_snaps_log", career, "career_snaps_log", cross_domain,
                     {"by_player": career["nfl_id"].to_numpy()}, metrics)
    oof_store["career_snaps_log"] = res5

    # ---- T6: athlete-level yards after catch --------------------------------
    print("T6 player_mean_yac", flush=True)
    if pl["player_mean_yac"].notna().sum() >= 40:
        res6 = _evaluate("player_mean_yac", pl, "player_mean_yac", cross_domain,
                         {"by_player": pl["nfl_id"].to_numpy()}, metrics)
        oof_store["player_mean_yac"] = res6

    # ---- T7: separation at the CATCH frame (tracking-derived PROXY) ---------
    print("T7 derived_sep_catch", flush=True)
    res7 = _evaluate("derived_sep_catch", pf, "derived_sep_catch",
                     kin_model + ctx,
                     {"by_player": pf["nfl_id"].to_numpy()}, metrics)
    oof_store["derived_sep_catch"] = res7
    v7 = pf[pf["derived_sep_catch"].notna() & pf["separation_at_pass_forward"].notna()]
    if len(v7) > 10:
        from scipy.stats import spearmanr as _sr
        val7 = dict(rows=int(len(v7)),
                    pearson=float(v7["derived_sep_catch"].corr(
                        v7["separation_at_pass_forward"])),
                    spearman=float(_sr(v7["derived_sep_catch"],
                                       v7["separation_at_pass_forward"]).statistic),
                    derived_mean_yd=float(v7["derived_sep_catch"].mean()),
                    official_mean_yd=float(v7["separation_at_pass_forward"].mean()))
        val7["status"] = ("ACCEPTED as target" if abs(val7["pearson"]) >= 0.4 else
                          "REJECTED as target: catch-frame nearest *tracked* "
                          "opponent fails validation against the official field")
        metrics["derived_sep_catch_validation_vs_official"] = val7
        res7["validation_vs_official"] = val7
        res7["eligible_for_headline"] = abs(val7["pearson"]) >= 0.4
        log_row("derived_catch_proxy_validation_pearson",
                round(val7["pearson"], 4),
                f"rows={val7['rows']} derived@catch vs official sep; "
                f"{'accepted' if abs(val7['pearson']) >= 0.4 else 'REJECTED'}")

    # ---- T8: official yards after catch (targeted + caught plays) -----------
    print("T8 yards_after_catch", flush=True)
    yac_pf = pf[(pf["target"] == True) & pf["yards_after_catch"].notna()]  # noqa: E712
    if len(yac_pf) >= 200:
        res8 = _evaluate("yards_after_catch", yac_pf, "yards_after_catch",
                         full_model,
                         {"by_player": yac_pf["nfl_id"].to_numpy(),
                          "by_game_play": yac_pf["game_play_id"].to_numpy()},
                         metrics)
        oof_store["yards_after_catch"] = res8

    # ---- legacy formulation (what produced R2=0.0355 / MAE=1.6355) ----------
    print("legacy formulation", flush=True)
    legacy = pp[pp["separation_at_pass_forward"].notna()].merge(
        comb, on="nfl_id", how="left")
    legacy["game_play_id"] = legacy.groupby(["game_id", "play_id"]).ngroup()
    resL = _evaluate("legacy_player_features_on_play_target", legacy,
                     "separation_at_pass_forward", cross_domain,
                     {"by_player": legacy["nfl_id"].to_numpy()}, metrics)
    oof_store["legacy_player_features_on_play_target"] = resL

    # ---- mixed effects ------------------------------------------------------
    print("MixedLM fits", flush=True)
    mm: Dict[str, dict] = {}
    cv_mm: Dict = {}
    if isinstance(metrics.get("mixed_effects"), dict) and metrics["mixed_effects"]:
        mm = metrics["mixed_effects"]
        cv_mm = mm.get("population_oof_by_player", {})
        print("  checkpoint hit: MixedLM already fitted", flush=True)
    if not mm:
        base_cols = ["speed_at_pf", "release_time_s", "path_efficiency", "max_speed",
                     "cushion", "snap_to_pf_s"]
        fml = ("separation_at_pass_forward ~ speed_at_pf + release_time_s + "
               "path_efficiency + max_speed + cushion + snap_to_pf_s + C(route_ran)")
        sub = pf.dropna(subset=[c for c in base_cols if c in pf.columns]).copy()
        mm["player_random_intercept"] = run_mixedlm(sub, fml, "nfl_id",
                                                    label="player RI")
        mm["game_random_intercept"] = run_mixedlm(sub, fml, "game_id",
                                                  label="game RI")
        mm["player_plus_team_vc"] = run_mixedlm(sub, fml, "nfl_id",
                                                vc={"team": "0 + C(team_abbr)"},
                                                label="player RI + team VC")
        try:
            mm["player_plus_game_vc"] = run_mixedlm(sub, fml, "nfl_id",
                                                    vc={"game": "0 + C(game_id)"},
                                                    label="player RI + game VC")
        except Exception as exc:
            mm["player_plus_game_vc"] = dict(label="player RI + game VC",
                                             error=str(exc)[:200])
        cv_mm = run_mixedlm_cv(sub, fml, base_cols, group_col="nfl_id",
                               label="mixedlm_population_oof_by_player")
        mm["population_oof_by_player"] = cv_mm
        oof_store["mixedlm_population"] = cv_mm
        # cut-level directional model. NOTE: drill_base was dropped from this
        # formula: alignment side is encoded in the drill NAME, so including
        # both makes the alignment coefficient non-identifiable (p = nan,
        # coefficient swings by 40 KCE units across optimisers).  Cut angle,
        # approach speed and speed retention are the physically meaningful
        # controls; alignment side then enters as a FEATURE.
        cuts = t["cuts"]
        mm["kce_directional"] = run_mixedlm(
            cuts, "kce ~ cut_angle_deg + approach_speed + speed_retention + "
                  "C(align_side)", "nfl_id",
            label="cut-level KCE ~ angle + speed + alignment side")
        metrics["mixed_effects"] = {k: strip_heavy(v) for k, v in mm.items()}
        _save_checkpoint(metrics)
        log_row("mixedlm_player_ri_converged",
                int(bool(mm["player_random_intercept"].get("converged"))),
                f"rows={mm['player_random_intercept'].get('rows')}")
        log_row("mixedlm_kce_align_beta",
                round(mm["kce_directional"]["params"].get(
                    "C(align_side)[T.RIGHT]", float("nan")), 5)
                if "params" in mm["kce_directional"] else "NA",
                "KCE ~ cut_angle + align_side + drill_base; alignment side is a FEATURE")
        log_row("mixedlm_population_oof_r2", round(cv_mm.get("r2", float("nan")), 4),
                f"folds_converged={cv_mm.get('folds_converged')} "
                f"fallback={cv_mm.get('folds_fallback')}")

    # ---- headline selection (validated official targets only) ---------------
    scores = {k: headline_scores(v) for k, v in metrics.items()
              if isinstance(v, dict) and "groupings" in v}
    best_name, best_r2 = None, -9e9
    for name, gs in scores.items():
        res = metrics[name]
        if res.get("eligible_for_headline") is False:
            continue                     # e.g. the rejected derived proxy
        for gname, row in gs.items():
            r2 = row.get("lightgbm", {}).get("r2")
            if r2 is not None and r2 > best_r2:
                best_name, best_r2 = f"{name}::{gname}", r2
    prior = dict(oof_r2=0.0355, mae=1.6352,
                 rows=5_006_995, note="previous run: player-level features "
                 "against a play-level target")
    cush = metrics.get("sep_gain_cushion_ablation", {})
    headline_note = (
        "selected by measured OOF R2 among validated official targets. "
        "sep_gain = separation_at_pass_forward - cushion and `cushion` is also a "
        "feature, so part of its score is algebraic: with-cushion R2="
        f"{cush.get('with_cushion_by_game_play', {}).get('r2')} vs "
        f"without-cushion R2={cush.get('without_cushion_by_game_play', {}).get('r2')} "
        "(sep_gain_no_cushion). The pure official target "
        "sep_at_pass_forward (no algebraic overlap) is reported separately and "
        "is the target statement used in WRITEUP.md."
    )
    metrics["headline"] = dict(target=best_name, oof_r2=round(float(best_r2), 4),
                               prior_run=prior,
                               improvement_over_prior_r2=round(
                                   float(best_r2) / 0.0355, 2),
                               headline_note=headline_note)
    metrics["primary_real_target"] = dict(
        target="separation_at_pass_forward",
        statement=("official player_play field, play level: pre-pass route "
                   "kinematics + pre-snap context + combine athleticism, "
                   "GroupKFold OOF"),
        scores=headline_scores(metrics["sep_at_pass_forward"]))
    metrics["summary_table"] = scores

    # ---- headline-target follow-ups: learning curve + per-position OOF ------
    frames = {"sep_at_pass_forward": pf, "sep_at_pass_forward_kinematics_only": pf,
              "sep_at_pass_forward_presnap_only": pf, "player_mean_sep": pl,
              "player_mean_sep_preregistered": pl, "sep_gain": pf,
              "sep_gain_no_cushion": pf, "derived_sep_pf": pf,
              "derived_sep_catch": pf, "career_snaps_log": career,
              "player_mean_yac": pl, "yards_after_catch": yac_pf,
              "legacy_player_features_on_play_target": legacy}
    hl_name, hl_grouping = best_name.split("::")
    hl_res = metrics[hl_name]
    dfh = frames.get(hl_name, pf)
    print(f"learning curve + per-position breakdown for {best_name}", flush=True)
    gcol = {"by_player": "nfl_id", "by_game_play": "game_play_id"}.get(hl_grouping,
                                                                      "nfl_id")
    if metrics.get("learning_curve"):
        print("  checkpoint hit: learning curve already measured", flush=True)
    else:
        try:
            lc = learning_curve(dfh, hl_res["target"], hl_res["feature_columns"],
                                dfh[gcol].to_numpy() if gcol in dfh.columns
                                else dfh["nfl_id"].to_numpy())
            metrics["learning_curve"] = lc
            for p in lc:
                log_row(f"learning_curve_r2::{hl_name}", p["r2"],
                        f"frac={p['frac']} rows={p['rows']} groups={p['groups']} "
                        f"mae={p['mae']} grouping={hl_grouping}")
        except Exception as exc:
            metrics["learning_curve"] = []
            log_row("learning_curve_failed", "NA", str(exc)[:160])

    if metrics.get("per_position_breakdown"):
        print("  checkpoint hit: per-position breakdown already measured",
              flush=True)
    else:
        pb_all: Dict[str, dict] = {}
        for gname_, gres in hl_res["groupings"].items():
            mres = gres.get("models", {}).get("lightgbm", {})
            s = per_position_scores(dfh, hl_res["target"], mres)
            if s:
                pb_all[gname_] = s
                for pos_, v_ in s.items():
                    log_row(f"per_position_r2::{hl_name}::{gname_}::{pos_}", v_["r2"],
                            f"n={v_['n']} mae={v_['mae']} spearman={v_['spearman']}")
        metrics["per_position_breakdown"] = pb_all

    # ---- persistence --------------------------------------------------------
    _save_checkpoint(metrics)
    print(f"wrote {METRICS_OUT}", flush=True)

    for name, gs in scores.items():
        for gname, row in gs.items():
            for model, m in row.items():
                log_row(f"cv_r2::{name}::{gname}::{model}", round(float(m["r2"]), 4),
                        f"mae={m['mae']:.4f} spearman={m['spearman']:.4f}")
    log_row("headline_target", best_name, f"oof_r2={best_r2:.4f} (prior run 0.0355)")
    if cush:
        log_row("sep_gain_cushion_ablation_r2",
                round(float(cush["with_cushion_by_game_play"].get("r2", np.nan)), 4),
                "with cushion vs without="
                f"{float(cush['without_cushion_by_game_play'].get('r2', np.nan)):.4f}"
                " (by_game_play, lightgbm)")

    # feature-target correlations for the headline target
    hl = best_name.split("::")[0]
    dfh = frames.get(hl, pf)
    rows = []
    tgt = metrics[hl]["target"]
    for c in metrics[hl]["feature_columns"]:
        if c not in dfh.columns or dfh[c].dtype.kind not in "if":
            continue
        s = dfh[[c, tgt]].dropna()
        if len(s) < 30:
            continue
        rows.append(dict(feature=c, n=len(s), target=tgt,
                         pearson=float(s[c].corr(s[tgt])),
                         spearman=float(s[c].corr(s[tgt], method="spearman"))))
    corr = pd.DataFrame(rows).sort_values("pearson", key=np.abs, ascending=False)
    corr.to_csv(os.path.join("outputs", "feature_target_correlations.csv"), index=False)

    from figures import make_figures
    make_figures(t=t, pf=pf, pl=pl, comb=comb, cuts=t["cuts"], corr=corr,
                 metrics=metrics, oof=oof_store, dv=t["dv"], headline=best_name)

    print(f"done in {time.time() - t0:.1f}s; headline={best_name} r2={best_r2:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
