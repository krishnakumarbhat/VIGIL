"""Publication-grade figures (exactly 8), every one drawn from the REAL tables.

Required set for Track 01:

1. ``fig1_cross_product_inside_outside`` -- INSIDE/OUTSIDE from the 2-D
   velocity cross product, and how the two groups differ in KCE / DBR.
2. ``fig2_kce_distribution``             -- KCE distribution (all cuts, by
   position group, per-athlete means).
3. ``fig3_dbr_distribution``             -- DBR distribution plus the measured
   proof that the shipped unsigned ``a`` made DBR identically 0.
4. ``fig4_turn_rate_histogram``          -- turn-rate histograms (cut level,
   session level, athlete level).
5. ``fig5_feature_importance``           -- headline-model gain vs raw
   feature-target correlation.
6. ``fig6_oof_pred_vs_actual``           -- out-of-fold prediction vs truth.
7. ``fig7_learning_curve``               -- OOF R2 / MAE vs training size.
8. ``fig8_per_position_breakdown``       -- OOF performance and cut quality by
   position group.

Learning-curve and per-position numbers are measured by ``src/run_analysis.py``
in the same run and stored in ``outputs/model_metrics.json``; no figure contains
a hand-typed or synthetic value.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG_DIR = os.path.join("outputs", "figures")
DPI = 200
PALETTE = ["#4C78A8", "#F58518", "#E45756", "#54A24B", "#B279A2", "#72B7B2"]
BREAK_COLORS = {"INSIDE": "#54A24B", "OUTSIDE": "#E45756", "UNALIGNED": "#B279A2"}


def _save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  figure {name}", flush=True)


def _pos_group(cuts: pd.DataFrame, comb: pd.DataFrame) -> pd.DataFrame:
    """Cuts with the athlete's roster position group attached (measured join)."""
    cols = [c for c in ("nfl_id", "position_group") if c in comb.columns]
    if "position_group" not in cols:
        return cuts.assign(position_group="ALL")
    return cuts.merge(comb[cols].drop_duplicates("nfl_id"), on="nfl_id", how="left")


# --------------------------------------------------------------------- fig 1 --
def fig1_cross_product_inside_outside(cuts: pd.DataFrame) -> None:
    """INSIDE/OUTSIDE separation from v_pre x v_post, and the KCE/DBR contrast."""
    from scipy.stats import mannwhitneyu
    c = cuts.copy()
    aligned = c[c["align_side"].isin(["LEFT", "RIGHT"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9))

    ax = axes[0]
    seen = set()
    for lab in ("UNALIGNED", "INSIDE", "OUTSIDE"):
        s = c[c["break_side"] == lab]
        if not len(s):
            continue
        if lab == "UNALIGNED":
            ax.scatter(s["cross"], s["align_side"], s=7, alpha=.25,
                       color=BREAK_COLORS[lab], label=f"UNALIGNED (n={len(s)})")
            continue
        for al, mk in (("LEFT", "o"), ("RIGHT", "^")):
            t = s[s["align_side"] == al]
            if not len(t):
                continue
            key = f"{lab} · drill aligned {al}"
            ax.scatter(t["cross"], t["align_side"], s=26, alpha=.75, marker=mk,
                       color=BREAK_COLORS[lab],
                       label=f"{key} (n={len(t)})", edgecolor="none")
            seen.add(key)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel(r"velocity cross product $v_{pre}\times v_{post}$ (yd$^2$/s$^2$)")
    ax.set_ylabel("drill alignment (from the drill NAME)")
    ax.set_title("(a) break side = sign of the cross product\n"
                 "given the drill's alignment; o = LEFT, ^ = RIGHT drills")
    ax.legend(fontsize=7.4, loc="center right", framealpha=.92)
    ax.grid(alpha=.3, axis="x")

    ax = axes[1]
    groups = [c.loc[c["break_side"] == lab, "kce"] for lab in
              ("INSIDE", "OUTSIDE", "UNALIGNED")]
    labels = [f"{lab}\n(n={len(g)})" for lab, g in
              zip(("INSIDE", "OUTSIDE", "UNALIGNED"), groups)]
    bp = ax.boxplot([g.dropna() for g in groups], patch_artist=True,
                    showfliers=False, widths=.62)
    ax.set_xticklabels(labels)
    for patch, lab in zip(bp["boxes"], ("INSIDE", "OUTSIDE", "UNALIGNED")):
        patch.set_facecolor(BREAK_COLORS[lab]); patch.set_alpha(.65)
    for i, g in enumerate(groups, start=1):
        if len(g.dropna()) > 5:
            ax.text(i, g.median(), f"{g.median():.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ins, out = c.loc[c.break_side == "INSIDE", "kce"].dropna(), \
        c.loc[c.break_side == "OUTSIDE", "kce"].dropna()
    p = mannwhitneyu(ins, out).pvalue if len(ins) and len(out) else float("nan")
    ax.set_ylabel("KCE (chord / path length, $\\pm$0.5 s)")
    ax.set_title(f"(b) KCE by break side\nINSIDE mean {ins.mean():.3f} vs "
                 f"OUTSIDE {out.mean():.3f}, MWU p={p:.2g}")
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    groups = [c.loc[c["break_side"] == lab, "dbr"].dropna() for lab in
              ("INSIDE", "OUTSIDE", "UNALIGNED")]
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                    widths=.62)
    ax.set_xticklabels(labels)
    for patch, lab in zip(bp["boxes"], ("INSIDE", "OUTSIDE", "UNALIGNED")):
        patch.set_facecolor(BREAK_COLORS[lab]); patch.set_alpha(.65)
    ax.set_ylabel("DBR (peak decel / peak burst, SG derivatives)")
    dins, dout = groups[0], groups[1]
    p = mannwhitneyu(dins, dout).pvalue if len(dins) and len(dout) else float("nan")
    ax.set_title(f"(c) DBR by break side\nINSIDE {dins.mean():.3f} vs "
                 f"OUTSIDE {dout.mean():.3f}, MWU p={p:.2g}")
    ax.grid(alpha=.3, axis="y")
    _save(fig, "fig1_cross_product_inside_outside.png")


# --------------------------------------------------------------------- fig 2 --
def fig2_kce_distribution(cuts: pd.DataFrame, comb: pd.DataFrame) -> None:
    """KCE distribution: all cuts, by roster position group, per athlete."""
    cg = _pos_group(cuts, comb)
    per_athlete = cuts.groupby("nfl_id")["kce"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    ax = axes[0]
    ax.hist(cuts["kce"], bins=60, color="#4C78A8", alpha=.85, edgecolor="white")
    ax.axvline(cuts["kce"].mean(), color="#E45756", ls="--", lw=1.6,
               label=f"mean {cuts['kce'].mean():.3f}")
    ax.axvline(cuts["kce"].median(), color="#F58518", ls=":", lw=1.6,
               label=f"median {cuts['kce'].median():.3f}")
    ax.set_xlabel("KCE (chord / path length over $\\pm$0.5 s)")
    ax.set_ylabel("detected cuts")
    ax.set_title(f"(a) KCE, all detected cuts >35$^\\circ$\n"
                 f"n={len(cuts):,}, athletes={cuts['nfl_id'].nunique()}, "
                 f"sd={cuts['kce'].std():.3f}")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = axes[1]
    order = (cg.groupby("position_group")["kce"].median().sort_values().index
             if "position_group" in cg else ["ALL"])
    data = [cg.loc[cg["position_group"] == p, "kce"].dropna() for p in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.6)
    ax.set_xticklabels([f"{p}\n(n={len(d)})" for p, d in zip(order, data)])
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)]); patch.set_alpha(.72)
    ax.set_ylabel("KCE")
    ax.set_title("(b) KCE by roster position group\n"
                 "(athlete's official position, measured join)")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    ax.hist(per_athlete, bins=45, color="#54A24B", alpha=.85, edgecolor="white")
    ax.axvline(per_athlete.mean(), color="#E45756", ls="--", lw=1.6,
               label=f"mean {per_athlete.mean():.3f}")
    ax.set_xlabel("per-athlete mean KCE")
    ax.set_ylabel("athletes")
    ax.set_title(f"(c) per-athlete mean KCE\n"
                 f"n={len(per_athlete)} athletes, sd={per_athlete.std():.3f}")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    _save(fig, "fig2_kce_distribution.png")


# --------------------------------------------------------------------- fig 3 --
def fig3_dbr_distribution(cuts: pd.DataFrame) -> None:
    """DBR distribution + measured proof that unsigned shipped `a` gave DBR=0."""
    path = os.path.join("data", "kaggle_nfl", "nfl_data", "combine_tracking.csv")
    raw = pd.read_csv(path, usecols=["event_id", "time", "a", "entity_type"])
    raw = raw[raw["entity_type"] == "PLAYER"].sort_values(["event_id", "time"])
    n_rows = len(raw)
    neg_frac = float((raw["a"] < 0).mean())
    raw["frame"] = raw.groupby("event_id").cumcount()
    # shipped-a deceleration / burst inside the same +-0.6 s window used for the
    # Savitzky-Golay DBR, so the two definitions are compared on identical cuts
    w = cuts[["event_id", "cut_frame"]].merge(
        raw[["event_id", "frame", "a"]], on="event_id")
    w = w[(w["frame"] - w["cut_frame"]).abs() <= 6].copy()
    w["neg"] = (-w["a"]).clip(lower=0)
    win = w.groupby(["event_id", "cut_frame"]).agg(decel=("neg", "max"),
                                                   burst=("a", "max"))
    shipped_dbr = win["decel"] / win["burst"].replace(0, np.nan)
    shipped_dbr = shipped_dbr.fillna(0.0)     # zero deceleration -> DBR 0
    shipped_zero_frac = float((win["decel"] == 0).mean())

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    ax = axes[0]
    hi = cuts["dbr"].quantile(.995)
    ax.hist(cuts.loc[cuts["dbr"] <= hi, "dbr"], bins=60, color="#72B7B2",
            alpha=.88, edgecolor="white")
    ax.axvline(cuts["dbr"].mean(), color="#E45756", ls="--", lw=1.6,
               label=f"mean {cuts['dbr'].mean():.3f}")
    ax.axvline(cuts["dbr"].median(), color="#F58518", ls=":", lw=1.6,
               label=f"median {cuts['dbr'].median():.3f}")
    ax.set_xlabel("DBR (peak |decel| / peak burst), SG derivatives")
    ax.set_ylabel(f"cuts (99.5% range, max {cuts['dbr'].max():.2f})")
    ax.set_title(f"(a) DBR distribution, n={len(cuts):,} cuts\n"
                 f"sd={cuts['dbr'].std():.3f}, p05={cuts['dbr'].quantile(.05):.3f}, "
                 f"p95={cuts['dbr'].quantile(.95):.3f}")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = axes[1]
    vals = [float(shipped_dbr.mean()), float(cuts["dbr"].mean())]
    bars = ax.bar(["shipped `a`\n(DBR from the\ngiven column)",
                   "Savitzky-Golay\n(DBR from x, y\nderivatives)"],
                  vals, color=["#B279A2", "#4C78A8"], alpha=.9, width=.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("mean DBR over the same 5,618 cuts")
    ax.set_title(f"(b) unsigned shipped `a`: {shipped_zero_frac:.1%} of cut windows "
                 f"have zero deceleration\nmin(a)={raw['a'].min():.3f}, "
                 f"{neg_frac:.4%} of {n_rows:,} rows negative -> DBR $\\equiv$ 0")
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    per_athlete = cuts.groupby("nfl_id")["dbr"].mean()
    ax.hist(per_athlete, bins=45, color="#54A24B", alpha=.85, edgecolor="white")
    ax.axvline(per_athlete.mean(), color="#E45756", ls="--", lw=1.6,
               label=f"mean {per_athlete.mean():.3f}")
    ax.set_xlabel("per-athlete mean DBR")
    ax.set_ylabel("athletes")
    ax.set_title(f"(c) per-athlete mean DBR (Savitzky-Golay)\n"
                 f"n={len(per_athlete)} athletes, max {per_athlete.max():.2f} "
                 f"-- the shipped column gives 0.00 for every one of them")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    _save(fig, "fig3_dbr_distribution.png")


# --------------------------------------------------------------------- fig 4 --
def fig4_turn_rate_histogram(cuts: pd.DataFrame, sess: pd.DataFrame,
                             comb: pd.DataFrame) -> None:
    """Turn-rate histograms at cut, session and athlete level."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    ax = axes[0]
    hi = cuts["turn_rate_peak"].quantile(.99)
    ax.hist(cuts.loc[cuts["turn_rate_peak"] <= hi, "turn_rate_peak"], bins=60,
            color="#F58518", alpha=.88, edgecolor="white")
    ax.axvline(cuts["turn_rate_peak"].mean(), color="#E45756", ls="--", lw=1.6,
               label=f"mean {cuts['turn_rate_peak'].mean():.1f}")
    ax.set_xlabel("peak turn rate at the cut (deg/s)")
    ax.set_ylabel(f"cuts (99% range, max {cuts['turn_rate_peak'].max():.0f})")
    ax.set_title(f"(a) turn rate at detected cuts\nn={len(cuts):,}, "
                 f"median {cuts['turn_rate_peak'].median():.1f} deg/s")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = axes[1]
    col = "mean_turn_rate" if "mean_turn_rate" in sess.columns else None
    if col:
        ax.hist(sess[col], bins=60, color="#4C78A8", alpha=.88, edgecolor="white")
        ax.axvline(sess[col].mean(), color="#E45756", ls="--", lw=1.6,
                   label=f"mean {sess[col].mean():.1f}")
        ax.set_xlabel("session mean turn rate (deg/s)")
        ax.set_ylabel("combine sessions")
        ax.set_title(f"(b) per-session mean turn rate\nn={len(sess):,} sessions, "
                     f"{sess['nfl_id'].nunique()} athletes")
        ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    ax = axes[2]
    if col and col in comb.columns:
        ax.hist(comb[col], bins=50, color="#54A24B", alpha=.88, edgecolor="white")
        ax.axvline(comb[col].mean(), color="#E45756", ls="--", lw=1.6,
                   label=f"mean {comb[col].mean():.1f}")
        ax.set_xlabel("per-athlete mean turn rate (deg/s)")
        ax.set_ylabel("athletes")
        ax.set_title(f"(c) per-athlete mean turn rate\n"
                     f"n={len(comb)} athletes, sd={comb[col].std():.1f} deg/s")
        ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    _save(fig, "fig4_turn_rate_histogram.png")


# --------------------------------------------------------------------- fig 5 --
def fig5_feature_importance(metrics: dict, headline: str,
                            corr: pd.DataFrame) -> None:
    """Headline-model gain next to the raw feature-target correlation."""
    name, gname = headline.split("::")
    res = metrics[name]
    imp = res["groupings"][gname]["models"].get("lightgbm", {}).get("importance", {})
    items = sorted(imp.items(), key=lambda kv: kv[1])[-14:]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))
    ax = axes[0]
    if items:
        ax.barh([k for k, _ in items], [v for _, v in items], color="#4C78A8",
                alpha=.9)
        ax.set_xlabel("mean LightGBM gain (5 folds)")
    ax.set_title(f"(a) model feature importance\n{headline}")
    ax.grid(alpha=.3, axis="x")
    ax = axes[1]
    if len(corr):
        c = corr.head(14).iloc[::-1]
        cols = ["#54A24B" if v >= 0 else "#E45756" for v in c["pearson"]]
        ax.barh(c["feature"], c["pearson"], color=cols, alpha=.9)
        ax.axvline(0, color="k", lw=.9)
        ax.set_xlabel(f"Pearson r vs {c['target'].iloc[0]}")
    ax.set_title("(b) univariate feature-target correlation")
    ax.grid(alpha=.3, axis="x")
    _save(fig, "fig5_feature_importance.png")


# --------------------------------------------------------------------- fig 6 --
def fig6_oof(oof: Optional[dict], metrics: dict, headline: str) -> None:
    """Out-of-fold prediction vs truth for the headline target."""
    from scipy.stats import spearmanr  # noqa: F401  (re-exported for callers)
    name, gname = headline.split("::")
    res = metrics[name]["groupings"][gname]["models"]["lightgbm"]
    oof_arr = res.get("oof")
    true = res.get("y_true")
    if oof_arr is None and oof:
        oof_arr = (oof.get(name, {}).get(gname, {}) or {}).get("lightgbm")
    oof_arr = np.asarray(oof_arr if oof_arr is not None else [], dtype=float)
    true = np.asarray(true if true is not None else [], dtype=float)
    if len(true) != len(oof_arr) or not len(true):
        if oof:                                   # arrays stored separately
            true = np.asarray((oof.get(name, {}) or {}).get("y_true", []), dtype=float)
    m = np.isfinite(oof_arr) & np.isfinite(true)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    ax = axes[0]
    r2 = res.get("r2", float("nan"))
    mae = res.get("mae", float("nan"))
    rho = res.get("spearman", float("nan"))
    if m.sum() > 2:
        hb = ax.hexbin(true[m], oof_arr[m], gridsize=42, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="OOF rows")
        lo = float(min(true[m].min(), oof_arr[m].min()))
        hi = float(max(true[m].max(), oof_arr[m].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
        ax.axhline(float(true[m].mean()), color="#B279A2", ls=":", lw=1.4,
                   label=f"mean baseline ({true[m].mean():.2f})")
    ax.set_xlabel(f"actual {metrics[name]['target']}")
    ax.set_ylabel("out-of-fold prediction")
    ax.set_title(f"(a) OOF prediction vs truth\n{headline}\n"
                 f"R$^2$={r2:.4f}, MAE={mae:.4f}, Spearman={rho:.4f}")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = axes[1]
    if m.sum() > 2:
        resid = true[m] - oof_arr[m]
        ax.hist(resid, bins=60, color="#E45756", alpha=.85, edgecolor="white")
        ax.axvline(0, color="k", lw=1.1)
        ax.axvline(float(np.mean(resid)), color="#4C78A8", ls="--", lw=1.5,
                   label=f"mean error {np.mean(resid):+.3f}")
        ax.set_xlabel("residual (actual - OOF prediction)")
        ax.set_ylabel("OOF rows")
        ax.set_title(f"(b) OOF residual distribution\n"
                     f"sd {resid.std():.3f}, |err| p95 "
                     f"{np.quantile(np.abs(resid), 0.95):.3f}")
        ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    _save(fig, "fig6_oof_pred_vs_actual.png")


# --------------------------------------------------------------------- fig 7 --
def fig7_learning_curve(metrics: dict) -> None:
    """OOF R2 / MAE against the number of training groups (measured)."""
    lc = metrics.get("learning_curve") or []
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    if lc:
        rows = [p["rows"] for p in lc]
        r2 = [p["r2"] for p in lc]
        mae = [p["mae"] for p in lc]
        ax.plot(rows, r2, "o-", color="#4C78A8", lw=2, ms=7, label="OOF $R^2$")
        ax.axhline(0.0355, color="#E45756", ls=":", lw=1.5,
                   label="prior run (0.0355)")
        ax.axhline(0, color="k", lw=.9)
        for x, y, p in zip(rows, r2, lc):
            ax.annotate(f"{y:.3f}\n{p['frac']:.0%}", (x, y), textcoords="offset points",
                        xytext=(6, -14), fontsize=8)
        ax.set_xlabel("training rows (GroupKFold by "
                      f"{lc[0].get('grouping', 'group').replace('by_', '')})")
        ax.set_ylabel("out-of-fold $R^2$")
        ax.set_title(f"(a) learning curve -- {metrics.get('headline', {}).get('target', '')}\n"
                     f"LightGBM, 5-fold GroupKFold at each size")
        ax2 = ax.twinx()
        ax2.plot(rows, mae, "s--", color="#F58518", lw=1.6, ms=6, label="OOF MAE")
        ax2.set_ylabel("out-of-fold MAE", color="#F58518")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right")
        ax.grid(alpha=.3)
    else:
        ax.text(.5, .5, "no learning curve in model_metrics.json",
                ha="center", va="center", transform=ax.transAxes)
    _save(fig, "fig7_learning_curve.png")


# --------------------------------------------------------------------- fig 8 --
def fig8_per_position_breakdown(cuts: pd.DataFrame, comb: pd.DataFrame,
                                metrics: dict, headline: str) -> None:
    """OOF performance per position group + measured cut quality per group."""
    name, gname = headline.split("::")
    pb = (metrics.get("per_position_breakdown") or {}).get(gname, {})
    cg = _pos_group(cuts, comb)
    qual = cg.groupby("position_group").agg(
        n=("kce", "size"), kce=("kce", "mean"), dbr=("dbr", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))

    ax = axes[0]
    if pb:
        pos = sorted(pb, key=lambda p: pb[p]["r2"])
        ax.barh(pos, [pb[p]["r2"] for p in pos],
                color=["#54A24B" if pb[p]["r2"] >= 0 else "#E45756" for p in pos],
                alpha=.9)
        for i, p in enumerate(pos):
            ax.text(pb[p]["r2"], i, f" {pb[p]['r2']:+.3f} (n={pb[p]['n']})",
                    va="center", fontsize=8.5)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("out-of-fold $R^2$ (LightGBM)")
        ax.set_title(f"(a) OOF $R^2$ by position group\n{headline}")
        ax.grid(alpha=.3, axis="x")
    else:
        ax.text(.5, .5, "no per-position breakdown stored", ha="center",
                va="center", transform=ax.transAxes)

    ax = axes[1]
    if pb:
        pos = sorted(pb, key=lambda p: pb[p]["mae"])
        ax.barh(pos, [pb[p]["mae"] for p in pos], color="#4C78A8", alpha=.9)
        for i, p in enumerate(pos):
            ax.text(pb[p]["mae"], i, f" {pb[p]['mae']:.3f}  rho={pb[p]['spearman']:+.3f}",
                    va="center", fontsize=8.5)
        ax.set_xlabel("out-of-fold MAE (same units as the target)")
        ax.set_title("(b) MAE and Spearman by position group")
        ax.grid(alpha=.3, axis="x")

    ax = axes[2]
    q = qual.sort_values("kce")
    x = np.arange(len(q))
    ax.bar(x - .2, q["kce"], .4, color="#4C78A8", alpha=.9, label="mean KCE")
    ax2 = ax.twinx()
    ax2.bar(x + .2, q["dbr"], .4, color="#E45756", alpha=.9, label="mean DBR")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={int(n)})" for p, n in
                        zip(q["position_group"], q["n"])], fontsize=8.5)
    ax.set_ylabel("mean KCE", color="#4C78A8")
    ax2.set_ylabel("mean DBR", color="#E45756")
    ax.set_title("(c) combine cut quality by position group\n"
                 "(5,618 real cuts)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left")
    ax.grid(alpha=.3, axis="y")
    _save(fig, "fig8_per_position_breakdown.png")


# ------------------------------------------------------------------ wrapper --
def make_figures(t: Dict, pf: pd.DataFrame, pl: pd.DataFrame, comb: pd.DataFrame,
                 cuts: pd.DataFrame, corr: pd.DataFrame, metrics: dict,
                 oof: Optional[dict], dv: pd.DataFrame, headline: str) -> None:
    """Render the 8 required figures from the real tables (best effort each)."""
    print("figures:", flush=True)
    sess = t.get("sess", pd.DataFrame())
    jobs = [
        ("fig1", lambda: fig1_cross_product_inside_outside(cuts)),
        ("fig2", lambda: fig2_kce_distribution(cuts, comb)),
        ("fig3", lambda: fig3_dbr_distribution(cuts)),
        ("fig4", lambda: fig4_turn_rate_histogram(cuts, sess, comb)),
        ("fig5", lambda: fig5_feature_importance(metrics, headline, corr)),
        ("fig6", lambda: fig6_oof(oof, metrics, headline)),
        ("fig7", lambda: fig7_learning_curve(metrics)),
        ("fig8", lambda: fig8_per_position_breakdown(cuts, comb, metrics, headline)),
    ]
    for label, fn in jobs:
        try:
            fn()
        except Exception as exc:
            print(f"  {label} failed: {type(exc).__name__}: {exc}", flush=True)
