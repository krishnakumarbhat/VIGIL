"""Compact, measured summary of outputs/model_metrics.json for reporting.

Prints every number needed for WRITEUP.md / the final report directly from the
stored JSON, so no figure or prose number is hand-typed.  Run after
``src/run_analysis.py``:

    python3 src/summarize_metrics.py
"""

from __future__ import annotations

import json
import os

METRICS = os.path.join("outputs", "model_metrics.json")


def _row(scores: dict, target: str, grouping: str) -> str:
    """One formatted line: LightGBM / HistGB / mean / shuffled for a grouping."""
    s = scores.get(target, {}).get(grouping, {})
    if not s:
        return f"  {target:38s} {grouping:12s} (missing)"
    parts = []
    for kind in ("lightgbm", "hist_gb", "mean_dummy", "shuffled_label_null"):
        m = s.get(kind)
        if m:
            parts.append(f"{kind} R2={m['r2']:+.4f} MAE={m['mae']:.4f} "
                         f"rho={m['spearman']:+.3f}")
    return f"  {target:38s} {grouping:12s} " + " | ".join(parts)


def main() -> None:
    """Print the headline, every target x grouping, the ablations and MixedLM."""
    with open(METRICS) as fh:
        m = json.load(fh)
    print("== headline ==")
    print(json.dumps(m["headline"], indent=2))
    print("\n== primary real target ==")
    print(json.dumps(m.get("primary_real_target", {}), indent=2))
    print("\n== all targets ==")
    scores = m.get("summary_table") or {
        k: v["groupings"] for k, v in m.items()
        if isinstance(v, dict) and isinstance(v.get("groupings"), dict)
        and all(isinstance(g, dict) for g in v["groupings"].values())}
    for name, res in scores.items():
        for gname in res:
            print(_row(scores, name, gname))
    print("\n== cushion ablation ==")
    print(json.dumps(m.get("sep_gain_cushion_ablation", {}), indent=2))
    print("\n== proxy validation ==")
    print(json.dumps(m.get("derived_sep_pf_validation_vs_official", {}), indent=2))
    print("\n== MixedLM ==")
    for k, v in m.get("mixed_effects", {}).items():
        if not isinstance(v, dict):
            continue
        if "error" in v and "params" not in v:
            print(f"  {k}: ERROR {v['error'][:120]}")
            continue
        print(f"  {k}: converged={v.get('converged')} singular="
              f"{v.get('singular_random_effects')} rows={v.get('rows')} "
              f"groups={v.get('groups')} r2_insample={v.get('r2_fitted_insample')}")
        if k == "population_oof_by_player":
            print(f"      OOF R2={v.get('r2'):.4f} MAE={v.get('mae'):.4f} "
                  f"spearman={v.get('spearman'):.4f} folds_converged="
                  f"{v.get('folds_converged')} fallback={v.get('folds_fallback')} "
                  f"n_rows={v.get('n_rows')}")
    print("\n== top feature-target correlations (headline) ==")
    corr = os.path.join("outputs", "feature_target_correlations.csv")
    if os.path.exists(corr):
        print(open(corr).read()[:1200])


if __name__ == "__main__":
    main()
