"""Regenerate outputs/figures/ WITHOUT re-running cross-validation.

Reads ``outputs/model_metrics.json`` (scores, learning curve, per-position
breakdown) and ``outputs/cache/oof_store.npz`` (out-of-fold arrays written by
``src/run_analysis.py``), re-injects the OOF arrays into the metrics structure
and calls ``figures.make_figures``.  Use this after editing ``src/figures.py``:

    python3 src/render_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figures import make_figures  # noqa: E402
from run_analysis import (CACHE, FIG_DIR, METRICS_OUT, build_frames,  # noqa: E402
                          load_tables, headline_scores, PLAY_KIN, PRE_SNAP_CTX)


def reinject_oof(metrics: dict, path: str) -> int:
    """Put the stored OOF / y_true arrays back into the metrics structure."""
    if not os.path.exists(path):
        return 0
    z = np.load(path)
    n = 0
    for key in z.files:
        parts = key.split("__")
        if len(parts) != 3:
            continue
        name, gname, field = parts
        if name not in metrics:
            continue
        bucket = ("models" if field in ("lightgbm", "hist_gb") else "baselines")
        if field in ("y_true",):
            continue
        g = metrics[name].get("groupings", {}).get(gname)
        if not g:
            continue
        slot = g.get(bucket, {}).get(field)
        if isinstance(slot, dict):
            slot["oof"] = z[key].tolist()
            n += 1
        if field == "lightgbm" and isinstance(slot, dict):
            slot["y_true"] = z[f"{name}__{gname}__y_true"].tolist()
    return n


def main() -> None:
    """Rebuild the 8 figures from the stored measurements."""
    with open(METRICS_OUT) as fh:
        metrics = json.load(fh)
    n = reinject_oof(metrics, os.path.join(CACHE, "oof_store.npz"))
    print(f"re-injected {n} OOF arrays", flush=True)
    t = load_tables()
    comb, pf, players, pp, pcs = build_frames(t)
    corr_path = os.path.join("outputs", "feature_target_correlations.csv")
    import pandas as pd
    corr = pd.read_csv(corr_path) if os.path.exists(corr_path) else pd.DataFrame()
    os.makedirs(FIG_DIR, exist_ok=True)
    make_figures(t=t, pf=pf, pl=pd.DataFrame(), comb=comb, cuts=t["cuts"],
                 corr=corr, metrics=metrics, oof=None, dv=t["dv"],
                 headline=metrics.get("headline", {}).get("target",
                                                          "sep_gain::by_game_play"))


if __name__ == "__main__":
    main()
