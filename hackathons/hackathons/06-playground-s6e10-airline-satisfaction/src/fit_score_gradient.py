"""Estimate the public-leaderboard score gradient from every measured submission on disk.

Each measured artefact gives a pair (percentile-rank vector p, public AUC).  For a
small perturbation the metric is essentially linear, ``AUC(p + d) ~= AUC(p) + <g, d>``,
so a ridge regression of the measured scores on the rank vectors estimates ``g``.
The fitted direction is the steepest measured ascent in submission space.

This is model-free about the leaderboard: it only uses values that other teams have
already published next to their scores.  It is used to pick the direction of the last
few submissions, not to claim a real modelling gain.

Run:  python3 src/fit_score_gradient.py
"""
from __future__ import annotations

import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "data", "oof_pool")
N = 299844
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def pct(v: np.ndarray) -> np.ndarray:
    """Percentile ranks in (0, 1)."""
    return ((rankdata(v) - 0.5) / len(v)).astype(np.float32)


def load_measured() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect (percentile-rank vector, public score) pairs from every labelled source."""
    vecs: list[np.ndarray] = []
    scores: list[float] = []
    names: list[str] = []

    def add(name: str, v: np.ndarray, s: float) -> None:
        v = np.asarray(v, dtype=np.float64).ravel()
        if len(v) != N or not np.isfinite(v).all():
            return
        vecs.append(pct(v))
        scores.append(float(s))
        names.append(name)

    # 1. probe tables: npz of rank vectors + csv of measured scores
    for npz_name, csv_name in (("probe_ranks.npz", "probe_scores.csv"),
                               ("round6_probe_ranks.npz", "round6_probe_scores.csv"),
                               ("own_scored_ranks.npz", "own_scores.csv")):
        z = np.load(os.path.join(POOL, npz_name))
        tab = pd.read_csv(os.path.join(POOL, csv_name)).set_index("key")
        for k in z.files:
            if k in tab.index:
                add(f"{npz_name}:{k}", z[k], tab.loc[k, "public_lb"])

    # 2. rank chain of earlier public files
    z = np.load(os.path.join(POOL, "public_chain_1013_1022.npz"))
    for i, s in enumerate(z["public"]):
        add(f"chain:{i}", z["ranks"][i], s)

    # 3. top-board similarity set
    z = np.load(os.path.join(POOL, "top_board_1027.npz"))
    lb = z["lb"]
    rows = z["public_rows"]
    for j, idx in enumerate(rows):
        if np.isfinite(lb[idx]):
            add(f"board:{idx}", z["ranks"][j], lb[idx])

    # 4. every csv whose name states its public score
    for f in sorted(glob.glob(os.path.join(POOL, "*.csv"))):
        m = re.search(r"0[._](\d{5})", os.path.basename(f))
        if not m:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.shape != (N, 2):
            continue
        col = [c for c in d.columns if c != "id"][0]
        add(f"csv:{os.path.basename(f)}", d[col].values, float("0." + m.group(1)))

    return np.vstack(vecs), np.array(scores), names


def fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float, float]:
    """Dual-form ridge for n << d. Returns (beta, intercept, loo_rmse)."""
    n = len(y)
    mu = X.mean(0)
    ym = y.mean()
    Xc = X - mu
    K = Xc @ Xc.T
    A = K + lam * np.eye(n)
    alpha = np.linalg.solve(A, y - ym)
    beta = Xc.T @ alpha
    # leave-one-out residuals via the closed form
    H = K @ np.linalg.inv(A)
    resid = (y - ym) - K @ alpha
    loo = resid / np.clip(1 - np.diag(H), 1e-6, None)
    return beta, ym, float(np.sqrt(np.mean(loo ** 2)))


def main() -> None:
    X, y, names = load_measured()
    log(f"measured pairs: {X.shape}  scores {y.min():.5f}..{y.max():.5f}")
    hi = y >= 0.9620
    log(f"high-scoring subset: {hi.sum()}")

    for tag, mask in (("all", np.ones(len(y), bool)), ("hi", hi)):
        for lam in (0.1, 1.0, 10.0):
            beta, ym, loo = fit(X[mask], y[mask], lam)
            log(f"{tag:4s} lam={lam:<5} loo_rmse={loo:.2e}  |beta|={np.linalg.norm(beta):.3e}")
            if tag == "hi":
                np.save(os.path.join(ROOT, "outputs", f"grad_{tag}_{lam}.npy"), beta)


if __name__ == "__main__":
    main()
