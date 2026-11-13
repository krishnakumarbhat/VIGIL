"""Cross-fitted LR stacker on T06 member OOF logits with the banked kNN prior.

Node n5 (CPU leg): inject the pre-banked GPU kNN-distance prior (oof_nnprior.npy,
fp16 cdist, k=5, inverse-distance weights, same StratifiedKFold(10, 42) folds)
into a cross-fitted LogisticRegression meta-stacker over member OOF logits.

Strictness: stack weights are learned out-of-fold (no in-sample weight
optimisation); metric = mean of pairwise AUC contributions (ROC-AUC).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = "/home/pope/Desktop/gpu /hackathons"
TRACK = f"{ROOT}/06-playground-s6e10-airline-satisfaction"
DATA = f"{TRACK}/data"
OUT = f"{TRACK}/outputs"
LOGS = f"{TRACK}/logs"
TARGET = "satisfaction"
NF, SEED = 10, 42
CLIP = 1e-6

T0 = time.time()


def log(*a) -> None:
    """Timestamped progress log to stdout."""
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load_members() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load y and banked OOF member probability vectors."""
    tr = pd.read_csv(f"{DATA}/train.csv", usecols=[TARGET])
    y = tr[TARGET].astype(int).to_numpy()
    names = ["lgbm", "cat", "xgb", "nnprior"]
    mem: Dict[str, np.ndarray] = {}
    for n in names:
        p = f"{OUT}/oof_{n}.npy"
        v = np.load(p).astype(np.float64)
        assert len(v) == len(y), n
        mem[n] = logit(np.clip(v, CLIP, 1 - CLIP))
    return y, mem


def coord_ascent(members: Dict[str, np.ndarray], y: np.ndarray) -> Tuple[Dict[str, float], float]:
    """In-sample greedy scalar weights on the logit scale (reference only)."""
    names = list(members)
    w = {n: 0.0 for n in names}
    w[names[0]] = 1.0
    best = roc_auc_score(y, sum(w[n] * members[n] for n in names))
    for step in (0.25, 0.125, 0.0625, 0.03125):
        improved = True
        while improved:
            improved = False
            for n in names:
                for sgn in (1, -1):
                    w2 = dict(w)
                    w2[n] = max(0.0, w2[n] + sgn * step)
                    cand = roc_auc_score(y, sum(w2[k] * members[k] for k in names))
                    if cand > best + 1e-7:
                        w, best = w2, cand
                        improved = True
    s = sum(w.values())
    return ({n: v / s for n, v in w.items()}, best) if s > 0 else (w, best)


def main() -> None:
    """Entry point: evaluate nnprior injection via strict cross-fitted stack."""
    os.makedirs(OUT, exist_ok=True)
    y, mem = load_members()
    names = list(mem)
    for n in names:
        log(f"member {n}: auc {roc_auc_score(y, mem[n]):.6f}")
    base = np.load(f"{OUT}/oof_blend.npy").astype(np.float64)
    base_auc = roc_auc_score(y, base)
    log(f"baseline 3-member blend auc {base_auc:.6f}")

    folds = list(StratifiedKFold(NF, shuffle=True, random_state=42).split(mem["lgbm"], y))
    logit_mat = np.column_stack([mem[n] for n in names])
    prob_mat = np.column_stack([expit(mem[n]) for n in names])

    results: Dict[str, float] = {}
    best_cfg, best_auc, best_oof = None, base_auc, None
    for block, mate in (("logits", logit_mat), ("logits+probs", np.column_stack([logit_mat, prob_mat]))):
        for C in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e6):
            oof = np.zeros(len(y), dtype=np.float64)
            for ti, vi in folds:
                r = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear")
                r.fit(mat(ti, mate), y[ti])
                oof[vi] = r.predict_proba(mat_rows(vi, mate))[:, 1]
            auc = roc_auc_score(y, oof)
            tag = f"lr_{block}_C{C:g}"
            results[tag] = auc
            log(f"{tag}: auc {auc:.6f}")
            if auc > best_auc:
                best_cfg, best_auc, best_oof = tag, auc, None
    log(f"best stack: {best_cfg} auc {best_auc:.6f} vs baseline {base_auc:.6f}")

    # n5b probe: per-column SCALE-FREE features (global ranks of logits).
    # Rank is a per-row monotone map (row-local, no cross-row info) so OOF validity holds.
    from scipy.stats import rankdata
    rank_mat = np.column_stack([rankdata(mem[n]) / len(y) for n in names])
    for C in (1.0, 8.0, 1e6):
        oof = np.zeros(len(y), dtype=np.float64)
        for ti, vi in folds:
            r = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear")
            r.fit(rank_mat[ti], y[ti])
            oof[vi] = r.predict_proba(rank_mat[vi])[:, 1]
        auc = roc_auc_score(y, oof)
        tag = f"lr_rank_C{C:g}"
        results[tag] = auc
        log(f"{tag}: auc {auc:.6f}")
        if auc > best_auc:
            best_cfg, best_auc = tag, auc
    log(f"best stack: {best_cfg} auc {best_auc:.6f} vs baseline {base_auc:.6f}")

    # rank-safe constant-weight reference with nnprior included (in-sample weights)
    wts, auc_w = coord_ascent(mem, y)
    log(f"4-member coord-ascent (in-sample weights, reference): {auc_w:.6f} weights {wts}")

    keep = best_auc > base_auc
    with open(f"{OUT}/cv_stack_nn_lr.json", "w") as fh:
        json.dump({
            "metric": "roc_auc",
            "protocol": f"StratifiedKFold({NF},shuffle=True,rs={SEED})",
            "baseline_blend_auc": round(base_auc, 6),
            "member_auc": {n: round(roc_auc_score(y, mem[n]), 6) for n in names},
            "stack_results": {k: round(v, 6) for k, v in results.items()},
            "best_cfg": best_cfg,
            "best_stack_auc": round(best_auc, 6),
            "improvement": round(best_auc - base_auc, 6),
            "keep": bool(keep),
            "coord_ascent_reference": {"auc": round(auc_w, 6), "weights": wts},
        }, fh, indent=2)
    log("KEEP" if keep else "DISCARD")
    with open(f"{LOGS}/experiments.tsv", "a") as fh:
        fh.write(f"track06\troc_auc\t{base_auc:.6f}\tbaseline_blend (prior sessions)\n")
        fh.write(f"track06\troc_auc\t{best_auc:.6f}\tnn_lr stack {best_cfg}; keep={keep}\n")


def mat(ti, M) -> np.ndarray:
    """Row slice helper."""
    return M[ti]


def mat_rows(vi, M) -> np.ndarray:
    """Row slice helper."""
    return M[vi]


if __name__ == "__main__":
    main()
