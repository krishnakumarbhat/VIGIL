"""Track 05 - frontier node n0 expansion: cross-fitted stacker + kNN-prior features.

Idea (stack6): the stage-1 baseline is a 0.8/0.2 simplex blend of two linear
members with a size-dependent bias repair.  This iteration replaces the hand
blend with a *cross-fitted stacker* that learns, per row, how to weigh

* ``svc_extra_claude_w50`` OOF probabilities (18),
* ``nbsvm_all``            OOF probabilities (18),
* ``svc35_w0``             OOF probabilities (18) - NEW diversity member
  (char 3-5 / word 1-3 view; typo-driven char n-grams dominate the 2-5 view,
  so the 3-5 view adds signal instead of duplicating it),
* kNN votes in the ``all`` TF-IDF space (18 unweighted + 18 similarity-weighted
  neighbour shares, neighbours drawn only from *other prompts* of the same fold,
  so the grouped protocol cannot leak),
* member margin / entropy features + log length.

The stacker is itself trained fold-wise (train on the other 4 folds' OOF rows,
predict the held-out fold), with class-balanced weighting so macro recall is
optimised directly.  Protocol stays StratifiedGroupKFold(5, groups=prompt_id,
seed 42); headline = balanced accuracy on the clean subset.

KEEP RULE: commit only if clean-subset balanced accuracy > 0.39061.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntl_features import LABELS, load_competition_frame, load_external_frame  # noqa: E402
from ntl_model import balanced_weights  # noqa: E402
from followups import extra_claude_variant, svc_variant  # noqa: E402
from run_experiments import (CLEAN, CACHE, COMP, EXT, FOLDS, Y, get_matrices, log,  # noqa: E402
                             log_experiment, run_nbsvm, save_oof, score)

OUT = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm/outputs"
BASELINE = 0.39061

os.makedirs(CACHE, exist_ok=True)


def load_or_run(tag: str, fn) -> np.ndarray:
    """Reuse a cached OOF matrix or compute it with the given callable."""
    path = f"{CACHE}/oof_{tag}.npy"
    if os.path.exists(path):
        log(f"cached member {tag}")
        return np.load(path)
    P = fn()
    save_oof(tag, P)
    return P


def knn_features(Xc: sp.csr_matrix, k: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour vote features from other prompts only (grouped folds)."""
    Xn = normalize(Xc)
    F_unw = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    F_w = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    extra = np.zeros((len(COMP), 4), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1).fit(Xn[ti])
        d, idx = nn.kneighbors(Xn[vi])
        s = np.clip(1.0 - d, 0.0, 1.0)                     # cosine similarity
        for row, j in enumerate(range(len(vi))):
            lab = Y[ti[idx[row]]]
            F_unw[vi[row]] = np.bincount(lab, minlength=len(LABELS)) / k
            w = s[row] / max(s[row].sum(), 1e-9)
            F_w[vi[row]] = np.bincount(lab, weights=w, minlength=len(LABELS))
            extra[vi[row]] = [s[row][0], s[row][0] - s[row][1], np.log1p(len(Y[ti])),
                              np.log1p((lab == np.bincount(lab).argmax()).sum())]
        log(f"  knn fold{f} neighbours={k}")
    return np.hstack([F_unw, F_w, extra])


def margin_features(Ps: dict[str, np.ndarray]) -> np.ndarray:
    """Top1-top2 gap, entropy, and log-length per member."""
    feats = []
    for tag, P in Ps.items():
        s = np.sort(P, axis=1)
        feats.append(s[:, -1] - s[:, -2])
        feats.append(-(P * np.log(np.clip(P, 1e-9, 1))).sum(1))
    nw = COMP["text"].str.split().str.len().to_numpy()
    feats.append(np.log1p(nw) / 5.0)
    feats.append(np.log1p(COMP["group"].map(
        pd.Series(COMP["group"].values).value_counts()).to_numpy()))
    return np.stack(feats, axis=1).astype(np.float32)


def run_stack(S: np.ndarray, name: str, C: float = 0.5, kind: str = "lr") -> np.ndarray:
    """Cross-fitted stacker: train fold model on other folds' OOF rows."""
    oof = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        if kind == "lr":
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                     n_jobs=-1)
        else:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                 num_leaves=31, colsample_bytree=0.8,
                                 subsample=0.8, random_state=42,
                                 verbose=-1)
        clf.fit(S[ti], Y[ti])
        if kind == "lr":
            oof[vi] = clf.predict_proba(S[vi])
        else:
            oof[vi] = clf.predict_proba(S[vi])
        log(f"  stack[{name}] fold{f} bal="
            f"{balanced_accuracy_score(Y[vi], oof[vi].argmax(1)):.4f}")
    return oof


def main() -> None:
    Xall, Xe_all = get_matrices("all")
    X35, Xe35 = get_matrices("all35")

    # ---- rebuild the stage-1 members (cache was cleaned between iterations) ----
    t0 = time.time()
    P_svc = load_or_run("svc_extra_claude_w50",
                        lambda: extra_claude_variant(0.5, 0.5, Xall, Xe_all))
    log(f"svc member ready {time.time()-t0:.0f}s")
    P_nb = load_or_run("nbsvm_all", lambda: run_nbsvm(Xall, Xe_all, 0.0))
    P_35 = load_or_run("svc35_w0", lambda: svc_variant("svc35_w0", 0.5, X35))

    Ps = {"svc_extra_claude_w50": P_svc, "nbsvm_all": P_nb, "svc35_w0": P_35}

    # ---- ablation 1: plain simplex blend with the 3rd member ----
    grid = [(w1, w2) for w1 in np.arange(0, 1.01, 0.05) for w2 in np.arange(0, 1.01, 0.05)
            if w1 + w2 <= 1.005]
    best_blend = (-1.0, None)
    for w1, w2 in grid:
        P = w1 * P_svc + w2 * P_nb + (1 - w1 - w2) * P_35
        s = score(P)
        if s > best_blend[0]:
            best_blend = (float(s), (w1, w2, 1 - w1 - w2))
    log(f"3-member blend best {best_blend}")
    log_experiment("bal_acc_blend3_svc35", best_blend[0],
                   f"svc_extra_claude_w50+nbsvm_all+svc35_w0 weights={best_blend[1]}; "
                   "grouped 5-fold clean subset")

    # ---- stacker ----
    Fknn = knn_features(Xall)
    np.save(f"{CACHE}/knn_feats.npy", Fknn)
    Fm = margin_features(Ps)
    S = np.hstack([np.hstack(list(Ps.values())), (Fknn * 2).astype(np.float32),
                   Fm]).astype(np.float32)
    log(f"stacker input S {S.shape}")

    oof_lr = run_stack(S, "lr", kind="lr")
    oof_lgb = run_stack(S, "lgbm", kind="lgbm")
    results = {"blend3": {"bal_acc_clean": best_blend[0]},
               "stack_lr": {"bal_acc_clean": score(oof_lr),
                            "bal_acc_all": score(oof_lr, np.ones(len(Y), bool)),
                            "acc": balanced_accuracy_score(
                                Y[CLEAN], oof_lr[CLEAN].argmax(1))},
               "stack_lgbm": {"bal_acc_clean": score(oof_lgb),
                              "bal_acc_all": score(oof_lgb, np.ones(len(Y), bool))}}
    log(f"results {results}")

    best_tag = max(("blend3", "stack_lr", "stack_lgbm"),
                   key=lambda t: results[t]["bal_acc_clean"])
    best_val = results[best_tag]["bal_acc_clean"]
    keep = best_val > BASELINE
    results["keep"] = keep
    results["baseline"] = BASELINE
    log(f"BEST={best_tag} val={best_val:.5f} keep={keep}")

    if keep and best_tag.startswith("stack"):
        np.save(f"{CACHE}/oof_stack_best.npy",
                oof_lr if best_tag == "stack_lr" else oof_lgb)
    log_experiment(f"bal_acc_{best_tag}", best_val,
                   f"{sorted(Ps)} + knn(2x18+4) + margins; cross-fitted stacker; "
                   f"grouped 5-fold clean subset; baseline={BASELINE}")
    json.dump(results, open(f"{OUT}/cv_stack_report.json", "w"), indent=1)
    log("stack iteration done")


if __name__ == "__main__":
    main()
