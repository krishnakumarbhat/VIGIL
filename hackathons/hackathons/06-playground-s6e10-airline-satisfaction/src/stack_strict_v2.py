"""Node n5 / idea I6 leg-2: STRICT cross-fitted re-verification of the 4-member stack.

realmlp_stack.py's coordinate-ascent weights (blend_v2 = 0.959628) were fitted
IN-SAMPLE on the full OOF AUC and are optimistic (Purucker & Beel 2023:
validation-set weight search "overfits drastically"). This script re-verifies
the {lgbm, cat, xgb, realmlp(, nnprior)} stack with weights learned strictly
out-of-fold: the same StratifiedKFold(10, shuffle, seed=42) splits as every
member, meta-learner fitted on meta-train rows only, prediction on meta-val
rows, concatenated as the OOF estimate.

Also quantifies the stack-weight optimism
    Delta_optimism = AUC_insample_weights - AUC_strict_OOF
on the fixed n=699,636 OOF panel, logged to equations.md.

Run: python3 src/stack_strict_v2.py
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

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
NF, SEED, CLIP = 10, 42, 1e-6
C_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1000.0)

T0 = time.time()


def log(*a) -> None:
    """Timestamped progress log."""
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load_y_members() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load ground truth and banked fold-honest OOF member probability vectors."""
    y = pd.read_csv(f"{DATA}/train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    mem: Dict[str, np.ndarray] = {}
    for n in ("lgbm", "cat", "xgb", "realmlp", "nnprior"):
        v = np.load(f"{OUT}/oof_{n}.npy").astype(np.float64)
        assert len(v) == len(y), n
        mem[n] = logit(np.clip(v, CLIP, 1 - CLIP))
    return y, mem


def strict_lr_stack(y: np.ndarray, M: np.ndarray, C: float) -> float:
    """Strict OOF AUC of an L2-regularised cross-fitted LR meta-stacker."""
    oof = np.zeros(len(y), dtype=np.float64)
    for ti, vi in StratifiedKFold(NF, shuffle=True, random_state=SEED).split(M, y):
        r = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                               solver="liblinear")
        r.fit(M[ti], y[ti])
        oof[vi] = r.decision_function(M[vi])
    return roc_auc_score(y, oof)


def coord_ascent(members: Dict[str, np.ndarray], y: np.ndarray) -> Tuple[Dict[str, float], float]:
    """Greedy scalar weights on the logit scale, evaluated on the rows given."""
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


def strict_coord_ascent(y: np.ndarray, mem: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
    """Cross-fitted coordinate ascent: per outer fold, weights fit on meta-train
    rows only, applied to meta-val rows; concatenated OOF blend reported."""
    names = list(mem)
    oof = np.zeros(len(y), dtype=np.float64)
    w_accum = {n: 0.0 for n in names}
    for ti, vi in StratifiedKFold(NF, shuffle=True, random_state=SEED).split(mem[names[0]], y):
        f_mem = {n: mem[n][ti] for n in names}
        w_f, _ = coord_ascent(f_mem, y[ti])
        for n in names:
            w_accum[n] += w_f[n] / NF
        oof[vi] = sum(w_f[n] * mem[n][vi] for n in names)
    return roc_auc_score(y, oof), w_accum


def main() -> None:
    """Entry point: strict stack sweep + optimism quantification."""
    y, mem = load_y_members()
    names4 = ["lgbm", "cat", "xgb", "realmlp"]
    names5 = names4 + ["nnprior"]
    base = np.load(f"{OUT}/oof_blend.npy").astype(np.float64)
    base_auc = roc_auc_score(y, base)
    log(f"baseline 3-member blend auc {base_auc:.6f}  (target to beat)")

    in_w, in_auc = coord_ascent({n: mem[n] for n in names4}, y)
    log(f"in-sample coord-ascent reference (realmlp_stack.py): {in_auc:.6f} weights {in_w}")

    mat4L = np.column_stack([mem[n] for n in names4])
    mat4LP = np.column_stack([mem[n] for n in names4] + [expit(mem[n]) for n in names4])
    mat5L = np.column_stack([mem[n] for n in names5])

    results: Dict[str, float] = {}
    best_cfg, best_auc, best_oof = "baseline_blend", base_auc, base
    for block, M in (("logits4", mat4L), ("logits+probs4", mat4LP), ("logits5", mat5L)):
        for C in C_SWEEP:
            auc = strict_lr_stack(y, M, C)
            tag = f"lr_{block}_C{C:g}"
            results[tag] = auc
            log(f"{tag}: auc {auc:.6f}")
            if auc > best_auc:
                best_cfg, best_auc = tag, auc

    s_ca_auc, s_ca_w = strict_coord_ascent(y, {n: mem[n].copy() for n in names4})
    tag = "strict_coord_ascent4"
    results[tag] = s_ca_auc
    log(f"{tag}: auc {s_ca_auc:.6f} mean-weights {s_ca_w}")
    if s_ca_auc > best_auc:
        best_cfg, best_auc = tag, s_ca_auc

    optimism = in_auc - results.get(tag, s_ca_auc)
    log(f"stack-weight optimism Delta = in-sample {in_auc:.6f} - strict {results[tag]:.6f}"
        f" = {optimism:+.6f}")

    # refit best config once on the full panel to stage its OOF blend
    if best_cfg.startswith("lr_"):
        block = "logits5" if "logits5" in best_cfg else (
            "logits+probs4" if "probs" in best_cfg else "logits4")
        M = {"logits4": mat4L, "logits+probs4": mat4LP, "logits5": mat5L}[block]
        C = float(best_cfg.split("_C")[1].replace("e+", "e"))
        oof = np.zeros(len(y), dtype=np.float64)
        for ti, vi in StratifiedKFold(NF, shuffle=True, random_state=SEED).split(M, y):
            r = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                   solver="liblinear")
            r.fit(M[ti], y[ti])
            oof[vi] = r.decision_function(M[vi])
        best_oof = oof
        np.save(f"{OUT}/oof_blend_strict.npy", oof)
    elif best_cfg == tag:
        pass  # strict coord ascent blend not staged; LR stacks carry test-side better

    keep = bool(best_auc > base_auc)
    report = {
        "metric": "roc_auc",
        "protocol": f"StratifiedKFold({NF},shuffle=True,rs={SEED}) strict cross-fitted meta",
        "baseline_blend_auc": round(base_auc, 6),
        "insample_coord_ascent_auc": round(in_auc, 6),
        "stack_weight_optimism_delta": round(optimism, 6),
        "strict_results": {k: round(v, 6) for k, v in results.items()},
        "best_cfg": best_cfg,
        "best_strict_auc": round(best_auc, 6),
        "improvement_vs_baseline": round(best_auc - base_auc, 6),
        "keep": keep,
        "member_auc": {n: round(roc_auc_score(y, v), 6) for n, v in mem.items()},
    }
    with open(f"{OUT}/cv_stack_strict.json", "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"BEST {best_cfg} strict auc {best_auc:.6f} vs baseline {base_auc:.6f} -> "
        f"{'KEEP' if keep else 'DISCARD'}")
    with open(f"{LOGS}/experiments.tsv", "a") as fh:
        fh.write(f"track06\troc_auc\t{best_auc:.6f}\tstrict cross-fitted re-verify "
                 f"{best_cfg}; optimism {optimism:+.6f}; keep={keep}\n")


if __name__ == "__main__":
    main()
