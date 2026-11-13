"""Addendum to stack_probe: rank-mean blend + probs-only stack + logit adjustment.

Same folds/protocol, uses the cached member OOFs only. KEEP rule unchanged:
clean-subset BA must beat 0.39061 by a margin (>+0.002) to count as a keep.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import CLEAN, CACHE, COMP, FOLDS, LABELS, Y, log, score  # noqa: E402

OUT = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm/outputs"
BASELINE = 0.39061
PRIOR = np.bincount(Y, minlength=len(LABELS)).astype(np.float64)
PRIOR /= PRIOR.sum()

res: dict[str, dict] = {}
tags = ["svc_extra_claude_w50", "nbsvm_all", "svc35_w0"]
Ps = {t: np.load(f"{CACHE}/oof_{t}.npy") for t in tags}


def logit_adj(P: np.ndarray, alpha: float) -> np.ndarray:
    return P / (PRIOR[None, :] ** alpha).astype(np.float32)


# ---- a) rank-mean blend + alpha grid ----
R = {}
for t, P in Ps.items():
    rp = np.column_stack([rankdata(P[:, c], method="average") for c in range(len(LABELS))])
    R[t] = rp / len(P)
best = (-1.0, None, None)
for w in ((0.1, 0.0, 0.9), (0.2, 0.1, 0.7), (0.3, 0.0, 0.7), (0.2, 0.0, 0.8),
          (0.1, 0.1, 0.8), (0.25, 0.25, 0.5), (0.5, 0.5, 0.0), (0.4, 0.1, 0.5),
          (0.3, 0.3, 0.4), (0.15, 0.05, 0.8)):
    P = sum(wi * R[t] for wi, t in zip(w, tags))
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = score(logit_adj(P, a))
        if s > best[0]:
            best = (s, w, a)
res["rankmean"] = {"bal_acc_clean": best[0], "weights": best[1], "alpha": best[2]}
log(f"rankmean best {best}")

# ---- b) stack on probs only (no knn/margins) ----
S = np.hstack(list(Ps.values())).astype(np.float32)
stk: dict[str, float] = {}
for name, clf in {
    "lr_c05": LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", n_jobs=-1),
    "lr_c1": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", n_jobs=-1),
    "lr_c2": LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced", n_jobs=-1),
}.items():
    oof = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        clf.fit(S[ti], Y[ti])
        oof[vi] = clf.predict_proba(S[vi])
    for a in (0.0, 0.25, 0.5):
        s = score(logit_adj(oof, a))
        stk[f"{name}_a{a}"] = s
        log(f"  {name} a={a} -> {s:.4f}")
oj = np.load(f"{CACHE}/knn_feats.npy")
S2 = np.hstack([S, (oj * 2).astype(np.float32)]).astype(np.float32)
oof2 = np.zeros_like(S2[:, :len(LABELS)], dtype=np.float32)
clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", n_jobs=-1)
for f, (ti, vi) in enumerate(FOLDS):
    clf.fit(S2[ti], Y[ti])
    oof2[vi] = clf.predict_proba(S2[vi])
for a in (0.0, 0.25, 0.5):
    stk[f"lr2_knn_a{a}"] = score(logit_adj(oof2, a))
    log(f"  lr2_knn a={a} -> {stk[f'lr2_knn_a{a}']:.4f}")
res["stack_probs_only"] = {"best": max(stk.values()), "grid": stk}

best_all = max(("rankmean", "stack"), key=lambda k: res[k].get("bal_acc_clean", res[k]["best"]))
top = res[best_all].get("bal_acc_clean", res[best_all]["best"])
res["verdict"] = {"best_tag": best_all, "value": top, "baseline": BASELINE,
                  "improvement": top - BASELINE, "keep": top > BASELINE + 0.002}
log(f"verdict {res['verdict']}")
json.dump(res, open(f"{OUT}/cv_stack_addendum.json", "w"), indent=1)
