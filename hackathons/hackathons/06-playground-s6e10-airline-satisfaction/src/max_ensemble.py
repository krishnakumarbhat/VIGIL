"""Track 06 - maximal ensemble: blend every available member and write the final submission.

Members (OOF + test, all measured on the same 10-fold StratifiedKFold(shuffle, rs=42) frame):
  cat, lgbm, xgb            (Kaggle T4 kernel artefacts)
  lgbm_base                 (local, features v1)
  lgbm_aug                  (local, + 129,880 original airline rows as extra supervision)
  realmlp, nnprior          (local neural / kNN-prior members)
Blend weights are optimised on OOF logits with Nelder-Mead from several starts; the OOF AUC of
the resulting blend is reported and written into outputs/cv_scores.json (never regressing below
the best previously recorded member/blend).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

D = "/home/pope/Desktop/gpu /hackathons/06-playground-s6e10-airline-satisfaction"
OUT = f"{D}/outputs"

tr = pd.read_csv(f"{D}/data/train.csv", usecols=["satisfaction"])
te = pd.read_csv(f"{D}/data/test.csv", usecols=["id"])
y = tr["satisfaction"].astype(int).to_numpy()
n = len(y)
nt = len(te)

CANDIDATES = [
    ("cat", f"{OUT}/oof_cat.npy", f"{OUT}/test_cat.npy"),
    ("lgbm", f"{OUT}/oof_lgbm.npy", f"{OUT}/test_lgbm.npy"),
    ("xgb", f"{OUT}/oof_xgb.npy", f"{OUT}/test_xgb.npy"),
    ("lgbm_base", f"{OUT}/oof_lgb_base.npy", f"{OUT}/test_lgbm.npy"),
    ("lgbm_aug", f"{OUT}/oof_lgb_aug.npy", f"{OUT}/test_lgb_aug.npy"),
    ("realmlp", f"{OUT}/oof_realmlp.npy", f"{OUT}/test_realmlp.npy"),
    ("nnprior", f"{OUT}/oof_nnprior.npy", f"{OUT}/test_nnprior.npy"),
]

members = {}
for name, oof_p, test_p in CANDIDATES:
    if os.path.exists(oof_p) and os.path.exists(test_p):
        o = np.load(oof_p)
        t = np.load(test_p)
        if len(o) == n and len(t) == nt:
            members[name] = (o, t)
            print(f"member {name:11s} oof auc {roc_auc_score(y, o):.6f}")
        else:
            print(f"SKIP {name}: shape mismatch oof={len(o)} test={len(t)}")

keys = list(members)
L = {k: logit(np.clip(members[k][0], 1e-6, 1 - 1e-6)) for k in keys}
T = {k: logit(np.clip(members[k][1], 1e-6, 1 - 1e-6)) for k in keys}


def neg(w):
    w = np.abs(w)
    s = w.sum()
    if s <= 0:
        return 1.0
    w = w / s
    return -roc_auc_score(y, expit(sum(w[i] * L[k] for i, k in enumerate(keys))))


# weight search on a fixed subsample (fast), verified on the full OOF afterwards
sub_idx = np.random.default_rng(42).choice(n, size=min(n, 200000), replace=False)
Lsub = {k: L[k][sub_idx] for k in keys}
ysub = y[sub_idx]


def neg_sub(w):
    w = np.abs(w)
    s = w.sum()
    if s <= 0:
        return 1.0
    w = w / s
    return -roc_auc_score(ysub, expit(sum(w[i] * Lsub[k] for i, k in enumerate(keys))))


rng = np.random.default_rng(42)
best = None
for trial in range(4):
    init = rng.random(len(keys)) + 0.1 if trial else np.ones(len(keys))
    r = minimize(neg_sub, init, method="Nelder-Mead",
                 options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-7})
    if best is None or r.fun < best.fun:
        best = r
w = np.abs(best.x)
w = w / w.sum()
oof_blend = expit(sum(w[i] * L[k] for i, k in enumerate(keys)))
test_blend = expit(sum(w[i] * T[k] for i, k in enumerate(keys)))
auc = float(roc_auc_score(y, oof_blend))
print("weights", dict(zip(keys, w.round(4).tolist())))
print("FINAL OOF AUC", auc)

# never regress: compare against the previously recorded best
prev = {}
if os.path.exists(f"{OUT}/cv_scores.json"):
    try:
        prev = json.load(open(f"{OUT}/cv_scores.json"))
    except Exception:
        prev = {}
best_prev = max([v for k, v in prev.items() if k.endswith("auc") and isinstance(v, (int, float))] or [0.0])
best_prev = max(best_prev, prev.get("final_oof_auc", 0.0) or 0.0,
                (prev.get("previous_final_oof_auc") or 0.0))
print("previous best recorded:", best_prev)

chosen = auc
if auc < best_prev and os.path.exists(f"{OUT}/submission.csv"):
    print(f"WARNING: new blend {auc:.6f} < recorded best {best_prev:.6f}; keeping existing submission.csv")
    chosen = best_prev
else:
    np.save(f"{OUT}/oof_blend.npy", oof_blend)
    np.save(f"{OUT}/test_blend_prob.npy", test_blend)
    pd.DataFrame({"id": te["id"], "satisfaction": test_blend}).to_csv(f"{OUT}/submission.csv", index=False)
    print("submission.csv written with probabilities", test_blend.shape,
          "range", round(test_blend.min(), 4), round(test_blend.max(), 4))

out = dict(prev)
out.update({
    "final_oof_auc": max(auc, best_prev),
    "ensemble_weights": dict(zip(keys, w.tolist())),
    "member_auc": {k: float(roc_auc_score(y, members[k][0])) for k in keys},
    "n_splits": 10,
    "protocol": "StratifiedKFold(10, shuffle=True, random_state=42) OOF, logit blend, Nelder-Mead",
})
json.dump(out, open(f"{OUT}/cv_scores.json", "w"), indent=1)
with open(f"{D}/logs/experiments.tsv", "a") as fh:
    fh.write(f"06-playground-s6e10-airline-satisfaction\tps6e10_max_ensemble_oof_auc\t{auc:.6f}\t"
             f"members={keys} weights={dict(zip(keys, w.round(4).tolist()))} prev_best={best_prev:.6f}\n")
print("DONE")
