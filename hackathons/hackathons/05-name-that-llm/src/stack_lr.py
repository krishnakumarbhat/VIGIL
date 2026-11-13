"""Track 05 - linear meta-learner stacker (v3) with nested model selection.

The v2 blender searched three things by hand: per-member temperature, blend weights, and a
per-class bias. That is a *constrained* model: every member contributes exactly one log-prob
vector scaled by a scalar, so a member cannot be trusted more on some classes than others.

v3 replaces it with a multinomial logistic regression over the concatenation of all members'
log-probabilities (d = n_members x 18). That learns a full cross-class weight matrix, which is
what the observed error structure demands: `glm-5.3` is confused with `glm-5.3-flash` and
`gpt-5.x`, so the meta-model can down-weight the members exactly on those columns.

Protocol (honest, no test leakage)
  * outer: StratifiedGroupKFold(5, shuffle, seed 42, groups=prompt_id) over the 2,699 real rows,
  * inner: the same splitter (4 folds) on the outer training folds to choose the L2 strength,
  * the held-out outer fold is predicted exactly once.

Variants compared: log-probs only vs log-probs+probs, and unweighted vs class-balanced loss.
The winner (by cross-fitted balanced accuracy) is exported to `outputs/ensemble_recipe.json` as
a plain linear map `z = W @ features + b`, which the submission notebook can apply to any member
prediction matrix.
"""
import glob
import json
import os

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

D = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{D}/outputs"
LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
C = len(LABELS)


def log(msg):
    print(msg, flush=True)


row_index = pd.read_parquet(f"{OUT}/kaggle_deberta/row_index_42.parquet")
n = len(row_index)
y = row_index["y"].to_numpy()
groups = row_index["prompt_id"].to_numpy()


def load_member(prefix, directory=OUT):
    mats, idxs = [], []
    for fold in range(5):
        op = f"{directory}/oof_{prefix}_fold{fold}.npy"
        ip = f"{directory}/oof_{prefix}_idx_fold{fold}.npy"
        if os.path.exists(op) and os.path.exists(ip):
            mats.append(np.load(op))
            idxs.append(np.load(ip))
    if len(mats) < 5:
        return None
    full = np.zeros((n, C))
    m = np.zeros(n, dtype=bool)
    for a, i in zip(mats, idxs):
        full[i] = np.asarray(a, dtype=np.float64)
        m[i] = True
    return np.clip(full, 1e-9, 1.0), m


members, masks = {}, {}
if os.path.exists(f"{OUT}/kaggle_deberta/oof_42.npy"):
    members["deberta_base"] = np.clip(np.load(f"{OUT}/kaggle_deberta/oof_42.npy").astype(np.float64), 1e-9, 1)
    masks["deberta_base"] = np.ones(n, dtype=bool)
for f0 in sorted(glob.glob(f"{OUT}/oof_*_fold0.npy")):
    prefix = os.path.basename(f0)[len("oof_"):-len("_fold0.npy")]
    if prefix.startswith(("real", "stack")) or prefix.endswith(("_idx", "_y")):
        continue
    got = load_member(prefix)
    if got is None:
        continue
    members[prefix], masks[prefix] = got
for extra, key in (("colab_pro_artifacts", "a100_7b"), ("kaggle_qwen_lora", "kaggle_7b"),
                   ("colab_artifacts", "colab_7b")):
    d = f"{OUT}/{extra}"
    mats, idxs = [], []
    for fold in range(5):
        op, ip = f"{d}/oof_real_fold{fold}.npy", f"{d}/oof_real_idx_fold{fold}.npy"
        if os.path.exists(op) and os.path.exists(ip):
            mats.append(np.load(op))
            idxs.append(np.load(ip))
    if len(mats) == 5:
        full = np.zeros((n, C))
        m = np.zeros(n, dtype=bool)
        for a, i in zip(mats, idxs):
            full[i] = np.asarray(a, dtype=np.float64)
            m[i] = True
        members[key], masks[key] = np.clip(full, 1e-9, 1.0), m

keys = sorted(members)
for k in keys:
    log(f"member {k:16s} bal_acc {balanced_accuracy_score(y, members[k].argmax(1)):.4f} "
        f"coverage {int(masks[k].sum())}")
valid = np.ones(n, dtype=bool)
for k in keys:
    valid &= masks[k]
log(f"full-coverage rows {int(valid.sum())}/{n}, members {len(keys)}")


def features(sel_idx, use_probs=True):
    """Concatenate member log-probs (always) and probs (optionally) for the given rows."""
    blocks = []
    for k in keys:
        lp = np.log(members[k][sel_idx])
        blocks.append(lp)
        if use_probs:
            blocks.append(members[k][sel_idx])
    return np.concatenate(blocks, axis=1)


outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
             .split(np.zeros(n), y, groups=groups))
C_GRID = [0.05, 0.2, 0.8]

variants = {(True, None): np.zeros((n, C)), (True, "balanced"): np.zeros((n, C)),
            (False, "balanced"): np.zeros((n, C))}
inner_scores = {k: [] for k in variants}

for fold, (tr, va) in enumerate(outer):
    tr = tr[valid[tr]]
    va = va[valid[va]]
    Xtr_all = {True: features(tr, True)}
    Xva_all = {True: features(va, True)}
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=7)
                 .split(np.zeros(len(tr)), y[tr], groups=groups[tr]))
    Xtr_all[False] = features(tr, False)
    Xva_all[False] = features(va, False)
    for (up, cw) in variants:
        Xtr = Xtr_all[up]
        best_c, best_s = C_GRID[0], -1.0
        for c in C_GRID:
            sc = []
            for itr, iva in inner:
                clf = LogisticRegression(C=c, max_iter=800, tol=1e-3, class_weight=cw)
                clf.fit(Xtr[itr], y[tr][itr])
                sc.append(balanced_accuracy_score(y[tr][iva], clf.predict(Xtr[iva])))
            m = float(np.mean(sc))
            if m > best_s:
                best_s, best_c = m, c
        clf = LogisticRegression(C=best_c, max_iter=800, tol=1e-3, class_weight=cw)
        clf.fit(Xtr, y[tr])
        variants[(up, cw)][va] = clf.predict_proba(Xva_all[up])
        inner_scores[(up, cw)].append(best_c)
    log(f"outer fold {fold}: " + ", ".join(
        f"{'lp+p' if up else 'lp'} cw={cw} val={balanced_accuracy_score(y[va], variants[(up, cw)][va].argmax(1)):.4f}"
        for (up, cw) in variants))

results = {}
for (up, cw), oof in variants.items():
    sel = valid & (oof.sum(axis=1) > 0)
    bal = float(balanced_accuracy_score(y[sel], oof[sel].argmax(1)))
    acc = float(accuracy_score(y[sel], oof[sel].argmax(1)))
    name = f"{'logprob+prob' if up else 'logprob'}|{cw or 'plain'}"
    results[name] = {"bal_acc": bal, "acc": acc, "C_per_fold": inner_scores[(up, cw)]}
    log(f"{name:24s} bal_acc {bal:.4f} acc {acc:.4f}")

best_name = max(results, key=lambda k: results[k]["bal_acc"])
best_key = [(up, cw) for (up, cw) in variants if f"{'logprob+prob' if up else 'logprob'}|{cw or 'plain'}" == best_name][0]
log(f"\nBEST: {best_name} bal_acc {results[best_name]['bal_acc']:.4f}")
oof_best = variants[best_key]
sel = valid & (oof_best.sum(axis=1) > 0)
per_class = recall_score(y[sel], oof_best[sel].argmax(1), labels=list(range(C)), average=None,
                         zero_division=0)
log("per-class recall: " + ", ".join(f"{LABELS[i]}={per_class[i]:.2f}" for i in range(C)))

# ---------------------------------------------------------------- export portable linear map
up, cw = best_key
prev_best = 0.0
for rep in ("stack_v2_report.json", "stack_v3_report.json"):
    fp = f"{OUT}/{rep}"
    if os.path.exists(fp):
        prev_best = max(prev_best, float(json.load(open(fp)).get("bal_acc", 0.0)))
log(f"best recorded stacker score (v2/v3): {prev_best:.4f}")
cv_best_c = float(np.median(inner_scores[best_key]))
clf = LogisticRegression(C=cv_best_c, max_iter=2000, tol=1e-4, class_weight=cw)
clf.fit(features(np.where(valid)[0], up), y[valid])

report = {"bal_acc": results[best_name]["bal_acc"], "acc": results[best_name]["acc"],
          "variant": best_name, "members": keys, "C": cv_best_c,
          "rows": int(sel.sum()), "all_variants": results,
          "per_class_recall": {LABELS[i]: float(per_class[i]) for i in range(C)},
          "protocol": "outer SGKF(5,s42,groups=prompt_id) x inner SGKF(4,s7) C selection; "
                      "per-fold predictions only on held-out rows"}
json.dump(report, open(f"{OUT}/stack_v3_report.json", "w"), indent=1)

if report["bal_acc"] > prev_best + 1e-9:
    recipe = {"type": "linear_meta", "members": keys, "use_probs": bool(up), "C": cv_best_c,
              "class_weight": cw, "labels": LABELS,
              "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
              "feature_spec": ("concat over members in `members` order of "
                               "[log p_i] (+ [p_i] when use_probs); then z = coef @ f + intercept; "
                               "pred = argmax(z). Drop members you cannot compute and refit or "
                               "zero their blocks."),
              "oof_bal_acc": report["bal_acc"]}
    json.dump(recipe, open(f"{OUT}/ensemble_recipe.json", "w"), indent=1)
    log("wrote ensemble_recipe.json (linear_meta)")
else:
    log(f"guard: {report['bal_acc']:.4f} <= previous {prev_best:.4f}, recipe kept")

np.save(f"{OUT}/oof_real_stack_v3.npy", oof_best)
with open(f"{D}/logs/experiments.tsv", "a") as fh:
    fh.write(f"05-name-that-llm\tstack_v3_{best_name}\t{results[best_name]['bal_acc']:.5f}"
             f"\tmembers={keys} rows={int(sel.sum())}\n")
log("DONE")
