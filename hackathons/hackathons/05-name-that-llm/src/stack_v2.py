"""Track 05 - cross-fitted stacker with per-class logit adjustment (v2).

Why v2
------
The v1 blender only searched blend weights. Its per-class recall table exposed a systematic
problem: several models (`glm-5.3`, `gpt-5.6-sol`, `glm-5.3-flash`, `gpt-5.6-luna`) were
almost never predicted (recall 0.047-0.17) even though accuracy was ~0.40. Because the
competition metric is a macro (balanced) average, that tail dominates the loss.

v2 fits, inside each cross-validation fold:
  1. a **temperature** per member (NLL-optimal, scipy bounded scalar search),
  2. **blend weights** on the temperature-scaled log-probabilities,
  3. a **per-class additive bias** vector tuned by coordinate ascent directly on balanced
     accuracy (step halving, bounded to |b| <= 2.0 to limit overfitting on the 540-row folds).

Everything is fit on 4 folds and applied to the held-out fold, so the reported number is
honest. The fitted recipe (temps, weights, biases) is exported to
`outputs/ensemble_recipe.json` so the submission notebook can reproduce it on the hidden test
set. A no-regression guard refuses to overwrite a better recorded score.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             recall_score)
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


# ------------------------------------------------------------------ members
row_idx = pd.read_parquet(f"{OUT}/kaggle_deberta/row_index_42.parquet")
n = len(row_idx)
y = row_idx["y"].to_numpy()
groups = row_idx["prompt_id"].to_numpy()

members, masks = {}, {}


def add(name, mat):
    members[name] = np.clip(np.asarray(mat, dtype=np.float64), 1e-9, 1.0)
    masks[name] = np.ones(n, dtype=bool)
    log(f"member {name:22s} shape {members[name].shape} "
        f"bal_acc {balanced_accuracy_score(y, members[name].argmax(1)):.4f}")


if os.path.exists(f"{OUT}/kaggle_deberta/oof_42.npy"):
    add("deberta_base", np.load(f"{OUT}/kaggle_deberta/oof_42.npy"))

def load_fold_member(prefix, directory=OUT):
    """Assemble a full OOF matrix from `oof_<prefix>_fold{i}.npy` (+ `_idx_fold{i}`).

    Returns (matrix, coverage_mask) or None when fewer than five folds are present. Requiring
    all five folds keeps every member's matrix comparable across the same 2,699 rows.
    """
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
    return full, m


for f0 in sorted(glob.glob(f"{OUT}/oof_*_fold0.npy")):
    prefix = os.path.basename(f0)[len("oof_"):-len("_fold0.npy")]
    if prefix.startswith(("real", "stack")) or prefix.endswith(("_idx", "_y")):
        continue
    got = load_fold_member(prefix)
    if got is None:
        log(f"member oof_{prefix}: incomplete folds, skipped")
        continue
    full, m = got
    members[prefix] = np.clip(full, 1e-9, 1.0)
    masks[prefix] = m
    log(f"member {prefix:22s} shape {full.shape} coverage {int(m.sum())} "
        f"bal_acc {balanced_accuracy_score(y[m], full[m].argmax(1)):.4f}")

for f0 in sorted(glob.glob(f"{OUT}/colab_pro_artifacts/oof_real_fold0.npy")):
    mats, idxs = [], []
    for fold in range(5):
        op = f"{OUT}/colab_pro_artifacts/oof_real_fold{fold}.npy"
        ip = f"{OUT}/colab_pro_artifacts/oof_real_idx_fold{fold}.npy"
        if os.path.exists(op) and os.path.exists(ip):
            mats.append(np.load(op))
            idxs.append(np.load(ip))
    if mats:
        full = np.zeros((n, C))
        m = np.zeros(n, dtype=bool)
        for a, i in zip(mats, idxs):
            full[i] = a
            m[i] = True
        members["a100_7b"] = np.clip(full, 1e-9, 1.0)
        masks["a100_7b"] = m
        log(f"member a100_7b shape {full.shape} coverage {int(m.sum())} "
            f"bal_acc {balanced_accuracy_score(y[m], full[m].argmax(1)):.4f}")

# 7B / Kaggle kernels produce oof_real_fold*.npy under their own directory
for extra in ("kaggle_qwen_lora", "colab_artifacts"):
    d = f"{OUT}/{extra}"
    mats, idxs = [], []
    for fold in range(5):
        for pat in (f"{d}/oof_real_fold{fold}.npy", f"{d}/oof_real_fold{fold}_oof.npy"):
            op = pat
            ip = f"{d}/oof_real_idx_fold{fold}.npy"
            if os.path.exists(op) and os.path.exists(ip):
                mats.append(np.load(op))
                idxs.append(np.load(ip))
                break
    if mats:
        full = np.zeros((n, C))
        m = np.zeros(n, dtype=bool)
        for a, i in zip(mats, idxs):
            full[i] = a
            m[i] = True
        key = f"lora_{extra}"
        members[key] = np.clip(full, 1e-9, 1.0)
        masks[key] = m
        log(f"member {key} shape {full.shape} coverage {int(m.sum())} "
            f"bal_acc {balanced_accuracy_score(y[m], full[m].argmax(1)):.4f}")

if not members:
    sys.exit("no OOF members found")

keys = list(members)
L = {k: np.log(members[k]) for k in keys}
valid = np.ones(n, dtype=bool)
for k in keys:
    valid &= masks[k]
log(f"rows with full coverage: {int(valid.sum())}/{n}")

# ------------------------------------------------------------------ fitting helpers
def fit_temperature(logp, yy, m):
    """NLL-optimal temperature for one member on rows m."""

    def nll(t):
        z = logp[m] / t
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        tgt = yy[m]
        return -np.log(np.maximum(p[np.arange(len(tgt)), tgt], 1e-12)).mean()

    r = minimize_scalar(nll, bounds=(0.5, 4.0), method="bounded")
    return float(r.x)


def combine(weights, temps, biases):
    z = sum(weights[i] * (L[k] / temps[i]) for i, k in enumerate(keys))
    return z + biases


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def tune_biases(z, yy, m, passes=3, cap=2.0):
    """Coordinate ascent on balanced accuracy with step halving."""
    b = np.zeros(C)
    best = balanced_accuracy_score(yy[m], (z[m] + b).argmax(1))
    steps = [1.0, 0.5, 0.25]
    for _ in range(passes):
        for j in range(C):
            for s in steps:
                for sign in (1.0, -1.0):
                    cand = b.copy()
                    cand[j] = float(np.clip(cand[j] + sign * s, -cap, cap))
                    if cand[j] == b[j]:
                        continue
                    score = balanced_accuracy_score(yy[m], (z[m] + cand).argmax(1))
                    if score > best + 1e-9:
                        best, b = score, cand
    return b, float(best)


def tune_weights(z_members, yy, m, rng, iters=250):
    """Random-restart simplex search on balanced accuracy over member weights."""
    k = len(z_members)
    w = np.ones(k) / k
    best = balanced_accuracy_score(yy[m], sum(w[i] * z_members[i][m] for i in range(k)).argmax(1))
    for _ in range(iters):
        cand = np.clip(w + rng.normal(0, 0.15, k), 0, None)
        if cand.sum() <= 0:
            continue
        cand /= cand.sum()
        score = balanced_accuracy_score(
            yy[m], sum(cand[i] * z_members[i][m] for i in range(k)).argmax(1))
        if score > best + 1e-9:
            best, w = score, cand
    return w, float(best)


# ------------------------------------------------------------------ cross-fitted loop
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
             .split(np.zeros(n), y, groups=groups))
oof = np.zeros((n, C))
rng = np.random.default_rng(0)
fold_report = []

for fold, (tr, va) in enumerate(folds):
    tr = tr[valid[tr]]
    va = va[valid[va]]
    if len(tr) < 100 or len(va) < 20:
        continue
    temps = [fit_temperature(L[k], y, tr) for k in keys]
    z_members = [L[k] / temps[i] for i, k in enumerate(keys)]
    w, w_score = tune_weights(z_members, y, tr, rng)
    z_tr = sum(w[i] * z_members[i] for i in range(len(keys)))
    b, b_score = tune_biases(z_tr, y, tr)
    oof[va] = softmax(z_tr[va] + b)
    fold_report.append({"fold": fold, "temps": dict(zip(keys, np.round(temps, 3))),
                        "weights": dict(zip(keys, np.round(w, 3))),
                        "bias_max_abs": float(np.abs(b).max()),
                        "train_bal_before_bias": round(w_score, 4),
                        "train_bal_after_bias": round(b_score, 4),
                        "val_bal_acc": round(float(balanced_accuracy_score(
                            y[va], oof[va].argmax(1))), 4),
                        "val_acc": round(float(accuracy_score(y[va], oof[va].argmax(1))), 4)})
    log(f"fold {fold}: temps {dict(zip(keys, np.round(temps, 2)))} "
        f"w {dict(zip(keys, np.round(w, 2)))} train {w_score:.4f}->{b_score:.4f} "
        f"val {fold_report[-1]['val_bal_acc']:.4f}")

sel = valid & (oof.sum(axis=1) > 0)
pred = oof.argmax(1)
bal = float(balanced_accuracy_score(y[sel], pred[sel]))
acc = float(accuracy_score(y[sel], pred[sel]))
per_class = recall_score(y[sel], pred[sel], labels=list(range(C)), average=None, zero_division=0)
log(f"\nCROSS-FITTED bal_acc {bal:.4f} acc {acc:.4f} on {int(sel.sum())} rows")
log("per-class recall: " + ", ".join(f"{LABELS[i]}={per_class[i]:.2f}" for i in range(C)))
cm = confusion_matrix(y[sel], pred[sel], labels=list(range(C)))

# ------------------------------------------------------------------ persist
prev = {}
if os.path.exists(f"{OUT}/stack_v2_report.json"):
    prev = json.load(open(f"{OUT}/stack_v2_report.json"))
prev_best = float(prev.get("bal_acc", 0.0))
log(f"previous stack_v2 best: {prev_best:.4f} -> new {bal:.4f}")

report = {"bal_acc": bal, "acc": acc, "rows": int(sel.sum()), "members": keys,
          "per_class_recall": {LABELS[i]: float(per_class[i]) for i in range(C)},
          "folds": fold_report,
          "protocol": "SGKF(5, shuffle, seed 42, groups=prompt_id); per-fold temperature, "
                      "weight search and per-class bias tuned on the 4 training folds only"}
json.dump(report, open(f"{OUT}/stack_v2_report.json", "w"), indent=1)

if bal > prev_best + 1e-9 or os.environ.get("FORCE_EXPORT") == "1":
    full_temps = [fit_temperature(L[k], y, sel) for k in keys]
    zm = [L[k] / full_temps[i] for i, k in enumerate(keys)]
    w_full, _ = tune_weights(zm, y, sel, rng)
    z_full = sum(w_full[i] * zm[i] for i in range(len(keys)))
    b_full, _ = tune_biases(z_full, y, sel)
    recipe = {"members": keys,
              "temperatures": dict(zip(keys, [float(t) for t in full_temps])),
              "weights": dict(zip(keys, [float(v) for v in w_full])),
              "biases": [float(v) for v in b_full],
              "labels": LABELS,
              "oof_bal_acc": bal,
              "how_to_apply": ("z = sum_i w_i * log(p_i) / T_i + b ; pred = argmax(z). "
                               "Apply to any member probability matrix with the same column "
                               "order; missing members must be dropped and the weights "
                               "renormalised.")}
    json.dump(recipe, open(f"{OUT}/ensemble_recipe.json", "w"), indent=1)
    log("wrote ensemble_recipe.json")
else:
    log("guard: new score not better, recipe not overwritten")

np.save(f"{OUT}/oof_real_stack_v2.npy", oof)
pd.DataFrame({"id": row_idx["id"], "prompt_id": row_idx["prompt_id"], "model": row_idx["model"],
              "y": y, "pred": [LABELS[i] if s else "" for i, s in zip(pred, sel)],
              "correct": (pred == y) & sel}).to_parquet(f"{OUT}/oof_predictions_stack_v2.parquet",
                                                        index=False)
with open(f"{D}/logs/experiments.tsv", "a") as fh:
    fh.write(f"05-name-that-llm\tstack_v2_crossfit_bal_acc\t{bal:.5f}\tmembers={keys} "
             f"rows={int(sel.sum())} per_class_min={per_class.min():.3f}\n")
log("DONE")
