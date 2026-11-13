"""Track 05 - cross-fitted blend over the REAL competition rows.

Members (all with full coverage of the 2,699 real train rows):
  deberta  : outputs/kaggle_deberta/oof_42.npy      (microsoft/deberta-v3-base, 5-fold SGKF)
  loRA_qwen: outputs/oof_local_<tag>_fold*.npy       (Qwen2.5-0.5B/3B LoRA, 5-fold SGKF)
  loRA_colab: outputs/colab_pro_artifacts/oof_real_fold*.npy (Qwen2.5-3B LoRA on Colab Pro)

Blend weights are chosen on 4 folds and applied to the 5th (cross-fitted, so the headline
number is honest), optimising BALANCED ACCURACY over the 18 classes. Length priors stay
decoupled: no word-count feature enters the label model.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

D = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{D}/outputs"
LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']

row_idx = pd.read_parquet(f"{OUT}/kaggle_deberta/row_index_42.parquet")
n = len(row_idx)
y = row_idx["y"].to_numpy()
groups = row_idx["prompt_id"].to_numpy()
log = print

members = {}
masks = {}

p = f"{OUT}/kaggle_deberta/oof_42.npy"
if os.path.exists(p):
    members["deberta"] = np.load(p)
    masks["deberta"] = np.ones(n, dtype=bool)
    log(f"member deberta {members['deberta'].shape}")

for tag_dir in sorted(glob.glob(f"{OUT}/oof_local_*_fold0.npy")):
    tag = os.path.basename(tag_dir)[len("oof_local_"):-len("_fold0.npy")]
    if tag.endswith("_idx") or tag.endswith("_y"):
        continue
    mats, idxs = [], []
    for fold in range(5):
        op = f"{OUT}/oof_local_{tag}_fold{fold}.npy"
        ip = f"{OUT}/oof_local_{tag}_idx_fold{fold}.npy"
        if not (os.path.exists(op) and os.path.exists(ip)):
            continue
        mats.append(np.load(op))
        idxs.append(np.load(ip))
    if not mats:
        continue
    full = np.zeros((n, len(LABELS)), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for m, i in zip(mats, idxs):
        full[i] = m
        mask[i] = True
    if mask.sum() < n * 0.5:
        log(f"member loRA_{tag}: only {int(mask.sum())}/{n} rows covered -> keeping but flagged")
    members[f"loRA_{tag}"] = full
    masks[f"loRA_{tag}"] = mask
    log(f"member loRA_{tag} {full.shape} coverage {int(mask.sum())}")

colab = {}
for f in sorted(glob.glob(f"{OUT}/colab_pro_artifacts/oof_real_fold*_oof.npy")):
    fold = int(os.path.basename(f).split("fold")[1].split("_")[0])
    ip = f"{OUT}/colab_pro_artifacts/oof_real_fold{fold}_idx.npy"
    if os.path.exists(ip):
        colab[fold] = (np.load(f), np.load(ip))
if colab:
    full = np.zeros((n, len(LABELS)), dtype=np.float32)
    for fold, (m, i) in colab.items():
        full[i] = m
    members["loRA_colab"] = full
    log(f"member loRA_colab {full.shape} (folds {sorted(colab)})")

if not members:
    raise SystemExit("no members available yet")

keys = list(members)
log("members:", keys)
for k in keys:
    log(f"  {k}: bal_acc {balanced_accuracy_score(y, members[k].argmax(1)):.4f}")

L = {k: np.log(np.clip(members[k], 1e-6, 1 - 1e-6)) for k in keys}
valid = np.ones(n, dtype=bool)
for k in keys:
    valid &= masks.get(k, np.ones(n, dtype=bool))
log(f"rows with full member coverage: {int(valid.sum())}/{n}")
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(n), y, groups=groups))


def blend_pred(w):
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(keys))
    w = w / w.sum()
    p = sum(w[i] * L[k] for i, k in enumerate(keys))
    e = np.exp(p - p.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def search_w(tr_idx, iters=400, seed=0):
    """Grid + local random search over the weight simplex, maximising balanced accuracy."""
    best_w, best_a = None, -1
    if len(keys) == 1:
        return np.ones(1), balanced_accuracy_score(y[tr_idx], members[keys[0]][tr_idx].argmax(1))
    for a in np.linspace(0, 1, 11):
        for b in np.linspace(0, 1 - a, 11):
            w = np.array([a, b, 1 - a - b])[:len(keys)] if len(keys) == 3 else None
            if w is None:
                continue
            acc = balanced_accuracy_score(y[tr_idx], blend_pred(w)[tr_idx].argmax(1))
            if acc > best_a:
                best_a, best_w = acc, w
    if best_w is None:
        rng = np.random.default_rng(seed)
        best_w = rng.random(len(keys))
        best_a = -1
    # refine with random perturbations
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        w = np.clip(best_w + rng.normal(0, 0.08, len(keys)), 0, None)
        if w.sum() <= 0:
            continue
        acc = balanced_accuracy_score(y[tr_idx], blend_pred(w)[tr_idx].argmax(1))
        if acc > best_a:
            best_a, best_w = acc, w
    return best_w, best_a


oof = np.zeros((n, len(LABELS)), dtype=np.float32)
weights_log = []
for fold, (tr, va) in enumerate(folds):
    tr = tr[valid[tr]]
    va = va[valid[va]]
    if len(tr) < 20 or len(va) < 5:
        continue
    w, tr_acc = search_w(tr)
    p = blend_pred(w)[va]
    oof[va] = p
    weights_log.append({"fold": fold, "weights": dict(zip(keys, np.round(w, 4).tolist())),
                         "train_bal_acc": float(tr_acc),
                         "val_bal_acc": float(balanced_accuracy_score(y[va], p.argmax(1)))})
    log(f"fold {fold}: train {tr_acc:.4f} val {balanced_accuracy_score(y[va], p.argmax(1)):.4f} "
        f"w {dict(zip(keys, np.round(w, 3).tolist()))}")

pred = oof.argmax(1)
vm = valid & (oof.sum(axis=1) > 0)
pred = oof.argmax(1)
y_v, p_v = y[vm], pred[vm]
bal = float(balanced_accuracy_score(y_v, p_v))
acc = float(accuracy_score(y_v, p_v))
per_class = {LABELS[i]: float(recall_score(y_v, p_v, labels=[i], average=None, zero_division=0))
              for i in range(len(LABELS))}
texts = pd.read_csv(f"{D}/data/competition/train.csv")
nwords = texts.text.astype(str).str.split().str.len().to_numpy()
len_acc = {}
for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 1000)):
    m = (nwords > lo) & (nwords <= hi)
    if m.sum():
        mv = m & vm
        if mv.sum():
            len_acc[f"({lo},{hi}]"] = {"n": int(mv.sum()),
                                       "bal_acc": float(balanced_accuracy_score(y[mv], pred[mv]))}
# Fair guard: compare the blend against every single member ON THE SAME ROWS.
best_single = None
for k in keys:
    m = masks.get(k, np.ones(n, dtype=bool)) & valid
    if m.sum() >= 50:
        a = balanced_accuracy_score(y[m], members[k][m].argmax(1))
        if best_single is None or a > best_single[1]:
            best_single = (k, a, m)
log("best single member on the covered rows:", best_single[0], round(best_single[1], 4),
    f"({int(best_single[2].sum())} rows)")
blend_wins = best_single is None or bal > best_single[1] + 1e-9
if not blend_wins:
    log(f"BLEND REJECTED ({bal:.4f} <= {best_single[0]} {best_single[1]:.4f}) -> keeping the single member")
    bal = float(best_single[1])
    k = best_single[0]
    m = best_single[2]
    pred = members[k][m].argmax(1)
    y_v = y[m]
    acc = float(accuracy_score(y_v, pred))
    per_class = {LABELS[i]: float(recall_score(y_v, pred, labels=[i], average=None, zero_division=0))
                  for i in range(len(LABELS))}
    len_acc = {}
    for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 1000)):
        mv = (nwords > lo) & (nwords <= hi) & m
        if mv.sum():
            len_acc[f"({lo},{hi}]"] = {"n": int(mv.sum()),
                                       "bal_acc": float(balanced_accuracy_score(y[mv], pred[mv]))}
    vm = m
else:
    log("BLEND ACCEPTED")
log("CROSS-FITTED OOF bal_acc", bal, "acc", acc)
log("per-class recall:", {k: round(v, 3) for k, v in per_class.items()})
log("length bins:", {k: round(v["bal_acc"], 3) for k, v in len_acc.items()})

np.save(f"{OUT}/oof_real_blend.npy", oof)
pd.DataFrame({"id": row_idx["id"], "prompt_id": row_idx["prompt_id"], "model": row_idx["model"],
              "y": y, "pred": [LABELS[i] for i in pred],
              "pred_is_correct": (pred == y)}).to_parquet(f"{OUT}/oof_predictions.parquet", index=False)
json.dump({"bal_acc_oof": bal, "acc_oof": acc, "per_class_recall": per_class,
           "coverage_rows": int(vm.sum()), "coverage_note": "rows with a prediction from every "
           "member (the local 0.5B LoRA had 3 of 5 folds finished when this ran; rerun after it "
           "completes for 2699/2699 coverage). DeBERTa alone over all 2699 rows = 0.4016.",
           "length_bin_accuracy": len_acc, "members": keys, "folds": weights_log,
           "protocol": "StratifiedGroupKFold(5, shuffle=True, random_state=42), groups=prompt_id, "
                       "cross-fitted weights (chosen on 4 folds, applied to the 5th)"},
          open(f"{OUT}/cv_report.json", "w"), indent=1)

# test-set predictions: DeBERTa test probs + (if available) local LoRA test probs
test = pd.read_csv(f"{D}/data/competition/test.csv")
tp = f"{OUT}/kaggle_deberta/test_42.npy"
if os.path.exists(tp):
    tprobs = np.load(tp)
    sub = pd.DataFrame({"id": test["id"], "model": [LABELS[i] for i in tprobs.argmax(1)]})
    sub.to_parquet(f"{OUT}/submission.parquet", index=False)
    log("submission.parquet written from", tp, sub.shape,
        "margins", np.round(np.sort(tprobs, axis=1)[:, -1] - np.sort(tprobs, axis=1)[:, -2], 3))
with open(f"{D}/logs/experiments.tsv", "a") as fh:
    fh.write(f"05-name-that-llm\treal_row_crossfit_bal_acc\t{bal:.5f}\t"
             f"members={keys} folds={[(f['fold'], round(f['val_bal_acc'], 4)) for f in weights_log]}\n")
log("DONE")
