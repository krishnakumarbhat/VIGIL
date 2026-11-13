"""Track 06: add the original-dataset-augmented LightGBM as a fourth ensemble member.

Trains on the full playground train rows PLUS the 129,880 original airline rows (labels mapped),
predicts the test set, then re-optimises the logit blend on OOF (base-lgb, augmented-lgb, cat, xgb)
and writes the final probability submission.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

D = "/home/pope/Desktop/gpu /hackathons/06-playground-s6e10-airline-satisfaction"
OUT = f"{D}/outputs"

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
                "Gate location", "Food and drink", "Online boarding", "Seat comfort",
                "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
                "Checkin service", "Cleanliness"]


def fe(df, med):
    X = df.copy()
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(med)
    X["delay_diff"] = X["Arrival Delay in Minutes"] - X["Departure Delay in Minutes"]
    X["Flight_Distance_Bin"] = (X["Flight Distance"] // 10).astype(str)
    X["Age_Bin"] = (X["Age"] // 5).astype(str)
    r = X[RATING_COLS]
    X["rating_mean"] = r.mean(axis=1)
    X["rating_std"] = r.std(axis=1)
    X["rating_min"] = r.min(axis=1)
    X["rating_max"] = r.max(axis=1)
    X["rating_sum"] = r.sum(axis=1)
    X["total_service_score"] = X[["Inflight wifi service", "Online boarding", "On-board service",
                                   "Leg room service", "Baggage handling", "Checkin service",
                                   "Cleanliness"]].sum(axis=1)
    X["inflight_comfort_score"] = X[["Seat comfort", "Inflight entertainment"]].sum(axis=1)
    X["boarding_entertainment_interaction"] = X["Online boarding"] * X["Inflight entertainment"]
    X["zero_rating_count"] = (r == 0).sum(axis=1)
    X["low_rating_count"] = (r <= 1).sum(axis=1)
    X["is_delayed"] = (X["Departure Delay in Minutes"] > 0).astype(int)
    X["is_long_flight"] = (X["Flight Distance"] > 2000).astype(int)
    X["type_x_class"] = X["Type of Travel"].astype(str) + "_" + X["Class"].astype(str)
    X["cust_x_age"] = X["Customer Type"].astype(str) + "_" + X["Age_Bin"].astype(str)
    X["gender_class"] = X["Gender"].astype(str) + "_" + X["Class"].astype(str)
    X["delay_ratio"] = X["Arrival Delay in Minutes"] / (X["Departure Delay in Minutes"] + 1)
    X["delay_per_1k"] = X["Departure Delay in Minutes"] / (X["Flight Distance"] / 1000 + 1)
    X["wifi_ent_interaction"] = X["Inflight wifi service"] * X["Inflight entertainment"]
    X["value_proxy"] = X["rating_sum"] / (X["Flight Distance"] / 1000 + 1)
    drop_cols = [c for c in ("satisfaction", "id", "Unnamed: 0") if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    return X


tr = pd.read_csv(f"{D}/data/train.csv")
te = pd.read_csv(f"{D}/data/test.csv")
y = tr["satisfaction"].astype(int).to_numpy()
med = tr["Departure Delay in Minutes"].median()

orig = pd.concat([pd.read_csv("/tmp/opencode/airline_orig/train.csv"),
                   pd.read_csv("/tmp/opencode/airline_orig/test.csv")], ignore_index=True)
orig = orig.drop(columns=[c for c in ("Unnamed: 0", "id") if c in orig.columns])
orig["satisfaction"] = (orig["satisfaction"].astype(str).str.strip().str.lower()
                        .map({"satisfied": 1, "neutral or dissatisfied": 0}))
orig = orig.dropna(subset=["satisfaction"]).reset_index(drop=True)
y_orig = orig["satisfaction"].astype(int).to_numpy()

X = fe(tr, med)
Xo = fe(orig, med).reindex(columns=X.columns)
Xte = fe(te, med)
for c in X.columns:
    if X[c].dtype == object:
        cats = pd.Index(X[c].astype(str).unique())
        Xo[c] = pd.Categorical(Xo[c].astype(str), categories=cats)
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
log("X", X.shape, "Xo", Xo.shape, "Xte", Xte.shape)

import lightgbm as lgb
PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
              subsample=0.8, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0,
              objective="binary", verbose=-1, n_jobs=8)

# --- OOF for the augmented model (recompute so it lines up exactly with the other members) ---
oof_aug = np.zeros(len(tr))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for f, (ti, vi) in enumerate(skf.split(X, y)):
    Xa = pd.concat([X.iloc[ti], Xo], ignore_index=True)
    ya = np.concatenate([y[ti], y_orig])
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xa, ya)
    oof_aug[vi] = m.predict_proba(X.iloc[vi])[:, 1]
    if (f + 1) % 5 == 0:
        log(f"aug fold {f+1}/10 auc {roc_auc_score(y[vi], oof_aug[vi]):.5f}")
log("aug OOF AUC", roc_auc_score(y, oof_aug))
np.save(f"{OUT}/oof_lgb_aug.npy", oof_aug)

# --- full-fit test prediction ---
Xa = pd.concat([X, Xo], ignore_index=True)
ya = np.concatenate([y, y_orig])
full = lgb.LGBMClassifier(**PARAMS)
full.fit(Xa, ya)
test_aug = full.predict_proba(Xte)[:, 1]
np.save(f"{OUT}/test_lgb_aug.npy", test_aug)
log("test prediction done", test_aug.shape)

# --- blend all available members ---
members = {}
for name, oof_path, test_path in [
    ("lgbm_base", f"{OUT}/oof_lgb_base.npy", f"{OUT}/kaggle/test_lgbm.npy"),
    ("lgbm_aug", f"{OUT}/oof_lgb_aug.npy", f"{OUT}/test_lgb_aug.npy"),
    ("cat", f"{OUT}/oof_cat.npy", f"{OUT}/kaggle/test_cat.npy"),
    ("xgb", f"{OUT}/oof_xgb.npy", f"{OUT}/kaggle/test_xgb.npy"),
]:
    import os
    if os.path.exists(oof_path) and os.path.exists(test_path):
        members[name] = (np.load(oof_path), np.load(test_path))
        log(f"member {name}: oof auc {roc_auc_score(y, members[name][0]):.6f}")

keys = list(members)
L = {k: logit(np.clip(members[k][0], 1e-6, 1 - 1e-6)) for k in keys}


def neg(w):
    w = np.abs(w)
    s = w.sum()
    if s <= 0:
        return 1.0
    w = w / s
    return -roc_auc_score(y, expit(sum(w[i] * L[k] for i, k in enumerate(keys))))


best = None
for init in (np.ones(len(keys)), np.array([1.0, 1.0, 2.0, 0.3]), np.array([1.0, 2.0, 3.0, 0.5])):
    r = minimize(neg, init, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    if best is None or r.fun < best.fun:
        best = r
w = np.abs(best.x)
w = w / w.sum()
oof_blend = expit(sum(w[i] * L[k] for i, k in enumerate(keys)))
final_auc = roc_auc_score(y, oof_blend)
log("weights", dict(zip(keys, w.round(4).tolist())), "FINAL OOF AUC", final_auc)

LT = {k: logit(np.clip(members[k][1], 1e-6, 1 - 1e-6)) for k in keys}
test_blend = expit(sum(w[i] * LT[k] for i, k in enumerate(keys)))
np.save(f"{OUT}/oof_blend.npy", oof_blend)
np.save(f"{OUT}/test_blend_prob.npy", test_blend)
sub = pd.DataFrame({"id": te["id"], "satisfaction": test_blend})
sub.to_csv(f"{OUT}/submission.csv", index=False)
log("submission written", sub.shape, "prob range", round(test_blend.min(), 4), round(test_blend.max(), 4))

prev = {}
if os.path.exists(f"{OUT}/cv_scores.json"):
    try:
        prev = json.load(open(f"{OUT}/cv_scores.json"))
    except Exception:
        prev = {}
out = {
    "oof_auc_per_model": {**{k: float(roc_auc_score(y, members[k][0])) for k in keys},
                          "lgb_aug_full": float(roc_auc_score(y, oof_aug))},
    "final_oof_auc": float(final_auc),
    "weights": dict(zip(keys, w.tolist())),
    "n_features": int(X.shape[1]),
    "n_splits": 10,
    "original_dataset_augmented_rows": int(len(orig)),
    "previous_final_oof_auc": prev.get("final_oof_auc"),
}
json.dump(out, open(f"{OUT}/cv_scores.json", "w"), indent=1)
with open(f"{D}/logs/experiments.tsv", "a") as fh:
    fh.write(f"06-playground-s6e10-airline-satisfaction\tps6e10_final_blend_oof_auc\t{final_auc:.6f}\t"
             f"members {list(keys)} weights {dict(zip(keys, w.round(4).tolist()))}; prev {prev.get('final_oof_auc')}\n")
log("DONE")
