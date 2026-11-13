"""Track 06 experiment: does adding the ORIGINAL airline dataset as training rows help?

The 129,880-row original Kaggle "Airline Passenger Satisfaction" dataset shares the same 21
features and has labels (neutral or dissatisfied / satisfied). The playground series is
synthetic data generated from the same distribution, so the source rows can be used as
extra supervision. This script measures the effect honestly with 10-fold
StratifiedKFold(shuffle=True, random_state=42) on the playground train rows only.
"""
import json
import time

import numpy as np
import pandas as pd
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
log("orig rows", len(orig), "pos rate", y_orig.mean())

X = fe(tr, med)
Xo = fe(orig, med)
Xo = Xo.reindex(columns=X.columns)
for c in X.columns:
    if X[c].dtype == object:
        cats = pd.Index(X[c].astype(str).unique())
        Xo[c] = pd.Categorical(Xo[c].astype(str), categories=cats)
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    else:
        Xo[c] = pd.to_numeric(Xo[c], errors="coerce")
log("X", X.shape, "Xo", Xo.shape)

import lightgbm as lgb

PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
              subsample=0.8, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0,
              objective="binary", verbose=-1, n_jobs=8)


def run(use_orig, tag):
    oof = np.zeros(len(tr))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for f, (ti, vi) in enumerate(skf.split(X, y)):
        if use_orig:
            Xa = pd.concat([X.iloc[ti], Xo], ignore_index=True)
            ya = np.concatenate([y[ti], y_orig])
        else:
            Xa, ya = X.iloc[ti], y[ti]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xa, ya)
        oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]
        if (f + 1) % 5 == 0:
            log(f"  {tag} fold {f+1}/10 running auc {roc_auc_score(y[vi], oof[vi]):.5f}")
    auc = roc_auc_score(y, oof)
    log(f"{tag} OOF AUC {auc:.6f}")
    np.save(f"{OUT}/oof_lgb_{tag}.npy", oof)
    return auc, oof


import os
if os.environ.get("SKIP_BASE") == "1" and os.path.exists(f"{OUT}/oof_lgb_base.npy"):
    o_base = np.load(f"{OUT}/oof_lgb_base.npy")
    a_base = roc_auc_score(y, o_base)
    log("base loaded from cache", a_base)
else:
    a_base, o_base = run(False, "base")
a_aug, o_aug = run(True, "augmented")
log(f"BASE {a_base:.6f}  AUGMENTED {a_aug:.6f}  DELTA {a_aug - a_base:+.6f}")

# blend base + augmented
from scipy.special import logit, expit
lb, la = logit(np.clip(o_base, 1e-6, 1-1e-6)), logit(np.clip(o_aug, 1e-6, 1-1e-6))
best = (0.0, None)
for w in np.linspace(0, 1, 21):
    auc = roc_auc_score(y, expit(w * la + (1 - w) * lb))
    if auc > best[0]:
        best = (auc, w)
log(f"blend base/aug best auc {best[0]:.6f} at w_aug={best[1]:.2f}")
np.save(f"{OUT}/oof_lgb_aug.npy", o_aug)
json.dump({"base_auc": float(a_base), "augmented_auc": float(a_aug), "delta": float(a_aug - a_base),
           "blend_best_auc": float(best[0]), "blend_w_aug": float(best[1]),
           "n_orig_rows": int(len(orig))},
          open(f"{OUT}/cv_scores_orig_aug.json", "w"), indent=1)
with open(f"{D}/logs/experiments.tsv", "a") as fh:
    fh.write(f"06-playground-s6e10-airline-satisfaction\tlgbm_orig_aug_oof_auc\t{a_aug:.6f}\t"
             f"base {a_base:.6f} + {len(orig)} original airline rows as extra supervision; delta {a_aug-a_base:+.6f}\n")
log("DONE")
