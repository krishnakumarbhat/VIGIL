"""Full-train refits of all stack members for test-side predictions.

Uses the final stack weights from outputs/cv_scores.json (tuned on OOF,
never on test) to build outputs/submission.csv as PROBABILITIES
(staged only - never submitted by policy). Local GPU lane, <3.2 GB VRAM.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch

from realmlp_stack import DATA, OUT, TARGET, RATING_COLS, SERVICE_COLS, SEED, fe, log, nn_prior

T0 = time.time()


def build_models() -> Dict[str, object]:
    """Baseline GBDT members with the kernel script's hyperparameters."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    return {
        "cat": CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=7, verbose=0,
                                  task_type="GPU", devices="0", l2_leaf_reg=3, random_seed=42),
        "lgbm": lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                                   colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                                   reg_alpha=0.1, reg_lambda=1.0, objective="binary",
                                   verbose=-1, n_jobs=16, random_state=42),
        "xgb": xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=7,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 tree_method="hist", device="cuda", eval_metric="auc",
                                 n_jobs=16, random_state=42),
    }


def main() -> None:
    """Entry point: refit members on full train, predict test, stage submission."""
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = tr[TARGET].astype(int).to_numpy()
    med = tr["Departure Delay in Minutes"].median()
    Xtr, Xte = fe(tr, med), fe(te, med)
    cat_cols = [c for c in Xtr.columns if Xtr[c].dtype == object]
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    for c in cat_cols:
        Xtr2[c] = Xtr2[c].astype(str)
        Xte2[c] = Xte2[c].astype(str).astype("category").cat.set_categories(
            Xtr2[c].astype(str).unique())
        Xtr2[c] = Xtr2[c].astype("category")
    log(f"test rows {len(te)} feats {Xtr.shape[1]}", T0)

    preds: Dict[str, np.ndarray] = {}

    rl_path = os.path.join(OUT, "test_realmlp.npy")
    if not os.path.exists(rl_path):
        from pytabkit import RealMLP_TD_Classifier
        clf = RealMLP_TD_Classifier(device="cuda", random_state=SEED, verbosity=0,
                                    n_epochs=128, batch_size=1024, predict_batch_size=4096)
        clf.fit(Xtr2, y)
        preds["realmlp"] = clf.predict_proba(Xte2)[:, 1]
        np.save(rl_path, preds["realmlp"])
    preds["realmlp"] = np.load(rl_path)
    log("realmlp test done", T0)

    nn_path = os.path.join(OUT, "test_nnprior.npy")
    if not os.path.exists(nn_path):
        preds["nnprior"] = nn_prior(Xtr, y, folds=[(np.arange(len(y)), [])],
                                    k=64, n_ref=100_000, Xquery=Xte)
        np.save(nn_path, preds["nnprior"])
    preds["nnprior"] = np.load(nn_path)
    log("nnprior test done", T0)

    for name, m in build_models().items():
        tp = os.path.join(OUT, f"test_{name}.npy")
        if not os.path.exists(tp):
            fit_kwargs = {"cat_features": cat_cols} if name == "cat" else {}
            m.fit(Xtr2, y, **fit_kwargs)
            np.save(tp, m.predict_proba(Xte2)[:, 1])
        preds[name] = np.load(tp)
        log(f"{name} test done", T0)

    with open(os.path.join(OUT, "cv_scores.json")) as fh:
        wts = {k: v for k, v in json.load(fh)["stack_weights"].items()}
    logits = sum(wts[k] * logit(np.clip(preds[k], 1e-6, 1 - 1e-6))
                 for k in wts if k in preds)
    prob = expit(logits)
    pd.DataFrame({"id": te["id"], "satisfaction": prob}).to_csv(
        os.path.join(OUT, "submission.csv"), index=False)
    log(f"submission.csv staged, mean prob {prob.mean():.5f}", T0)


if __name__ == "__main__":
    main()
