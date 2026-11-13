"""Frontier node n5 / idea I6: RealMLP + kNN-prior stacking for PS6E10.

Adds two learned non-tree members to the existing rank/logit GBDT blend
(oof_lgbm 0.957583 / oof_cat 0.959046 / oof_xgb 0.956385 -> blend 0.959265):

  1. RealMLP-TD (Holzmüller et al. 2024) trained on the identical feature
     space and the identical 10-fold StratifiedKFold(shuffle, seed=42)
     protocol as the baseline members, via pytabkit on the local RTX 3050
     (<3.2 GB VRAM lane).
  2. "TabPFN-style" nearest-neighbour prior: a distance-weighted label prior
     of the k=64 nearest neighbours of each row computed against a
     stratified 100k-row reference set drawn OOF-safe (within training
     folds only) on the GPU.

Members are combined on the logit/rank scale with deterministic coordinate
ascent over OOF AUC. Strict validation: OOF AUC over the same folds as the
baseline; no leakage because all member scores are fold-honest.

Run:  cd track06 && python3 src/realmlp_stack.py [--fast]
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torch

ROOT = "/home/pope/Desktop/gpu /hackathons/06-playground-s6e10-airline-satisfaction"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
LOGS = os.path.join(ROOT, "logs")
FAST = "--fast" in os.sys.argv

TARGET = "satisfaction"
RATING_COLS = [
    c for c in pd.read_csv(os.path.join(DATA, "train.csv"), nrows=1).columns
    if c not in (
        "id", "Age", "Flight Distance", "Departure Delay in Minutes",
        "Arrival Delay in Minutes", "Gender", "Customer Type", "Type of Travel",
        "Class", TARGET,
    )
]
SERVICE_COLS = [
    c for c in RATING_COLS if c not in ("Inflight entertainment", "Food and drink", "Seat comfort")
]
NF = 10
SEED = 42


def log(msg: str, t0: float = None) -> None:
    """Print a timestamped progress line."""
    p = f"[{time.time() - t0:7.1f}s]" if t0 is not None else "[timeless]"
    print(p, msg, flush=True)


def fe(df: pd.DataFrame, med: float = None) -> pd.DataFrame:
    """Identical feature engineering to kaggle_track06_stack.py."""
    X = df.copy()
    if med is None:
        med = X["Departure Delay in Minutes"].median()
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
    X["total_service_score"] = X[SERVICE_COLS].sum(axis=1)
    X["inflight_comfort_score"] = X[["Seat comfort", "Inflight entertainment"]].sum(axis=1)
    X["boarding_entertainment_interaction"] = X["Online boarding"] * X["Inflight entertainment"]
    X["zero_rating_count"] = (r == 0).sum(axis=1)
    X["low_rating_count"] = (r <= 1).sum(axis=1)
    X["is_delayed"] = (X["Departure Delay in Minutes"] > 0).astype(int)
    X["is_long_flight"] = (X["Flight Distance"] > 2000).astype(int)
    X["type_x_class"] = X["Type of Travel"].astype(str) + "_" + X["Class"].astype(str)
    X["cust_x_age"] = X["Customer Type"].astype(str) + "_" + X["Age_Bin"].astype(str)
    X["gender_class"] = X["Gender"].astype(str) + "_" + X["Class"].astype(str)
    return X.drop(columns=[TARGET] if TARGET in X.columns else [])


def nn_prior(X: pd.DataFrame, y: np.ndarray, folds: List[Tuple[np.ndarray, np.ndarray]],
             k: int = 64, n_ref: int = 100_000,
             Xquery: pd.DataFrame = None) -> np.ndarray:
    """Distance-weighted kNN label prior, computed fold-honestly on GPU.

    With folds: OOF prior for every row of X (query rows = validation rows,
    reference set drawn from that fold's training rows only).
    With Xquery (folds=None): prior of Xquery rows against a stratified
    reference set drawn from the full X/y (train side).
    """
    frames = X if Xquery is None else pd.concat([X, Xquery], ignore_index=True)
    cat_cols = [c for c in frames.columns if frames[c].dtype == object]
    Xn = frames.copy()
    for c in cat_cols:
        codes, _ = pd.factorize(Xn[c].astype(str))
        Xn[c] = codes.astype(np.float32)
    Xn = Xn.astype(np.float32).to_numpy()
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    col_w = np.ones(Xn.shape[1], dtype=np.float32)
    for c in cat_cols:
        col_w[frames.columns.get_loc(c)] = 0.15  # damp neighbours sharing only a bin code
    oof = np.zeros(len(y) if Xquery is None else len(Xquery), dtype=np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    do_oof = Xquery is None
    for f, (ti, vi) in enumerate(folds):
        rng = np.random.RandomState(SEED + f)
        pos = ti[y[ti] == 1]
        neg = ti[y[ti] == 0]
        n_ref_half = n_ref // 2
        ref = np.concatenate([
            pos[rng.permutation(len(pos))[:min(n_ref_half, len(pos))]],
            neg[rng.permutation(len(neg))[:min(n_ref_half, len(neg))]],
        ])
        mu = Xn[ti].mean(axis=0)
        sd = Xn[ti].std(axis=0) + 1e-6
        Xall = np.clip((Xn - mu) / sd, -3.0, 3.0) * col_w
        if do_oof:
            Q = torch.tensor(Xall[vi], dtype=torch.float16, device=dev)
        else:
            Q = torch.tensor(Xall[len(X):], dtype=torch.float16, device=dev)
        R = torch.tensor(Xall[ref], dtype=torch.float16, device=dev)
        y_ref = torch.tensor(y[ref], dtype=torch.float32, device=dev)
        prior = torch.zeros(Q.shape[0], dtype=torch.float32, device=dev)
        CHUNK = 2048
        for s in range(0, Q.shape[0], CHUNK):
            d = torch.cdist(Q[s:s + CHUNK], R, p=2)  # (chunk, ref) fp16
            dk, ik = torch.topk(d, k, largest=False)
            w = 1.0 / (dk.float() + 1e-3)
            prior[s:s + CHUNK] = (w * y_ref[ik]).sum(dim=1) / w.sum(dim=1)
        if do_oof:
            oof[vi] = prior.cpu().numpy()
            log(f"nn_prior fold {f}: auc {roc_auc_score(y[vi], oof[vi]):.5f}", T0)
        else:
            oof = prior.cpu().numpy()
        del Q, R, prior
        torch.cuda.empty_cache()
        if not do_oof:
            break
    return oof


def run_realmlp(X: pd.DataFrame, y: np.ndarray,
                folds: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """RealMLP-TD OOF predictions with the baseline fold protocol."""
    from pytabkit import RealMLP_TD_Classifier

    cat_cols = [c for c in X.columns if X[c].dtype == object]
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype("category")
    oof = np.zeros(len(y), dtype=np.float32)
    for f, (ti, vi) in enumerate(folds):
        clf = RealMLP_TD_Classifier(
            device="cuda", random_state=SEED, verbosity=0,
            n_epochs=64 if FAST else 128, val_metric_name="1-auc_ovr",
            batch_size=1024, predict_batch_size=4096, val_fraction=0.08,
        )
        clf.fit(Xc.iloc[ti], y[ti])
        oof[vi] = clf.predict_proba(Xc.iloc[vi])[:, 1]
        log(f"realmlp fold {f}: auc {roc_auc_score(y[vi], oof[vi]):.5f}", T0)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return oof


def coord_ascent(members: Dict[str, np.ndarray], y: np.ndarray,
                 steps: np.ndarray) -> Tuple[Dict[str, float], float]:
    """Deterministic greedy scalar weights on the logit scale."""
    names = list(members)
    w = {n: 0.0 for n in names}
    love = 0.25
    w[names[0]] = 1.0
    best = roc_auc_score(y, sum(w[n] * members[n] for n in names))
    for _ in range(60):
        improved = False
        for n in names:
            for d in steps:
                for sgn in (1, -1):
                    w2 = dict(w)
                    w2[n] = max(0.0, w2[n] + sgn * d)
                    cand = roc_auc_score(y, sum(w2[k] * members[k] for k in names))
                    if cand > best + 1e-7:
                        w, best = w2, cand
                        improved = True
        if not improved:
            break
    s = sum(w.values())
    return ({n: v / s for n, v in w.items()}, best) if s > 0 else (w, best)


T0 = time.time()


def main() -> None:
    """Entry point: build OOF members, stack, write artefacts."""
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    y = tr[TARGET].astype(int).to_numpy()
    med = tr["Departure Delay in Minutes"].median()
    X = fe(tr, med)
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    for c in cat_cols:
        X[c] = X[c].astype(str)
    log(f"features {X.shape} cat {len(cat_cols)}", T0)
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    nnp_path = os.path.join(OUT, "oof_nnprior.npy")
    rl_path = os.path.join(OUT, "oof_realmlp.npy")

    if FAST:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(y), 120_000, replace=False)
        Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
        sf = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(Xs, ys))
        sub = {"lgbm": np.load(os.path.join(OUT, "oof_lgbm.npy"))[idx],
               "cat": np.load(os.path.join(OUT, "oof_cat.npy"))[idx],
               "xgb": np.load(os.path.join(OUT, "oof_xgb.npy"))[idx]}
        mp = run_realmlp(Xs, ys, sf)
        np_mod = nn_prior(Xs, ys, sf)
        stack = {"realmlp": mp, "nnprior": np_mod, **sub}
        wts, auc = coord_ascent(stack, ys, np.arange(1, -0.15, -0.05) * 0.25)
        log(f"FAST smoke blend auc {auc:.5f} weights {wts}", T0)
        return

    # bank cheap GPU member first so partial runs keep their artefacts
    if not os.path.exists(nnp_path):
        nnp = nn_prior(X, y, folds)
        np.save(nnp_path, nnp)

    if not os.path.exists(rl_path):
        rl = run_realmlp(X, y, folds)
        np.save(rl_path, rl)
    rl = np.load(rl_path)
    log(f"realmlp oof auc {roc_auc_score(y, rl):.6f}", T0)

    members: Dict[str, np.ndarray] = {
        "lgbm": np.load(os.path.join(OUT, "oof_lgbm.npy")),
        "cat": np.load(os.path.join(OUT, "oof_cat.npy")),
        "xgb": np.load(os.path.join(OUT, "oof_xgb.npy")),
    }
    base_blend = np.load(os.path.join(OUT, "oof_blend.npy"))
    log(f"baseline blend auc {roc_auc_score(y, base_blend):.6f}", T0)

    members["realmlp"] = rl
    members["nnprior"] = nnp
    logits = {k: logit(np.clip(v, 1e-6, 1 - 1e-6)) for k, v in members.items()}
    wts, auc_w = coord_ascent(
        {k: v.astype(np.float64) for k, v in logits.items()},
        y, np.arange(0.25, -0.01, -0.025),
    )
    log(f"stacked logit-blend auc {auc_w:.6f} weights {wts}", T0)

    blend_v2 = np.zeros(len(y))
    for k, wv in wts.items():
        blend_v2 += wv * logits[k]
    blend_v2 = expit(blend_v2)
    np.save(os.path.join(OUT, "oof_blend_v2.npy"), blend_v2)

    with open(os.path.join(OUT, "cv_scores.json"), "w") as fh:
        json.dump({
            "metric": "roc_auc",
            "folds": NF, "seed": SEED,
            "members": {k: round(roc_auc_score(y, v), 6) for k, v in members.items()},
            "stack_weights": wts, "stack_auc": round(auc_w, 6),
            "baseline_blend_auc": round(roc_auc_score(y, base_blend), 6),
            "improvement": round(auc_w - roc_auc_score(y, base_blend), 6),
        }, fh, indent=2)

    log(f"FINAL blend_v2 auc {roc_auc_score(y, blend_v2):.6f} "
        f"vs baseline {roc_auc_score(y, base_blend):.6f}", T0)

    with open(os.path.join(LOGS, "experiments.tsv"), "a") as fh:
        fh.write("track06\troc_auc\t" + f"{roc_auc_score(y, blend_v2):.6f}\t"
                 "realmlp+nnprior logit stack, 10-fold seed42, weights {wts}\n")

    keep = roc_auc_score(y, blend_v2) > roc_auc_score(y, base_blend)
    log("KEEP" if keep else "DISCARD", T0)


if __name__ == "__main__":
    main()
