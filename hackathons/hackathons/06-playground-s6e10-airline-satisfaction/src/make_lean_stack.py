"""Fit a lean meta stack over the k strongest published members and write a submission.

Lean beats wide here: with 148 members the nested score is 0.96213, with the top 10 it is
0.96221, because correlated families let a wide logistic regression fit member-specific
noise. The subset is chosen by OOF AUC only, never by leaderboard feedback.

Run:  python3 src/make_lean_stack.py --k 10 --C 0.3 --out outputs/submission_lean10.csv
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from stack_experiments import ROOT, load_all, matrix, nested_lr

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission_lean.csv"))
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["satisfaction"])
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"])
    y = tr.satisfaction.astype(int).values
    names, oofs, tests = load_all(len(y), len(te))
    aucs = np.array([roc_auc_score(y, o) for o in oofs])
    order = np.argsort(-aucs)[: args.k]
    log = lambda m: print(f"{m}", flush=True)
    log(f"members: {[names[i] for i in order]}")
    log("aucs: " + ", ".join(f"{aucs[i]:.5f}" for i in order))

    X = matrix([oofs[i] for i in order])
    XT = matrix([tests[i] for i in order])
    auc, _, _ = nested_lr(X, y, args.C)
    log(f"nested AUC k={args.k} C={args.C}: {auc:.6f}")

    model = LogisticRegression(C=args.C, max_iter=5000, n_jobs=-1).fit(X, y)
    pred = 1 / (1 + np.exp(-model.decision_function(XT)))
    pd.DataFrame({"id": te.id.values, "satisfaction": pred}).to_csv(args.out, index=False)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
