"""Track 05 - frontier node n9, leg 2: length-bin conditional prior offsets.

Hypothesis (n9b): the three-member OOF-probability meta stacker treats every
class as equally likely at inference, but passage LENGTH is a stratifying
variable whose per-prompt generation distribution shifts the class marginals
(model families differ in typical passage length).  Adding *length-bin
conditional* class-prior offsets to the stacker lets it re-weight the
posteriors the way label-shift correction does, per stratum:

    P(y=c | x, bin)  proportional to  P(y=c | x) * P(y=c | bin) / P(y=c)

implemented as features ``log((n_{b,c} + m P_c) / (N_b + m) / P_c)`` (Dirichlet
shrinkage m) plus a one-hot of the length bin, appended to the 54 member
probabilities, fitted fold-wise by ``LogisticRegression(class_weight=balanced)``.

Leakage protocol (STACKER-LENPRIORS protocol, audited 2026-10-22):
  * fold split      : StratifiedGroupKFold(5, shuffle, seed 42), groups=prompt_id
  * validation rows : prior table from TRAIN fold rows only
  * training rows   : leave-one-prompt-out tables (T minus own prompt), so the
                      meta-LR cannot over-trust a prior feature computed with
                      the horse's own labels in the race
  * word counts     : text.split() length (house convention, LENGTH_BINS edges)
  * comparability   : baseline re-anchored in the SAME script/session with
                      identical FOLDS and cached OOFs (recorded anchor 0.40863)
  * selection       : (C, m) grid reported in full, CV-selected with the same
                      house precedence as the C=8 meta (no nested CV claim)

KEEP rule (house): best config clean-subset BA > 0.40863 + 0.002.
"""
from __future__ import annotations

import json
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import (CLEAN, CACHE, COMP, FOLDS, LABELS, Y,  # noqa: E402
                             log, log_experiment, score)
from ntl_data import LENGTH_BINS  # noqa: E402

OUT = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm/outputs"
RECORDED_ANCHOR = 0.40863
KEEP_RULE = RECORDED_ANCHOR + 0.002

NB = len(LENGTH_BINS) - 1          # 4 length bins
NC = len(LABELS)                   # 18 classes

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
tags = ["svc_extra_claude_w50", "nbsvm_all", "svc35_w0"]
Ps = {t: np.load(f"{CACHE}/oof_{t}.npy") for t in tags}
assert all(P.shape == (len(COMP), NC) for P in Ps.values()), {t: P.shape for t, P in Ps.items()}

S54 = np.hstack(list(Ps.values())).astype(np.float32)
nw = COMP["text"].str.split().str.len().to_numpy()
BIN = np.digitize(nw, LENGTH_BINS[1:-1])          # 0..NB-1
assert BIN.min() == 0 and BIN.max() == NB - 1, (BIN.min(), BIN.max())
GROUP = COMP["group"].to_numpy()

log(f"S54 {S54.shape}, bins {np.bincount(BIN, minlength=NB)}, "
    f"clean={CLEAN.sum()}, prompts={pd.Series(GROUP).nunique()}")


# --------------------------------------------------------------------------- #
# fold-local prior tables (train rows only; LO-prompt variant for train rows)
# --------------------------------------------------------------------------- #
def count_table(idx: np.ndarray) -> np.ndarray:
    """(NB, NC) label counts for rows in ``idx``."""
    t = np.zeros((NB, NC), dtype=np.float64)
    np.add.at(t, (BIN[idx], Y[idx]), 1.0)
    return t


def class_marginal(idx: np.ndarray) -> np.ndarray:
    p = np.bincount(Y[idx], minlength=NC).astype(np.float64)
    return p / p.sum()


def ratio_table(counts: np.ndarray, N_b: np.ndarray, P_c: np.ndarray,
                m: float) -> np.ndarray:
    """log((n_{b,c} + m P_c) / (N_b + m) / P_c) — Dirichlet-shrunk log ratios."""
    r = np.log((counts + m * P_c[None, :]) / (N_b[:, None] + m) / P_c[None, :])
    assert np.isfinite(r).all(), "non-finite log ratios"
    return r


def bin_reports(tbl: np.ndarray, N_b: np.ndarray, P_c: np.ndarray) -> str:
    """Max absolute deviation of smoothed posteriors from a simplex row (sanity)."""
    denom = N_b[:, None] + 1e-12
    return f"max|sum_c P(c|b)-1|={np.abs((tbl / denom).sum(1) - 1).max():.2e}"


# --------------------------------------------------------------------------- #
# math checks (equations.md protocol): Bayes consistency of the prior table
# --------------------------------------------------------------------------- #
def verify_prior_math(m: float = 40.0) -> None:
    tbl_all = count_table(np.arange(len(Y)))
    N_all = tbl_all.sum(1)
    P_all = class_marginal(np.arange(len(Y)))
    r = ratio_table(tbl_all, N_all, P_all, m)

    # (1) Bayes consistency on the full frame: P(c|b) built from the same table
    #     must be a valid simplex row and reproduce the empirical P(c|b).
    post = (tbl_all + m * P_all[None, :]) / (N_all[:, None] + m)
    assert abs(post.sum(1) - 1).max() < 1e-9
    emp = tbl_all / N_all[:, None]
    assert np.abs(post - emp).max() < 0.02, np.abs(post - emp).max()

    # (2) shrinkage correctness: as m -> 1e9 posterior -> global prior.
    post_inf = (tbl_all + 1e9 * P_all[None, :]) / (N_all[:, None] + 1e9)
    assert np.abs(post_inf - P_all[None, :]).max() < 1e-6, \
        np.abs(post_inf - P_all[None, :]).max()

    # (3) LO-prompt arithmetic: T-minus-p equals a fresh recomputation that
    #     excludes one prompt entirely.
    rng = np.random.default_rng(7)
    sample_prompts = rng.choice(pd.Series(GROUP).unique(), size=8, replace=False)
    for p in sample_prompts:
        keep = GROUP != p
        rows = np.where(keep)[0]
        counts = tbl_all - count_table(np.where(GROUP == p)[0])
        direct = ratio_table(count_table(rows), count_table(rows).sum(1),
                             class_marginal(rows), m)
        lo = ratio_table(counts, counts.sum(1), class_marginal(rows), m)
        assert np.abs(lo - direct).max() < 1e-9, np.abs(lo - direct).max()
        assert np.shape(lo) == (NB, NC)
    log(f"math checks OK {bin_reports(tbl_all, N_all, P_all)}")


# --------------------------------------------------------------------------- #
# fold-wise features
# --------------------------------------------------------------------------- #
def build_features(m: float) -> np.ndarray:
    """76-dim stack features: 54 member probs + 4 bin one-hot + 18 prior ratios."""
    per_prompt = {}
    for p, sub in pd.DataFrame({"g": GROUP, "b": BIN, "y": Y}).groupby("g"):
        t = np.zeros((NB, NC))
        np.add.at(t, (sub["b"].to_numpy(), sub["y"].to_numpy()), 1.0)
        per_prompt[p] = t

    F = np.zeros((len(COMP), 54 + NB + NC), dtype=np.float64)
    F[:, :54] = S54
    F[:, 54:54 + NB] = np.eye(NB)[BIN]           # row's own bin one-hot

    for f, (ti, vi) in enumerate(FOLDS):
        # validation rows: train-fold table (no validation labels touched)
        T = count_table(ti)
        N_b, P_c = T.sum(1), class_marginal(ti)
        r_val = ratio_table(T, N_b, P_c, m)
        F[vi, 54 + NB:] = r_val[BIN[vi]]

        # training rows: leave-one-prompt-out tables (positional -> original idx)
        by_prompt = {}
        for pos, p in enumerate(GROUP[ti]):
            by_prompt.setdefault(p, []).append(ti[pos])
        for p, rows in by_prompt.items():
            rows = np.asarray(rows)
            lo_tbl = T - per_prompt[p]
            lo_N = lo_tbl.sum(1)
            lo_P = lo_tbl.sum(0) / lo_tbl.sum()   # LO population's own marginal
            F[rows, 54 + NB:] = ratio_table(lo_tbl, lo_N, lo_P, m)[BIN[rows]]
        if f == 0:
            log(f"    fold0 sanity: bin hot ~ rows {np.bincount(BIN[vi], minlength=NB)}")
    assert np.isfinite(F).all()
    return F


def cross_fit_lr(F: np.ndarray, C: float) -> np.ndarray:
    oof = np.zeros((len(COMP), NC), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                 n_jobs=-1)
        clf.fit(F[ti], Y[ti])
        oof[vi] = clf.predict_proba(F[vi])
    return oof


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    res: dict = {"recipe": "54 member probs + 4 length-bin one-hot + 18 shrinkage "
                           "log-prior-ratio features (LO-prompt tables for train rows), "
                           "cross-fitted LR class_weight=balanced, SGKF5 seed 42",
                 "length_bins": list(LENGTH_BINS)}

    # ---- re-anchor baseline: recorded stack (54-dim, C=8) in this session ----
    oof0 = cross_fit_lr(S54.copy(), 8.0)
    anchor = score(oof0)
    res["anchor_recomputed"] = anchor
    res["anchor_recorded"] = RECORDED_ANCHOR
    drift = anchor - RECORDED_ANCHOR
    log(f"anchor recomputed {anchor:.5f} (recorded {RECORDED_ANCHOR}, drift {drift:+.5f})")
    if abs(drift) > 1.0e-4:
        log(f"WARNING: anchor drift {drift:+.5f} -> task has environment drift; "
            "comparisons vs recorded 0.40863 flagged in verdict")

    verify_prior_math()
    res["math_checks"] = "bayes-simplex + shrinkage-limit + LO-prompt parity: OK"

    # ---- V1: stacker with bin-prior feature blocks ----
    v1: dict[str, float] = {}
    for C, m in product((4.0, 8.0, 16.0, 32.0), (10.0, 40.0, 200.0)):
        oof = cross_fit_lr(build_features(m), C)
        v1[f"c{C:g}_m{m:g}"] = score(oof)
        log(f"  v1 C={C:g} m={m:g} -> {v1[f'c{C:g}_m{m:g}']:.5f}")
    res["v1_grid"] = v1
    best = max(v1, key=v1.get)
    res["v1_best"] = {"tag": best, "bal_acc_clean": v1[best]}

    # ---- V2 (reference): pure post-hoc bin prior exponents on the anchor OOF ----
    v2: dict[str, float] = {}
    for alpha in (0.05, 0.10, 0.20):
        oof_adj = np.zeros_like(oof0)
        for f, (ti, vi) in enumerate(FOLDS):
            T = count_table(ti)
            N_b, P_c = T.sum(1), class_marginal(ti)
            r = ratio_table(T, N_b, P_c, 40.0)[BIN[vi]]       # train-fold table
            oof_adj[vi] = oof0[vi] * np.clip(np.exp(alpha * r), 1e-6, None)
            oof_adj[vi] /= np.clip(oof_adj[vi].sum(1, keepdims=True), 1e-9, None)
        v2[f"alpha{alpha:g}"] = score(oof_adj)
        log(f"  v2 alpha={alpha:g} -> {v2[f'alpha{alpha:g}']:.5f}")
    res["v2_posthoc_reference"] = v2

    keep = v1[best] > max(anchor, RECORDED_ANCHOR) + 0.002
    res["verdict"] = {"frontier_before": RECORDED_ANCHOR,
                      "anchor_this_session": anchor,
                      "best_v1": v1[best], "best_v1_tag": best,
                      "keep": keep, "improvement": v1[best] - RECORDED_ANCHOR}
    log(f"verdict {res['verdict']}")

    if keep:
        np.save(f"{CACHE}/oof_stack_lenprior.npy", oof)
        log_experiment("bal_acc_stack_lenprior", v1[best],
                       f"n9b length-bin prior offsets ON TOP of 3-member stack "
                       f"({best}; grid C{{4,8,16,32}} x m{{10,40,200}} all logged); "
                       f"LO-prompt prior tables; anchor {anchor:.5f}/recorded "
                       f"{RECORDED_ANCHOR} -> KEEP")
        json.dump(res, open(f"{OUT}/cv_lenprior_report.json", "w"), indent=1)
    else:
        log_experiment("bal_acc_stack_lenprior", v1[best],
                       f"n9b length-bin prior offsets REJECTED ({best} best of "
                       f"12-config grid; alpha-grid posthoc {v2}); anchor "
                       f"{anchor:.5f}; baseline {RECORDED_ANCHOR} stands")
        json.dump(res, open(f"{OUT}/cv_lenprior_report.json", "w"), indent=1)
    log("lenprior iteration done")


if __name__ == "__main__":
    main()
