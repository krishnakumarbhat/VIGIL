"""Track 05 - grouped-CV experiment grid for Name That LLM (18-way authorship attribution).

Protocol (fixed, non-negotiable):

* validation rows  = the 9,290 competition-style passages (510 prompts)
* groups           = ``prompt_id``  -> StratifiedGroupKFold(5, shuffle, seed 42)
* extra training   = external 18-label corpus (21k rows, prompt ids disjoint
                     from every competition prompt) mixed in with a weight ``w``
* features         = fitted on the external corpus only, so the validation text
                     never influences vocabulary or IDF
* headline metric  = balanced accuracy (macro recall) on the *clean* subset,
                     i.e. excluding the 170 off-protocol ``pilot_range`` rows;
                     the all-rows number is logged next to it.

Each variant writes its OOF probability matrix to ``outputs/cache/`` so blends
can be evaluated without refitting, and appends one row to ``logs/experiments.tsv``.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntl_features import (LABELS, FeatureStack, load_competition_frame,  # noqa: E402
                          load_external_frame)

ROOT = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{ROOT}/outputs"
CACHE = f"{OUT}/cache"
LOG = f"{ROOT}/logs/experiments.tsv"
os.makedirs(CACHE, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def log_experiment(metric: str, value: float, notes: str) -> None:
    """Append one experiment row (track, metric, value, notes) to the TSV."""
    header = not os.path.exists(LOG) or os.path.getsize(LOG) == 0
    with open(LOG, "a") as fh:
        if header:
            fh.write("track\tmetric\tvalue\tnotes\n")
        fh.write(f"05-name-that-llm\t{metric}\t{value:.5f}\t{notes}\n")


# --------------------------------------------------------------------------- #
# data + folds
# --------------------------------------------------------------------------- #
COMP = load_competition_frame()
EXT = load_external_frame(max_per_class=1200)
CLEAN = (COMP["split"] != "pilot_range").to_numpy()
Y = COMP["y"].to_numpy()
GROUPS = COMP["group"].to_numpy()
log(f"comp {COMP.shape} clean={CLEAN.sum()} ext {EXT.shape}")

FOLDS = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42
                                  ).split(np.zeros(len(COMP)), Y, groups=GROUPS))
log("fold sizes", [len(v) for _, v in FOLDS])

FEATURE_CONFIGS = {
    # name: dict for FeatureStack
    "cwS": dict(views=("char", "word", "sty"), char_ngram=(2, 5), word_ngram=(1, 2)),
    "all": dict(views=("char", "word", "shape", "sty"), char_ngram=(2, 5), word_ngram=(1, 2)),
    "all35": dict(views=("char", "word", "shape", "sty"), char_ngram=(3, 5), word_ngram=(1, 3)),
}


def get_matrices(cfg_name: str):
    """Build (or load) the competition and external feature matrices for a config."""
    key = f"{cfg_name}_{len(COMP)}_{len(EXT)}"
    fc, fe = f"{CACHE}/Xc_{key}.npz", f"{CACHE}/Xe_{key}.npz"
    if os.path.exists(fc) and os.path.exists(fe):
        return sp.load_npz(fc), sp.load_npz(fe)
    fs = FeatureStack(**FEATURE_CONFIGS[cfg_name]).fit(EXT["text"].tolist())
    Xc = fs.transform(COMP["text"].tolist())
    Xe = fs.transform(EXT["text"].tolist())
    sp.save_npz(fc, Xc)
    sp.save_npz(fe, Xe)
    log(f"built {cfg_name}: {Xc.shape} {Xc.nnz} / {Xe.shape} {Xe.nnz}")
    return Xc, Xe


# --------------------------------------------------------------------------- #
# weighting
# --------------------------------------------------------------------------- #
def class_weights(y: np.ndarray, w_ext: float, mask_ext: np.ndarray | None = None):
    """Per-row weights so every class carries total weight 1 (comp/ext split w).

    ``w_ext`` is the share of each class's total weight given to external rows.
    With ``w_ext=0`` the competition rows are class-balanced among themselves.
    """
    w = np.zeros(len(y), dtype=np.float64)
    idx_comp = np.ones(len(y), bool) if mask_ext is None else ~mask_ext
    idx_ext = ~idx_comp

    def _fill(idx: np.ndarray, share: float) -> None:
        if share <= 0 or not idx.any():
            return
        labels, counts = np.unique(y[idx], return_counts=True)
        cnt = dict(zip(labels, counts))
        w[idx] = share / np.array([cnt[v] for v in y[idx]], dtype=np.float64)

    if mask_ext is None:
        _fill(idx_comp, 1.0)
    else:
        _fill(idx_comp, 1.0 - w_ext)
        _fill(idx_ext, w_ext)
    w *= len(LABELS)
    return w / w.mean()          # mean weight 1 keeps C comparable across variants


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) * t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- #
# model runners
# --------------------------------------------------------------------------- #
def run_linear(Xc, Xe, kind: str, w_ext: float, seed: int = 0) -> np.ndarray:
    """Fit one linear model per fold on [fold-train + external] and return OOF probs."""
    oof = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        if w_ext > 0:
            Xtr = sp.vstack([Xc[ti], Xe], format="csr")
            ytr = np.concatenate([Y[ti], EXT["y"].to_numpy()])
            sw = class_weights(ytr, w_ext, np.r_[np.zeros(len(ti), bool), np.ones(Xe.shape[0], bool)])
        else:
            Xtr, ytr = Xc[ti], Y[ti]
            sw = class_weights(ytr, 0.0)
        if kind == "svc":
            clf = LinearSVC(C=0.5, dual=True, max_iter=4000, class_weight=None)
            clf.fit(Xtr, ytr, sample_weight=sw)
            oof[vi] = softmax(clf.decision_function(Xc[vi]), 1.0)
        elif kind == "svc_unw":
            clf = LinearSVC(C=0.5, dual=True, max_iter=4000, class_weight=None)
            clf.fit(Xtr, ytr)
            oof[vi] = softmax(clf.decision_function(Xc[vi]), 1.0)
        elif kind == "svc_c1":
            clf = LinearSVC(C=1.0, dual=True, max_iter=4000)
            clf.fit(Xtr, ytr, sample_weight=sw)
            oof[vi] = softmax(clf.decision_function(Xc[vi]), 1.0)
        elif kind == "svc_c025":
            clf = LinearSVC(C=0.25, dual=True, max_iter=4000)
            clf.fit(Xtr, ytr, sample_weight=sw)
            oof[vi] = softmax(clf.decision_function(Xc[vi]), 1.0)
        elif kind == "sgd":
            clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=40, tol=1e-4,
                                random_state=42, average=True)
            clf.fit(Xtr, ytr, sample_weight=sw)
            oof[vi] = clf.predict_proba(Xc[vi])
        else:
            raise ValueError(kind)
        log(f"  {kind} w={w_ext} fold{f} n_tr={Xtr.shape[0]} bal="
            f"{balanced_accuracy_score(Y[vi], oof[vi].argmax(1)):.4f}")
        del Xtr
    return oof


def run_nbsvm(Xc, Xe, w_ext: float, alpha: float = 1.0, C: float = 0.1) -> np.ndarray:
    """NB-SVM: per-class log-count-ratio scaled features + one-vs-rest LinearSVC.

    The log-count ratio r_c = log( (p_c + a) / (q_c + a) ) is computed from the
    binarised training matrix of each fold; features become X * r_c, which is the
    standard strong baseline for authorship/style attribution.
    """
    oof = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        if w_ext > 0:
            Xtr = sp.vstack([Xc[ti], Xe], format="csr")
            ytr = np.concatenate([Y[ti], EXT["y"].to_numpy()])
            sw = class_weights(ytr, w_ext, np.r_[np.zeros(len(ti), bool), np.ones(Xe.shape[0], bool)])
        else:
            Xtr, ytr = Xc[ti], Y[ti]
            sw = class_weights(ytr, 0.0)
        b = (Xtr > 0).astype(np.float32)
        dec = np.zeros((len(vi), len(LABELS)), dtype=np.float32)
        for c in range(len(LABELS)):
            pos = (ytr == c)
            p = np.asarray(b[pos].sum(0)).ravel() + alpha
            q = np.asarray(b[~pos].sum(0)).ravel() + alpha
            r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
            Xr = Xtr.multiply(r).tocsr()
            clf = LinearSVC(C=C, dual=True, max_iter=3000)
            clf.fit(Xr, pos.astype(int), sample_weight=sw)
            dec[:, c] = clf.decision_function(Xc[vi].multiply(r).tocsr())
            del Xr
        oof[vi] = softmax(dec, 2.0)
        log(f"  nbsvm w={w_ext} fold{f} bal="
            f"{balanced_accuracy_score(Y[vi], oof[vi].argmax(1)):.4f}")
        del Xtr, b
    return oof


def run_centroid(Xc, w_ext: float) -> np.ndarray:
    """Cosine similarity to class centroids of the fold-train competition rows."""
    oof = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    Xn = normalize(Xc)
    for f, (ti, vi) in enumerate(FOLDS):
        cent = np.vstack([np.asarray(Xn[ti][Y[ti] == c].mean(0)).ravel()
                          for c in range(len(LABELS))])
        cent = normalize(cent)
        oof[vi] = softmax(Xn[vi] @ cent.T, 12.0)
    return oof


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def score(oof: np.ndarray, mask: np.ndarray = CLEAN) -> float:
    return float(balanced_accuracy_score(Y[mask], oof[mask].argmax(1)))


def prior_correct(P: np.ndarray, alpha: float) -> np.ndarray:
    """Divide by class priors^alpha, pushing the argmax toward uniform recall."""
    prior = np.bincount(Y, minlength=len(LABELS)).astype(np.float64)
    prior /= prior.sum()
    return (P / prior[None, :] ** alpha).astype(np.float32)


def save_oof(tag: str, P: np.ndarray) -> None:
    np.save(f"{CACHE}/oof_{tag}.npy", P.astype(np.float32))


def load_oof(tag: str) -> np.ndarray:
    return np.load(f"{CACHE}/oof_{tag}.npy")


if __name__ == "__main__":
    results: dict[str, dict] = {}
    XcwS, Xe_cwS = get_matrices("cwS")
    Xall, Xe_all = get_matrices("all")

    # (tag, feature config, model kind, external weight)
    grid = [
        ("svc_cwS_w0", "cwS", "svc", 0.0),
        ("svc_unw_all_w0", "all", "svc_unw", 0.0),
        ("svc_all_w0", "all", "svc", 0.0),
        ("sgd_all_w0", "all", "sgd", 0.0),
        ("sgd_all_w50", "all", "sgd", 0.5),
    ]
    for tag, cfg, kind, w in grid:
        cache = f"{CACHE}/oof_{tag}.npy"
        Xc, Xe = (XcwS, Xe_cwS) if cfg == "cwS" else (Xall, Xe_all)
        if os.path.exists(cache):
            P = np.load(cache)
        else:
            P = run_linear(Xc, Xe, kind, w)
            save_oof(tag, P)
        b_clean, b_all = score(P), score(P, np.ones(len(Y), bool))
        results[tag] = {"bal_acc_clean": b_clean, "bal_acc_all": b_all}
        log(f"{tag}: clean {b_clean:.4f} all {b_all:.4f}")
        log_experiment(f"bal_acc_{tag}", b_clean,
                       f"views={cfg} kind={kind} w_ext={w}; grouped 5-fold, clean subset "
                       f"(n={int(CLEAN.sum())}); all-rows={b_all:.4f}")

    if not os.path.exists(f"{CACHE}/oof_nbsvm_all.npy"):
        P_nb = run_nbsvm(Xall, Xe_all, 0.0)
        save_oof("nbsvm_all", P_nb)
    P_nb = np.load(f"{CACHE}/oof_nbsvm_all.npy")
    results["nbsvm_all"] = {"bal_acc_clean": score(P_nb),
                            "bal_acc_all": score(P_nb, np.ones(len(Y), bool))}
    log(f"nbsvm_all: {results['nbsvm_all']}")
    log_experiment("bal_acc_nbsvm_all", results["nbsvm_all"]["bal_acc_clean"],
                   "NB-SVM log-count-ratio + OvR LinearSVC C=0.1, views=all, comp-only, "
                   "prior-softmax T=2")

    P_cen = run_centroid(Xall, 0.0)
    save_oof("centroid_all", P_cen)
    results["centroid_all"] = {"bal_acc_clean": score(P_cen)}
    log_experiment("bal_acc_centroid_all", results["centroid_all"]["bal_acc_clean"],
                   "cosine to class centroids, views=all, comp-only")

    json.dump(results, open(f"{OUT}/cv_variants.json", "w"), indent=1)
    log("variants done")
