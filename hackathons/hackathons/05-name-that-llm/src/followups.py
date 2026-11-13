"""Two cheap follow-up experiments, run after the main grid.

1. ``svc_c1`` / ``svc_c025``: LinearSVC regularisation sweep on the competition-only
   all-view features (the best single model was C=0.5 at 0.3754).
2. ``svc_extra_claude_w50``: external rows restricted to the protocol-matched
   ``ntl-extra-api-passages`` block (one paragraph, per-generation length target),
   mixed at w=0.5 for the five Claude classes only.

Both keep the fixed grouped folds and write OOF matrices for the blend search.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntl_features import LABELS, load_competition_frame, load_external_frame  # noqa: E402
from run_experiments import (COMP, FOLDS, Y, get_matrices, log,  # noqa: E402
                             log_experiment, save_oof, score)

CLAUDE = {"claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
          "claude-sonnet-5"}


def svc_variant(tag: str, C: float, Xc: sp.csr_matrix) -> np.ndarray:
    """Comp-only LinearSVC at a given C, fixed grouped folds."""
    from ntl_model import balanced_weights
    oof = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        clf = LinearSVC(C=C, dual=True, max_iter=3000)
        clf.fit(Xc[ti], Y[ti], sample_weight=balanced_weights(Y[ti]))
        d = clf.decision_function(Xc[vi])
        d -= d.max(1, keepdims=True)
        e = np.exp(d)
        oof[vi] = (e / e.sum(1, keepdims=True)).astype(np.float32)
        log(f"  {tag} fold{f} bal={balanced_accuracy_score(Y[vi], oof[vi].argmax(1)):.4f}")
    save_oof(tag, oof)
    log_experiment(f"bal_acc_{tag}", score(oof),
                   f"LinearSVC C={C}, char2-5+word1-2+shape+sty, comp-only, class-balanced weights")
    return oof


def extra_claude_variant(w: float, C: float, Xc: sp.csr_matrix, Xe: sp.csr_matrix) -> np.ndarray:
    """Protocol-matched extra-corpus rows for the 5 Claude classes at weight w."""
    ext = load_external_frame(max_per_class=1200)
    keep = ((ext["source"] == "extra_claude") & ext["model"].isin(CLAUDE)).to_numpy()
    Xe_c, y_e = Xe[keep], ext.loc[keep, "y"].to_numpy()
    log(f"extra_claude rows {Xe_c.shape[0]} (of {len(ext)})")
    claude_ids = {LABELS.index(m) for m in CLAUDE}
    oof = np.zeros((len(COMP), len(LABELS)), dtype=np.float32)
    for f, (ti, vi) in enumerate(FOLDS):
        Xtr = sp.vstack([Xc[ti], Xe_c], format="csr")
        ytr = np.concatenate([Y[ti], y_e])
        mask = np.r_[np.zeros(len(ti), bool), np.ones(Xe_c.shape[0], bool)]
        sw = np.zeros(len(ytr))
        for share, m in ((1 - w, ~mask), (w, mask)):
            lab, cnt = np.unique(ytr[m], return_counts=True)
            d = dict(zip(lab, cnt))
            sw[m] = share / np.array([d[v] for v in ytr[m]])
        for c in range(len(LABELS)):        # non-Claude classes stay competition-only
            if c not in claude_ids:
                sw[(~mask) & (ytr == c)] /= (1 - w)
        sw *= len(LABELS)
        sw /= sw.mean()
        clf = LinearSVC(C=C, dual=True, max_iter=3000).fit(Xtr, ytr, sample_weight=sw)
        d = clf.decision_function(Xc[vi])
        d -= d.max(1, keepdims=True)
        e = np.exp(d)
        oof[vi] = (e / e.sum(1, keepdims=True)).astype(np.float32)
        log(f"  extra_claude fold{f} bal={balanced_accuracy_score(Y[vi], oof[vi].argmax(1)):.4f}")
        del Xtr
    save_oof("svc_extra_claude_w50", oof)
    log_experiment("bal_acc_svc_extra_claude_w50", score(oof),
                   f"LinearSVC C={C}, external restricted to protocol-matched extra_claude rows, "
                   f"Claude classes only, w={w}")
    return oof


if __name__ == "__main__":
    Xc, Xe = get_matrices("all")
    svc_variant("svc_c1", 1.0, Xc)
    svc_variant("svc_c025", 0.25, Xc)
    extra_claude_variant(0.5, 0.5, Xc, Xe)
    log("follow-ups done")
