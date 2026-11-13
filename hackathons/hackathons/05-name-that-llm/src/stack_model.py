"""Picklable stacked final model for Track 05.

Extends the member-blend ``NTLModel`` with a *meta stacker*: a multinomial
LogisticRegression trained on the cross-validated OOF probability matrices of
the three members.  Two feature stacks are held (the ``all`` view for svc +
NB-SVM and the ``all35`` view for the svc35 diversity member).
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntl_features import LABELS, FeatureStack  # noqa: E402
from ntl_model import NTLModel, _softmax, balanced_weights  # noqa: E402

CLAUDE = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
          "claude-sonnet-5"]


class StackedNTLModel(NTLModel):
    """svc + nbsvm + svc35 fitted separately, combined by a C=8 LR meta stacker."""

    def __init__(self, feature_config: dict, cfg35: dict,
                 extra_claude_weight: float, meta_C: float = 8.0) -> None:
        super().__init__(feature_config, members=("svc", "nbsvm"), weights=(1, 1))
        self.cfg35 = dict(cfg35)
        self.extra_claude_weight = float(extra_claude_weight)
        self.meta_C = float(meta_C)
        self.meta: LogisticRegression | None = None
        self.stack35: FeatureStack | None = None
        self.fitted["svc35"] = None

    # ------------------------------------------------------------------ #
    def _extra_rows(self, ext_texts, ext_y, ext_raw, ext_source) -> tuple:
        """Restrict external rows to the protocol-matched extra-claude block."""
        if ext_texts is None:
            return None, None
        keep = (ext_source == "extra_claude") & np.isin(
            ext_y, [LABELS.index(m) for m in CLAUDE])
        return ext_texts[keep], ext_y[keep]

    def fit(self, comp, ext, oof_probs: dict[str, np.ndarray]) -> "StackedNTLModel":
        """Fit stacks on external corpus, members on all comp rows, meta on OOF.

        ``oof_probs`` maps ``svc``/``nbsvm``/``svc35`` to their cross-validated
        OOF probability matrices (from ``run_experiments`` / ``followups``).
        """
        texts = comp["text"].astype(str).tolist()
        y = comp["y"].to_numpy()
        e_texts = ext["text"].astype(str).tolist()
        e_y = ext["y"].to_numpy()
        self.stack = FeatureStack(**self.feature_config).fit(e_texts)
        self.stack35 = FeatureStack(**self.cfg35).fit(e_texts)
        X = self.stack.transform(texts)
        X35 = self.stack35.transform(texts)
        Xe = self.stack.transform(e_texts)
        Xe35 = self.stack35.transform(e_texts)
        self.prior = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
        self.prior /= self.prior.sum()

        # svc: extra-claude rows for the 5 Claude classes at half weight
        keep_xc = np.where((ext["source"] == "extra_claude").to_numpy()
                           & np.isin(e_y, [LABELS.index(m) for m in CLAUDE]))[0]
        share = np.zeros(len(LABELS))
        for m in CLAUDE:
            share[LABELS.index(m)] = self.extra_claude_weight
        Xm = sp.vstack([X, Xe[keep_xc]], format="csr")
        ym = np.concatenate([y, e_y[keep_xc]])
        sw = self._mix_weights(y, e_y[keep_xc], share, np.ones(len(keep_xc)))
        clf = LinearSVC(C=0.5, dual=True, max_iter=4000).fit(Xm, ym, sample_weight=sw)
        self.fitted["svc"] = clf
        del Xm

        # nbsvm on competition rows only
        b = (X > 0).astype(np.float32)
        ratios, clfs = [], []
        for c in range(len(LABELS)):
            pos = y == c
            p = np.asarray(b[pos].sum(0)).ravel() + 1.0
            q = np.asarray(b[~pos].sum(0)).ravel() + 1.0
            r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
            c_c = LinearSVC(C=0.1, dual=True, max_iter=3000)
            c_c.fit(X.multiply(r).tocsr(), pos.astype(int), sample_weight=balanced_weights(y))
            ratios.append(r)
            clfs.append(c_c)
        del b
        self.fitted["nbsvm"] = (ratios, clfs)

        # svc35: competition rows only, C=0.5
        c35 = LinearSVC(C=0.5, dual=True, max_iter=4000)
        c35.fit(X35, y, sample_weight=balanced_weights(y, len(LABELS)))
        self.fitted["svc35"] = c35

        # meta stacker on the cross-validated OOF probabilities
        S_meta = np.hstack([oof_probs["svc"], oof_probs["nbsvm"],
                            oof_probs["svc35"]]).astype(np.float32)
        self.meta = LogisticRegression(C=self.meta_C, max_iter=3000,
                                       class_weight="balanced", n_jobs=-1)
        self.meta.fit(S_meta, y)
        return self

    # ------------------------------------------------------------------ #
    def _member_proba(self, name: str, X: sp.csr_matrix, X35: sp.csr_matrix) -> np.ndarray:
        if name == "svc35":
            return _softmax(self.fitted["svc35"].decision_function(X35), 1.0)
        return super()._member_proba(name, X)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        X = self.stack.transform(texts)
        X35 = self.stack35.transform(texts)
        S = np.hstack([self._member_proba("svc", X, X35),
                       self._member_proba("nbsvm", X, X35),
                       self._member_proba("svc35", X, X35)]).astype(np.float32)
        return self.meta.predict_proba(S).astype(np.float32)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([LABELS[i] for i in self.predict_proba(texts).argmax(1)])
