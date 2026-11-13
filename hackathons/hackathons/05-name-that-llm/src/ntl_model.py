"""Picklable final model for Track 05 (Name That LLM).

Wraps the feature stack plus the linear members selected by grouped CV, so the
inference notebook only needs ``joblib.load`` and ``predict_proba(texts)``.
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntl_features import LABELS, FeatureStack  # noqa: E402


def _softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) * t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def balanced_weights(y: np.ndarray, n_classes: int = len(LABELS)) -> np.ndarray:
    """Weights that give every class the same total weight."""
    labels, counts = np.unique(y, return_counts=True)
    cnt = dict(zip(labels, counts))
    w = np.array([1.0 / cnt[v] for v in y], dtype=np.float64)
    return w * n_classes


class NTLModel:
    """TF-IDF features + blended linear classifiers with prior correction.

    ``members`` subsets ``{"svc", "sgd", "nbsvm", "centroid"}``; ``weights`` are
    the non-negative blend weights (normalised at predict time).  ``prior_alpha``
    divides the blended probabilities by the training class priors, which trades
    raw accuracy for macro recall - the competition metric.
    """

    def __init__(self, feature_config: dict, members: Sequence[str] = ("svc", "nbsvm"),
                 weights: Sequence[float] | None = None, prior_alpha: float = 0.0) -> None:
        self.feature_config = dict(feature_config)
        self.members = tuple(members)
        self.weights = tuple(weights) if weights is not None else tuple([1.0] * len(members))
        self.prior_alpha = float(prior_alpha)
        self.stack: FeatureStack | None = None
        self.fitted: dict = {}
        self.prior: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _member_fit(self, name: str, X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray) -> None:
        if name == "svc":
            clf = LinearSVC(C=0.5, dual=True, max_iter=4000)
            clf.fit(X, y, sample_weight=sw)
            self.fitted["svc"] = clf
        elif name == "sgd":
            clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=40, tol=1e-4,
                                random_state=42, average=True)
            clf.fit(X, y, sample_weight=sw)
            self.fitted["sgd"] = clf
        elif name == "centroid":
            Xn = normalize(X)
            cent = np.vstack([np.asarray(Xn[y == c].mean(0)).ravel() for c in range(len(LABELS))])
            self.fitted["centroid"] = normalize(cent)
        elif name == "nbsvm":
            b = (X > 0).astype(np.float32)
            ratios, clfs = [], []
            for c in range(len(LABELS)):
                pos = (y == c)
                p = np.asarray(b[pos].sum(0)).ravel() + 1.0
                q = np.asarray(b[~pos].sum(0)).ravel() + 1.0
                r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
                clf = LinearSVC(C=0.1, dual=True, max_iter=3000)
                clf.fit(X.multiply(r).tocsr(), pos.astype(int), sample_weight=sw)
                ratios.append(r)
                clfs.append(clf)
            self.fitted["nbsvm"] = (ratios, clfs)
        else:
            raise ValueError(f"unknown member {name}")

    @staticmethod
    def _mix_weights(y_comp: np.ndarray, y_ext: np.ndarray, share: np.ndarray,
                     raw: np.ndarray | None = None) -> np.ndarray:
        """Per-row weights giving each class total 1, with ``share[c]`` on external rows.

        ``share`` is a length-18 vector of the per-class mass handed to external
        rows (0 keeps a class competition-only); ``raw`` scales individual
        external rows before that mass is split.
        """
        y_all = np.concatenate([y_comp, y_ext])
        is_ext = np.r_[np.zeros(len(y_comp), bool), np.ones(len(y_ext), bool)]
        raw = np.ones(len(y_ext)) if raw is None else np.asarray(raw, float)
        raw = np.concatenate([np.zeros(len(y_comp)), raw])    # rows are indexed over all rows
        sw = np.zeros(len(y_all))
        for c in range(len(LABELS)):
            comp_m = (~is_ext) & (y_all == c)
            ext_m = is_ext & (y_all == c)
            s = float(share[c]) if c < len(share) else 0.0
            if comp_m.any():
                sw[comp_m] = (1.0 - s) / comp_m.sum()
            if ext_m.any() and s > 0 and raw[ext_m].sum() > 0:
                sw[ext_m] = s * raw[ext_m] / raw[ext_m].sum()
        sw *= len(LABELS)
        return sw / sw.mean()

    def fit(self, texts: Sequence[str], y: np.ndarray,
            ext_texts: Sequence[str] | None = None,
            ext_y: np.ndarray | None = None, w_ext: float = 0.0,
            w_ext_map: dict[str, float] | None = None,
            ext_share_map: dict[str, np.ndarray] | None = None,
            ext_raw: np.ndarray | None = None) -> "NTLModel":
        """Fit features on the external corpus, then every member on comp (+weighted external).

        ``w_ext_map`` gives each member a scalar external mixing weight; the more
        precise ``ext_share_map`` gives a per-class share vector (and optional
        per-row ``ext_raw``), which is how the protocol-matched extra corpus is
        used for the five Claude classes only.
        """
        self.w_ext_map = dict(w_ext_map or {})
        self.ext_share_map = dict(ext_share_map or {})
        self.stack = FeatureStack(**self.feature_config).fit(
            list(ext_texts) if ext_texts is not None else list(texts))
        X = self.stack.transform(list(texts))
        Xe = self.stack.transform(list(ext_texts)) if ext_texts is not None else None
        self.prior = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
        self.prior /= self.prior.sum()

        for name in self.members:
            if name in self.ext_share_map and Xe is not None:
                raw = np.ones(Xe.shape[0]) if ext_raw is None else np.asarray(ext_raw, float)
                keep = raw > 0
                Xm = sp.vstack([X, Xe[keep]], format="csr")
                ym = np.concatenate([y, ext_y[keep]])
                sw = self._mix_weights(y, ext_y[keep], self.ext_share_map[name], raw[keep])
            elif self.w_ext_map.get(name, w_ext) > 0 and Xe is not None:
                w = self.w_ext_map.get(name, w_ext)
                Xm = sp.vstack([X, Xe], format="csr")
                ym = np.concatenate([y, ext_y])
                share = np.full(len(LABELS), w)
                sw = self._mix_weights(y, ext_y, share)
            else:
                Xm, ym, sw = X, y, balanced_weights(y)
            self._member_fit(name, Xm, ym, sw)
            if Xm is not X:
                del Xm
        return self

    # ------------------------------------------------------------------ #
    def _member_proba(self, name: str, X: sp.csr_matrix) -> np.ndarray:
        if name == "svc":
            return _softmax(self.fitted["svc"].decision_function(X), 1.0)
        if name == "sgd":
            return self.fitted["sgd"].predict_proba(X)
        if name == "centroid":
            return _softmax(normalize(X) @ self.fitted["centroid"].T, 12.0)
        if name == "nbsvm":
            ratios, clfs = self.fitted["nbsvm"]
            dec = np.zeros((X.shape[0], len(LABELS)), dtype=np.float32)
            for c, (r, clf) in enumerate(zip(ratios, clfs)):
                dec[:, c] = clf.decision_function(X.multiply(r).tocsr())
            return _softmax(dec, 2.0)
        raise ValueError(name)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self.stack.transform(list(texts))
        out = np.zeros((X.shape[0], len(LABELS)), dtype=np.float32)
        wsum = sum(self.weights)
        for name, w in zip(self.members, self.weights):
            if w:
                out += (w / wsum) * self._member_proba(name, X)
        if self.prior_alpha:
            out = out / self.prior[None, :] ** self.prior_alpha
        return out.astype(np.float32)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([LABELS[i] for i in self.predict_proba(texts).argmax(1)])
