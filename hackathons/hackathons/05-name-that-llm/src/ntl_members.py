"""Linear classifier members for Track 05 with class-balanced sample weighting.

Each member exposes ``fit(X, y, sw)`` and ``proba(X)``.  The NB-SVM member
implements the multinomial log-count-ratio reweighting of the feature matrix
(one binary problem per class), which is the cheapest strong probe for
class-conditional token weighting on this task.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

N_CLASSES = 18


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with temperature, numerically stable."""
    z = np.asarray(z, dtype=np.float64) * float(temperature)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def balanced_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Sample weights giving every class equal total mass."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    per_row = np.array([1.0 / max(counts[c], 1.0) for c in y])
    return per_row * n_classes


class SVCMember:
    """One-vs-rest LinearSVC with softmax over decision values."""

    def __init__(self, C: float = 0.5, temperature: float = 1.0) -> None:
        self.C = float(C)
        self.temperature = float(temperature)
        self.clf: LinearSVC | None = None

    def fit(self, X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray) -> "SVCMember":
        """Fit the classifier on weighted rows."""
        self.clf = LinearSVC(C=self.C, dual=True, max_iter=6000).fit(X, y, sample_weight=sw)
        return self

    def proba(self, X: sp.csr_matrix) -> np.ndarray:
        """Softmax of the OvR decision function."""
        assert self.clf is not None
        return softmax(self.clf.decision_function(X), self.temperature)


class LogisticMember:
    """Multinomial logistic regression (lbfgs or saga) on sparse features."""

    def __init__(self, C: float = 2.0, solver: str = "lbfgs", max_iter: int = 400) -> None:
        self.C = float(C)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.clf: LogisticRegression | None = None

    def fit(self, X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray) -> "LogisticMember":
        """Fit multinomial logistic regression."""
        self.clf = LogisticRegression(C=self.C, solver=self.solver, max_iter=self.max_iter,
                                      n_jobs=1).fit(X, y, sample_weight=sw)
        return self

    def proba(self, X: sp.csr_matrix) -> np.ndarray:
        """Predicted class probabilities."""
        assert self.clf is not None
        return self.clf.predict_proba(X).astype(np.float32)


class NBSVMMember:
    """NB-SVM: per-class log-count-ratio features + binary LinearSVC."""

    def __init__(self, C: float = 0.1, temperature: float = 2.0, binarise: bool = True) -> None:
        self.C = float(C)
        self.temperature = float(temperature)
        self.binarise = bool(binarise)
        self.ratios: list[np.ndarray] = []
        self.clfs: list[LinearSVC] = []

    def fit(self, X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray) -> "NBSVMMember":
        """Fit one log-count-ratio vector and one binary SVM per class."""
        base = (X > 0).astype(np.float32) if self.binarise else X
        for c in range(N_CLASSES):
            pos = y == c
            p = np.asarray(base[pos].sum(0)).ravel() + 1.0
            q = np.asarray(base[~pos].sum(0)).ravel() + 1.0
            r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
            clf = LinearSVC(C=self.C, dual=True, max_iter=4000).fit(
                X.multiply(r).tocsr(), pos.astype(np.int8), sample_weight=sw)
            self.ratios.append(r)
            self.clfs.append(clf)
        return self

    def proba(self, X: sp.csr_matrix) -> np.ndarray:
        """Softmax over the 18 binary decision values."""
        dec = np.zeros((X.shape[0], N_CLASSES), dtype=np.float32)
        for c, (r, clf) in enumerate(zip(self.ratios, self.clfs)):
            dec[:, c] = clf.decision_function(X.multiply(r).tocsr())
        return softmax(dec, self.temperature)


class RidgeMember:
    """Ridge classifier with softmax over class scores (cheap, well-calibrated rank)."""

    def __init__(self, alpha: float = 1.0, temperature: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.clf: RidgeClassifier | None = None

    def fit(self, X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray) -> "RidgeMember":
        """Fit the ridge classifier."""
        self.clf = RidgeClassifier(alpha=self.alpha, solver="sparse_cg",
                                   max_iter=400).fit(X, y, sample_weight=sw)
        return self

    def proba(self, X: sp.csr_matrix) -> np.ndarray:
        """Softmax over class scores."""
        assert self.clf is not None
        return softmax(self.clf.decision_function(X), self.temperature)


class CentroidMember:
    """Cosine similarity to class centroids, sharpened by temperature."""

    def __init__(self, temperature: float = 12.0) -> None:
        self.temperature = float(temperature)
        self.centroids: np.ndarray | None = None

    def fit(self, X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray) -> "CentroidMember":
        """Compute weighted class centroids on L2-normalised features."""
        Xn = normalize(X)
        rows = []
        for c in range(N_CLASSES):
            m = y == c
            w = sw[m][:, None]
            rows.append(np.asarray(Xn[m].multiply(w).sum(0)).ravel() / max(w.sum(), 1e-9))
        self.centroids = normalize(np.vstack(rows))
        return self

    def proba(self, X: sp.csr_matrix) -> np.ndarray:
        """Softmax over cosine similarities."""
        assert self.centroids is not None
        return softmax(normalize(X) @ self.centroids.T, self.temperature)


class PriorCorrector:
    """Divides probabilities by ``prior ** alpha`` to trade accuracy for macro recall."""

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = float(alpha)
        self.prior: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "PriorCorrector":
        """Estimate the training class prior."""
        p = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        self.prior = p / p.sum()
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Apply the prior correction."""
        if self.alpha == 0.0 or self.prior is None:
            return proba
        return (proba / self.prior[None, :] ** self.alpha).astype(np.float32)
