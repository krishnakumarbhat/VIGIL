"""Grouped cross-validation harness over the REAL competition rows.

Every variant receives features that were fitted on the disjoint mirror corpus
only, so no validation text ever touches the vocabulary or the IDF weights.
The split is ``StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)``
with ``groups = prompt_id``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
SEED = 42


@dataclass
class FoldResult:
    """Out-of-fold probabilities plus the fold assignment used to build them."""

    proba: np.ndarray
    folds: np.ndarray


@dataclass
class VariantScore:
    """Scored variant: OOF probabilities and derived metrics."""

    name: str
    notes: str
    proba: np.ndarray
    folds: np.ndarray
    metrics: dict = field(default_factory=dict)


def grouped_folds(y: np.ndarray, groups: np.ndarray,
                  n_splits: int = N_SPLITS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the fixed (train_idx, valid_idx) split list for the protocol."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


class GroupedCVRunner:
    """Runs any ``fit_predict(Xtr, ytr, Xva) -> proba`` callable across folds."""

    def __init__(self, y: np.ndarray, groups: np.ndarray, n_splits: int = N_SPLITS,
                 seed: int = SEED) -> None:
        self.y = np.asarray(y)
        self.groups = np.asarray(groups)
        self.folds = grouped_folds(self.y, self.groups, n_splits, seed)

    def run(self, fit_predict: Callable[..., np.ndarray], X: sp.csr_matrix,
            n_classes: int = 18) -> FoldResult:
        """Evaluate one fit/predict callable and return OOF probabilities."""
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float32)
        fold_id = np.zeros(X.shape[0], dtype=np.int8)
        for k, (tr, va) in enumerate(self.folds):
            proba[va] = fit_predict(X[tr], self.y[tr], X[va], k)
            fold_id[va] = k
        return FoldResult(proba=proba, folds=fold_id)


class MetricsCalculator:
    """Balanced accuracy, accuracy, per-class recall and length-bin breakdown."""

    def __init__(self, labels: Sequence[str], length_bins: Sequence[int]) -> None:
        self.labels = list(labels)
        self.bins = list(length_bins)

    def score(self, proba: np.ndarray, y: np.ndarray, n_words: np.ndarray) -> dict:
        """Compute every metric reported in ``cv_report.json``."""
        pred = proba.argmax(1)
        recalls = recall_score(y, pred, labels=list(range(len(self.labels))),
                               average=None, zero_division=0)
        out = {
            "bal_acc": float(recall_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "n_rows": int(len(y)),
            "per_class_recall": {lab: float(r) for lab, r in zip(self.labels, recalls)},
            "length_bin_accuracy": {},
        }
        for lo, hi in zip(self.bins[:-1], self.bins[1:]):
            m = (n_words > lo) & (n_words <= hi)
            name = f"({lo},{hi}]"
            if m.any():
                out["length_bin_accuracy"][name] = {
                    "n": int(m.sum()),
                    "acc": float(accuracy_score(y[m], pred[m])),
                    "bal_acc": float(recall_score(y[m], pred[m], average="macro", zero_division=0)),
                }
            else:
                out["length_bin_accuracy"][name] = {"n": 0, "acc": None, "bal_acc": None}
        return out
