"""Learned ranking prior over candidate programs.

The v1 cascade returned the first hypothesis that survived training-pair
verification.  That turns a precision problem into a *ranking* problem only if
we can score hypotheses.  This module builds a small supervised model from
real solver runs on the training split:

    features(task, candidate) -> candidate correct on the held-out test output?

Training tasks provide the labels.  The evaluation split is never used for
fitting, and validation is a task-level split so that no task contributes both
a training and a validation row.

Features are deliberately gold-free: family identity, program complexity,
leave-one-out evidence, output shape statistics, agreement between candidate
families, and coarse task descriptors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.search import expected_output_shape

Grid = np.ndarray

FAMILIES: Tuple[str, ...] = (
    "depth1", "beam", "mosaic", "select_crop", "select_bbox", "recolour_rule",
    "move_all", "duplicate_offset", "delete_rule", "fill_holes", "outline",
    "stamp", "connect", "cell_tree", "cell_tree_d4", "object_tree", "patch_lookup",
    "mlp", "fallback",
)

#: Fallback ordering used before the learned model is available, derived from
#: the v1 per-method precision table (symbolic methods first).
MANUAL_PRIOR: Dict[str, float] = {
    "depth1": 1.00, "beam": 0.95, "mosaic": 0.90, "select_crop": 0.75,
    "select_bbox": 0.72, "recolour_rule": 0.65, "move_all": 0.60,
    "duplicate_offset": 0.60, "delete_rule": 0.60, "fill_holes": 0.70,
    "outline": 0.55, "stamp": 0.50, "connect": 0.50, "cell_tree_d4": 0.55,
    "cell_tree": 0.30, "patch_lookup": 0.30, "object_tree": 0.12, "mlp": 0.25,
    "fallback": 0.05,
}


_SHAPE_CACHE: Dict[Tuple[str, int], Any] = {}


def _expected_shape(task, index: int) -> Optional[Tuple[int, int]]:
    """Cached output-shape prediction for one task/test index.

    ``fit_shape_model`` is cheap but is queried once per candidate per test
    input; caching keeps the ranker's feature extraction O(1) per candidate.
    """
    key = (getattr(task, "task_id", "?"), index)
    if key not in _SHAPE_CACHE:
        try:
            _SHAPE_CACHE[key] = expected_output_shape(task, index)
        except Exception:
            _SHAPE_CACHE[key] = None
        if len(_SHAPE_CACHE) > 4096:
            _SHAPE_CACHE.clear()
    return _SHAPE_CACHE[key]


def _one_hot(name: str) -> np.ndarray:
    """One-hot encoding of a family name over the fixed family list."""
    vector = np.zeros(len(FAMILIES), dtype=np.float32)
    if name in FAMILIES:
        vector[FAMILIES.index(name)] = 1.0
    return vector


def task_features(task) -> np.ndarray:
    """Gold-free descriptors of a task used by every candidate of that task."""
    ins = [a for a, _ in task.train]
    outs = [b for _, b in task.train]
    shapes_same = np.mean([a.shape == b.shape for a, b in zip(ins, outs)])
    area = np.mean([a.size for a in ins])
    colours = np.mean([len(np.unique(a)) for a in ins])
    out_colours = np.mean([len(np.unique(b)) for b in outs])
    n_train = float(len(task.train))
    n_test = float(len(task.test_inputs))
    test_area = np.mean([g.size for g in task.test_inputs]) if task.test_inputs else 0.0
    # Coarse output-size relation, which drives which families are plausible.
    h_ratio = np.median([b.shape[0] / max(a.shape[0], 1) for a, b in zip(ins, outs)])
    w_ratio = np.median([b.shape[1] / max(a.shape[1], 1) for a, b in zip(ins, outs)])
    return np.array([n_train, n_test, np.log1p(area), np.log1p(test_area),
                     shapes_same, colours, out_colours, h_ratio, w_ratio,
                     float(np.mean([len(np.unique(a)) > 2 for a in ins]))],
                    dtype=np.float32)


def candidate_features(task, candidate, all_candidates: Sequence[Any],
                       method_prior: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Gold-free feature vector for one candidate of one task.

    Includes the agreement between the candidate's output shape and the task's
    fitted output-shape model, which is what separates two same-family
    candidates (e.g. ``crop_content`` versus ``crop_largest_obj``) when only one
    of them can possibly be right.
    """
    outputs = candidate.outputs
    test_inputs = task.test_inputs
    identical_duplicates = 0
    for other in all_candidates:
        if other is candidate or len(other.outputs) != len(outputs):
            continue
        if all(a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b)
               for a, b in zip(outputs, other.outputs)):
            identical_duplicates += 1
    shape_equal = np.mean([o is not None and o.shape == g.shape
                           for o, g in zip(outputs, test_inputs)]) if outputs else 0.0
    area_ratio = np.mean([o.size / max(g.size, 1) for o, g in zip(outputs, test_inputs)]) \
        if outputs else 0.0
    palette = np.mean([len(np.unique(o)) for o in outputs]) if outputs else 0.0
    changed = np.mean([float((o != g).mean()) if o is not None and o.shape == g.shape else 1.0
                       for o, g in zip(outputs, test_inputs)]) if outputs else 0.0
    is_identity = np.mean([float(o is not None and o.shape == g.shape and np.array_equal(o, g))
                           for o, g in zip(outputs, test_inputs)]) if outputs else 0.0
    prior = (method_prior or MANUAL_PRIOR).get(candidate.family, 0.1)
    expected = []
    for index in range(len(test_inputs)):
        want = _expected_shape(task, index)
        produced = outputs[index] if index < len(outputs) else None
        expected.append(float(want is not None and produced is not None
                              and produced.shape == want))
    shape_model_match = float(np.mean(expected)) if expected else 0.0
    extra = np.array([
        np.log1p(max(candidate.complexity, 0)),
        float(candidate.loo_pass),
        float(getattr(candidate, "loo_checked", False)),
        shape_equal, np.log1p(max(area_ratio, 0.0)), palette, changed, is_identity,
        float(identical_duplicates), float(len(all_candidates)),
        prior, float(candidate.family == "fallback"), shape_model_match,
    ], dtype=np.float32)
    return np.concatenate([task_features(task), extra, _one_hot(candidate.family)])


def prior_features(candidate, all_candidates: Sequence[Any],
                   family_prior: Dict[str, float],
                   detail_prior: Dict[str, float]) -> np.ndarray:
    """Prior-strength features computed from training-split success rates.

    Appended separately from :func:`candidate_features` so that dataset rows can
    be re-scored with priors fitted later without regenerating the dataset.
    """
    same = [other for other in all_candidates if other.family == candidate.family]
    rank = same.index(candidate) if candidate in same else 0
    return np.array([
        family_prior.get(candidate.family, 0.1),
        detail_prior.get(candidate.detail, family_prior.get(candidate.family, 0.1)),
        float(len(same)),
        float(rank) / max(len(same), 1),
    ], dtype=np.float32)


def add_prior_features(rows: List[Dict[str, Any]], family_prior: Dict[str, float],
                       detail_prior: Dict[str, float]) -> List[List[float]]:
    """Append prior features to stored dataset rows grouped by task."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    out: List[List[float]] = []
    for row in rows:
        group = grouped[row["task_id"]]
        same = [other for other in group if other["family"] == row["family"]]
        rank = same.index(row) if row in same else 0
        tail = [
            family_prior.get(row["family"], 0.1),
            detail_prior.get(row.get("detail", ""), family_prior.get(row["family"], 0.1)),
            float(len(same)),
            float(rank) / max(len(same), 1),
        ]
        out.append(list(row["features"]) + tail)
    return out


class ProgramRanker:
    """Logistic / gradient-boosted ranker over candidate programs."""

    def __init__(self, model: Optional[Any] = None,
                 method_prior: Optional[Dict[str, float]] = None,
                 checkpoint: Optional[str | Path] = None,
                 family_prior: Optional[Dict[str, float]] = None,
                 detail_prior: Optional[Dict[str, float]] = None) -> None:
        """Load a fitted model from ``checkpoint`` when given."""
        self.model = model
        self.method_prior = dict(method_prior or MANUAL_PRIOR)
        self.family_prior: Dict[str, float] = dict(family_prior or {})
        self.detail_prior: Dict[str, float] = dict(detail_prior or {})
        self.metrics: Dict[str, Any] = {}
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        if checkpoint is not None and Path(checkpoint).exists():
            self.load(checkpoint)

    # -- fit / predict -----------------------------------------------------

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int],
            kind: str = "logistic") -> Dict[str, Any]:
        """Fit the ranker; returns in-sample statistics only."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        x = np.asarray(rows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0)
        self._std[self._std < 1e-6] = 1.0
        xs = (x - self._mean) / self._std
        if kind == "boost":
            self.model = HistGradientBoostingClassifier(max_depth=3, max_iter=120,
                                                         learning_rate=0.08,
                                                         min_samples_leaf=20,
                                                         random_state=0)
        else:
            self.model = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        self.model.fit(xs, y)
        prob = self.predict_proba_rows(rows)
        self.metrics = {"rows": int(len(y)), "positives": int(y.sum()), "kind": kind,
                        "train_logloss": float(-np.mean(
                            y * np.log(np.clip(prob, 1e-6, 1)) +
                            (1 - y) * np.log(np.clip(1 - prob, 1e-6, 1))))}
        return self.metrics

    def predict_proba_rows(self, rows: Sequence[Sequence[float]]) -> np.ndarray:
        """P(correct) for raw feature rows."""
        x = np.asarray(rows, dtype=np.float32)
        if self._mean is not None:
            x = (x - self._mean) / self._std
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)

    def score(self, task, candidate, all_candidates: Sequence[Any]) -> float:
        """Score one candidate; a manual prior is used when no model is fitted."""
        if self.model is None:
            return self.method_prior.get(candidate.family, 0.1)
        row = self._row(task, candidate, all_candidates)
        return float(self.predict_proba_rows([row])[0])

    def _row(self, task, candidate, all_candidates: Sequence[Any]) -> np.ndarray:
        """Full feature row: gold-free features plus prior-strength features."""
        base = candidate_features(task, candidate, all_candidates, self.method_prior)
        tail = prior_features(candidate, all_candidates, self.family_prior, self.detail_prior)
        return np.concatenate([base, tail])

    def rank(self, task, candidates: Sequence[Any]) -> List[int]:
        """Return candidate indices sorted by decreasing score."""
        if self.model is None:
            return sorted(range(len(candidates)),
                          key=lambda i: (-self.family_prior.get(candidates[i].family,
                                                               self.method_prior.get(candidates[i].family, 0.1)),
                                         -self.detail_prior.get(candidates[i].detail, 0.0),
                                         candidates[i].complexity))
        rows = [self._row(task, c, candidates) for c in candidates]
        scores = self.predict_proba_rows(rows)
        return sorted(range(len(candidates)),
                      key=lambda i: (-float(scores[i]), candidates[i].complexity))

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the model and its standardisation statistics."""
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "mean": self._mean, "std": self._std,
                     "method_prior": self.method_prior, "metrics": self.metrics,
                     "family_prior": self.family_prior,
                     "detail_prior": self.detail_prior}, path)

    def load(self, path: str | Path) -> None:
        """Restore a persisted model."""
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self._mean = bundle["mean"]
        self._std = bundle["std"]
        self.method_prior = bundle.get("method_prior", self.method_prior)
        self.family_prior = bundle.get("family_prior", {})
        self.detail_prior = bundle.get("detail_prior", {})
        self.metrics = bundle.get("metrics", {})
