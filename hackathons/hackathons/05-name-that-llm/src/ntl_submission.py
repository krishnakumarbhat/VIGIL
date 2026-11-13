"""Picklable inference artefact and submission writer for Track 05.

``EnsembleModel`` bundles the fitted feature stack, one or more linear members
with blend weights, and the optional class-prior correction.  The notebook only
needs ``joblib.load(...)`` followed by ``predict_proba(texts)``.

``SubmissionWriter`` enforces the competition schema: columns ``id,model``, one
row per test id, no nulls, every value in the 18 official label strings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ntl_data import LABELS
from ntl_features import FeatureStack


class EnsembleModel:
    """Weighted blend of fitted members + prior correction, ready for inference."""

    def __init__(self, stack: FeatureStack, members: Sequence[tuple[str, object, float]],
                 prior_alpha: float = 0.0, prior: np.ndarray | None = None) -> None:
        """Store the fitted stack and ``(name, member, weight)`` triples."""
        self.stack = stack
        self.members = list(members)
        self.prior_alpha = float(prior_alpha)
        self.prior = prior
        self.labels = list(LABELS)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 4096) -> np.ndarray:
        """Blend member probabilities for every text (batched to bound memory)."""
        texts = [str(t) for t in texts]
        total_w = sum(w for _, _, w in self.members if w)
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            X = self.stack.transform(chunk)
            acc = np.zeros((len(chunk), len(self.labels)), dtype=np.float32)
            for _, member, w in self.members:
                if w:
                    acc += (w / total_w) * member.proba(X)
            if self.prior_alpha and self.prior is not None:
                acc = acc / self.prior[None, :] ** self.prior_alpha
            out[start:start + batch_size] = acc
        return out

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return the arg-max label string per text."""
        idx = self.predict_proba(texts).argmax(1)
        return np.array([self.labels[i] for i in idx])


class SubmissionWriter:
    """Validates and writes the ``id,model`` parquet for any test frame."""

    def __init__(self, labels: Sequence[str] = LABELS) -> None:
        self.labels = list(labels)

    def build(self, test: pd.DataFrame, model: EnsembleModel) -> pd.DataFrame:
        """Score ``test`` (columns ``id,text``) and return a validated frame."""
        if not {"id", "text"}.issubset(test.columns):
            raise ValueError(f"test frame needs id,text; got {list(test.columns)}")
        test = test.dropna(subset=["id"]).copy()
        proba = model.predict_proba(test["text"].fillna("").astype(str).tolist())
        sub = pd.DataFrame({
            "id": test["id"].astype(str).to_numpy(),
            "model": [self.labels[i] for i in proba.argmax(1)],
        })
        self.validate(sub, expected_rows=len(test))
        return sub

    def validate(self, sub: pd.DataFrame, expected_rows: int | None = None) -> None:
        """Assert the competition schema; raise ``ValueError`` on any violation."""
        problems: list[str] = []
        if list(sub.columns) != ["id", "model"]:
            problems.append(f"columns are {list(sub.columns)}, expected ['id', 'model']")
        if expected_rows is not None and len(sub) != expected_rows:
            problems.append(f"{len(sub)} rows, expected {expected_rows}")
        if sub["id"].isna().any() or sub["model"].isna().any():
            problems.append("null id or model value")
        if sub["id"].duplicated().any():
            problems.append("duplicated id values")
        unknown = set(sub["model"].unique()) - set(self.labels)
        if unknown:
            problems.append(f"unknown labels {sorted(unknown)}")
        if problems:
            raise ValueError("submission invalid: " + "; ".join(problems))

    def write(self, sub: pd.DataFrame, path: Path) -> Path:
        """Validate then write the parquet, returning the path."""
        self.validate(sub)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(path, index=False)
        return Path(path)
