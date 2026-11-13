"""Tiny torch MLP over cell features (test-time adaptation baseline).

Shares the feature extractor of :class:`src.cell_rule_learner.CellRuleLearner`
and is accepted only when it reproduces every training cell exactly.  It is the
"neuro" component of the cascade; the decision tree usually wins on small data,
but the MLP occasionally captures a smooth rule the tree splits badly.

An experimental CNN variant is documented in :mod:`src.cnn_cell_learner`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.cell_rule_learner import CellRuleLearner

Grid = np.ndarray
Pair = Tuple[Grid, Grid]


class MlpCellLearner:
    """Tiny torch MLP over cell features; the 'neuro' half of the solver."""

    def __init__(self, max_seconds: float = 3.0) -> None:
        """Store the wall-clock cap for training."""
        self.max_seconds = max_seconds

    def fit(self, pairs: Sequence[Pair]) -> Optional[Dict[str, Any]]:
        """Train an MLP to reproduce every training cell; require 100% accuracy."""
        try:
            import torch
            from torch import nn
        except Exception:
            return None
        learner = CellRuleLearner()
        xs, ys = [], []
        for src, dst in pairs:
            if src.shape != dst.shape:
                return None
            xs.append(learner._features(src))
            ys.append(dst.ravel().astype(np.int64))
        x = torch.from_numpy(np.concatenate(xs, axis=0))
        y = torch.from_numpy(np.concatenate(ys, axis=0))
        if x.shape[0] < 16 or int(torch.unique(y).numel()) < 2:
            return None
        model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(),
                              nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        loss_fn = nn.CrossEntropyLoss()
        start = time.monotonic()
        model.train()
        for _ in range(400):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            if time.monotonic() - start > self.max_seconds:
                break
            if float(loss.detach()) < 1e-4:
                break
        with torch.no_grad():
            accuracy = float((model(x).argmax(dim=1) == y).float().mean())
        if accuracy < 1.0:
            return None
        return {"model": model, "learner": learner}

    def predict(self, bundle: Dict[str, Any], grid: Grid) -> Optional[Grid]:
        """Predict a grid with the trained MLP."""
        import torch

        learner: CellRuleLearner = bundle["learner"]
        with torch.no_grad():
            logits = bundle["model"](torch.from_numpy(learner._features(grid)))
        return dsl.as_grid(logits.argmax(dim=1).numpy().reshape(grid.shape))
