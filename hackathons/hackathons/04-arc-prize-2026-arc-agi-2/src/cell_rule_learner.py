"""Decision-tree learner over per-cell neighbourhood features.

The tree predicts the output colour of each cell from its colour, normalised
position, border flag, 3x3 neighbourhood and D4 symmetry partners.  Optional D4
augmentation at fit time and D4 voting at decode time turn the rule into a
symmetry-equivariant one, which is a strong regulariser for ARC grids.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.canonical import D4, D4_INVERSE

Grid = np.ndarray
Pair = Tuple[Grid, Grid]


class CellRuleLearner:
    """Per-cell classifier over colour + 3x3 neighbourhood + position features."""

    def __init__(self, max_depth: int = 10) -> None:
        """Store the tree depth cap; deeper trees are treated as memorisation."""
        self.max_depth = max_depth

    @staticmethod
    def _features(grid: Grid) -> np.ndarray:
        """Build the per-cell feature matrix (colour, position, 3x3 patch, mirrors).

        Features, in order: 10 colour one-hots, normalised row/col, border flag,
        90 neighbourhood one-hots (3x3 x 10), and 40 symmetry-partner one-hots
        (left-right, top-bottom, 180-degree partners as extra neighbourhood
        colours).  The partner block lets shallow trees express symmetric
        repair rules without needing a deep spatial split.
        """
        h, w = grid.shape
        colors = np.eye(10, dtype=np.float32)[grid.ravel()]
        rr, cc = np.mgrid[0:h, 0:w]
        pos = np.stack([rr.ravel() / max(h - 1, 1), cc.ravel() / max(w - 1, 1)], axis=1).astype(np.float32)
        padded = np.full((h + 2, w + 2), 255, dtype=np.int16)
        padded[1:-1, 1:-1] = grid
        patches = np.zeros((h * w, 9, 10), dtype=np.float32)
        for dr in range(3):
            for dc in range(3):
                window = padded[dr:dr + h, dc:dc + w].ravel().astype(np.int16)
                valid = window < 10
                patches[valid, dr * 3 + dc, window[valid]] = 1.0
        patches = patches.reshape(h * w, -1)
        border = ((rr.ravel() == 0) | (cc.ravel() == 0) | (rr.ravel() == h - 1) |
                  (cc.ravel() == w - 1)).astype(np.float32)[:, None]
        partners = np.floor(
            np.stack([
                np.eye(11, dtype=np.float32)[np.fliplr(grid).ravel()],
                np.eye(11, dtype=np.float32)[np.flipud(grid).ravel()],
                np.eye(11, dtype=np.float32)[np.rot90(grid, 2).ravel()],
                np.eye(11, dtype=np.float32)[grid.T.ravel()] if h == w else np.zeros((h * w, 11), dtype=np.float32),
            ], axis=1).reshape(h * w, -1)[:, :40], dtype=np.float32)
        return np.concatenate([colors, pos, border, patches, partners], axis=1)

    def fit(self, pairs: Sequence[Pair], augment: bool = True) -> Optional[Any]:
        """Fit a decision tree; return ``None`` unless it fits all cells exactly.

        With ``augment`` the training cells are expanded over the eight D4
        images of every pair.  This bakes rotation/reflection equivariance into
        the learned rule, which is the single strongest regulariser for ARC
        style grids, and costs only 8x a very small feature matrix.
        """
        from sklearn.tree import DecisionTreeClassifier

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        views = [fn for _, fn in D4] if augment else [dsl.as_grid]
        for src, dst in pairs:
            if src.shape != dst.shape:
                return None
            for view in views:
                xs.append(self._features(view(src)))
            for view in views:
                ys.append(view(dst).ravel().astype(np.int64))
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        if x.shape[0] < 8 or len(np.unique(y)) < 2:
            return None
        tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=0)
        tree.fit(x, y)
        if float(tree.score(x, y)) < 1.0:
            return None
        # Reject trees with more leaves than half the cells: pure memorisation.
        if int(tree.get_n_leaves()) > max(8, x.shape[0] // 2):
            return None
        return tree

    def predict_single(self, tree: Any, grid: Grid) -> Optional[Grid]:
        """Apply the tree to one grid without test-time augmentation."""
        pred = tree.predict(self._features(grid))
        return dsl.as_grid(pred.reshape(grid.shape))

    def predict(self, tree: Any, grid: Grid, vote: bool = True) -> Optional[Grid]:
        """Apply the tree with D4 test-time augmentation and majority voting."""
        if not vote:
            return self.predict_single(tree, grid)
        counts = np.zeros(grid.shape + (10,), dtype=np.int16)
        for name, view in D4:
            pred = self.predict_single(tree, view(grid))
            if pred is None or pred.shape != grid.shape:
                return None
            back = D4_INVERSE[name](pred)
            if back.shape != grid.shape:
                return None
            np.add.at(counts, (np.arange(grid.shape[0])[:, None],
                               np.arange(grid.shape[1])[None, :], back.astype(np.int64)), 1)
        return dsl.as_grid(counts.argmax(axis=-1))
