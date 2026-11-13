"""Decision-tree learner over connected-component properties.

Predicts the output colour of each component from size, bounding box, border
contact, palette width and size rank.  This is the symbolic analogue of
"recolour every object by a property" and generalises from very few pairs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.canonical import D4, D4_INVERSE

Grid = np.ndarray
Pair = Tuple[Grid, Grid]


class ObjectRuleLearner:
    """Per-component classifier: predicts the output colour of each object."""

    def __init__(self) -> None:
        """Initialise an empty feature cache (features are computed on demand)."""

    @staticmethod
    def _object_table(grid: Grid, connectivity: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Return component masks and an (n_obj, 23) feature matrix.

        Features: relative size, normalised bbox extent and centre, border
        contact, palette width, size rank (ascending and descending), the
        largest/smallest flags and the 10-bin colour histogram.  Rank features
        are what make "recolour by size order" tasks learnable from two pairs.
        """
        masks = dsl.components(grid, connectivity=connectivity, same_color=False)
        h, w = grid.shape
        rows: List[np.ndarray] = []
        total = max(sum(int(m.sum()) for m in masks), 1)
        sizes = [float(m.sum()) for m in masks]
        order = sorted(range(len(masks)), key=lambda i: sizes[i])
        rank_asc = {idx: rank / max(len(masks) - 1, 1) for rank, idx in enumerate(order)}
        rank_desc = {idx: 1.0 - rank_asc[idx] for idx in rank_asc}
        n_objects = float(len(masks))
        for pos, mask in enumerate(masks):
            box = dsl.bbox(mask)
            assert box is not None
            r0, c0, r1, c1 = box
            values = grid[mask]
            hist = np.bincount(values, minlength=10).astype(np.float32)
            size = float(mask.sum())
            features = np.concatenate([
                np.array([size / total, (r1 - r0) / h, (c1 - c0) / w,
                          (r0 + r1) / (2 * h), (c0 + c1) / (2 * w),
                          float(int(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())),
                          float(len(np.unique(values))), rank_asc[pos], rank_desc[pos],
                          float(pos == 0), float(pos == len(masks) - 1),
                          size / max(max(sizes), 1.0), 1.0 / n_objects], dtype=np.float32),
                hist / max(size, 1.0),
            ])
            rows.append(features)
        return masks, (np.stack(rows) if rows else np.zeros((0, 23), dtype=np.float32))

    def fit(self, pairs: Sequence[Pair], connectivity: int = 4,
            augment: bool = True) -> Optional[Dict[str, Any]]:
        """Fit the component classifier; requires near-uniform target colours.

        D4 augmentation turns one two-pair task into sixteen observed object
        instances, which is what makes rank/shape rules identifiable instead of
        memorised.
        """
        from sklearn.tree import DecisionTreeClassifier

        xs: List[np.ndarray] = []
        ys: List[int] = []
        views = [fn for _, fn in D4] if augment else [dsl.as_grid]
        for src, dst in pairs:
            if src.shape != dst.shape:
                return None
            for view in views:
                masks, features = self._object_table(view(src), connectivity)
                if not masks:
                    return None
                transformed = view(dst)
                for mask, feature in zip(masks, features):
                    targets = transformed[mask]
                    values, counts = np.unique(targets, return_counts=True)
                    if counts.max() / counts.sum() < 0.95:
                        return None
                    xs.append(feature)
                    ys.append(int(values[counts.argmax()]))
        if len(xs) < 4 or len(set(ys)) < 2:
            return None
        x, y = np.stack(xs), np.asarray(ys)
        tree = DecisionTreeClassifier(max_depth=6, random_state=0)
        tree.fit(x, y)
        if float(tree.score(x, y)) < 1.0 or int(tree.get_n_leaves()) > max(4, len(xs) // 3):
            return None
        return {"tree": tree, "connectivity": connectivity}

    def predict_single(self, model: Dict[str, Any], grid: Grid) -> Optional[Grid]:
        """Recolour components in one grid without augmentation."""
        masks, features = self._object_table(grid, int(model["connectivity"]))
        if not masks:
            return None
        colors = model["tree"].predict(features)
        out = dsl.as_grid(grid)
        for mask, color in zip(masks, colors):
            out[mask] = int(color)
        return out

    def predict(self, model: Dict[str, Any], grid: Grid, vote: bool = True) -> Optional[Grid]:
        """Recolour components with D4 test-time augmentation and voting."""
        if not vote:
            return self.predict_single(model, grid)
        counts = np.zeros(grid.shape + (10,), dtype=np.int16)
        for name, view in D4:
            pred = self.predict_single(model, view(grid))
            if pred is None or pred.shape != grid.shape:
                return None
            back = D4_INVERSE[name](pred)
            if back.shape != grid.shape:
                return None
            np.add.at(counts, (np.arange(grid.shape[0])[:, None],
                               np.arange(grid.shape[1])[None, :], back.astype(np.int64)), 1)
        return dsl.as_grid(counts.argmax(axis=-1))
