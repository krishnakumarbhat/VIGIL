"""Exact local-pattern lookup with abstention.

Cells are predicted from their 5x5 neighbourhood, falling back to the 3x3
neighbourhood and finally to the input colour when a patch was never seen in
training.  Acceptance requires patch consistency across every training pair and
real compression (fewer unique patches than training cells).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl

Grid = np.ndarray
Pair = Tuple[Grid, Grid]


class PatchLookupLearner:
    """Translation-invariant local-rule learner with abstention.

    A cell's output colour is looked up from its exact 5x5 neighbourhood; the
    rule is accepted only when every training patch maps to a single colour.
    At prediction time unseen patches fall back to the 3x3 neighbourhood and
    finally to the input colour, which makes the model behave like a local
    cellular-automaton rule rather than a global memoriser.
    """

    def __init__(self) -> None:
        """Initialise an empty lookup table holder."""

    @staticmethod
    def _padded(grid: Grid, radius: int) -> np.ndarray:
        """Pad the grid with an out-of-palette sentinel (10)."""
        return np.pad(grid, radius, mode="constant", constant_values=10)

    def _extract(self, grid: Grid, radius: int) -> List[bytes]:
        """Return the window bytes of every cell at the given radius."""
        padded = self._padded(grid, radius)
        h, w = grid.shape
        width = 2 * radius + 1
        keys: List[bytes] = []
        for r in range(h):
            for c in range(w):
                window = padded[r:r + width, c:c + width]
                keys.append(np.ascontiguousarray(window, dtype=np.int8).tobytes())
        return keys

    def fit(self, pairs: Sequence[Pair]) -> Optional[Dict[str, Any]]:
        """Build radius-2 and radius-1 tables; reject conflicting patches."""
        tables: Dict[int, Dict[bytes, int]] = {}
        totals = 0
        for radius in (2, 1):
            table: Dict[bytes, int] = {}
            consistent = True
            for src, dst in pairs:
                if src.shape != dst.shape:
                    return None
                for key, value in zip(self._extract(src, radius), dst.ravel().tolist()):
                    if key in table and table[key] != value:
                        consistent = False
                        break
                    table[key] = value
                if not consistent:
                    break
            if consistent and table:
                tables[radius] = table
                totals = max(totals, sum(src.size for src, _ in pairs))
        if 2 not in tables:
            return None
        # Require real compression: a table with one entry per training cell is
        # memorisation and will not transfer to the test grid.
        if totals and len(tables[2]) > 0.8 * totals:
            return None
        return {"tables": tables, "radius": 2}

    def predict(self, model: Dict[str, Any], grid: Grid) -> Optional[Grid]:
        """Predict with abstention: r=2 patch, then r=1 patch, then input colour."""
        out = dsl.as_grid(grid)
        for radius in (2, 1):
            table = model["tables"].get(radius)
            if not table:
                continue
            keys = self._extract(grid, radius)
            for index, key in enumerate(keys):
                value = table.get(key)
                if value is not None:
                    out.flat[index] = value
        return out
