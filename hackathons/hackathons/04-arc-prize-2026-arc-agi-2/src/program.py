"""Shared program representation and tiny grid-mapping helpers.

Kept separate from the family implementations so that several modules can
construct :class:`Program` objects without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Grid = np.ndarray
Pair = Tuple[Grid, Grid]
ProgramFn = Callable[[Grid], Optional[Grid]]


@dataclass(eq=False)
class Program:
    """A fitted program plus the evidence collected while fitting it."""

    family: str
    fn: ProgramFn
    detail: str
    complexity: int = 1
    loo_pass: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def apply(self, grid: Grid) -> Optional[Grid]:
        """Run the program on one grid and sanitise the output."""
        try:
            out = self.fn(grid)
        except Exception:
            return None
        if out is None:
            return None
        out = np.asarray(out)
        if out.ndim != 2 or out.size == 0:
            return None
        return np.ascontiguousarray(np.clip(out, 0, 9).astype(np.int8))


def grids_equal(a: Optional[Grid], b: Grid) -> bool:
    """Exact grid equality with shape check."""
    return a is not None and a.shape == b.shape and np.array_equal(a, b)


def fit_cell_map(pairs: Sequence[Pair]) -> Optional[Dict[int, int]]:
    """Learn a consistent colour→colour map between equal-shaped grid pairs."""
    mapping: Dict[int, int] = {}
    for src, dst in pairs:
        if src.shape != dst.shape:
            return None
        for a, b in zip(src.ravel().tolist(), dst.ravel().tolist()):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping


def apply_map(grid: Grid, mapping: Dict[int, int]) -> Optional[Grid]:
    """Apply a colour map, returning ``None`` when the grid has unseen colours."""
    out = np.empty_like(grid)
    for src, dst in mapping.items():
        out[grid == src] = dst
    for value in np.unique(grid).tolist():
        if int(value) not in mapping:
            return None
    return out
