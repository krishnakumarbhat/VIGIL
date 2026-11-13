"""D4 and color-permutation canonicalisation utilities.

Every ARC grid is an ``int8`` numpy array of values ``0..9``.  Two grids that
are equal up to a dihedral symmetry of the square or a renaming of colours are
the *same* hypothesis state; collapsing them shrinks the search space by up to
16x (8 D4 elements x colour relabelling).  All functions here are pure and
return new arrays; callers must never mutate the result of a canonicaliser.
"""

from __future__ import annotations

from typing import Callable, Dict, Final, List, Tuple

import numpy as np

Grid = np.ndarray

# ---------------------------------------------------------------------------
# Dihedral group D4
# ---------------------------------------------------------------------------


def identity(grid: Grid) -> Grid:
    """Return the grid unchanged (already a contiguous int8 view copy)."""
    return np.ascontiguousarray(grid)


def rot90(grid: Grid) -> Grid:
    """Rotate the grid 90 degrees counter-clockwise."""
    return np.ascontiguousarray(np.rot90(grid, 1))


def rot180(grid: Grid) -> Grid:
    """Rotate the grid 180 degrees."""
    return np.ascontiguousarray(np.rot90(grid, 2))


def rot270(grid: Grid) -> Grid:
    """Rotate the grid 270 degrees counter-clockwise."""
    return np.ascontiguousarray(np.rot90(grid, 3))


def fliplr(grid: Grid) -> Grid:
    """Mirror the grid left-to-right."""
    return np.ascontiguousarray(np.fliplr(grid))


def flipud(grid: Grid) -> Grid:
    """Mirror the grid top-to-bottom."""
    return np.ascontiguousarray(np.flipud(grid))


def transpose(grid: Grid) -> Grid:
    """Transpose the grid across the main diagonal."""
    return np.ascontiguousarray(grid.T)


def antidiag(grid: Grid) -> Grid:
    """Transpose the grid across the anti-diagonal."""
    return np.ascontiguousarray(np.rot90(grid, 2).T)


D4: Final[Tuple[Tuple[str, Callable[[Grid], Grid]], ...]] = (
    ("id", identity),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("fliplr", fliplr),
    ("flipud", flipud),
    ("transpose", transpose),
    ("antidiag", antidiag),
)

D4_NAMES: Final[Tuple[str, ...]] = tuple(name for name, _ in D4)


def d4_variants(grid: Grid) -> List[Grid]:
    """Return all eight D4 images of ``grid`` (may contain duplicates)."""
    return [fn(grid) for _, fn in D4]


# ---------------------------------------------------------------------------
# Colour canonicalisation
# ---------------------------------------------------------------------------


def canonical_colors(grid: Grid) -> Grid:
    """Relabel colours by order of first appearance (row-major scan).

    The resulting grid starts at colour 0 and is invariant to any bijective
    renaming of the input palette, so it can be hashed for deduplication.
    """
    flat = np.ascontiguousarray(grid, dtype=np.int8).ravel()
    mapping = np.full(10, -1, dtype=np.int8)
    nxt = 0
    for value in flat.tolist():
        if mapping[value] < 0:
            mapping[value] = nxt
            nxt += 1
    return mapping[flat].reshape(grid.shape).astype(np.int8)


def color_signature(grid: Grid) -> Tuple[int, ...]:
    """Return a permutation-invariant signature of the colour multiset."""
    counts = np.bincount(np.asarray(grid, dtype=np.int8).ravel(), minlength=10)
    return tuple(sorted(int(c) for c in counts if c > 0))


def canonical_key(grid: Grid) -> bytes:
    """Hashable key: min over D4 of the colour-canonicalised grid."""
    colored = canonical_colors(grid)
    best: bytes | None = None
    for _, fn in D4:
        raw = np.ascontiguousarray(fn(colored), dtype=np.int8).tobytes()
        if best is None or raw < best:
            best = raw
    assert best is not None
    return best


def canonical_shape_key(grid: Grid) -> Tuple[int, int]:
    """Sorted (rows, cols) so transposed states deduplicate cheaply."""
    h, w = grid.shape
    return (min(h, w), max(h, w))


def canonical_pair_key(pair: Tuple[Grid, Grid]) -> bytes:
    """Canonical key for an (input, output) pair under simultaneous D4 action."""
    a, b = pair
    best: bytes | None = None
    for name, fn in D4:
        ra, rb = fn(a), fn(b)
        key = canonical_key(ra) + b"|" + canonical_key(rb)
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def orbit_size(grid: Grid) -> int:
    """Number of distinct D4 images of the grid (1, 2, 4 or 8)."""
    return len({fn(grid).tobytes() + bytes(grid.shape) for _, fn in D4})


def dedupe_by_orbit(grids: List[Grid]) -> List[Grid]:
    """Keep one representative per D4 orbit, preserving input order."""
    seen: set[bytes] = set()
    kept: List[Grid] = []
    for grid in grids:
        key = canonical_key(grid)
        if key in seen:
            continue
        seen.add(key)
        kept.append(grid)
    return kept


def shape_relation(inputs: List[Grid], outputs: List[Grid]) -> str:
    """Classify the shape mapping of a task as same, up, down or mixed."""
    flags: List[str] = []
    for a, b in zip(inputs, outputs):
        area_a, area_b = a.size, b.size
        if a.shape == b.shape:
            flags.append("same")
        elif area_b > area_a:
            flags.append("up")
        elif area_b < area_a:
            flags.append("down")
        else:
            flags.append("reshaped")
    uniq = set(flags)
    if len(uniq) == 1:
        return uniq.pop()
    return "mixed"


D4_SHAPE_MAP: Dict[str, Tuple[int, int]] = {
    name: (0 if name in ("id", "rot90", "rot180", "rot270") else 1, 0) for name, _ in D4
}


def apply_by_name(name: str, grid: Grid) -> Grid:
    """Apply the D4 element called ``name`` to ``grid``."""
    for candidate, fn in D4:
        if candidate == name:
            return fn(grid)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Inverse D4 action (used to undo test-time augmentation)
# ---------------------------------------------------------------------------

D4_INVERSE: Final[Dict[str, Callable[[Grid], Grid]]] = {
    "id": lambda g: np.array(g, dtype=np.int8, copy=True),
    "rot90": lambda g: np.ascontiguousarray(np.rot90(g, 3), dtype=np.int8),
    "rot180": lambda g: np.ascontiguousarray(np.rot90(g, 2), dtype=np.int8),
    "rot270": lambda g: np.ascontiguousarray(np.rot90(g, 1), dtype=np.int8),
    "fliplr": lambda g: np.ascontiguousarray(np.fliplr(g), dtype=np.int8),
    "flipud": lambda g: np.ascontiguousarray(np.flipud(g), dtype=np.int8),
    "transpose": lambda g: np.ascontiguousarray(g.T, dtype=np.int8),
    "antidiag": lambda g: np.ascontiguousarray(np.rot90(g, 2).T, dtype=np.int8),
}
