"""Object-centric segmentation and attribute extraction for ARC grids.

The v1 solver reasoned mostly about whole grids and individual cells.  This
module supplies the missing middle level: a grid is decomposed into connected
components, each carrying the attributes an ARC rule normally talks about
(size, shape up to D4, colour set, bounding box, border contact, holes,
symmetry, position rank).  Everything downstream -- the program inducer, the
beam search and the ranking model -- consumes :class:`ArcObject` lists.

All grids are ``int8`` numpy arrays.  Segmentation is a small BFS over at most
30x30 cells, so it is cheap enough to run inside a per-task search loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl

Grid = np.ndarray


# ---------------------------------------------------------------------------
# Object representation
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class ArcObject:
    """One connected component plus the attributes ARC rules refer to."""

    mask: np.ndarray
    box: Tuple[int, int, int, int]
    colors: Tuple[int, ...]
    size: int
    shape_key: bytes
    holes: int
    symmetric: str          # "", "h", "v", "hv", "d", "a", "r180"
    touches_border: bool
    fill_ratio: float
    centroid: Tuple[float, float]
    cells: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def height(self) -> int:
        """Bounding-box height in cells."""
        return self.box[2] - self.box[0]

    @property
    def width(self) -> int:
        """Bounding-box width in cells."""
        return self.box[3] - self.box[1]

    @property
    def color(self) -> int:
        """Single colour of the object, or -1 when it is multicoloured."""
        return self.colors[0] if len(self.colors) == 1 else -1

    @property
    def is_rect(self) -> bool:
        """True when the component fills its whole bounding box."""
        return self.size == self.height * self.width

    def signature(self) -> Tuple[bytes, Tuple[int, ...], int]:
        """Colour-plus-shape signature used to match objects across grids."""
        return (self.shape_key, self.colors, self.size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _d4_masks(mask: np.ndarray) -> List[np.ndarray]:
    """The eight D4 images of a boolean mask, cropped to their bounding box."""
    out: List[np.ndarray] = []
    base = np.ascontiguousarray(mask)
    variants = [base, np.rot90(base, 1), np.rot90(base, 2), np.rot90(base, 3),
                np.fliplr(base), np.flipud(base), base.T, np.rot90(base, 2).T]
    for variant in variants:
        box = dsl.bbox(variant)
        if box is None:
            continue
        out.append(np.ascontiguousarray(variant[box[0]:box[2], box[1]:box[3]]))
    return out


def shape_key(mask: np.ndarray) -> bytes:
    """Canonical byte key of a component's shape, invariant under D4."""
    variants = _d4_masks(mask)
    if not variants:
        return b""
    best = min(variants, key=lambda m: (m.shape, m.tobytes()))
    return bytes(best.shape) + best.tobytes()


def _symmetry(mask: np.ndarray) -> str:
    """Return the strongest symmetry label of a component mask."""
    if mask.size == 0:
        return ""
    h, w = mask.shape
    labels: List[str] = []
    if np.array_equal(mask, np.fliplr(mask)):
        labels.append("h")
    if np.array_equal(mask, np.flipud(mask)):
        labels.append("v")
    if h == w and np.array_equal(mask, mask.T):
        labels.append("d")
    if h == w and np.array_equal(mask, np.rot90(mask, 2).T):
        labels.append("a")
    if np.array_equal(mask, np.rot90(mask, 2)):
        labels.append("r180")
    if "h" in labels and "v" in labels:
        return "hv"
    return labels[0] if labels else ""


def _hole_count(mask: np.ndarray) -> int:
    """Number of background regions fully enclosed by the component mask."""
    box = dsl.bbox(mask)
    if box is None:
        return 0
    sub = mask[box[0]:box[2], box[1]:box[3]]
    h, w = sub.shape
    if h < 3 or w < 3:
        return 0
    free = ~sub
    seen = np.zeros_like(free)
    stack = [(r, c) for r in range(h) for c in (0, w - 1) if free[r, c]]
    stack += [(r, c) for c in range(w) for r in (0, h - 1) if free[r, c]]
    for r, c in stack:
        seen[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return int((free & ~seen).sum())


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment(grid: Grid, same_color: bool = True, connectivity: int = 4,
            background: Optional[int] = None, min_size: int = 1,
            max_objects: int = 60) -> List[ArcObject]:
    """Decompose ``grid`` into :class:`ArcObject` components, largest first."""
    grid = dsl.as_grid(grid)
    bg = dsl.background_color(grid) if background is None else background
    masks = dsl.components(grid, background=bg, connectivity=connectivity,
                           same_color=same_color)[:max_objects]
    objects: List[ArcObject] = []
    h, w = grid.shape
    for mask in masks:
        size = int(mask.sum())
        if size < min_size:
            continue
        box = dsl.bbox(mask)
        if box is None:
            continue
        values = np.unique(grid[mask])
        rows, cols = np.nonzero(mask)
        objects.append(ArcObject(
            mask=mask,
            box=box,
            colors=tuple(int(v) for v in values),
            size=size,
            shape_key=shape_key(mask),
            holes=_hole_count(mask),
            symmetric=_symmetry(mask),
            touches_border=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
            fill_ratio=size / max((box[2] - box[0]) * (box[3] - box[1]), 1),
            centroid=(float(rows.mean()), float(cols.mean())),
            cells=list(zip(rows.tolist(), cols.tolist())),
        ))
    return objects


def content_objects(grid: Grid, same_color: bool = True) -> List[ArcObject]:
    """Segment with the grid's own background as the excluded colour."""
    return segment(grid, same_color=same_color, background=dsl.background_color(grid))


# ---------------------------------------------------------------------------
# Selection predicates
# ---------------------------------------------------------------------------


def _rank_key(objects: Sequence[ArcObject], index: int, attr: str) -> float:
    """Scalar used for ordered selection (ties broken by index)."""
    obj = objects[index]
    if attr == "size":
        return obj.size
    if attr == "width":
        return obj.width
    if attr == "height":
        return obj.height
    if attr == "holes":
        return obj.holes
    if attr == "colors":
        return len(obj.colors)
    if attr == "row":
        return obj.centroid[0]
    if attr == "col":
        return obj.centroid[1]
    raise KeyError(attr)


def _shape_counts(objects: Sequence[ArcObject]) -> Dict[bytes, int]:
    """How often each canonical shape occurs in the object list."""
    counts: Dict[bytes, int] = {}
    for obj in objects:
        counts[obj.shape_key] = counts.get(obj.shape_key, 0) + 1
    return counts


def _color_counts(objects: Sequence[ArcObject]) -> Dict[int, int]:
    """How often each single colour occurs across objects."""
    counts: Dict[int, int] = {}
    for obj in objects:
        if obj.color >= 0:
            counts[obj.color] = counts.get(obj.color, 0) + 1
    return counts


def selection_predicates() -> Dict[str, Callable[[Sequence[ArcObject], Grid], Optional[int]]]:
    """Named deterministic object-selection rules.

    Each predicate takes the object list plus the source grid and returns the
    index of the chosen object.  They are the reusable vocabulary behind
    "output the odd one out", "output the largest shape", "output the unique
    colour" and similar ARC families.
    """
    def picker(attr: str, largest: bool):
        """Build an extremum picker over one scalar attribute."""
        def choose(objects: Sequence[ArcObject], _grid: Grid) -> Optional[int]:
            """Return the index maximising (or minimising) ``attr``."""
            if not objects:
                return None
            key = lambda i: (_rank_key(objects, i, attr), -i if largest else i)
            return max(range(len(objects)), key=key) if largest else min(range(len(objects)), key=key)
        choose.__name__ = f"{'max' if largest else 'min'}_{attr}"
        return choose

    def unique_by(fn: Callable[[ArcObject], object]):
        """Build a picker for the object whose key occurs exactly once."""
        def choose(objects: Sequence[ArcObject], _grid: Grid) -> Optional[int]:
            """Return the singleton key's object, or ``None`` if ambiguous."""
            keys = [fn(o) for o in objects]
            counts: Dict[object, int] = {}
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
            uniques = [i for i, key in enumerate(keys) if counts[key] == 1]
            return uniques[0] if len(uniques) == 1 else None
        return choose

    def holes_present(objects: Sequence[ArcObject], _grid: Grid) -> Optional[int]:
        """Return the single object that encloses a hole."""
        found = [i for i, o in enumerate(objects) if o.holes > 0]
        return found[0] if len(found) == 1 else None

    def odd_shape(objects: Sequence[ArcObject], _grid: Grid) -> Optional[int]:
        """Return the object whose canonical shape is the rarest."""
        counts = _shape_counts(objects)
        return unique_by(lambda o: o.shape_key)(objects, _grid) if len(counts) > 1 else None

    def odd_color(objects: Sequence[ArcObject], _grid: Grid) -> Optional[int]:
        """Return the object whose colour occurs once while others repeat."""
        counts = _color_counts(objects)
        return unique_by(lambda o: o.color)(objects, _grid) if len(counts) > 1 else None

    def odd_size(objects: Sequence[ArcObject], _grid: Grid) -> Optional[int]:
        """Return the object whose exact size occurs once."""
        return unique_by(lambda o: o.size)(objects, _grid)

    preds: Dict[str, Callable[[Sequence[ArcObject], Grid], Optional[int]]] = {
        "max_size": picker("size", True), "min_size": picker("size", False),
        "max_width": picker("width", True), "min_width": picker("width", False),
        "max_height": picker("height", True), "min_height": picker("height", False),
        "max_holes": picker("holes", True), "max_colors": picker("colors", True),
        "min_colors": picker("colors", False),
        "top": picker("row", False), "bottom": picker("row", True),
        "left": picker("col", False), "right": picker("col", True),
        "has_holes": holes_present, "odd_shape": odd_shape,
        "odd_color": odd_color, "odd_size": odd_size,
    }
    preds["symmetric"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.symmetric), None)
    preds["asymmetric"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if not o.symmetric), None)
    preds["interior"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if not o.touches_border), None)
    preds["border"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.touches_border), None)
    preds["square"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.height == o.width), None)
    preds["non_square"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.height != o.width), None)
    preds["rect"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.is_rect), None)
    preds["singleton"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.size == 1), None)
    preds["small"] = lambda objects, _g: min(range(len(objects)),
                                             key=lambda i: objects[i].size) if objects else None
    preds["wide"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.width > o.height), None)
    preds["tall"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.height > o.width), None)
    preds["hollow"] = lambda objects, _g: next(
        (i for i, o in enumerate(objects) if o.holes > 0 and o.fill_ratio < 0.9), None)
    preds["common_shape"] = lambda objects, _g: (
        max(range(len(objects)), key=lambda i: sum(
            1 for o in objects if o.shape_key == objects[i].shape_key))
        if objects else None)
    return preds


PREDICATES = selection_predicates()


# ---------------------------------------------------------------------------
# Cropping / rendering
# ---------------------------------------------------------------------------


def crop_mask(grid: Grid, obj: ArcObject, background: Optional[int] = None) -> Grid:
    """Crop ``grid`` to the object's bounding box, blanking cells outside it."""
    bg = dsl.background_color(grid) if background is None else background
    r0, c0, r1, c1 = obj.box
    out = np.full((r1 - r0, c1 - c0), bg, dtype=np.int8)
    sub = obj.mask[r0:r1, c0:c1]
    out[sub] = grid[r0:r1, c0:c1][sub]
    return out


def crop_bbox(grid: Grid, obj: ArcObject) -> Grid:
    """Crop ``grid`` to the object's bounding box, keeping overlapped cells."""
    r0, c0, r1, c1 = obj.box
    return np.ascontiguousarray(grid[r0:r1, c0:c1])


def render_objects(grid: Grid, objects: Sequence[ArcObject],
                   colors: Optional[Sequence[int]] = None,
                   keep: Optional[Sequence[int]] = None) -> Grid:
    """Paint ``objects`` back onto a background grid, optionally recoloured."""
    out = np.full(grid.shape, dsl.background_color(grid), dtype=np.int8)
    for index, obj in enumerate(objects):
        if keep is not None and index not in keep:
            continue
        color = int(colors[index]) if colors is not None else -1
        if color < 0:
            out[obj.mask] = grid[obj.mask]
        else:
            out[obj.mask] = color
    return out


def shift_object(out: Grid, obj: ArcObject, dr: int, dc: int, background: int) -> None:
    """Translate one object in place by ``(dr, dc)``, clipped to the grid."""
    h, w = out.shape
    values = np.full(obj.size, 0, dtype=np.int8)
    for n, (r, c) in enumerate(obj.cells):
        values[n] = out[r, c]
    out[obj.mask] = background
    for (r, c), value in zip(obj.cells, values):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = value


def bbox_of(predicate_name: str, grid: Grid) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the object chosen by a named predicate."""
    objects = content_objects(grid)
    index = PREDICATES[predicate_name](objects, grid)
    return None if index is None else objects[index].box
