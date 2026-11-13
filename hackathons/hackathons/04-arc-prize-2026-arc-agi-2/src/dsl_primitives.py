"""Grid primitive DSL for ARC-AGI-2.

The module exposes parameter-free and parametrised grid transformations over
``int8`` numpy arrays.  Every primitive is a pure callable ``Grid -> Grid|None``
returning ``None`` when it is undefined for the given input (e.g. an object
crop on a grid without objects).  ``enumerate_primitives`` builds the candidate
set from the training pairs, so data-dependent operators such as a learned
colour bijection or the panel layout of the task are included.

Design notes
------------
* All hot loops operate on whole arrays with numpy; Python loops are limited to
  grids of at most a few hundred cells (ARC max is 30x30).
* Connected components use an iterative raster flood fill (4- or 8-connected),
  which is exact and allocation-light for this size.
* Primitives never copy the input unless they must; ``np.ascontiguousarray``
  keeps the dtype at ``int8`` so later equality checks stay cheap.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Grid = np.ndarray
Primitive = Callable[[Grid], Optional[Grid]]

BACKGROUNDS: Tuple[int, ...] = tuple(range(10))


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def as_grid(array: np.ndarray) -> Grid:
    """Return a contiguous ``int8`` *copy* of ``array`` shaped as a 2-D grid.

    The copy is deliberate: several primitives write into their working array,
    and ARC training grids are at most 900 bytes, so the copy is a single cheap
    C-level memcpy that keeps every operator pure.
    """
    out = np.array(array, dtype=np.int8, order="C", copy=True)
    if out.ndim != 2:
        raise ValueError(f"grid must be 2-D, got shape {out.shape}")
    return out


def background_color(grid: Grid) -> int:
    """Most frequent colour of the grid (ties broken by lowest colour id)."""
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts.argmax())


def color_counts(grid: Grid) -> np.ndarray:
    """Return the 10-bin colour histogram of the grid."""
    return np.bincount(grid.ravel(), minlength=10)


def bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box ``(r0, c0, r1, c1)`` of a boolean mask, or ``None``."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def crop(grid: Grid, box: Tuple[int, int, int, int]) -> Grid:
    """Crop ``grid`` to the half-open box ``(r0, c0, r1, c1)``."""
    r0, c0, r1, c1 = box
    return as_grid(grid[r0:r1, c0:c1])


def trim_border(grid: Grid) -> Grid:
    """Remove rows/columns of a uniform outer colour (up to two layers)."""
    out = as_grid(grid)
    for _ in range(2):
        if out.shape[0] < 3 or out.shape[1] < 3:
            break
        if np.all(out[0] == out[0, 0]) and np.all(out[-1] == out[-1, 0]) and np.all(out[:, 0] == out[0, 0]) and np.all(out[:, -1] == out[0, 0]):
            out = as_grid(out[1:-1, 1:-1])
        else:
            break
    return out


def components(grid: Grid, background: Optional[int] = None, connectivity: int = 4,
               same_color: bool = True) -> List[np.ndarray]:
    """Return boolean masks of connected components.

    ``same_color`` restricts a component to cells sharing one colour; when it is
    false a component is any connected region of non-background cells.
    """
    bg = background_color(grid) if background is None else background
    h, w = grid.shape
    foreground = grid != bg if bg >= 0 else np.ones_like(grid, dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    masks: List[np.ndarray] = []
    if connectivity == 8:
        neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    values = grid.tolist()
    fg = foreground.tolist()
    for r in range(h):
        for c in range(w):
            if not fg[r][c] or visited[r, c]:
                continue
            target = values[r][c]
            mask = np.zeros((h, w), dtype=bool)
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                mask[cr, cc] = True
                for dr, dc in neighbours:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and fg[nr][nc]:
                        if same_color and values[nr][nc] != target:
                            continue
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            masks.append(mask)
    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    return masks


def flood_from(grid: Grid, seeds: Sequence[Tuple[int, int]], target: int, color: int,
               connectivity: int = 4) -> Grid:
    """Flood fill ``target``-coloured cells reachable from ``seeds`` with ``color``."""
    out = as_grid(grid)
    h, w = out.shape
    if connectivity == 8:
        neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    seen = np.zeros((h, w), dtype=bool)
    stack = [s for s in seeds if 0 <= s[0] < h and 0 <= s[1] < w and out[s] == target]
    for r, c in stack:
        seen[r, c] = True
    while stack:
        r, c = stack.pop()
        out[r, c] = color
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and out[nr, nc] == target:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return out


def enclosed_mask(grid: Grid, background: Optional[int] = None) -> np.ndarray:
    """Boolean mask of background cells not reachable from the grid border."""
    bg = background_color(grid) if background is None else background
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    if bg < 0:
        return outside
    stack = [(r, c) for r in range(h) for c in (0, w - 1) if grid[r, c] == bg]
    stack += [(r, c) for c in range(w) for r in (0, h - 1) if grid[r, c] == bg]
    for r, c in stack:
        outside[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not outside[nr, nc] and grid[nr, nc] == bg:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return (grid == bg) & ~outside


def gravity(grid: Grid, axis: int, positive: bool) -> Grid:
    """Slide every non-background cell along ``axis`` until it hits an edge."""
    out = as_grid(grid)
    bg = background_color(grid)
    if axis == 0:
        out = as_grid(out.T)
    h, w = out.shape
    for col in range(w):
        vals = [v for v in out[:, col].tolist() if v != bg]
        if positive:
            fill = [bg] * (h - len(vals)) + vals
        else:
            fill = vals + [bg] * (h - len(vals))
        out[:, col] = np.asarray(fill, dtype=np.int8)
    return as_grid(out.T if axis == 0 else out)


def mirror_complete(grid: Grid, mode: str) -> Grid:
    """Fill background cells using horizontal/vertical/diagonal symmetry."""
    out = as_grid(grid)
    bg = background_color(grid)
    h, w = out.shape
    if mode == "h":
        partner = np.fliplr(out)
    elif mode == "v":
        partner = np.flipud(out)
    elif mode == "d":
        partner = out.T if out.shape[0] == out.shape[1] else out
    elif mode == "a":
        partner = np.rot90(out, 2).T if out.shape[0] == out.shape[1] else out
    else:
        raise KeyError(mode)
    if partner.shape != out.shape:
        return out
    holes = out == bg
    return as_grid(np.where(holes & (partner != bg), partner, out))


def raycast(grid: Grid, direction: str, color: int = -1, stop: bool = True) -> Grid:
    """Cast a ray from every foreground cell; ``color=-1`` keeps the source colour."""
    out = as_grid(grid)
    bg = background_color(grid)
    h, w = out.shape
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = deltas[direction]
    sources = list(zip(*np.nonzero(out != bg)))
    for r, c in sources:
        src = int(out[r, c])
        paint = src if color < 0 else color
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            if stop and out[nr, nc] != bg:
                break
            out[nr, nc] = paint
            nr, nc = nr + dr, nc + dc
    return out


def upscale(grid: Grid, k: int) -> Grid:
    """Nearest-neighbour upscale by integer factor ``k``."""
    return as_grid(np.kron(grid, np.ones((k, k), dtype=np.int8)))


def tile(grid: Grid, rows: int, cols: int) -> Grid:
    """Tile the grid ``rows`` x ``cols`` times."""
    return as_grid(np.tile(grid, (rows, cols)))


def concat_mirror(grid: Grid, mode: str) -> Grid:
    """Concatenate the grid with one or two of its mirrors."""
    if mode == "h":
        return as_grid(np.hstack([grid, np.fliplr(grid)]))
    if mode == "v":
        return as_grid(np.vstack([grid, np.flipud(grid)]))
    if mode == "hv":
        top = np.hstack([grid, np.fliplr(grid)])
        return as_grid(np.vstack([top, np.flipud(top)]))
    if mode == "vh":
        left = np.vstack([grid, np.flipud(grid)])
        return as_grid(np.hstack([left, np.fliplr(left)]))
    raise KeyError(mode)


def split_panels(grid: Grid, axis: int, separator: Optional[int] = None) -> List[Grid]:
    """Split the grid along uniform separator lines.

    ``axis=0`` splits on row separators, ``axis=1`` on columns.  A separator is
    a full line whose cells are all equal to the same colour (``separator`` when
    given, otherwise the most common line colour that is not the task
    background).  Panels are returned in raster order.
    """
    out = as_grid(grid)
    if axis == 1:
        out = as_grid(out.T)
    h, w = out.shape
    bg = background_color(grid)
    sep_rows: List[int] = []
    uniform: Dict[int, int] = {}
    for r in range(h):
        row = out[r]
        if np.all(row == row[0]):
            color = int(row[0])
            if separator is not None and color != separator:
                continue
            if separator is None and color == bg and h > 3:
                # A background band is only a separator if it is a thin strip.
                pass
            uniform[r] = color
            sep_rows.append(r)
    if not sep_rows:
        return []
    # Keep separator runs whose colour is consistent, then cut between runs.
    cuts: List[Tuple[int, int]] = []
    start = 0
    for idx, r in enumerate(sep_rows):
        if idx + 1 < len(sep_rows) and sep_rows[idx + 1] == r + 1:
            continue
        cuts.append((start, r))
        start = r + 1
    if start <= h - 1:
        cuts.append((start, h - 1))
    panels: List[Grid] = []
    for r0, r1 in cuts:
        if r1 < r0:
            continue
        block = out[r0:r1 + 1]
        if block.size == 0:
            continue
        panels.append(as_grid(block.T if axis == 1 else block))
    return [p for p in panels if p.size > 0 and p.shape[0] > 0 and p.shape[1] > 0]


def combine_panels(panels: List[Grid], rule: str) -> Optional[Grid]:
    """Combine equal-shaped panels per cell using a reduction rule."""
    if len(panels) < 2:
        return None
    shape = panels[0].shape
    if any(p.shape != shape for p in panels):
        return None
    stack = np.stack([p.astype(np.int16) for p in panels], axis=0)
    bg = background_color(panels[0])
    if rule == "first_non_bg":
        out = stack[0].copy()
        for layer in stack[1:]:
            out = np.where((out == bg) & (layer != bg), layer, out)
        return as_grid(out)
    if rule == "last_non_bg":
        out = stack[-1].copy()
        for layer in stack[:-1][::-1]:
            out = np.where((out == bg) & (layer != bg), layer, out)
        return as_grid(out)
    if rule == "majority":
        counts = np.zeros(shape + (10,), dtype=np.int16)
        for layer in stack:
            for value in range(10):
                counts[..., value] += (layer == value)
        return as_grid(counts.argmax(axis=-1))
    if rule == "difference":
        out = np.where(stack[0] == stack[1], bg, stack[0] if rule == "difference" else stack[1])
        return as_grid(out)
    if rule == "xor_non_bg":
        out = np.full(shape, bg, dtype=np.int8)
        for layer in stack:
            out = np.where((out != bg) ^ (layer != bg), np.where(layer != bg, layer, out), out)
        return as_grid(out)
    raise KeyError(rule)


def overlay_masks(masks: List[np.ndarray], shape: Tuple[int, int], color: int) -> Grid:
    """Paint the union of boolean masks with ``color`` on a background grid."""
    out = np.zeros(shape, dtype=np.int8)
    for mask in masks:
        out[mask] = color
    return out


# ---------------------------------------------------------------------------
# Data-dependent primitive families
# ---------------------------------------------------------------------------


def fit_color_bijection(pairs: Sequence[Tuple[Grid, Grid]]) -> Optional[Dict[int, int]]:
    """Learn a colour map if every cell maps deterministically to one colour."""
    mapping: Dict[int, int] = {}
    for src, dst in pairs:
        if src.shape != dst.shape:
            return None
        flat_src = src.ravel().tolist()
        flat_dst = dst.ravel().tolist()
        for a, b in zip(flat_src, flat_dst):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping or None


def apply_color_map(grid: Grid, mapping: Dict[int, int]) -> Grid:
    """Apply a colour dictionary to every cell."""
    lut = np.arange(10, dtype=np.int8)
    for src, dst in mapping.items():
        lut[src] = dst
    return as_grid(lut[grid])


def fit_panel_structure(pairs: Sequence[Tuple[Grid, Grid]]) -> Optional[Dict[str, int]]:
    """Detect a constant panel layout shared by all training inputs."""
    for axis in (0, 1):
        counts: Optional[int] = None
        ok = True
        for src, _ in pairs:
            panels = split_panels(src, axis)
            if len(panels) < 2 or any(p.shape != panels[0].shape for p in panels):
                ok = False
                break
            if counts is None:
                counts = len(panels)
            elif counts != len(panels):
                ok = False
                break
        if ok and counts:
            return {"axis": axis, "panels": counts}
    return None


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------


def _structural() -> List[Tuple[str, Primitive]]:
    """Parameter-free structural candidates (grid in, grid out)."""
    ops: List[Tuple[str, Primitive]] = [("identity", as_grid)]
    for name, fn in (("rot90", lambda g: np.rot90(g, 1)), ("rot180", lambda g: np.rot90(g, 2)),
                     ("rot270", lambda g: np.rot90(g, 3)), ("fliplr", np.fliplr),
                     ("flipud", np.flipud), ("transpose", lambda g: g.T),
                     ("antidiag", lambda g: np.rot90(g, 2).T)):
        ops.append((name, lambda g, f=fn: as_grid(f(g))))
    ops += [
        ("trim_border", trim_border),
        ("crop_content", lambda g: (lambda b: crop(g, b) if b else None)(bbox(g != background_color(g)))),
        ("crop_largest_obj", lambda g: (lambda m: crop(g, bbox(m[0])) if m and bbox(m[0]) else None)(
            components(g))),
        ("crop_smallest_obj", lambda g: (lambda m: crop(g, bbox(m[-1])) if m and bbox(m[-1]) else None)(
            components(g))),
        ("mirror_h", lambda g: mirror_complete(g, "h")),
        ("mirror_v", lambda g: mirror_complete(g, "v")),
        ("mirror_d", lambda g: mirror_complete(g, "d")),
        ("gravity_down", lambda g: gravity(g, 1, True)),
        ("gravity_up", lambda g: gravity(g, 1, False)),
        ("gravity_right", lambda g: gravity(g, 0, True)),
        ("gravity_left", lambda g: gravity(g, 0, False)),
        ("outline_objs", None),  # replaced below
        ("fill_enclosed_bg", None),
        ("concat_h", lambda g: concat_mirror(g, "h")),
        ("concat_v", lambda g: concat_mirror(g, "v")),
        ("concat_hv", lambda g: concat_mirror(g, "hv")),
        ("concat_vh", lambda g: concat_mirror(g, "vh")),
        ("upscale2", lambda g: upscale(g, 2)),
        ("upscale3", lambda g: upscale(g, 3)),
        ("tile2x2", lambda g: tile(g, 2, 2)),
        ("tile1x2", lambda g: tile(g, 1, 2)),
        ("tile2x1", lambda g: tile(g, 2, 1)),
        ("remove_dup_rows", lambda g: as_grid(np.unique(g, axis=0)) if g.shape[0] > 1 else g),
        ("transpose_shape", lambda g: as_grid(g.reshape(g.shape[1], g.shape[0])) if g.size else None),
    ]

    def _outline(g: Grid) -> Optional[Grid]:
        """Draw a one-cell outline around every foreground component."""
        out = as_grid(g)
        bg = background_color(g)
        for mask in components(g):
            box = bbox(mask)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            r0, c0 = max(0, r0 - 1), max(0, c0 - 1)
            r1, c1 = min(out.shape[0], r1 + 1), min(out.shape[1], c1 + 1)
            color = int(np.bincount(g[mask], minlength=10).argmax()) if mask.any() else bg
            out[r0, c0:c1] = color
            out[r1 - 1, c0:c1] = color
            out[r0:r1, c0] = color
            out[r0:r1, c1 - 1] = color
        return out

    def _fill_enclosed(g: Grid) -> Optional[Grid]:
        """Fill enclosed background regions with the dominant foreground colour."""
        holes = enclosed_mask(g)
        if not holes.any():
            return None
        fg = g[~holes & (g != background_color(g))]
        if fg.size == 0:
            return None
        color = int(np.bincount(fg, minlength=10).argmax())
        out = as_grid(g)
        out[holes] = color
        return out

    return [(n, f) for n, f in ops if f is not None] + [("outline_objs", _outline),
                                                       ("fill_enclosed_bg", _fill_enclosed)]


def enumerate_primitives(pairs: Sequence[Tuple[Grid, Grid]], depth1_budget: int = 400) -> List[Tuple[str, Primitive]]:
    """Build the candidate primitive list, including data-dependent operators."""
    ops: List[Tuple[str, Primitive]] = list(_structural())
    ops = [(n, f) for n, f in ops if f is not None]

    # Learn a global colour bijection.
    mapping = fit_color_bijection(pairs)
    if mapping and mapping != {i: i for i in range(10)}:
        ops.append(("color_map_learned", lambda g, m=mapping: apply_color_map(g, m)))

    # Single colour swaps for every ordered pair (covers one-colour recolours).
    for src in range(10):
        for dst in range(10):
            if src == dst:
                continue
            ops.append((f"swap_{src}_{dst}", lambda g, a=src, b=dst: apply_color_map(g, {a: b})))

    # Replace the most common colour with each other colour.
    for dst in range(10):
        ops.append((f"bg_to_{dst}", lambda g, d=dst: apply_color_map(g, {background_color(g): d})))

    # Enclosed-region fill with each colour.
    for color in range(10):
        def fill_with(g: Grid, c: int = color) -> Optional[Grid]:
            """Replace enclosed background cells with colour ``c``."""
            holes = enclosed_mask(g)
            if not holes.any():
                return None
            out = as_grid(g)
            out[holes] = c
            return out
        ops.append((f"fill_enclosed_{color}", fill_with))

    # Symmetric ray casting with each colour and direction.
    for direction in ("up", "down", "left", "right"):
        ops.append((f"ray_{direction}_own", lambda g, d=direction: raycast(g, d, -1)))
        ops.append((f"ray_{direction}_1", lambda g, d=direction: raycast(g, d, 1)))

    # Panel handling (data dependent layout, all deterministic selectors).
    layout = fit_panel_structure(pairs)
    if layout is not None:
        axis = layout["axis"]
        selectors: List[Tuple[str, Primitive]] = []

        def _panel_index(g: Grid, idx: int, ax: int = axis) -> Optional[Grid]:
            """Return panel ``idx`` of the split grid, or ``None``."""
            panels = split_panels(g, ax)
            if len(panels) <= idx:
                return None
            return panels[idx]

        for idx in range(int(layout["panels"])):
            selectors.append((f"panel_{axis}_{idx}", lambda g, i=idx: _panel_index(g, i)))

        def _panel_most_colors(g: Grid, ax: int = axis) -> Optional[Grid]:
            """Return the panel with the largest palette."""
            panels = split_panels(g, ax)
            if len(panels) < 2:
                return None
            return max(panels, key=lambda p: len(np.unique(p)))

        def _panel_most_fg(g: Grid, ax: int = axis) -> Optional[Grid]:
            """Return the panel with the most non-background cells."""
            panels = split_panels(g, ax)
            if len(panels) < 2:
                return None
            return max(panels, key=lambda p: int((p != background_color(p)).sum()))

        selectors += [("panel_most_colors", _panel_most_colors), ("panel_most_fg", _panel_most_fg)]
        for rule in ("first_non_bg", "last_non_bg", "majority", "difference"):
            def _combined(g: Grid, r: str = rule, ax: int = axis) -> Optional[Grid]:
                """Combine the panels of ``g`` with reduction rule ``r``."""
                panels = split_panels(g, ax)
                return combine_panels(panels, r)
            selectors.append((f"panels_{rule}", _combined))
        ops += selectors

    # Object selection / removal families.
    def _keep_largest(g: Grid) -> Optional[Grid]:
        """Blank the grid except for the largest connected component."""
        masks = components(g)
        if len(masks) < 2:
            return None
        out = np.zeros_like(g)
        out[masks[0]] = g[masks[0]]
        return as_grid(out)

    def _remove_largest(g: Grid) -> Optional[Grid]:
        """Blank the largest connected component."""
        masks = components(g)
        if len(masks) < 2:
            return None
        out = as_grid(g)
        out[masks[0]] = background_color(g)
        return as_grid(out)

    def _keep_unique_color(g: Grid) -> Optional[Grid]:
        """Keep only cells whose colour occurs exactly once in the grid."""
        counts = color_counts(g)
        keep = np.isin(g, np.flatnonzero(counts == 1).astype(np.int8))
        if not keep.any():
            return None
        return as_grid(np.where(keep, g, background_color(g)))

    ops += [("keep_largest", _keep_largest), ("remove_largest", _remove_largest),
            ("keep_unique_color", _keep_unique_color)]

    # Row/column border trimming variants.
    for k in (1, 2):
        ops.append((f"crop_border_{k}", lambda g, kk=k: as_grid(g[kk:-kk, kk:-kk])
                    if g.shape[0] > 2 * kk and g.shape[1] > 2 * kk else None))

    # De-duplicate labels while keeping first occurrence.
    seen: set[str] = set()
    unique: List[Tuple[str, Primitive]] = []
    for name, fn in ops:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, fn))
    return unique[:depth1_budget]


def composition_left_set(pairs: Sequence[Tuple[Grid, Grid]]) -> List[Tuple[str, Primitive]]:
    """Left-hand candidates for depth-2 composition.

    Single colour swaps are excluded here because the right-hand fast set
    already contains them; keeping the left set short and structural buys room
    for the data-dependent operators (panels, object selection, learned map)
    that were previously truncated by the depth-1 budget.
    """
    ops = enumerate_primitives(pairs, depth1_budget=400)
    return [(name, fn) for name, fn in ops if not name.startswith("swap_")
            and not name.startswith("bg_to_")]


def primitive_fast_set() -> List[Tuple[str, Primitive]]:
    """Cheap, high-yield operators used on the right-hand side of depth-2 chains."""
    ops: List[Tuple[str, Primitive]] = [(n, f) for n, f in _structural() if f is not None]
    for src in range(10):
        for dst in range(10):
            if src != dst:
                ops.append((f"swap_{src}_{dst}", lambda g, a=src, b=dst: apply_color_map(g, {a: b})))
    for dst in range(10):
        ops.append((f"bg_to_{dst}", lambda g, d=dst: apply_color_map(g, {background_color(g): d})))
    return ops


def apply_chain(grid: Grid, chain: Sequence[Primitive]) -> Optional[Grid]:
    """Apply a chain of primitives left to right; ``None`` if any step fails."""
    current: Optional[Grid] = as_grid(grid)
    for fn in chain:
        if current is None:
            return None
        current = fn(current)
        if current is None or current.size == 0:
            return None
    return current


def block_mosaic_fit(pairs: Sequence[Tuple[Grid, Grid]]) -> Optional[Dict[str, object]]:
    """Fit an output that is a k x m mosaic of transformed copies of the input.

    Returns a mapping ``(block_row, block_col) -> (d4 name, colour map)`` when
    every training output splits evenly into blocks that each equal a D4 image
    of the corresponding input, with a consistent per-position transform.
    """
    plan: Dict[Tuple[int, int], Tuple[str, Dict[int, int]]] = {}
    k = m = 0
    for src, dst in pairs:
        if dst.shape[0] % src.shape[0] or dst.shape[1] % src.shape[1]:
            return None
        rows, cols = dst.shape[0] // src.shape[0], dst.shape[1] // src.shape[1]
        if rows < 1 or cols < 1 or (rows == 1 and cols == 1):
            return None
        if k == 0:
            k, m = rows, cols
        elif (k, m) != (rows, cols):
            return None
        for br in range(rows):
            for bc in range(cols):
                block = dst[br * src.shape[0]:(br + 1) * src.shape[0],
                           bc * src.shape[1]:(bc + 1) * src.shape[1]]
                found: Optional[Tuple[str, Dict[int, int]]] = None
                for name, fn in (("id", lambda g: g), ("rot90", lambda g: np.rot90(g, 1)),
                                 ("rot180", lambda g: np.rot90(g, 2)), ("rot270", lambda g: np.rot90(g, 3)),
                                 ("fliplr", np.fliplr), ("flipud", np.flipud),
                                 ("transpose", lambda g: g.T), ("antidiag", lambda g: np.rot90(g, 2).T)):
                    cand = np.asarray(fn(src), dtype=np.int8)
                    if cand.shape != block.shape:
                        continue
                    pair_map = fit_color_bijection([(cand, block)])
                    if pair_map is not None and np.array_equal(apply_color_map(cand, pair_map), block):
                        found = (name, pair_map)
                        break
                if found is None:
                    return None
                key = (br, bc)
                if key in plan and plan[key] != found:
                    return None
                plan[key] = found
    if not plan:
        return None
    return {"rows": k, "cols": m, "plan": plan}


def block_mosaic_apply(grid: Grid, plan: Dict[str, object]) -> Optional[Grid]:
    """Render a fitted block mosaic for a new input grid."""
    rows, cols = int(plan["rows"]), int(plan["cols"])
    mapping = plan["plan"]
    blocks: List[List[np.ndarray]] = []
    for br in range(rows):
        row_blocks: List[np.ndarray] = []
        for bc in range(cols):
            entry = mapping.get((br, bc))
            if entry is None:
                return None
            name, cmap = entry
            transformed = np.asarray(_D4_LOOKUP[name](grid), dtype=np.int8)
            row_blocks.append(apply_color_map(transformed, cmap))
        blocks.append(row_blocks)
    if any(b.shape != blocks[0][0].shape for row in blocks for b in row):
        return None
    return as_grid(np.vstack([np.hstack(row) for row in blocks]))


_D4_LOOKUP: Dict[str, Callable[[Grid], Grid]] = {
    "id": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "fliplr": np.fliplr,
    "flipud": np.flipud,
    "transpose": lambda g: g.T,
    "antidiag": lambda g: np.rot90(g, 2).T,
}
