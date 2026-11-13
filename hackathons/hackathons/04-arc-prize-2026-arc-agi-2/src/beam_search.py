"""Best-first / beam search over composed grid and object operators.

v1 enumerated depth-2 programs exhaustively, which capped the reachable
program length at two and spent its budget on operators that could never
match.  This module replaces that with a beam search scored by a cheap
distance heuristic between the current state and the training target:

    h(state) = shape penalty + colour-histogram L1 + object-count error
               + symmetry error + depth penalty

The heuristic is evaluated on the *cheapest* training pair only; any program
that reaches that target exactly is then verified on every pair.  Operators
include the object-level vocabulary (object selection, deletion, cropping,
filling, outlining, gravity, mirror completion) alongside global D4, tiling
and colour operators, so compositions such as "delete the odd object, then
crop to the remaining one" become reachable at depth 3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.canonical import canonical_key
from src.object_model import (PREDICATES, content_objects, crop_bbox, crop_mask,
                              render_objects, segment)

Grid = np.ndarray
Pair = Tuple[Grid, Grid]
Op = Tuple[str, Callable[[Grid], Optional[Grid]]]


# ---------------------------------------------------------------------------
# Operator vocabulary
# ---------------------------------------------------------------------------


def _object_ops() -> List[Op]:
    """Object-level operators: select, delete, crop, fill, outline, gravity."""
    ops: List[Op] = []

    def keep(pred_name: str):
        """Keep only the object chosen by ``pred_name``."""
        def op(grid: Grid) -> Optional[Grid]:
            """Render the selected object alone on the background."""
            objects = content_objects(grid)
            index = PREDICATES[pred_name](objects, grid)
            if index is None:
                return None
            return render_objects(grid, objects, keep=[index])
        return op

    def delete(pred_name: str):
        """Erase the object chosen by ``pred_name``."""
        def op(grid: Grid) -> Optional[Grid]:
            """Paint the selected object with the background colour."""
            objects = content_objects(grid)
            index = PREDICATES[pred_name](objects, grid)
            if index is None:
                return None
            out = dsl.as_grid(grid)
            out[objects[index].mask] = dsl.background_color(grid)
            return out
        return op

    def crop(pred_name: str, masked: bool):
        """Crop to the selected object's bounding box."""
        def op(grid: Grid) -> Optional[Grid]:
            """Return the crop of the selected object."""
            objects = content_objects(grid)
            index = PREDICATES[pred_name](objects, grid)
            if index is None:
                return None
            return crop_mask(grid, objects[index]) if masked else crop_bbox(grid, objects[index])
        return op

    for name in PREDICATES:
        ops.append((f"keep_{name}", keep(name)))
        ops.append((f"del_{name}", delete(name)))
        ops.append((f"crop_{name}", crop(name, False)))

    def fill_holes(grid: Grid) -> Optional[Grid]:
        """Fill enclosed background with the most frequent foreground colour."""
        enclosed = dsl.enclosed_mask(grid)
        if not enclosed.any():
            return None
        values = grid[grid != dsl.background_color(grid)]
        if values.size == 0:
            return None
        colors, counts = np.unique(values, return_counts=True)
        out = dsl.as_grid(grid)
        out[enclosed] = int(colors[counts.argmax()])
        return out

    def outline(grid: Grid, shape: bool = False) -> Optional[Grid]:
        """Draw each object's bounding box or shape outline on background."""
        bg = dsl.background_color(grid)
        out = np.full(grid.shape, bg, dtype=np.int8)
        objects = content_objects(grid)
        if not objects:
            return None
        for obj in objects:
            r0, c0, r1, c1 = obj.box
            color = int(grid[obj.mask][0])
            if shape:
                inner = obj.mask.copy()
                inner[1:-1, 1:-1] = False
                out[inner] = color
            else:
                out[r0, c0:c1] = color
                out[r1 - 1, c0:c1] = color
                out[r0:r1, c0] = color
                out[r0:r1, c1 - 1] = color
        return out

    def gravity_all(axis: int, positive: bool):
        """Slide all cells (not objects) along one axis."""
        return lambda grid: dsl.gravity(grid, axis, positive)

    def ray_both(grid: Grid, vertical: bool) -> Optional[Grid]:
        """Cast rays in both directions along one axis from every source cell."""
        first = dsl.raycast(grid, "up" if vertical else "left", -1, True)
        return dsl.raycast(first, "down" if vertical else "right", -1, True)

    def interpolate(grid: Grid, axis: int) -> Optional[Grid]:
        """Fill background gaps between two equal colours along rows or columns."""
        out = dsl.as_grid(grid)
        bg = dsl.background_color(grid)
        lines = out if axis == 0 else out.T
        changed = False
        for index in range(lines.shape[0]):
            row = lines[index]
            positions = np.flatnonzero(row != bg)
            for left, right in zip(positions.tolist(), positions.tolist()[1:]):
                if right - left > 1 and row[left] == row[right]:
                    row[left + 1:right] = row[left]
                    changed = True
        if not changed:
            return None
        return out if axis == 0 else np.ascontiguousarray(out.T)

    def object_gravity(grid: Grid, dr: int, dc: int) -> Optional[Grid]:
        """Slide whole objects one step at a time until they hit an obstacle."""
        bg = dsl.background_color(grid)
        out = dsl.as_grid(grid)
        objects = content_objects(grid)
        if not objects:
            return None
        h, w = out.shape
        order = sorted(objects, key=lambda o: (o.centroid[0] * (1 if dr > 0 else -1) +
                                               o.centroid[1] * (1 if dc > 0 else -1)),
                       reverse=True)
        moved_any = False
        for obj in order:
            cells = list(obj.cells)
            values = [int(grid[r, c]) for r, c in cells]
            out[obj.mask] = bg
            while True:
                nxt = [(r + dr, c + dc) for r, c in cells]
                if all(0 <= r < h and 0 <= c < w and out[r, c] == bg for r, c in nxt):
                    cells = nxt
                    moved_any = True
                else:
                    break
            for (r, c), value in zip(cells, values):
                out[r, c] = value
        return out if moved_any else None

    ops.extend([
        ("fill_holes", fill_holes),
        ("ray_both_v", lambda g: ray_both(g, True)),
        ("ray_both_h", lambda g: ray_both(g, False)),
        ("interp_rows", lambda g: interpolate(g, 0)),
        ("interp_cols", lambda g: interpolate(g, 1)),
        ("obj_grav_down", lambda g: object_gravity(g, 1, 0)),
        ("obj_grav_up", lambda g: object_gravity(g, -1, 0)),
        ("obj_grav_right", lambda g: object_gravity(g, 0, 1)),
        ("obj_grav_left", lambda g: object_gravity(g, 0, -1)),
        ("crop_content_all", dsl.trim_border),
        ("outline_bbox", lambda g: outline(g, False)),
        ("outline_shape", lambda g: outline(g, True)),
        ("gravity_down", gravity_all(0, True)),
        ("gravity_up", gravity_all(0, False)),
        ("gravity_right", gravity_all(1, True)),
        ("gravity_left", gravity_all(1, False)),
        ("mirror_h", lambda g: dsl.mirror_complete(g, "h")),
        ("mirror_v", lambda g: dsl.mirror_complete(g, "v")),
        ("mirror_d", lambda g: dsl.mirror_complete(g, "d")),
        ("mirror_a", lambda g: dsl.mirror_complete(g, "a")),
        ("trim", dsl.trim_border),
        ("fill_enclosed_major", fill_holes),
        ("ray_up", lambda g: dsl.raycast(g, "up", -1, True)),
        ("ray_down", lambda g: dsl.raycast(g, "down", -1, True)),
        ("ray_left", lambda g: dsl.raycast(g, "left", -1, True)),
        ("ray_right", lambda g: dsl.raycast(g, "right", -1, True)),
        ("ray_down_nostop", lambda g: dsl.raycast(g, "down", -1, False)),
        ("ray_right_nostop", lambda g: dsl.raycast(g, "right", -1, False)),
        ("largest_only", lambda g: render_objects(
            g, content_objects(g), keep=[0]) if content_objects(g) else None),
    ])
    for color in range(10):
        ops.append((f"fill_enclosed_{color}",
                    lambda g, c=color: _fill_enclosed(g, c)))
    return ops


def _fill_enclosed(grid: Grid, color: int) -> Optional[Grid]:
    """Fill enclosed background regions with a fixed colour."""
    enclosed = dsl.enclosed_mask(grid)
    if not enclosed.any():
        return None
    out = dsl.as_grid(grid)
    out[enclosed] = color
    return out


def _global_ops() -> List[Op]:
    """Shape-changing and D4 operators."""
    return [
        ("id", dsl.as_grid),
        ("rot90", lambda g: np.ascontiguousarray(np.rot90(g, 1))),
        ("rot180", lambda g: np.ascontiguousarray(np.rot90(g, 2))),
        ("rot270", lambda g: np.ascontiguousarray(np.rot90(g, 3))),
        ("fliplr", lambda g: np.ascontiguousarray(np.fliplr(g))),
        ("flipud", lambda g: np.ascontiguousarray(np.flipud(g))),
        ("transpose", lambda g: np.ascontiguousarray(g.T)),
        ("up2", lambda g: dsl.upscale(g, 2)),
        ("up3", lambda g: dsl.upscale(g, 3)),
        ("tile2x2", lambda g: dsl.tile(g, 2, 2)),
        ("tile1x2", lambda g: dsl.tile(g, 1, 2)),
        ("tile2x1", lambda g: dsl.tile(g, 2, 1)),
        ("tile3x3", lambda g: dsl.tile(g, 3, 3)),
    ]


def _colour_ops(pairs: Sequence[Pair]) -> List[Op]:
    """Data-driven colour operators: learned bijection and frequent swaps."""
    ops: List[Op] = []

    def majority_swap(grid: Grid) -> Optional[Grid]:
        """Swap the two most frequent colours of the training input."""
        counts = dsl.color_counts(pairs[0][0])
        present = np.flatnonzero(counts)
        if present.size < 2:
            return None
        top = sorted(present.tolist(), key=lambda c: -counts[c])[:2]
        return dsl.apply_color_map(grid, {int(top[0]): int(top[1]), int(top[1]): int(top[0])})

    ops.append(("swap_majority", majority_swap))

    def to_bg(grid: Grid) -> Optional[Grid]:
        """Map the least frequent colour to the background colour."""
        counts = dsl.color_counts(grid)
        present = np.flatnonzero(counts)
        if present.size < 2:
            return None
        rare = int(present[np.argmin(counts[present])])
        return dsl.apply_color_map(grid, {rare: dsl.background_color(grid)})

    ops.append(("rare_to_bg", to_bg))
    return ops


def build_operators(pairs: Sequence[Pair]) -> List[Op]:
    """Full operator vocabulary for one task."""
    return _global_ops() + _object_ops() + _colour_ops(pairs)


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


def _histogram(grid: Grid) -> np.ndarray:
    """Normalised 10-bin colour histogram."""
    counts = np.bincount(grid.ravel(), minlength=10).astype(np.float32)
    return counts / max(float(counts.sum()), 1.0)


def heuristic(state: Grid, target: Grid) -> float:
    """Cheap distance from ``state`` to ``target``.

    Shape mismatch dominates because no downstream operator except tiling or
    upscaling can repair it.  When shapes agree the score combines a colour
    histogram L1 distance, an object-count mismatch and a symmetry penalty.
    """
    if state is None or state.size == 0:
        return 1e6
    score = 0.0
    if state.shape != target.shape:
        dh = abs(state.shape[0] - target.shape[0])
        dw = abs(state.shape[1] - target.shape[1])
        score += 100.0 + 10.0 * (dh + dw)
        state_box = dsl.bbox(state != dsl.background_color(state))
        target_box = dsl.bbox(target != dsl.background_color(target))
        if state_box and target_box:
            score += 2.0 * (abs((state_box[2] - state_box[0]) - (target_box[2] - target_box[0])) +
                            abs((state_box[3] - state_box[1]) - (target_box[3] - target_box[1])))
        return score
    hist_distance = float(np.abs(_histogram(state) - _histogram(target)).sum())
    score += 5.0 * hist_distance
    diff = int((state != target).sum())
    score += diff / max(target.size, 1)
    # Segmentation is the expensive part of this heuristic, so the object-count
    # term is only evaluated for states that already look close; those are the
    # only ones whose ranking the term can change.
    if hist_distance < 0.6:
        try:
            score += 0.5 * abs(len(content_objects(state)) - len(content_objects(target)))
        except Exception:
            pass
    for fn in (np.fliplr, np.flipud):
        try:
            if np.array_equal(target, fn(target)) and not np.array_equal(state, fn(state)):
                score += 0.3
        except Exception:
            continue
    return score


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@dataclass
class BeamProgram:
    """A verified operator chain produced by the beam search."""

    chain: List[Op]
    outputs: List[Grid] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Human-readable composition string."""
        return " -> ".join(name for name, _ in self.chain)

    def apply(self, grid: Grid) -> Optional[Grid]:
        """Apply the whole chain, aborting on failure."""
        return dsl.apply_chain(grid, [fn for _, fn in self.chain])


class BeamSolver:
    """Best-first search with a heuristic beam, then exact multi-pair verification."""

    def __init__(self, beam_width: int = 8, max_depth: int = 3,
                 max_nodes: int = 900, time_budget: float = 4.0) -> None:
        """Configure beam width, depth, node cap and wall-clock budget."""
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.time_budget = time_budget

    def solve(self, pairs: Sequence[Pair], test_inputs: Sequence[Grid],
              max_programs: int = 2) -> List[BeamProgram]:
        """Search for chains that map the probe input to its output exactly."""
        if not pairs:
            return []
        start_time = time.monotonic()
        probe = min(pairs, key=lambda pair: pair[0].size)
        ops = build_operators(pairs)
        target = probe[1]

        # Depth-0 check: the identity program is a legitimate answer.
        best: List[BeamProgram] = []
        frontier: List[Tuple[Grid, List[Op]]] = [(probe[0], [])]
        visited = {canonical_key(probe[0])}
        nodes = 0

        for _depth in range(self.max_depth):
            candidate_states: List[Tuple[float, Grid, List[Op]]] = []
            for state, chain in frontier:
                for name, fn in ops:
                    if nodes >= self.max_nodes or time.monotonic() - start_time > self.time_budget:
                        break
                    nodes += 1
                    try:
                        nxt = fn(state)
                    except Exception:
                        continue
                    if nxt is None or not isinstance(nxt, np.ndarray) or nxt.ndim != 2 or nxt.size == 0:
                        continue
                    nxt = np.ascontiguousarray(np.clip(nxt, 0, 9).astype(np.int8))
                    try:
                        key = canonical_key(nxt)
                    except Exception:
                        key = nxt.tobytes() + bytes(nxt.shape)
                    if key in visited:
                        continue
                    visited.add(key)
                    new_chain = chain + [(name, fn)]
                    if nxt.shape == target.shape and np.array_equal(nxt, target):
                        program = BeamProgram(chain=new_chain)
                        if self._verify(program, pairs, test_inputs):
                            best.append(program)
                            if len(best) >= max_programs:
                                return best
                        continue
                    candidate_states.append((heuristic(nxt, target) + 0.05 * len(new_chain), nxt, new_chain))
                if nodes >= self.max_nodes or time.monotonic() - start_time > self.time_budget:
                    break
            if not candidate_states:
                break
            candidate_states.sort(key=lambda item: item[0])
            frontier = [(state, chain) for _score, state, chain in candidate_states[:self.beam_width]]
            if time.monotonic() - start_time > self.time_budget:
                break
        return best

    def _verify(self, program: BeamProgram, pairs: Sequence[Pair],
                test_inputs: Sequence[Grid]) -> bool:
        """Check the chain on every training pair and collect test outputs."""
        for src, dst in pairs:
            out = program.apply(src)
            if out is None or out.shape != dst.shape or not np.array_equal(out, dst):
                return False
        outputs: List[Grid] = []
        for grid in test_inputs:
            out = program.apply(grid)
            if out is None:
                return False
            outputs.append(out)
        program.outputs = outputs
        return True
