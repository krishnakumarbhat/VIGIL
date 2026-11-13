"""Structural ARC families beyond single-object rules.

These are the compositional layouts that ARC-AGI-2 leans on and that neither
v1's depth-2 enumeration nor a per-object recolouring rule can express:

``panels``    the grid splits into 2-3 panels (separator lines or equal
              division); the output is one selected panel, or a *learned
              element-wise function of the panel values* (XOR, AND, overlay)
``rays``      every foreground cell shoots a ray until it hits something
``window``    the output is a window of the input chosen by a consistent rule
              (content bbox, per-colour bbox, fixed offsets)
``resize``    the output is a nearest-neighbour rescale by a consistent factor
``denoise``   objects below a consistent size threshold are removed

Every family is verified on all training pairs before being returned.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.object_model import content_objects, crop_bbox, crop_mask
from src.program import Program, apply_map, fit_cell_map, grids_equal

Grid = np.ndarray
Pair = Tuple[Grid, Grid]


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------


def panel_layouts(grid: Grid) -> List[Tuple[str, List[Grid]]]:
    """All plausible panel decompositions of a grid.

    Two mechanisms are considered: cutting along uniform separator rows/columns
    and equal division into halves or thirds.  Each layout is returned with a
    name so the same layout can be required across training pairs.
    """
    h, w = grid.shape
    layouts: List[Tuple[str, List[Grid]]] = []

    def cut_uniform(axis: int) -> List[Grid]:
        """Split along uniform rows (axis 0) or columns (axis 1)."""
        if axis == 0:
            lines = [r for r in range(h) if len(np.unique(grid[r])) == 1]
            positions = [r for r in lines if 0 < r < h - 1]
        else:
            lines = [c for c in range(w) if len(np.unique(grid[:, c])) == 1]
            positions = [c for c in lines if 0 < c < w - 1]
        panels: List[Grid] = []
        previous = -1
        for position in positions + [h if axis == 0 else w]:
            block = grid[previous + 1:position, :] if axis == 0 else grid[:, previous + 1:position]
            if block.size:
                panels.append(np.ascontiguousarray(block))
            previous = position
        return panels

    for axis, label in ((0, "rows"), (1, "cols")):
        try:
            panels = cut_uniform(axis)
        except Exception:
            panels = []
        if 2 <= len(panels) <= 4:
            layouts.append((f"sep_{label}_{len(panels)}", panels))

    if h % 2 == 0 and h >= 4:
        layouts.append(("half_rows", [np.ascontiguousarray(grid[:h // 2]), np.ascontiguousarray(grid[h // 2:])]))
    if w % 2 == 0 and w >= 4:
        layouts.append(("half_cols", [np.ascontiguousarray(grid[:, :w // 2]), np.ascontiguousarray(grid[:, w // 2:])]))
    if h % 3 == 0 and h >= 6:
        layouts.append(("third_rows", [np.ascontiguousarray(grid[i * h // 3:(i + 1) * h // 3]) for i in range(3)]))
    if w % 3 == 0 and w >= 6:
        layouts.append(("third_cols", [np.ascontiguousarray(grid[:, i * w // 3:(i + 1) * w // 3]) for i in range(3)]))
    return layouts


# ---------------------------------------------------------------------------
# Family class
# ---------------------------------------------------------------------------


class StructuralFamilies:
    """Fit panel, ray, window, resize and denoise families."""

    def __init__(self, max_programs: int = 2, loo: bool = True) -> None:
        """Cap the number of returned programs and enable leave-one-out checks."""
        self.max_programs = max_programs
        self.loo = loo
        self._in_loo = False

    def fit(self, pairs: Sequence[Pair]) -> List[Program]:
        """Return every structural program that fits all training pairs."""
        found: List[Program] = []
        for fitter in (self._fit_panels, self._fit_rays, self._fit_window,
                       self._fit_resize, self._fit_denoise):
            try:
                found.extend(fitter(pairs)[:self.max_programs])
            except Exception:
                continue
            if len(found) >= self.max_programs * 3:
                break
        return found

    # -- helpers -----------------------------------------------------------

    def _emit(self, family: str, fn: Callable[[Grid], Optional[Grid]], detail: str,
              pairs: Sequence[Pair], complexity: int = 2,
              refit: Optional[Callable[[Sequence[Pair]], Optional[Program]]] = None,
              meta: Optional[Dict[str, Any]] = None) -> List[Program]:
        """Verify a program on every pair and attach leave-one-out evidence."""
        for src, dst in pairs:
            if not grids_equal(fn(src), dst):
                return []
        loo_pass = False
        if refit is not None and self.loo and len(pairs) >= 3 and not self._in_loo:
            loo_pass = True
            self._in_loo = True
            try:
                for index in range(len(pairs)):
                    subset = list(pairs[:index]) + list(pairs[index + 1:])
                    model = refit(subset)
                    if model is None or not grids_equal(model.apply(pairs[index][0]), pairs[index][1]):
                        loo_pass = False
                        break
            finally:
                self._in_loo = False
        return [Program(family, fn, detail, complexity, loo_pass, dict(meta or {}))]

    # -- panels ------------------------------------------------------------

    def _fit_panels(self, pairs: Sequence[Pair]) -> List[Program]:
        """Output = one selected panel, or a learned function of panel values."""
        layouts_first = panel_layouts(pairs[0][0])
        results: List[Program] = []
        for layout_name, panels_first in layouts_first:
            k = len(panels_first)
            per_pair: List[List[Grid]] = []
            ok = True
            for src, _dst in pairs:
                match = [p for name, p in panel_layouts(src) if name == layout_name]
                if not match or len(match[0]) != k:
                    ok = False
                    break
                per_pair.append(match[0])
            if not ok:
                continue

            # 1) output is a selected panel (optionally D4 / recoloured).
            for rule in self._panel_rules(k):
                crops: List[Optional[Grid]] = []
                for panels, (_src, dst) in zip(per_pair, pairs):
                    index = rule(panels)
                    crops.append(None if index is None else panels[index])
                if any(crop is None or crop.shape != dst.shape
                       for crop, (_s, dst) in zip(crops, pairs)):
                    continue
                for view_name, view in (("id", dsl.as_grid), ("rot90", lambda g: np.rot90(g, 1)),
                                        ("flipud", np.flipud), ("fliplr", np.fliplr)):
                    view_crops = [view(crop) for crop in crops if crop is not None]
                    if any(v.shape != d.shape for v, (_s, d) in zip(view_crops, pairs)):
                        continue
                    for mode in ("identity", "mapped"):
                        mapping = {0: 0} if mode == "identity" else fit_cell_map(
                            list(zip(view_crops, [d for _s, d in pairs])))
                        if mapping is None:
                            continue
                        if not all(grids_equal(apply_map(v, mapping), d)
                                   for v, (_s, d) in zip(view_crops, pairs)):
                            continue

                        def program(grid: Grid, name=layout_name, sel=rule, vw=view_name,
                                    mp=mapping) -> Optional[Grid]:
                            """Select a panel, transform it and recolour it."""
                            panels = [p for layout, p in panel_layouts(grid) if layout == name]
                            if not panels:
                                return None
                            index = sel(panels[0])
                            if index is None:
                                return None
                            crop = panels[0][index]
                            view_fn = {"id": dsl.as_grid, "rot90": lambda g: np.rot90(g, 1),
                                       "flipud": np.flipud, "fliplr": np.fliplr}[vw]
                            return apply_map(view_fn(crop), mp)

                        detail = f"select:{layout_name}:{rule.__name__}:{view_name}:{mode}"
                        results.extend(self._emit("panels", program, detail, pairs, 3,
                                                  lambda sub, n=layout_name: self._first_panel(sub, n),
                                                  {"layout": layout_name}))
                        if len(results) >= self.max_programs:
                            return results

            # 2) output is a learned element-wise function of all panel values.
            aligned = True
            for panels, (_src, dst) in zip(per_pair, pairs):
                if any(panel.shape != dst.shape for panel in panels):
                    aligned = False
                    break
            if aligned:
                table: Dict[Tuple[int, ...], int] = {}
                consistent = True
                for panels, (_src, dst) in zip(per_pair, pairs):
                    stacked = np.stack([panel.ravel() for panel in panels], axis=1)
                    for key, value in zip(map(tuple, stacked.tolist()), dst.ravel().tolist()):
                        if key in table and table[key] != value:
                            consistent = False
                            break
                        table[key] = value
                    if not consistent:
                        break
                if consistent and table:
                    def program(grid: Grid, name=layout_name, learned=table) -> Optional[Grid]:
                        """Apply the learned panel-value table element-wise."""
                        panels = [p for layout, p in panel_layouts(grid) if layout == name]
                        if not panels:
                            return None
                        chosen = panels[0]
                        if any(panel.shape != chosen[0].shape for panel in chosen):
                            return None
                        stacked = np.stack([panel.ravel() for panel in chosen], axis=1)
                        out = np.zeros(chosen[0].size, dtype=np.int8)
                        for index, key in enumerate(map(tuple, stacked.tolist())):
                            value = learned.get(key)
                            if value is None:
                                return None
                            out[index] = value
                        return out.reshape(chosen[0].shape)

                    results.extend(self._emit("panels", program,
                                              f"table:{layout_name}:{len(table)}entries",
                                              pairs, 2 + len(table) // 10,
                                              lambda sub, n=layout_name: self._first_panel(sub, n),
                                              {"layout": layout_name, "table": len(table)}))
                    if len(results) >= self.max_programs:
                        return results
        if results:
            return results
        # 3) output is a panel with a constant colour added/removed, handled by
        #    the recolour family; nothing else to try here.
        return []

    @staticmethod
    def _panel_rules(k: int) -> List[Callable[[List[Grid]], Optional[int]]]:
        """Selection rules over a panel list."""
        def most_foreground(panels: List[Grid]) -> Optional[int]:
            """Panel with the largest number of foreground cells."""
            scores = [int((panel != dsl.background_color(panel)).sum()) for panel in panels]
            return int(np.argmax(scores))

        def least_foreground(panels: List[Grid]) -> Optional[int]:
            """Panel with the fewest foreground cells."""
            scores = [int((panel != dsl.background_color(panel)).sum()) for panel in panels]
            return int(np.argmin(scores))

        def odd_panel(panels: List[Grid]) -> Optional[int]:
            """Panel that differs most from the others (the odd one out)."""
            best: Optional[Tuple[float, int]] = None
            for index in range(len(panels)):
                others = [j for j in range(len(panels)) if j != index]
                distance = 0.0
                for other in others:
                    if panels[index].shape != panels[other].shape:
                        return None
                    distance += float((panels[index] != panels[other]).mean())
                if best is None or distance > best[0]:
                    best = (distance, index)
            return None if best is None else best[1]

        rules: List[Callable[[List[Grid]], Optional[int]]] = [
            lambda panels: 0, lambda panels: len(panels) - 1,
            most_foreground, least_foreground, odd_panel]
        return rules

    def _first_panel(self, pairs: Sequence[Pair], layout_name: str) -> Optional[Program]:
        """Refit helper for LOO over panel programs."""
        candidates = self._fit_panels(pairs)
        for program in candidates:
            if program.meta.get("layout") == layout_name:
                return program
        return candidates[0] if candidates else None

    # -- rays --------------------------------------------------------------

    def _fit_rays(self, pairs: Sequence[Pair]) -> List[Program]:
        """Every foreground cell casts a ray in one consistent direction."""
        directions = ("up", "down", "left", "right")
        for direction in directions:
            for stop in (True, False):
                for color in (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                    if color >= 0 and color == dsl.background_color(pairs[0][0]):
                        continue
                    fn = lambda grid, d=direction, s=stop, c=color: dsl.raycast(grid, d, c, s)
                    found = self._emit("rays", fn, f"{direction}:stop={stop}:color={color}",
                                       pairs, 2)
                    if found:
                        return found
        return []

    # -- windows -----------------------------------------------------------

    def _window_candidates(self, grid: Grid) -> List[Tuple[str, Grid]]:
        """Windows of ``grid`` proposed by simple consistent rules."""
        out: List[Tuple[str, Grid]] = []
        bg = dsl.background_color(grid)
        content = dsl.bbox(grid != bg)
        if content is not None:
            out.append(("content", crop_bbox(grid, _Box(content))))
        for color in range(10):
            if color == bg:
                continue
            box = dsl.bbox(grid == color)
            if box is not None and (box[2] - box[0]) * (box[3] - box[1]) < grid.size:
                out.append((f"color_{color}", crop_bbox(grid, _Box(box))))
        objects = content_objects(grid)
        if objects:
            out.append(("largest_obj", crop_bbox(grid, objects[0])))
            out.append(("largest_obj_mask", crop_mask(grid, objects[0])))
        trimmed = dsl.trim_border(grid)
        if trimmed.shape != grid.shape:
            out.append(("trim", trimmed))
        return out

    def _fit_window(self, pairs: Sequence[Pair]) -> List[Program]:
        """Output equals one of the rule-selected windows, up to D4 and colour."""
        names = [name for name, _ in self._window_candidates(pairs[0][0])]
        for name in names:
            windows: List[Optional[Grid]] = []
            for src, _dst in pairs:
                match = [window for label, window in self._window_candidates(src) if label == name]
                windows.append(match[0] if match else None)
            if any(window is None for window in windows):
                continue
            if any(window.shape != dst.shape for window, (_s, dst) in zip(windows, pairs)):
                continue
            for view_name, view in (("id", dsl.as_grid), ("rot90", lambda g: np.rot90(g, 1)),
                                    ("rot180", lambda g: np.rot90(g, 2)),
                                    ("rot270", lambda g: np.rot90(g, 3)),
                                    ("fliplr", np.fliplr), ("flipud", np.flipud)):
                views = [view(window) for window in windows if window is not None]
                if any(v.shape != d.shape for v, (_s, d) in zip(views, pairs)):
                    continue
                for mode in ("identity", "mapped"):
                    mapping = {0: 0} if mode == "identity" else fit_cell_map(
                        list(zip(views, [d for _s, d in pairs])))
                    if mapping is None:
                        continue
                    if not all(grids_equal(apply_map(v, mapping), d)
                               for v, (_s, d) in zip(views, pairs)):
                        continue

                    def program(grid: Grid, label=name, vw=view_name, mp=mapping) -> Optional[Grid]:
                        """Select a window, transform it and recolour it."""
                        match = [w for l, w in self._window_candidates(grid) if l == label]
                        if not match:
                            return None
                        view_fn = {"id": dsl.as_grid, "rot90": lambda g: np.rot90(g, 1),
                                   "rot180": lambda g: np.rot90(g, 2),
                                   "rot270": lambda g: np.rot90(g, 3),
                                   "fliplr": np.fliplr, "flipud": np.flipud}[vw]
                        return apply_map(view_fn(match[0]), mp)

                    found = self._emit("window", program, f"{name}:{view_name}:{mode}", pairs, 2,
                                       lambda sub, l=name: self._first_window(sub, l))
                    if found:
                        return found
        return []

    def _first_window(self, pairs: Sequence[Pair], name: str) -> Optional[Program]:
        """Refit helper for LOO over window programs."""
        candidates = self._fit_window(pairs)
        return candidates[0] if candidates else None

    # -- resize ------------------------------------------------------------

    def _fit_resize(self, pairs: Sequence[Pair]) -> List[Program]:
        """Nearest-neighbour rescale by a factor consistent across pairs."""
        factors: List[Tuple[float, float]] = []
        for src, dst in pairs:
            factors.append((dst.shape[0] / src.shape[0], dst.shape[1] / src.shape[1]))
        if not factors or len({(round(a, 3), round(b, 3)) for a, b in factors}) != 1:
            return []
        kh, kw = factors[0]
        if (kh, kw) == (1.0, 1.0) or kh < 0.2 or kw < 0.2 or kh > 8 or kw > 8:
            return []

        def program(grid: Grid, rh=kh, rw=kw, exact=all(
                f[0] == int(f[0]) and f[1] == int(f[1]) for f in
                [(dst.shape[0] / src.shape[0], dst.shape[1] / src.shape[1])
                 for src, dst in pairs])) -> Optional[Grid]:
            """Resize by nearest-neighbour indexing to the learned shape."""
            h, w = grid.shape
            new_h, new_w = int(round(h * rh)), int(round(w * rw))
            if new_h < 1 or new_w < 1:
                return None
            rows = np.minimum((np.arange(new_h) / rh).astype(int), h - 1)
            cols = np.minimum((np.arange(new_w) / rw).astype(int), w - 1)
            return np.ascontiguousarray(grid[np.ix_(rows, cols)])

        return self._emit("resize", program, f"factor=({kh:.2f},{kw:.2f})", pairs, 2)

    # -- denoise -----------------------------------------------------------

    def _fit_denoise(self, pairs: Sequence[Pair]) -> List[Program]:
        """Remove objects below a size threshold, or the rarest shape."""
        if any(src.shape != dst.shape for src, dst in pairs):
            return []
        max_size = max(max((obj.size for obj in content_objects(src)), default=0)
                       for src, _ in pairs)
        for threshold in range(1, max(2, max_size)):
            fn = lambda grid, t=threshold: self._drop_small(grid, t)
            found = self._emit("denoise", fn, f"drop_size<{threshold}", pairs, 2)
            if found:
                return found
        for keep_modal in (True, False):
            fn = lambda grid, mode=keep_modal: self._keep_modal_shape(grid, mode)
            found = self._emit("denoise", fn, f"modal_shape={keep_modal}", pairs, 3)
            if found:
                return found
        return []

    @staticmethod
    def _drop_small(grid: Grid, threshold: int) -> Optional[Grid]:
        """Erase every object smaller than ``threshold`` cells."""
        out = dsl.as_grid(grid)
        bg = dsl.background_color(grid)
        changed = False
        for obj in content_objects(grid):
            if obj.size < threshold:
                out[obj.mask] = bg
                changed = True
        return out if changed else None

    @staticmethod
    def _keep_modal_shape(grid: Grid, modal: bool) -> Optional[Grid]:
        """Keep only objects whose shape is the most (or least) frequent."""
        objects = content_objects(grid)
        if len(objects) < 2:
            return None
        counts: Dict[bytes, int] = {}
        for obj in objects:
            counts[obj.shape_key] = counts.get(obj.shape_key, 0) + 1
        target = max(counts.values()) if modal else min(counts.values())
        out = np.full(grid.shape, dsl.background_color(grid), dtype=np.int8)
        for obj in objects:
            if counts[obj.shape_key] == target:
                out[obj.mask] = grid[obj.mask]
        return out


class _Box:
    """Minimal object shim exposing a bounding box for :func:`crop_bbox`."""

    def __init__(self, box: Tuple[int, int, int, int]) -> None:
        """Store the bounding box."""
        self.box = box
