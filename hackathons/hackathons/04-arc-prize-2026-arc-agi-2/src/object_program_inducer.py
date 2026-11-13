"""Object-centric program induction.

Given a task's training pairs this module searches a *structured* family of
hypotheses that all speak about whole objects rather than cells:

======================  =====================================================
family                  hypothesis
======================  =====================================================
``select_crop``         output is the bounding box of the object picked by a
                        selection predicate, optionally D4-transformed and
                        recoloured through a learned bijection
``select_bbox``         same but crops the full grid box (keeps neighbours)
``recolour_rule``       every object is recoloured by one attribute rule that
                        is *consistent across all training pairs*
``move_all``            every object translates by one shared offset
``duplicate_offset``    every object is copied by one shared offset
``delete_rule``         objects matching a predicate are erased
``fill_holes``          enclosed background is filled with a learned colour
``outline``             each object is replaced by its bbox/shape outline
``stamp``               a unique template object is stamped at marker cells
``connect``             collinear objects are joined by lines
======================  =====================================================

Every fitter is parameter-free apart from discrete choices read off the
training pairs, and every returned program reproduces *all* training pairs
exactly.  Fitters with a large discrete search space are additionally checked
with leave-one-out refitting (``loo_pass``) when the task has at least three
training pairs; that flag is a feature for the ranking model and a hard gate
for the highest-risk families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.canonical import D4
from src.object_model import (ArcObject, PREDICATES, content_objects, crop_bbox,
                              crop_mask, render_objects, segment)

Grid = np.ndarray
Pair = Tuple[Grid, Grid]
ProgramFn = Callable[[Grid], Optional[Grid]]


@dataclass(eq=False)
class Program:
    """A fitted object-level program plus the evidence collected while fitting."""

    family: str
    fn: ProgramFn
    detail: str
    complexity: int = 1
    loo_pass: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def apply(self, grid: Grid) -> Optional[Grid]:
        """Run the program on one grid, sanitising the output shape."""
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


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _equal(a: Optional[Grid], b: Grid) -> bool:
    """Exact grid equality with shape check."""
    return a is not None and a.shape == b.shape and np.array_equal(a, b)


def _fit_cell_map(pairs: Sequence[Tuple[Grid, Grid]]) -> Optional[Dict[int, int]]:
    """Learn a consistent colour→colour map between equal-shaped grids."""
    mapping: Dict[int, int] = {}
    for src, dst in pairs:
        if src.shape != dst.shape:
            return None
        for a, b in zip(src.ravel().tolist(), dst.ravel().tolist()):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping


def _apply_map(grid: Grid, mapping: Dict[int, int]) -> Optional[Grid]:
    """Apply a colour map, returning ``None`` when the grid has unseen colours."""
    out = np.empty_like(grid)
    for src, dst in mapping.items():
        out[grid == src] = dst
    for value in np.unique(grid).tolist():
        if int(value) not in mapping:
            return None
    return out


def _consistent_map(observations: Sequence[Tuple[Any, int]]) -> Optional[Dict[Any, int]]:
    """Turn (feature value, colour) observations into a conflict-free dict."""
    mapping: Dict[Any, int] = {}
    for value, color in observations:
        if value in mapping and mapping[value] != color:
            return None
        mapping[value] = color
    return mapping


# ---------------------------------------------------------------------------
# Inducer
# ---------------------------------------------------------------------------


class ObjectProgramInducer:
    """Search object-level programs and keep only those that fit every pair."""

    def __init__(self, max_programs_per_family: int = 1, loo: bool = True,
                 time_budget: float = 6.0) -> None:
        """Configure how many alternatives per family to keep and the LOO flag."""
        self.max_per_family = max_programs_per_family
        self.loo = loo
        self._in_loo = False

    # -- public API --------------------------------------------------------

    def fit(self, pairs: Sequence[Pair]) -> List[Program]:
        """Fit every object-level family; return all programs that fit exactly."""
        if not pairs:
            return []
        programs: List[Program] = []
        from src.structural_families import StructuralFamilies

        fitters = (
            self._fit_select_object, self._fit_select_bbox, self._fit_recolour_rule,
            self._fit_move_all, self._fit_duplicate_offset, self._fit_delete_rule,
            self._fit_fill_holes, self._fit_outline, self._fit_stamp, self._fit_connect,
        )
        programs.extend(StructuralFamilies(max_programs=2).fit(pairs))
        for fitter in fitters:
            try:
                found = fitter(pairs)
            except Exception:
                continue
            programs.extend(found[:self.max_per_family])
        return programs

    # -- verification / LOO ------------------------------------------------

    def _verify(self, fn: ProgramFn, pairs: Sequence[Pair]) -> bool:
        """True when ``fn`` reproduces every training output exactly."""
        for src, dst in pairs:
            if not _equal(fn(src), dst):
                return False
        return True

    def _emit(self, family: str, fn: ProgramFn, detail: str, pairs: Sequence[Pair],
              refit: Optional[Callable[[Sequence[Pair]], Optional["Program"]]] = None,
              complexity: int = 1, meta: Optional[Dict[str, Any]] = None) -> List[Program]:
        """Validate ``fn`` on all pairs, run LOO when possible, build a Program."""
        if not self._verify(fn, pairs):
            return []
        loo_pass = False
        if refit is not None and self.loo and len(pairs) >= 3 and not self._in_loo:
            loo_pass = True
            self._in_loo = True
            try:
                for index in range(len(pairs)):
                    subset = list(pairs[:index]) + list(pairs[index + 1:])
                    model = refit(subset)
                    if model is None or not _equal(model.apply(pairs[index][0]), pairs[index][1]):
                        loo_pass = False
                        break
            finally:
                self._in_loo = False
        return [Program(family=family, fn=fn, detail=detail, complexity=complexity,
                        loo_pass=loo_pass, meta=dict(meta or {}))]

    # -- family 1: select an object and crop -------------------------------

    def _select_candidates(self, pairs: Sequence[Pair], masked: bool):
        """Search (predicate, D4 transform, colour mode) combinations.

        Object lists and crops are cached per (pair, D4 view) so that the
        predicate sweep costs pure array indexing rather than repeated
        segmentation -- the difference between ~20 s and ~0.2 s per task.
        """
        cache: List[Dict[str, Any]] = []
        for src, _dst in pairs:
            views: Dict[str, Tuple[Grid, List[ArcObject]]] = {}
            for view_name, view in D4:
                view_grid = view(src)
                views[view_name] = (view_grid, content_objects(view_grid))
            cache.append(views)

        for name, predicate in PREDICATES.items():
            for view_name, _view in D4:
                crops: List[Optional[Grid]] = []
                for index_pair, (_src, _dst) in enumerate(pairs):
                    view_grid, objects = cache[index_pair][view_name]
                    obj_index = predicate(objects, view_grid)
                    if obj_index is None:
                        crops = []
                        break
                    obj = objects[obj_index]
                    crops.append(crop_mask(view_grid, obj) if masked else crop_bbox(view_grid, obj))
                if not crops or any(c is None for c in crops):
                    continue
                if any(c.shape != d.shape for c, (_s, d) in zip(crops, pairs)):
                    continue
                for mode in ("identity", "mapped"):
                    mapping = {0: 0} if mode == "identity" else _fit_cell_map(
                        [(c, d) for c, (_s, d) in zip(crops, pairs)])
                    if mapping is None:
                        continue
                    if not all(_equal(_apply_map(c, mapping), d)
                               for c, (_s, d) in zip(crops, pairs)):
                        continue

                    def program(grid: Grid, pred=predicate, vw_name=view_name, mp=mapping,
                                msk=masked) -> Optional[Grid]:
                        """Crop the selected object, transform it and recolour it."""
                        vw = dict(D4)[vw_name]
                        view_grid = vw(grid)
                        objects = content_objects(view_grid)
                        index = pred(objects, view_grid)
                        if index is None:
                            return None
                        obj = objects[index]
                        crop = crop_mask(view_grid, obj) if msk else crop_bbox(view_grid, obj)
                        return _apply_map(crop, mp)

                    detail = f"{'mask' if masked else 'bbox'}:{name}:{view_name}:{mode}"
                    meta = {"predicate": name, "view": view_name, "colour_mode": mode,
                            "n_objects": len(cache[0][view_name][1]) if cache else 0}
                    yield program, detail, 3, meta

    def _fit_select_object(self, pairs: Sequence[Pair]) -> List[Program]:
        """Output = cropped, transformed, recoloured selected object."""
        found: List[Program] = []
        for program, detail, complexity, meta in self._select_candidates(pairs, masked=True):
            refit = lambda sub, m=True: self._first_select(sub, m)
            found.extend(self._emit("select_crop", program, detail, pairs, refit,
                                    complexity, meta))
            if len(found) >= self.max_per_family:
                break
        return found

    def _fit_select_bbox(self, pairs: Sequence[Pair]) -> List[Program]:
        """Output = full-grid bbox crop of the selected object."""
        found: List[Program] = []
        for program, detail, complexity, meta in self._select_candidates(pairs, masked=False):
            refit = lambda sub, m=False: self._first_select(sub, m)
            found.extend(self._emit("select_bbox", program, detail, pairs, refit,
                                    complexity, meta))
            if len(found) >= self.max_per_family:
                break
        return found

    def _first_select(self, pairs: Sequence[Pair], masked: bool) -> Optional[Program]:
        """Refit helper for LOO: the first selection program that fits."""
        for program, detail, complexity, meta in self._select_candidates(pairs, masked):
            if self._verify(program, pairs):
                return Program("select_crop" if masked else "select_bbox", program,
                               detail, complexity, meta=meta)
        return None

    # -- family 3: per-object recolouring by one attribute rule ------------

    @staticmethod
    def _rule_values(objects: Sequence[ArcObject], grid: Grid, kind: str) -> List[Any]:
        """Feature value per object for one attribute rule."""
        sizes = [o.size for o in objects]
        order = sorted(range(len(objects)), key=lambda i: (-sizes[i], i))
        rank = {index: position / max(len(objects) - 1, 1) for position, index in enumerate(order)}
        counts: Dict[bytes, int] = {}
        colors: Dict[int, int] = {}
        for obj in objects:
            counts[obj.shape_key] = counts.get(obj.shape_key, 0) + 1
            colors[obj.color] = colors.get(obj.color, 0) + 1
        values: List[Any] = []
        for index, obj in enumerate(objects):
            if kind == "constant":
                values.append(0)
            elif kind == "color":
                values.append(obj.color)
            elif kind == "size":
                values.append(obj.size)
            elif kind == "size_rank":
                values.append(round(rank[index], 3))
            elif kind == "size_rank_rev":
                values.append(round(1.0 - rank[index], 3))
            elif kind == "holes":
                values.append(obj.holes)
            elif kind == "border":
                values.append(bool(obj.touches_border))
            elif kind == "symmetry":
                values.append(obj.symmetric)
            elif kind == "shape":
                values.append(obj.shape_key)
            elif kind == "n_colors":
                values.append(len(obj.colors))
            elif kind == "shape_count":
                values.append(counts[obj.shape_key])
            elif kind == "color_count":
                values.append(colors.get(obj.color, 0))
            elif kind == "fill":
                values.append(round(obj.fill_ratio, 2))
            elif kind == "is_rect":
                values.append(bool(obj.is_rect))
            elif kind == "width":
                values.append(obj.width)
            elif kind == "height":
                values.append(obj.height)
            else:
                raise KeyError(kind)
        return values

    RULES = ("color", "size_rank", "size_rank_rev", "holes", "border", "symmetry",
             "shape", "n_colors", "shape_count", "color_count", "fill", "is_rect",
             "width", "height", "size")

    def _fit_recolour_rule(self, pairs: Sequence[Pair]) -> List[Program]:
        """Recolour objects by a single attribute rule consistent across pairs."""
        if any(src.shape != dst.shape for src, dst in pairs):
            return []
        for multicolor in (False, True):
            for rule in self.RULES:
                for blank in (False, True):
                    observations: List[Tuple[Any, int]] = []
                    ok = True
                    for src, dst in pairs:
                        objects = segment(src, same_color=not multicolor, connectivity=4)
                        values = self._rule_values(objects, src, rule)
                        for index, obj in enumerate(objects):
                            targets = dst[obj.mask]
                            colors, counts = np.unique(targets, return_counts=True)
                            if counts.max() / counts.sum() < 0.95:
                                ok = False
                                break
                            observations.append((values[index], int(colors[counts.argmax()])))
                        if not ok:
                            break
                    if not ok or len(observations) < 2:
                        continue
                    mapping = _consistent_map(observations)
                    if mapping is None:
                        continue

                    def program(grid: Grid, rule_name=rule, table=mapping,
                                mc=multicolor, blank_bg=blank) -> Optional[Grid]:
                        """Repaint every object with its rule-predicted colour."""
                        objects = segment(grid, same_color=not mc, connectivity=4)
                        if not objects:
                            return None
                        values = self._rule_values(objects, grid, rule_name)
                        bg = dsl.background_color(grid)
                        out = np.full(grid.shape, bg, dtype=np.int8) if blank_bg else dsl.as_grid(grid)
                        for value, obj in zip(values, objects):
                            if value not in table:
                                return None
                            out[obj.mask] = table[value]
                        return out

                    meta = {"rule": rule, "multicolor": multicolor, "blank": blank,
                            "n_objects": len(observations)}
                    refit = lambda sub, mc=multicolor: self._first_recolour(sub, mc)
                    found = self._emit("recolour_rule", program,
                                       f"{rule}:multicolor={multicolor}:blank={blank}",
                                       pairs, refit, 2 + len(mapping), meta)
                    if found:
                        return found
        return []

    def _first_recolour(self, pairs: Sequence[Pair], multicolor: bool) -> Optional[Program]:
        """Refit helper for LOO over recolouring rules."""
        found = [p for p in self._fit_recolour_rule(pairs)
                 if p.meta.get("multicolor") == multicolor]
        return found[0] if found else None

    # -- family 4: translate every object by one offset --------------------

    def _match_objects(self, src_objects: Sequence[ArcObject], dst_objects: Sequence[ArcObject],
                       src: Grid, dst: Grid) -> Optional[List[Tuple[int, int]]]:
        """Greedily match destination objects to source objects by signature."""
        used = set()
        pairs: List[Tuple[int, int]] = []
        for dst_obj in dst_objects:
            best: Optional[Tuple[float, int]] = None
            for index, src_obj in enumerate(src_objects):
                if index in used or src_obj.shape_key != dst_obj.shape_key:
                    continue
                distance = abs(src_obj.centroid[0] - dst_obj.centroid[0]) + \
                    abs(src_obj.centroid[1] - dst_obj.centroid[1])
                if best is None or distance < best[0]:
                    best = (distance, index)
            if best is None:
                return None
            used.add(best[1])
            pairs.append((best[1], dst_objects.index(dst_obj)))
        return pairs

    def _fit_move_all(self, pairs: Sequence[Pair]) -> List[Program]:
        """Every object shifted by one shared (dr, dc), border-clipped."""
        offsets: List[Tuple[int, int]] = []
        for src, dst in pairs:
            if src.shape != dst.shape:
                return []
            src_objects = content_objects(src)
            dst_objects = content_objects(dst)
            if not src_objects or len(src_objects) != len(dst_objects):
                return []
            if [o.shape_key for o in src_objects] != [o.shape_key for o in dst_objects]:
                return []
            dr = dst_objects[0].centroid[0] - src_objects[0].centroid[0]
            dc = dst_objects[0].centroid[1] - src_objects[0].centroid[1]
            offsets.append((int(round(dr)), int(round(dc))))
        unique = set(offsets)
        if len(unique) != 1:
            return []
        dr, dc = offsets[0]
        if dr == 0 and dc == 0:
            return []

        def program(grid: Grid, shift_r=dr, shift_c=dc) -> Optional[Grid]:
            """Translate every foreground object by the learned offset."""
            bg = dsl.background_color(grid)
            out = np.full(grid.shape, bg, dtype=np.int8)
            h, w = grid.shape
            for obj in content_objects(grid):
                for r, c in obj.cells:
                    nr, nc = r + shift_r, c + shift_c
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = grid[r, c]
            return out

        return self._emit("move_all", program, f"shift=({dr},{dc})", pairs, None, 2,
                          {"shift_r": dr, "shift_c": dc})

    # -- family 5: duplicate every object by one offset --------------------

    def _fit_duplicate_offset(self, pairs: Sequence[Pair]) -> List[Program]:
        """Copy every object by one shared offset while keeping the original."""
        offsets: List[Tuple[int, int]] = []
        for src, dst in pairs:
            if src.shape != dst.shape:
                return []
            src_objects = content_objects(src)
            dst_objects = content_objects(dst)
            if not src_objects or len(dst_objects) != 2 * len(src_objects):
                return []
            extra = [o for o in dst_objects if o.size]
            keyed = {(o.shape_key, round(o.centroid[0], 1), round(o.centroid[1], 1))
                     for o in src_objects}
            news = [o for o in extra
                    if (o.shape_key, round(o.centroid[0], 1), round(o.centroid[1], 1)) not in keyed]
            if len(news) != len(src_objects):
                return []
            deltas = set()
            for obj in news:
                best = min(src_objects, key=lambda s: (abs(s.centroid[0] - obj.centroid[0]) +
                                                       abs(s.centroid[1] - obj.centroid[1])))
                deltas.add((int(round(obj.centroid[0] - best.centroid[0])),
                            int(round(obj.centroid[1] - best.centroid[1]))))
            if len(deltas) != 1:
                return []
            offsets.append(next(iter(deltas)))
        if len(set(offsets)) != 1:
            return []
        dr, dc = offsets[0]
        if dr == 0 and dc == 0:
            return []

        def program(grid: Grid, shift_r=dr, shift_c=dc) -> Optional[Grid]:
            """Keep the grid and add a translated copy of every object."""
            out = dsl.as_grid(grid)
            h, w = out.shape
            for obj in content_objects(grid):
                for r, c in obj.cells:
                    nr, nc = r + shift_r, c + shift_c
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = grid[r, c]
            return out

        return self._emit("duplicate_offset", program, f"copy_shift=({dr},{dc})", pairs,
                          None, 2, {"shift_r": dr, "shift_c": dc})

    # -- family 6: delete objects matching a predicate ---------------------

    DELETE_PREDICATES = (
        "max_size", "min_size", "border", "interior", "has_holes", "odd_shape",
        "odd_color", "symmetric", "asymmetric", "max_holes", "max_colors",
        "min_colors", "square", "non_square", "rect",
    )

    def _fit_delete_rule(self, pairs: Sequence[Pair]) -> List[Program]:
        """Erase objects matching one selection predicate."""
        if any(src.shape != dst.shape for src, dst in pairs):
            return []
        for name in self.DELETE_PREDICATES:
            for mode in ("delete", "keep"):
                ok = True
                for src, dst in pairs:
                    objects = content_objects(src)
                    index = PREDICATES[name](objects, src)
                    if index is None:
                        ok = False
                        break
                    predicted = dsl.as_grid(src)
                    if mode == "delete":
                        predicted[objects[index].mask] = dsl.background_color(src)
                    else:
                        for pos, obj in enumerate(objects):
                            if pos != index:
                                predicted[obj.mask] = dsl.background_color(src)
                    if not _equal(predicted, dst):
                        ok = False
                        break
                if not ok:
                    continue

                def program(grid: Grid, pred=PREDICATES[name], keep=(mode == "keep")) -> Optional[Grid]:
                    """Erase (or keep only) the object chosen by the predicate."""
                    objects = content_objects(grid)
                    index = pred(objects, grid)
                    if index is None:
                        return None
                    out = dsl.as_grid(grid)
                    bg = dsl.background_color(grid)
                    if keep:
                        for pos, obj in enumerate(objects):
                            if pos != index:
                                out[obj.mask] = bg
                    else:
                        out[objects[index].mask] = bg
                    return out

                meta = {"predicate": name, "mode": mode}
                refit = lambda sub, n=name, m=mode: self._first_delete(sub, n, m)
                found = self._emit("delete_rule", program, f"{name}:{mode}", pairs, refit, 3, meta)
                if found:
                    return found
        return []

    def _first_delete(self, pairs: Sequence[Pair], name: str, mode: str) -> Optional[Program]:
        """Refit helper for LOO over deletion rules."""
        for candidate in self._fit_delete_rule(pairs):
            if candidate.meta.get("predicate") == name and candidate.meta.get("mode") == mode:
                return candidate
        return None

    # -- family 7: fill enclosed background --------------------------------

    def _fit_fill_holes(self, pairs: Sequence[Pair]) -> List[Program]:
        """Fill enclosed background regions with a learned colour."""
        colors: List[int] = []
        for src, dst in pairs:
            if src.shape != dst.shape:
                return []
            enclosed = dsl.enclosed_mask(src)
            if not enclosed.any():
                return []
            values = np.unique(dst[enclosed])
            if values.size != 1:
                return []
            allowed = src == dsl.background_color(src)
            if not np.array_equal(dst[allowed & ~enclosed], src[allowed & ~enclosed]):
                return []
            if not np.array_equal(dst[~allowed], src[~allowed]):
                return []
            colors.append(int(values[0]))
        if len(set(colors)) != 1:
            return []
        color = colors[0]

        def program(grid: Grid, fill=color) -> Optional[Grid]:
            """Paint every enclosed background region with the fill colour."""
            out = dsl.as_grid(grid)
            out[dsl.enclosed_mask(grid)] = fill
            return out

        return self._emit("fill_holes", program, f"colour={color}", pairs, None, 1,
                          {"colour": color})

    # -- family 8: object outlines -----------------------------------------

    def _fit_outline(self, pairs: Sequence[Pair]) -> List[Program]:
        """Replace each object by its bounding-box outline on the background."""
        for variant in ("bbox", "shape"):
            ok = True
            for src, dst in pairs:
                if src.shape != dst.shape:
                    ok = False
                    break
                bg = dsl.background_color(src)
                predicted = np.full(src.shape, bg, dtype=np.int8)
                for obj in content_objects(src):
                    r0, c0, r1, c1 = obj.box
                    if variant == "bbox":
                        predicted[r0, c0:c1] = src[obj.mask][0] if obj.size else bg
                        predicted[r1 - 1, c0:c1] = src[obj.mask][0] if obj.size else bg
                        predicted[r0:r1, c0] = src[obj.mask][0] if obj.size else bg
                        predicted[r0:r1, c1 - 1] = src[obj.mask][0] if obj.size else bg
                    else:
                        inner = obj.mask.copy()
                        inner[1:-1, 1:-1] = False
                        values = src[obj.mask]
                        predicted[inner] = values[0] if values.size else bg
                if not _equal(predicted, dst):
                    ok = False
                    break
            if ok:
                def program(grid: Grid, mode=variant) -> Optional[Grid]:
                    """Draw bbox or shape outlines of every object."""
                    bg = dsl.background_color(grid)
                    out = np.full(grid.shape, bg, dtype=np.int8)
                    for obj in content_objects(grid):
                        r0, c0, r1, c1 = obj.box
                        color = int(grid[obj.mask][0]) if obj.size else bg
                        if mode == "bbox":
                            out[r0, c0:c1] = color
                            out[r1 - 1, c0:c1] = color
                            out[r0:r1, c0] = color
                            out[r0:r1, c1 - 1] = color
                        else:
                            inner = obj.mask.copy()
                            inner[1:-1, 1:-1] = False
                            out[inner] = color
                    return out

                return self._emit("outline", program, variant, pairs, None, 2)
        return []

    # -- family 9: stamp template at markers -------------------------------

    def _fit_stamp(self, pairs: Sequence[Pair]) -> List[Program]:
        """Stamp a unique template object at every singleton marker cell."""
        plans: List[Tuple[Tuple[int, ...], int, int]] = []
        for src, dst in pairs:
            if src.shape != dst.shape:
                return []
            objects = content_objects(src)
            if len(objects) < 2:
                return []
            templates = [o for o in objects if o.size >= 3 and o.size > 1]
            markers = [o for o in objects if o.size <= 2]
            if len(templates) < 1 or len(markers) < 1:
                return []
            template = max(templates, key=lambda o: o.size)
            if any(t.shape_key == template.shape_key for t in templates if t is not template):
                return []
            relative = tuple(sorted((r - template.box[0], c - template.box[1])
                                    for r, c in template.cells))
            plans.append((relative, len(markers), int(template.colors[0])))
        if len({p[2] for p in plans}) != 1 or len({p[0] for p in plans}) != 1:
            return []
        relative, _, color = plans[0]
        template_h = max(r for r, _ in relative) + 1
        template_w = max(c for _, c in relative) + 1

        def program(grid: Grid, cells=relative, fill=color,
                    th=template_h, tw=template_w) -> Optional[Grid]:
            """Copy the template shape onto every marker cell."""
            objects = content_objects(grid)
            if len(objects) < 2:
                return None
            markers = [o for o in objects if o.size <= 2]
            out = dsl.as_grid(grid)
            h, w = out.shape
            for marker in markers:
                r, c = marker.cells[0]
                r0, c0 = r - th // 2, c - tw // 2
                for dr, dc in cells:
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = fill
            return out

        return self._emit("stamp", program, f"template={len(relative)}cells", pairs, None, 3)

    # -- family 10: connect collinear objects ------------------------------

    def _fit_connect(self, pairs: Sequence[Pair]) -> List[Program]:
        """Join object centres with horizontal/vertical lines."""
        for grouping in ("colour", "all", "same_shape"):
            for axis in ("row", "col", "both"):
                for color_mode in ("object", "constant"):
                    ok = True
                    constant: Optional[int] = None
                    for src, dst in pairs:
                        if src.shape != dst.shape:
                            ok = False
                            break
                        objects = content_objects(src)
                        if len(objects) < 2:
                            ok = False
                            break
                        out = dsl.as_grid(src)
                        for a in range(len(objects)):
                            for b in range(a + 1, len(objects)):
                                oa, ob = objects[a], objects[b]
                                if grouping == "colour" and oa.color != ob.color:
                                    continue
                                if grouping == "same_shape" and oa.shape_key != ob.shape_key:
                                    continue
                                if axis in ("row", "both") and \
                                        abs(oa.centroid[0] - ob.centroid[0]) < 0.8:
                                    row = int(round(oa.centroid[0]))
                                    c0, c1 = sorted((int(round(oa.centroid[1])),
                                                     int(round(ob.centroid[1]))))
                                    line_color = oa.color if color_mode == "object" else -1
                                    for c in range(c0 + 1, c1):
                                        if out[row, c] == dsl.background_color(src):
                                            if color_mode == "constant":
                                                constant = constant if constant is not None else \
                                                    int(dst[row, c])
                                            out[row, c] = line_color if color_mode == "object" else (constant or 0)
                                if axis in ("col", "both") and \
                                        abs(oa.centroid[1] - ob.centroid[1]) < 0.8:
                                    col = int(round(oa.centroid[1]))
                                    r0, r1 = sorted((int(round(oa.centroid[0])),
                                                     int(round(ob.centroid[0]))))
                                    line_color = oa.color if color_mode == "object" else -1
                                    for r in range(r0 + 1, r1):
                                        if out[r, col] == dsl.background_color(src):
                                            if color_mode == "constant":
                                                constant = constant if constant is not None else \
                                                    int(dst[r, col])
                                            out[r, col] = line_color if color_mode == "object" else (constant or 0)
                        if not _equal(out, dst):
                            ok = False
                            break
                    if not ok:
                        continue

                    def program(grid: Grid, grp=grouping, ax=axis, cm=color_mode,
                                const=constant) -> Optional[Grid]:
                        """Draw connector lines between objects."""
                        objects = content_objects(grid)
                        if len(objects) < 2:
                            return None
                        out = dsl.as_grid(grid)
                        bg = dsl.background_color(grid)
                        for a in range(len(objects)):
                            for b in range(a + 1, len(objects)):
                                oa, ob = objects[a], objects[b]
                                if grp == "colour" and oa.color != ob.color:
                                    continue
                                if grp == "same_shape" and oa.shape_key != ob.shape_key:
                                    continue
                                color = oa.color if cm == "object" else (const or 0)
                                if ax in ("row", "both") and abs(oa.centroid[0] - ob.centroid[0]) < 0.8:
                                    row = int(round(oa.centroid[0]))
                                    lo, hi = sorted((int(round(oa.centroid[1])),
                                                     int(round(ob.centroid[1]))))
                                    for c in range(lo + 1, hi):
                                        if out[row, c] == bg:
                                            out[row, c] = color
                                if ax in ("col", "both") and abs(oa.centroid[1] - ob.centroid[1]) < 0.8:
                                    col = int(round(oa.centroid[1]))
                                    lo, hi = sorted((int(round(oa.centroid[0])),
                                                     int(round(ob.centroid[0]))))
                                    for r in range(lo + 1, hi):
                                        if out[r, col] == bg:
                                            out[r, col] = color
                        return out

                    found = self._emit("connect", program,
                                       f"{grouping}:{axis}:{color_mode}", pairs, None, 3)
                    if found:
                        return found
        return []
