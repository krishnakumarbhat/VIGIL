"""Bidirectional DSL search plus test-time-learned neuro-symbolic hypotheses.

The solver is a cascade of complementary strategies, each of which must fit
*every* training pair exactly before it may touch a test input:

1. ``depth1``    - single primitive from :mod:`src.dsl_primitives`.
2. ``bidir``     - forward states from inputs meet inverse states from outputs,
                   synthesising short compositions (true bidirectional search).
3. ``depth2``    - ordered pair enumeration with a fast right-hand operator set,
                   verified pair-by-pair (pair 0 acts as an O(1) filter).
4. ``mosaic``    - output is a k x m mosaic of transformed copies of the input.
5. ``cell_tree`` - per-cell decision tree over local 3x3 neighbourhood features.
6. ``object_tree`` - per-component decision tree over shape/colour properties.
7. ``mlp``       - small torch MLP over the same cell features (neuro-symbolic
                   test-time adaptation; accepted only on a perfect train fit).

All strategies are pure functions of the task, so the runner can execute tasks
in parallel worker processes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.canonical import canonical_key, shape_relation
from src.cell_rule_learner import CellRuleLearner
from src.mlp_cell_learner import MlpCellLearner
from src.object_rule_learner import ObjectRuleLearner
from src.patch_lookup_learner import PatchLookupLearner

Grid = np.ndarray
Pair = Tuple[Grid, Grid]



# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ArcTask:
    """One ARC task: training pairs plus the test inputs to predict.

    ``test_outputs`` is populated when the JSON happens to include gold test
    answers (the bundled offline data does).  The solver never reads it; it is
    used only by the runner to grade predictions after the fact.
    """

    task_id: str
    train: List[Pair]
    test_inputs: List[Grid]
    source: str = "unknown"
    test_outputs: Optional[List[Grid]] = None

    @property
    def signature(self) -> str:
        """Coarse task family label used for the metrics breakdown."""
        ins = [a for a, _ in self.train]
        outs = [b for _, b in self.train]
        relation = shape_relation(ins, outs)
        max_dim = max(max(a.shape[0], a.shape[1]) for a in ins)
        bucket = "small" if max_dim <= 10 else ("medium" if max_dim <= 20 else "large")
        return f"{relation}|{bucket}|ntr={min(len(self.train), 5)}"


@dataclass
class Hypothesis:
    """A verified program: its name, the callable, and the produced test grids."""

    method: str
    outputs: List[Grid]
    detail: str = ""


@dataclass
class SolverConfig:
    """Runtime knobs for :class:`DslSolver`."""

    time_budget: float = 12.0
    max_depth: int = 2
    max_hypotheses: int = 2
    enable_depth2: bool = True
    enable_neuro: bool = True
    enable_learners: bool = True
    pair_filter: bool = True

    def budget_for(self, n_train: int, n_test: int) -> float:
        """Scale the budget with the number of pairs (more pairs, more cost)."""
        factor = 1.0 + 0.12 * max(0, n_train - 2)
        return self.time_budget * factor


@dataclass
class TaskResult:
    """Outcome of solving one task."""

    task_id: str
    signature: str
    solved: bool
    method: str
    runtime_s: float
    attempts: List[List[Grid]] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Loading / serialisation helpers
# ---------------------------------------------------------------------------


def load_tasks(data_dir: str | Path, split: str = "training", limit: Optional[int] = None) -> List[ArcTask]:
    """Load ARC task JSON files from ``data_dir/<split>`` sorted by filename."""
    root = Path(data_dir) / split
    files = sorted(root.glob("*.json"))
    if limit:
        files = files[:limit]
    tasks: List[ArcTask] = []
    for path in files:
        raw = json.loads(path.read_text())
        train = [(dsl.as_grid(np.asarray(p["input"])), dsl.as_grid(np.asarray(p["output"]))) for p in raw["train"]]
        test_inputs = [dsl.as_grid(np.asarray(p["input"])) for p in raw["test"]]
        gold = [dsl.as_grid(np.asarray(p["output"])) for p in raw["test"] if "output" in p]
        tasks.append(ArcTask(task_id=path.stem, train=train, test_inputs=test_inputs, source=split,
                             test_outputs=gold if len(gold) == len(test_inputs) else None))
    return tasks


def _content_bbox(grid: Grid) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of every non-background cell."""
    return dsl.bbox(grid != dsl.background_color(grid))


def _object_bbox(grid: Grid, index: int, multicolor: bool) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the ``index``-th component (size order), if any."""
    masks = dsl.components(grid, connectivity=4, same_color=not multicolor)
    if len(masks) <= index:
        return None
    return dsl.bbox(masks[index])


def _rarest_color_bbox(grid: Grid) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the least frequent colour."""
    counts = dsl.color_counts(grid)
    present = np.flatnonzero(counts)
    if present.size < 2:
        return None
    color = int(present[np.argmin(counts[present])])
    return dsl.bbox(grid == color)


SHAPE_ROWS: Tuple[Tuple[str, Any], ...] = (
    ("height", lambda g: g.shape[0]),
    ("width", lambda g: g.shape[1]),
    ("content_h", lambda g: (lambda b: b[2] - b[0] if b else None)(_content_bbox(g))),
    ("content_w", lambda g: (lambda b: b[3] - b[1] if b else None)(_content_bbox(g))),
    ("largest_h", lambda g: (lambda b: b[2] - b[0] if b else None)(_object_bbox(g, 0, False))),
    ("largest_w", lambda g: (lambda b: b[3] - b[1] if b else None)(_object_bbox(g, 0, False))),
    ("largest_multi_h", lambda g: (lambda b: b[2] - b[0] if b else None)(_object_bbox(g, 0, True))),
    ("largest_multi_w", lambda g: (lambda b: b[3] - b[1] if b else None)(_object_bbox(g, 0, True))),
    ("rarest_h", lambda g: (lambda b: b[2] - b[0] if b else None)(_rarest_color_bbox(g))),
    ("rarest_w", lambda g: (lambda b: b[3] - b[1] if b else None)(_rarest_color_bbox(g))),
    ("content_rows", lambda g: int((g != dsl.background_color(g)).any(axis=1).sum())),
    ("content_cols", lambda g: int((g != dsl.background_color(g)).any(axis=0).sum())),
    ("half_h", lambda g: g.shape[0] // 2 if g.shape[0] % 2 == 0 else None),
    ("half_w", lambda g: g.shape[1] // 2 if g.shape[1] % 2 == 0 else None),
    ("double_h", lambda g: g.shape[0] * 2 if g.shape[0] * 2 <= 30 else None),
    ("double_w", lambda g: g.shape[1] * 2 if g.shape[1] * 2 <= 30 else None),
    ("n_objects", lambda g: len(dsl.components(g, connectivity=4, same_color=False))),
    ("n_same_color_objects", lambda g: len(dsl.components(g, connectivity=4, same_color=True))),
    ("n_colors", lambda g: int((dsl.color_counts(g) > 0).sum())),
    ("n_fg_colors", lambda g: int((dsl.color_counts(g) > 0).sum()) - 1),
    ("max_color_count", lambda g: int(dsl.color_counts(g).max())),
)


def fit_shape_model(task: ArcTask) -> Tuple[Optional[Any], Optional[Any]]:
    """Learn row/column predictors that reproduce every training output shape.

    Row and column rules are selected independently, which covers tasks whose
    output is, e.g., the bounding box of the largest object on one axis and the
    full grid on the other.  Returns ``(row_fn, col_fn)`` with ``None`` entries
    when no candidate generalises.
    """
    row_fn: Optional[Any] = None
    for name, fn in SHAPE_ROWS:
        try:
            if all(fn(src) == dst.shape[0] for src, dst in task.train):
                row_fn = fn
                break
        except Exception:
            continue
    col_fn: Optional[Any] = None
    for name, fn in SHAPE_ROWS:
        try:
            if all(fn(src) == dst.shape[1] for src, dst in task.train):
                col_fn = fn
                break
        except Exception:
            continue
    return row_fn, col_fn


def expected_output_shape(task: ArcTask, index: int = 0) -> Tuple[int, int]:
    """Best guess of a test output shape when no hypothesis is available."""
    shapes = [b.shape for _, b in task.train]
    # Constant output shape wins (many ARC tasks expand to a fixed canvas).
    if len(set(shapes)) == 1:
        return shapes[0]
    test = task.test_inputs[index]
    row_fn, col_fn = fit_shape_model(task)
    rows = row_fn(test) if row_fn is not None else None
    cols = col_fn(test) if col_fn is not None else None
    if rows and cols and 0 < rows <= 30 and 0 < cols <= 30:
        return int(rows), int(cols)
    if rows and 0 < rows <= 30:
        return int(rows), test.shape[1]
    if cols and 0 < cols <= 30:
        return test.shape[0], int(cols)
    # Constant integer scale factor (upscale / tiling tasks).
    factors: set[Tuple[float, float]] = set()
    for src, dst in task.train:
        if src.shape[0] and src.shape[1]:
            factors.add((round(dst.shape[0] / src.shape[0], 3),
                         round(dst.shape[1] / src.shape[1], 3)))
    if len(factors) == 1:
        fr, fc = factors.pop()
        if fr == int(fr) and fc == int(fc) and 0 < fr <= 30 and 0 < fc <= 30:
            return int(test.shape[0] * fr), int(test.shape[1] * fc)
    return test.shape


def fallback_grid(task: ArcTask, index: int = 0) -> Grid:
    """Majority-colour grid of the expected shape for test input ``index``."""
    test = task.test_inputs[index]
    color = dsl.background_color(test)
    shape = expected_output_shape(task, index)
    return np.full(shape, color, dtype=np.int8)


def task_from_json(task_id: str, payload: Dict[str, Any], source: str = "inline") -> ArcTask:
    """Build an :class:`ArcTask` from a raw ARC JSON payload."""
    train = [(dsl.as_grid(np.asarray(p["input"])), dsl.as_grid(np.asarray(p["output"]))) for p in payload["train"]]
    tests = [dsl.as_grid(np.asarray(p["input"])) for p in payload["test"]]
    gold = [dsl.as_grid(np.asarray(p["output"])) for p in payload["test"] if "output" in p]
    return ArcTask(task_id=task_id, train=train, test_inputs=tests, source=source,
                   test_outputs=gold if len(gold) == len(tests) else None)


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------


def fits_all(fn, pairs: Sequence[Pair]) -> bool:
    """True when ``fn`` maps every training input to its output exactly."""
    try:
        for src, dst in pairs:
            out = fn(src)
            if out is None or out.shape != dst.shape or not np.array_equal(out, dst):
                return False
    except Exception:
        return False
    return True


def sanitize(grid: Optional[Grid]) -> Optional[Grid]:
    """Clamp to 0..9, force int8 and reject degenerate grids."""
    if grid is None:
        return None
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    arr = np.clip(arr, 0, 9).astype(np.int8)
    return np.ascontiguousarray(arr)


def distinct_grids(grids: Iterable[Grid]) -> List[Grid]:
    """Deduplicate grids by exact bytes."""
    seen: set[bytes] = set()
    out: List[Grid] = []
    for grid in grids:
        key = np.ascontiguousarray(grid, dtype=np.int8).tobytes() + bytes(grid.shape)
        if key not in seen:
            seen.add(key)
            out.append(grid)
    return out


# ---------------------------------------------------------------------------
# Hypothesis builders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class DslSolver:
    """Cascade solver combining symbolic search with test-time learned rules."""

    def __init__(self, config: Optional[SolverConfig] = None) -> None:
        """Create a solver; ``config`` controls budget, depth and ablations."""
        self.config = config or SolverConfig()
        self._cell_learner = CellRuleLearner()
        self._object_learner = ObjectRuleLearner()
        self._patch_learner = PatchLookupLearner()
        self._mlp = MlpCellLearner(max_seconds=2.5)

    # -- public API --------------------------------------------------------

    def solve(self, task: ArcTask) -> TaskResult:
        """Solve ``task`` and return attempts for every test input."""
        start = time.monotonic()
        budget = self.config.budget_for(len(task.train), len(task.test_inputs))
        deadline = start + budget
        hypotheses: List[Hypothesis] = []
        notes: List[str] = []

        strategies = [
            ("depth1", self._search_depth1),
            ("bidir", self._search_bidirectional),
            ("depth2", self._search_depth2),
            ("mosaic", self._search_mosaic),
            ("cell_tree", self._search_cell_tree),
            ("object_tree", self._search_object_tree),
            ("patch_lookup", self._search_patch_lookup),
            ("cell_tree_d4", self._search_cell_tree_d4),
            ("object_tree_d4", self._search_object_tree_d4),
            ("mlp", self._search_mlp),
        ]
        for name, strategy in strategies:
            if len(hypotheses) >= self.config.max_hypotheses:
                break
            if time.monotonic() >= deadline:
                notes.append(f"deadline_before_{name}")
                break
            # Once one hypothesis is in hand, a second opinion only gets a
            # short slice: it must not eat the budget of the remaining tasks.
            strategy_deadline = deadline
            if hypotheses:
                strategy_deadline = min(deadline, time.monotonic() + 1.5)
            if name == "depth2" and not self.config.enable_depth2:
                continue
            if name in ("cell_tree", "object_tree", "patch_lookup",
                        "cell_tree_d4", "object_tree_d4") and not self.config.enable_learners:
                continue
            if name == "mlp" and not self.config.enable_neuro:
                continue
            try:
                found = strategy(task, strategy_deadline)
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"{name}_error:{type(exc).__name__}")
                continue
            for hypothesis in found:
                duplicate = any(
                    all(np.array_equal(hypothesis.outputs[i], existing.outputs[i])
                        for i in range(len(task.test_inputs)))
                    for existing in hypotheses
                )
                if duplicate:
                    continue
                hypothesis.method = name
                if len(hypotheses) < self.config.max_hypotheses:
                    hypotheses.append(hypothesis)

        attempts: List[List[Grid]] = []
        for index in range(len(task.test_inputs)):
            fb = fallback_grid(task, index)
            first = sanitize(hypotheses[0].outputs[index]) if hypotheses else None
            second = sanitize(hypotheses[1].outputs[index]) if len(hypotheses) > 1 else None
            attempts.append([first if first is not None else fb,
                             second if second is not None else fb])

        method = hypotheses[0].method if hypotheses else "fallback"
        if len(hypotheses) > 1:
            method = f"{hypotheses[0].method}+{hypotheses[1].method}"
        return TaskResult(task_id=task.task_id, signature=task.signature, solved=bool(hypotheses),
                          method=method, runtime_s=time.monotonic() - start, attempts=attempts,
                          notes=";".join(notes))

    def solve_json(self, task_id: str, payload: Dict[str, Any]) -> TaskResult:
        """Convenience wrapper for inline JSON tasks (used by the notebook)."""
        return self.solve(task_from_json(task_id, payload))

    # -- strategy 1: single primitive -------------------------------------

    def _search_depth1(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Try every enumerated primitive as a complete program."""
        primitives = dsl.enumerate_primitives(task.train)
        found: List[Hypothesis] = []
        for name, fn in primitives:
            if time.monotonic() >= deadline:
                break
            if not fits_all(fn, task.train):
                continue
            outputs = [sanitize(fn(t)) for t in task.test_inputs]
            if any(o is None for o in outputs):
                continue
            found.append(Hypothesis(method="depth1", outputs=[o for o in outputs if o is not None],
                                    detail=name))
            if len(found) >= self.config.max_hypotheses:
                break
        return found

    # -- strategy 2: bidirectional search ---------------------------------

    @staticmethod
    def _inverse_ops() -> List[Tuple[str, Any]]:
        """Inverse operators mapping a target grid back towards the input."""
        ops: List[Tuple[str, Any]] = [
            ("id", dsl.as_grid),
            ("rot90", lambda g: np.ascontiguousarray(np.rot90(g, 3))),
            ("rot180", lambda g: np.ascontiguousarray(np.rot90(g, 2))),
            ("rot270", lambda g: np.ascontiguousarray(np.rot90(g, 1))),
            ("fliplr", lambda g: dsl.as_grid(np.fliplr(g))),
            ("flipud", lambda g: dsl.as_grid(np.flipud(g))),
            ("transpose", lambda g: dsl.as_grid(g.T)),
            ("antidiag", lambda g: dsl.as_grid(np.rot90(g, 2).T)),
            ("trim_border", dsl.trim_border),
        ]

        def _inverse_color_map(grid: Grid, mapping: Dict[int, int]) -> Optional[Grid]:
            """Invert an injective colour map, or return ``None``."""
            inverse: Dict[int, int] = {}
            for src, dst in mapping.items():
                if dst in inverse and inverse[dst] != src:
                    return None
                inverse[dst] = src
            return dsl.apply_color_map(grid, inverse)

        for src in range(10):
            for dst in range(10):
                if src != dst:
                    ops.append((f"swap_{src}_{dst}", lambda g, a=src, b=dst: _inverse_color_map(g, {a: b})))
        for k in (2, 3):
            def _split_upscale(grid: Grid, kk: int = k) -> Optional[Grid]:
                """Invert nearest-neighbour upscaling when blocks agree."""
                if grid.shape[0] % kk or grid.shape[1] % kk:
                    return None
                h, w = grid.shape[0] // kk, grid.shape[1] // kk
                first = grid[:h, :w]
                for i in range(kk):
                    for j in range(kk):
                        if not np.array_equal(grid[i * h:(i + 1) * h, j * w:(j + 1) * w], first):
                            return None
                return dsl.as_grid(first)
            ops.append((f"upscale_{k}", _split_upscale))
        return ops

    def _search_bidirectional(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Meet-forward/backward search: find p_b with p_b(output) == p_f(input)."""
        primitives = dsl.enumerate_primitives(task.train, depth1_budget=220)
        inverses = self._inverse_ops()
        src, dst = task.train[0]
        forward: Dict[bytes, Tuple[str, Any]] = {}
        for name, fn in primitives:
            if time.monotonic() >= deadline:
                return []
            state = fn(src)
            if state is None or state.size == 0:
                continue
            key = canonical_key(state)
            forward.setdefault(key, (name, fn))
        backward: Dict[bytes, Tuple[str, Any]] = {}
        for name, fn in inverses:
            if time.monotonic() >= deadline:
                return []
            state = fn(dst)
            if state is None or state.size == 0:
                continue
            key = canonical_key(state)
            backward.setdefault(key, (name, fn))
        found: List[Hypothesis] = []
        shared = set(forward) & set(backward)
        # Prefer shorter-preferable backward ops (D4 first, colour next).
        for key in shared:
            fname, ffn = forward[key]
            bname, bfn = backward[key]

            def program(grid: Grid, f=ffn, b=bfn) -> Optional[Grid]:
                """Compose forward state with the backward operator."""
                mid = f(grid)
                if mid is None or mid.size == 0:
                    return None
                return b(mid)

            if not fits_all(program, task.train):
                continue
            outputs = [sanitize(program(t)) for t in task.test_inputs]
            if any(o is None for o in outputs):
                continue
            found.append(Hypothesis(method="bidir", outputs=[o for o in outputs if o is not None],
                                    detail=f"{bname}({fname}(x))"))
            if len(found) >= self.config.max_hypotheses:
                break
        return found

    # -- strategy 3: depth-2 enumeration ----------------------------------

    def _search_depth2(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Enumerate ordered primitive pairs, verifying pair 0 first."""
        left = dsl.composition_left_set(task.train)
        right = dsl.primitive_fast_set()
        src0, dst0 = task.train[0]
        found: List[Hypothesis] = []
        for lname, lfn in left:
            if time.monotonic() >= deadline:
                break
            mid = lfn(src0)
            if mid is None or mid.size == 0:
                continue
            for rname, rfn in right:
                if time.monotonic() >= deadline:
                    return found
                try:
                    out = rfn(mid)
                except Exception:
                    continue
                if out is None or out.shape != dst0.shape or not np.array_equal(out, dst0):
                    continue
                detail = f"{rname}({lname}(x))"

                def program(grid: Grid, l=lfn, r=rfn) -> Optional[Grid]:
                    """Apply the left then the right primitive."""
                    inter = l(grid)
                    if inter is None or inter.size == 0:
                        return None
                    return r(inter)

                if not fits_all(program, task.train):
                    continue
                outputs = [sanitize(program(t)) for t in task.test_inputs]
                if any(o is None for o in outputs):
                    continue
                found.append(Hypothesis(method="depth2", outputs=[o for o in outputs if o is not None],
                                        detail=detail))
                if len(found) >= self.config.max_hypotheses:
                    return found
        return found

    # -- strategy 4: block mosaic -----------------------------------------

    def _search_mosaic(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Fit output = k x m mosaic of transformed input copies."""
        if time.monotonic() >= deadline:
            return []
        plan = dsl.block_mosaic_fit(task.train)
        if plan is None:
            return []
        outputs = [sanitize(dsl.block_mosaic_apply(t, plan)) for t in task.test_inputs]
        if any(o is None for o in outputs):
            return []
        return [Hypothesis(method="mosaic", outputs=[o for o in outputs if o is not None],
                           detail=f"{plan['rows']}x{plan['cols']}")]

    # -- strategy 5/6/7: learned rules ------------------------------------

    def _search_cell_tree(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Fit a per-cell decision tree (no augmentation, higher coverage)."""
        return self._cell_tree_hypotheses(task, deadline, augment=False, method="cell_tree")

    def _search_cell_tree_d4(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Fit a D4-equivariant cell tree and decode with symmetry voting."""
        return self._cell_tree_hypotheses(task, deadline, augment=True, method="cell_tree_d4")

    def _cell_tree_hypotheses(self, task: ArcTask, deadline: float, augment: bool,
                              method: str) -> List[Hypothesis]:
        """Shared implementation for the plain and D4-augmented cell trees."""
        if time.monotonic() >= deadline:
            return []
        tree = self._cell_learner.fit(task.train, augment=augment)
        if tree is None:
            return []
        outputs = [sanitize(self._cell_learner.predict(tree, t, vote=augment))
                   for t in task.test_inputs]
        if any(o is None for o in outputs):
            return []
        if any(o.shape != t.shape for o, t in zip(outputs, task.test_inputs)):
            return []
        return [Hypothesis(method=method, outputs=[o for o in outputs if o is not None],
                           detail=f"leaves={tree.get_n_leaves()}")]

    def _search_object_tree(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Fit a per-component recolouring tree (no augmentation)."""
        return self._object_tree_hypotheses(task, deadline, augment=False, method="object_tree")

    def _search_object_tree_d4(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Fit a D4-equivariant component tree and decode with symmetry voting."""
        return self._object_tree_hypotheses(task, deadline, augment=True, method="object_tree_d4")

    def _object_tree_hypotheses(self, task: ArcTask, deadline: float, augment: bool,
                                method: str) -> List[Hypothesis]:
        """Shared implementation for the plain and D4-augmented object trees."""
        if time.monotonic() >= deadline:
            return []
        for connectivity in (4, 8):
            model = self._object_learner.fit(task.train, connectivity=connectivity, augment=augment)
            if model is None:
                continue
            outputs = [sanitize(self._object_learner.predict(model, t, vote=augment))
                       for t in task.test_inputs]
            if any(o is None for o in outputs):
                continue
            return [Hypothesis(method=method, outputs=[o for o in outputs if o is not None],
                               detail=f"conn={connectivity}")]
        return []

    def _search_patch_lookup(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Fit an exact 5x5/3x3 local lookup with abstention to the input."""
        if time.monotonic() >= deadline:
            return []
        model = self._patch_learner.fit(task.train)
        if model is None:
            return []
        outputs = [sanitize(self._patch_learner.predict(model, t)) for t in task.test_inputs]
        if any(o is None for o in outputs):
            return []
        if any(o.shape != t.shape for o, t in zip(outputs, task.test_inputs)):
            return []
        return [Hypothesis(method="patch_lookup", outputs=[o for o in outputs if o is not None],
                           detail="r=2")]

    def _search_mlp(self, task: ArcTask, deadline: float) -> List[Hypothesis]:
        """Train a small MLP per task and accept only a perfect train fit."""
        budget = min(self._mlp.max_seconds, max(0.0, deadline - time.monotonic()))
        if budget < 0.4:
            return []
        self._mlp.max_seconds = budget
        bundle = self._mlp.fit(task.train)
        if bundle is None:
            return []
        outputs = [sanitize(self._mlp.predict(bundle, t)) for t in task.test_inputs]
        if any(o is None for o in outputs):
            return []
        if any(o.shape != t.shape for o, t in zip(outputs, task.test_inputs)):
            return []
        return [Hypothesis(method="mlp", outputs=[o for o in outputs if o is not None], detail="mlp128")]


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def solve_task(task: ArcTask, config: Optional[SolverConfig] = None) -> TaskResult:
    """Module-level helper mirroring :meth:`DslSolver.solve`."""
    return DslSolver(config).solve(task)
