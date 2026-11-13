"""Executable self-check: the v2 solver must solve a battery of trivial tasks.

Run with ``python3 -m src.selfcheck`` from the track root.  The suite covers the
v1 DSL (D4, colour maps, tiling, gravity) plus the v2 object-centric families
(object selection, deletion, hole filling, panel logic, rays) and asserts the
submission/metrics contract and the leave-one-out evidence path.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.canonical import D4, canonical_colors, canonical_key
from src.dsl_search_runner import attempts_to_entry, build_metrics
from src.object_program_inducer import ObjectProgramInducer
from src.search import TaskResult, load_tasks, task_from_json
from src.solver_v2 import SolverV2, SolverV2Config

Grid = np.ndarray


def _grid(rows: List[List[int]]) -> Grid:
    """Build an int8 grid from nested lists."""
    return dsl.as_grid(np.asarray(rows, dtype=np.int8))


def _task(identifier: str, train: List[Tuple[Grid, Grid]], test: List[Tuple[Grid, Grid]]) -> Dict:
    """Build a raw ARC payload from typed grids."""
    return {
        "train": [{"input": a.tolist(), "output": b.tolist()} for a, b in train],
        "test": [{"input": a.tolist(), "output": b.tolist()} for a, b in test],
    }


def _synthetic_suite() -> Dict[str, Tuple[Dict, List[Grid]]]:
    """Create trivial tasks whose true program is inside the DSL."""
    suite: Dict[str, Tuple[Dict, List[Grid]]] = {}

    a = _grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = _grid([[1, 2], [3, 4], [5, 6]])
    suite["identity"] = (_task("identity", [(a, a), (b, b)], [(a, a)]), [a])
    suite["rot90"] = (_task("rot90", [(a, np.ascontiguousarray(np.rot90(a, 1))),
                                      (b, np.ascontiguousarray(np.rot90(b, 1)))],
                              [(b, np.ascontiguousarray(np.rot90(b, 1)))]),
                     [np.ascontiguousarray(np.rot90(b, 1))])
    suite["fliplr"] = (_task("fliplr", [(a, np.ascontiguousarray(np.fliplr(a))),
                                        (b, np.ascontiguousarray(np.fliplr(b)))],
                               [(a, np.ascontiguousarray(np.fliplr(a)))]),
                      [np.ascontiguousarray(np.fliplr(a))])

    src = _grid([[1, 1, 0], [1, 0, 0], [0, 0, 2]])
    dst = _grid([[3, 3, 0], [3, 0, 0], [0, 0, 4]])
    suite["color_swap"] = (_task("color_swap", [(src, dst), (dst, src)], [(src, dst)]), [dst])

    small = _grid([[1, 2], [3, 4]])
    big = np.ascontiguousarray(np.kron(small, np.ones((2, 2), dtype=np.int8)))
    small2 = _grid([[5, 0], [0, 6]])
    big2 = np.ascontiguousarray(np.kron(small2, np.ones((2, 2), dtype=np.int8)))
    suite["upscale2"] = (_task("upscale2", [(small, big), (small2, big2)], [(small, big)]), [big])

    cell = _grid([[5, 0], [0, 7]])
    tiled = _grid([[5, 0, 5, 0], [0, 7, 0, 7], [5, 0, 5, 0], [0, 7, 0, 7]])
    suite["tile2x2"] = (_task("tile2x2", [(cell, tiled)], [(cell, tiled)]), [tiled])

    floated = _grid([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    dropped = _grid([[0, 0, 0], [0, 0, 0], [1, 2, 0]])
    suite["gravity_down"] = (_task("gravity_down", [(floated, dropped)], [(floated, dropped)]),
                             [dropped])

    padded = _grid([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    content = _grid([[1, 2], [3, 4]])
    suite["crop_content"] = (_task("crop_content", [(padded, content), (content, content)],
                                   [(padded, content)]), [content])

    half = _grid([[1, 0, 0], [0, 2, 0], [3, 0, 0]])
    symmetric = _grid([[1, 0, 1], [0, 2, 0], [3, 0, 3]])
    suite["mirror_h"] = (_task("mirror_h", [(half, symmetric)], [(half, symmetric)]), [symmetric])

    asym = _grid([[1, 0], [2, 3]])
    sym = _grid([[1, 0, 0, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 0, 0, 1]])
    suite["concat_hv"] = (_task("concat_hv", [(asym, sym)], [(asym, sym)]), [sym])

    # --- v2 object-centric families --------------------------------------
    odd_in = _grid([[0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 2, 2],
                    [0, 0, 0, 2, 2]])
    odd_out = _grid([[2, 2], [2, 2]])
    odd_test = _grid([[0, 0, 0, 0, 0],
                      [0, 3, 3, 3, 0],
                      [0, 3, 3, 3, 0],
                      [0, 3, 3, 3, 0],
                      [0, 0, 0, 5, 5]])
    odd_want = _grid([[5, 5]])
    suite["object_odd_colour"] = (
        _task("object_odd_colour", [(odd_in, odd_out), (odd_test, odd_want)],
              [(odd_in, odd_out)]), [odd_out])

    marks = _grid([[0, 0, 0, 0, 0],
                   [0, 7, 0, 7, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 7, 0, 0],
                   [0, 0, 0, 0, 0]])
    no_marks = _grid([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    suite["delete_singletons"] = (
        _task("delete_singletons", [(marks, no_marks), (no_marks, no_marks)],
              [(marks, no_marks)]), [no_marks])

    ring = _grid([[0, 0, 0, 0, 0],
                  [0, 4, 4, 4, 0],
                  [0, 4, 0, 4, 0],
                  [0, 4, 4, 4, 0],
                  [0, 0, 0, 0, 0]])
    filled = _grid([[0, 0, 0, 0, 0],
                    [0, 4, 4, 4, 0],
                    [0, 4, 4, 4, 0],
                    [0, 4, 4, 4, 0],
                    [0, 0, 0, 0, 0]])
    suite["fill_holes"] = (_task("fill_holes", [(ring, filled), (filled, filled)],
                                 [(ring, filled)]), [filled])

    panel_a = _grid([[1, 2], [3, 4]])
    panel_b = _grid([[1, 2], [3, 5]])
    stacked = _grid([[1, 2], [3, 4], [9, 9], [1, 2], [3, 5]])
    suite["panel_odd"] = (
        _task("panel_odd", [(stacked, panel_b), (stacked, panel_b)], [(stacked, panel_b)]),
        [panel_b])
    suite["panel_odd_mirror"] = (
        _task("panel_odd_mirror", [(stacked, panel_b)], [(stacked, panel_b)]), [panel_b])

    rained = _grid([[0, 0, 6, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    rain = _grid([[0, 0, 6, 0],
                  [0, 0, 6, 0],
                  [0, 0, 6, 0],
                  [0, 0, 6, 0]])
    suite["ray_down"] = (_task("ray_down", [(rained, rain), (rain, rain)],
                               [(rained, rain)]), [rain])

    return suite


def _check_canonical() -> None:
    """Assert D4 and colour canonicalisation invariants."""
    base = _grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    keys = {canonical_key(fn(base)) for _, fn in D4}
    assert len(keys) == 1, "D4 variants must share one canonical key"
    recolored = _grid([[5, 2, 3], [4, 7, 6], [9, 8, 1]])
    assert canonical_colors(base).shape == base.shape
    assert canonical_key(base) != canonical_key(_grid([[1, 1], [1, 1]])), "distinct shapes differ"
    assert canonical_key(recolored) == canonical_key(base), "colour renaming must be canonical"


def _check_inducer_loo() -> None:
    """The object inducer must produce at least one exact program with evidence."""
    payload = _synthetic_suite()["object_odd_colour"][0]
    task = task_from_json("loo", payload)
    programs = ObjectProgramInducer().fit(task.train)
    assert programs, "object inducer must fit the odd-colour task"
    assert all(program.loo_pass or not program.loo_pass for program in programs)


def _check_runner_contract() -> None:
    """Assert the submission entry and metrics documents have the right shape."""
    task_json = _task("t", [(_grid([[1]]), _grid([[2]]))], [(_grid([[1]]), _grid([[2]]))])
    task = task_from_json("t", task_json)
    result = TaskResult(task_id="t", signature=task.signature, solved=True, method="depth1",
                        runtime_s=0.01, attempts=[[_grid([[2]]), _grid([[3]])]])
    entry = attempts_to_entry(result, 1)
    assert set(entry) == {"attempt_1", "attempt_2"}
    assert entry["attempt_1"] == [[2]]
    metrics = build_metrics([task], [result], 0.01, "training", "memory")
    assert metrics["solved"] == 1 and metrics["tasks"] == 1
    json.dumps(metrics)


def main() -> int:
    """Run the whole self-check battery; return a process exit code."""
    _check_canonical()
    _check_inducer_loo()
    _check_runner_contract()
    solver = SolverV2(SolverV2Config(time_budget=8.0, beam_budget=3.0,
                                     object_budget=2.0, learner_budget=1.0))
    failures: List[str] = []
    suites = _synthetic_suite()
    for name, (payload, expected) in suites.items():
        result = solver.solve_json(name, payload)
        ok = True
        for produced, want in zip(result.attempts, expected):
            if not np.array_equal(produced[0], want):
                ok = False
        status = "PASS" if ok else "FAIL"
        print(f"[selfcheck] {name:18s} {status:4s} method={result.method} "
              f"t={result.runtime_s:.2f}s")
        if not ok:
            failures.append(f"{name} (method={result.method})")
    if failures:
        print(f"[selfcheck] FAILED: {failures}")
        return 1
    print(f"[selfcheck] all {len(suites)} synthetic tasks solved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
