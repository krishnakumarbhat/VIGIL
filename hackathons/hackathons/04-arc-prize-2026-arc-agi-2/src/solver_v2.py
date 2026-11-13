"""v2 solver: object-centric induction + heuristic beam search + learned ranking.

The cascade generates *several* verified candidates per task instead of
returning the first one, then ranks them with :class:`ProgramRanker`.  Ranking
is what converts the low-precision v1 learners (cell trees at 10-15%, object
trees at 3%) into usable second attempts, and it lets the object-centric
inducer contribute without being able to veto a symbolic program.

Order of generation is by expected precision, and duplicates are collapsed, so
a cheap high-precision program is never displaced by a learned rule that
happens to fit the training pairs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.beam_search import BeamSolver
from src.cell_rule_learner import CellRuleLearner
from src.object_program_inducer import ObjectProgramInducer, Program
from src.object_rule_learner import ObjectRuleLearner
from src.patch_lookup_learner import PatchLookupLearner
from src.program_ranker import ProgramRanker
from src.search import ArcTask, SolverConfig, TaskResult, fallback_grid, sanitize

Grid = np.ndarray
Pair = Tuple[Grid, Grid]


@dataclass(eq=False)
class Candidate:
    """One verified hypothesis with the evidence collected while fitting it."""

    family: str
    outputs: List[Grid]
    detail: str = ""
    complexity: int = 1
    loo_pass: bool = False
    loo_checked: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverV2Config:
    """Runtime knobs for :class:`SolverV2`."""

    time_budget: float = 16.0
    beam_budget: float = 5.0
    object_budget: float = 3.5
    learner_budget: float = 5.0
    max_candidates: int = 16
    enable_beam: bool = True
    enable_objects: bool = True
    enable_learners: bool = True
    enable_object_tree: bool = True
    enable_mlp: bool = False


def _dedupe_key(outputs: Sequence[Grid]) -> Tuple[Any, ...]:
    """Hashable key of a candidate's predictions."""
    return tuple((o.shape, o.tobytes()) for o in outputs if o is not None)


class SolverV2:
    """Object-centric, beam-searched, rank-ordered ARC solver."""

    def __init__(self, config: Optional[SolverV2Config] = None,
                 ranker: Optional[ProgramRanker] = None) -> None:
        """Create a solver; the ranker may be pre-fitted from the training split."""
        self.config = config or SolverV2Config()
        self.ranker = ranker or ProgramRanker()
        self._last_ranking: List[Candidate] = []
        self.last_candidates: List[Candidate] = []
        self._cell = CellRuleLearner()
        self._object = ObjectRuleLearner()
        self._patch = PatchLookupLearner()

    # -- public API --------------------------------------------------------

    def solve(self, task: ArcTask) -> TaskResult:
        """Generate, verify and rank candidates; return two attempts per test input."""
        start = time.monotonic()
        deadline = start + self.config.time_budget
        candidates: List[Candidate] = []
        notes: List[str] = []

        stages = [("depth1", self._stage_depth1, 3.0),
                  ("depth2", self._stage_depth2, 2.0),
                  ("mosaic", self._stage_mosaic, 0.5),
                  ("objects", self._stage_objects, self.config.object_budget),
                  ("beam", self._stage_beam, self.config.beam_budget),
                  ("learners", self._stage_learners, self.config.learner_budget)]
        for name, stage, budget in stages:
            if name == "beam" and not self.config.enable_beam:
                continue
            if name == "objects" and not self.config.enable_objects:
                continue
            if name == "learners" and not self.config.enable_learners:
                continue
            if time.monotonic() >= deadline:
                notes.append(f"deadline_before_{name}")
                break
            stage_deadline = min(deadline, time.monotonic() + budget)
            try:
                found = stage(task, stage_deadline)
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"{name}_error:{type(exc).__name__}")
                continue
            for candidate in found:
                if len(candidates) >= self.config.max_candidates:
                    break
                if any(_dedupe_key(candidate.outputs) == _dedupe_key(existing.outputs)
                       for existing in candidates):
                    continue
                candidates.append(candidate)

        self.last_candidates = list(candidates)
        attempts = self._attempts(task, candidates)
        method = "+".join(c.family for c in self._last_ranking[:2]) if candidates else "fallback"
        return TaskResult(task_id=task.task_id, signature=task.signature,
                          solved=bool(candidates), method=method,
                          runtime_s=time.monotonic() - start, attempts=attempts,
                          notes=";".join(notes))

    def solve_json(self, task_id: str, payload: Dict[str, Any]) -> TaskResult:
        """Solve an inline task payload (used by the notebook)."""
        from src.search import task_from_json

        return self.solve(task_from_json(task_id, payload))

    # -- attempts ----------------------------------------------------------

    def _attempts(self, task: ArcTask, candidates: Sequence[Candidate]) -> List[List[Grid]]:
        """Rank candidates and build (attempt_1, attempt_2) for every test input."""
        order = self.ranker.rank(task, candidates) if candidates else []
        self._last_ranking = [candidates[i] for i in order]
        attempts: List[List[Grid]] = []
        for index in range(len(task.test_inputs)):
            fallback = fallback_grid(task, index)
            first: Optional[Grid] = None
            second: Optional[Grid] = None
            if order:
                first = sanitize(candidates[order[0]].outputs[index])
            for position in order:
                produced = sanitize(candidates[position].outputs[index])
                if produced is None:
                    continue
                if first is not None and np.array_equal(produced, first):
                    continue
                second = produced
                break
            if first is None:
                first = fallback
            if second is None or np.array_equal(second, first):
                second = fallback
            attempts.append([first, second])
        return attempts

    # -- stage 1: single primitives ---------------------------------------

    def _stage_depth1(self, task: ArcTask, deadline: float) -> List[Candidate]:
        """Every enumerated primitive that fits all training pairs."""
        found: List[Candidate] = []
        for name, fn in dsl.enumerate_primitives(task.train):
            if time.monotonic() >= deadline:
                break
            try:
                if not all(np.array_equal(fn(src), dst) for src, dst in task.train):
                    continue
                outputs = [sanitize(fn(grid)) for grid in task.test_inputs]
            except Exception:
                # A primitive that raises on one pair must not discard the
                # candidates found by the others (v1 wrapped this check).
                continue
            if any(o is None for o in outputs):
                continue
            found.append(Candidate("depth1", [o for o in outputs if o is not None],
                                   detail=name, complexity=1))
        return found

    # -- stage 1b: depth-2 composition -------------------------------------

    def _stage_depth2(self, task: ArcTask, deadline: float) -> List[Candidate]:
        """Exhaustive ordered primitive pairs, pair-0 filtered (v1 strategy)."""
        left = dsl.composition_left_set(task.train)
        right = dsl.primitive_fast_set()
        src0, dst0 = task.train[0]
        found: List[Candidate] = []
        for left_name, left_fn in left:
            if time.monotonic() >= deadline:
                break
            try:
                middle = left_fn(src0)
            except Exception:
                continue
            if middle is None or middle.size == 0:
                continue
            for right_name, right_fn in right:
                if time.monotonic() >= deadline:
                    return found
                try:
                    out = right_fn(middle)
                    if out is None or out.shape != dst0.shape or not np.array_equal(out, dst0):
                        continue
                    def program(grid: Grid, l=left_fn, r=right_fn) -> Optional[Grid]:
                        """Apply the left then the right primitive."""
                        inter = l(grid)
                        if inter is None or inter.size == 0:
                            return None
                        return r(inter)
                    if not all(np.array_equal(program(src), dst) for src, dst in task.train):
                        continue
                    outputs = [sanitize(program(grid)) for grid in task.test_inputs]
                except Exception:
                    continue
                if any(o is None for o in outputs):
                    continue
                found.append(Candidate("depth2", [o for o in outputs if o is not None],
                                       detail=f"{right_name}({left_name}(x))", complexity=2))
                if len(found) >= 2:
                    return found
        return found

    # -- stage 2: mosaic ---------------------------------------------------

    def _stage_mosaic(self, task: ArcTask, deadline: float) -> List[Candidate]:
        """Output is a k x m mosaic of transformed input copies."""
        if time.monotonic() >= deadline:
            return []
        plan = dsl.block_mosaic_fit(task.train)
        if plan is None:
            return []
        outputs = [sanitize(dsl.block_mosaic_apply(grid, plan)) for grid in task.test_inputs]
        if any(o is None for o in outputs):
            return []
        return [Candidate("mosaic", [o for o in outputs if o is not None],
                          detail=f"{plan['rows']}x{plan['cols']}", complexity=2)]

    # -- stage 3: object-centric induction ---------------------------------

    def _stage_objects(self, task: ArcTask, deadline: float) -> List[Candidate]:
        """All object-level programs that reproduce every training pair."""
        inducer = ObjectProgramInducer()
        programs: List[Program] = []
        for program in inducer.fit(task.train):
            if time.monotonic() >= deadline:
                break
            programs.append(program)
        found: List[Candidate] = []
        for program in programs:
            outputs = [program.apply(grid) for grid in task.test_inputs]
            if any(o is None for o in outputs):
                continue
            if not all(program.apply(src) is not None for src, _ in task.train):
                continue
            found.append(Candidate(program.family, [o for o in outputs if o is not None],
                                   detail=program.detail, complexity=program.complexity,
                                   loo_pass=program.loo_pass,
                                   loo_checked=len(task.train) >= 3, meta=program.meta))
        return found

    # -- stage 4: beam search ----------------------------------------------

    def _stage_beam(self, task: ArcTask, deadline: float) -> List[Candidate]:
        """Heuristic beam search over composed grid and object operators."""
        budget = max(0.2, min(self.config.beam_budget, deadline - time.monotonic()))
        solver = BeamSolver(time_budget=budget)
        programs = solver.solve(task.train, task.test_inputs, max_programs=2)
        return [Candidate("beam", program.outputs, detail=program.name,
                          complexity=len(program.chain)) for program in programs]

    # -- stage 5: learned rules --------------------------------------------

    def _stage_learners(self, task: ArcTask, deadline: float) -> List[Candidate]:
        """Cell trees, local patch lookup and (optionally) object trees.

        Each learner gets its own slice so that an expensive D4-augmented tree
        cannot starve the cheap ones (a v2-pre regression measured at 13 lost
        tasks on the training split).
        """
        found: List[Candidate] = []
        variants = [("cell_tree", self._cell, {"augment": False}, False, 2.0),
                    ("patch_lookup", self._patch, {}, False, 0.8),
                    ("cell_tree_d4", self._cell, {"augment": True}, True, 1.5)]
        if self.config.enable_object_tree:
            variants.append(("object_tree", self._object, {"augment": False}, False, 0.8))
        for name, learner, kwargs, vote, slice_s in variants:
            if time.monotonic() >= deadline:
                break
            slice_deadline = min(deadline, time.monotonic() + slice_s)
            candidate = self._fit_learner(task, name, learner, kwargs, vote, slice_deadline)
            if candidate is not None:
                found.append(candidate)
        return found

    def _fit_learner(self, task: ArcTask, name: str, learner: Any, kwargs: Dict[str, Any],
                     vote: bool, deadline: Optional[float] = None) -> Optional[Candidate]:
        """Fit one learner, verify on train, and measure leave-one-out evidence."""
        try:
            model = learner.fit(task.train, **kwargs)
        except Exception:
            return None
        if model is None:
            return None
        if deadline is not None and time.monotonic() >= deadline:
            return None
        outputs = [self._predict(learner, model, grid, vote) for grid in task.test_inputs]
        if any(o is None for o in outputs):
            return None
        if deadline is not None and time.monotonic() >= deadline:
            return Candidate(name, [o for o in outputs if o is not None], detail="loo=timeout",
                             complexity=4, loo_pass=False, loo_checked=False)
        loo_pass, loo_checked = self._loo(task, learner, kwargs, vote)
        return Candidate(name, [o for o in outputs if o is not None],
                         detail=f"loo={loo_pass}", complexity=4,
                         loo_pass=loo_pass, loo_checked=loo_checked)

    @staticmethod
    def _predict(learner: Any, model: Any, grid: Grid, vote: bool) -> Optional[Grid]:
        """Predict with a learner, tolerating learners without a vote argument."""
        try:
            return sanitize(learner.predict(model, grid, vote=vote))
        except TypeError:
            return sanitize(learner.predict(model, grid))

    def _loo(self, task: ArcTask, learner: Any, kwargs: Dict[str, Any],
             vote: bool) -> Tuple[bool, bool]:
        """Leave-one-out refit evidence: does the learner predict a held-out pair?"""
        if len(task.train) < 2:
            return (False, False)
        for index in range(len(task.train)):
            subset = list(task.train[:index]) + list(task.train[index + 1:])
            try:
                model = learner.fit(subset, **kwargs)
            except Exception:
                return (False, True)
            if model is None:
                return (False, True)
            predicted = self._predict(learner, model, task.train[index][0], vote)
            gold = task.train[index][1]
            if predicted is None or predicted.shape != gold.shape or not np.array_equal(predicted, gold):
                return (False, True)
        return (True, True)
