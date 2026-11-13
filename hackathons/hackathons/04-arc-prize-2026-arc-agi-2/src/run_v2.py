"""Runner for the v2 object-centric ARC-AGI-2 solver.

Responsibilities
----------------
* generate candidates for every task in parallel and checkpoint the submission
  file every N tasks so an interrupted run still leaves a complete file;
* grade attempt_1 / attempt_2 *before* writing metrics, using the fixed
  definition ``pass@2 = attempt_1 correct or attempt_2 correct`` (v1 reported
  only attempt_2 in that slot, understating its own pass@2);
* collect a gold-free feature / label dataset for the ranking model from the
  training split only, then fit and validate that model on a task-level split;
* emit per-family precision and per-signature pass rates for the paper track.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.program_ranker import ProgramRanker, candidate_features
from src.search import ArcTask, TaskResult, fallback_grid, load_tasks
from src.solver_v2 import Candidate, SolverV2, SolverV2Config

Grid = np.ndarray


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _as_json(grid: Optional[Grid]) -> List[List[int]]:
    """Serialise a grid; ``None`` becomes an empty grid."""
    if grid is None:
        return [[0]]
    return np.asarray(grid, dtype=np.int8).tolist()


def _solve_one(payload: Tuple[ArcTask, SolverV2Config, Optional[str], bool]) -> Dict[str, Any]:
    """Worker: solve one task, grade its candidates, optionally emit ranker rows."""
    task, config, ranker_path, collect = payload
    ranker = ProgramRanker(checkpoint=ranker_path) if ranker_path else ProgramRanker()
    solver = SolverV2(config, ranker)
    try:
        result = solver.solve(task)
    except Exception as exc:  # pragma: no cover - defensive
        result = TaskResult(task_id=task.task_id, signature=task.signature, solved=False,
                            method="error", runtime_s=0.0,
                            attempts=[[fallback_grid(task, i),
                                       fallback_grid(task, i)] for i in range(len(task.test_inputs))],
                            notes=f"worker_error:{type(exc).__name__}:{exc}"[:200])
        return {"result": result, "candidates": [], "rows": []}

    candidates = list(getattr(solver, "last_candidates", []))
    rank_order = [id(c) for c in getattr(solver, "_last_ranking", [])]

    def correct(candidate: Candidate) -> bool:
        """True when the candidate predicts every gold test output exactly."""
        if task.test_outputs is None or len(candidate.outputs) != len(task.test_outputs):
            return False
        return all(o is not None and o.shape == g.shape and np.array_equal(o, g)
                   for o, g in zip(candidate.outputs, task.test_outputs))

    rows: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []
    for candidate in candidates:
        is_correct = correct(candidate)
        stats.append({"family": candidate.family, "complexity": candidate.complexity,
                      "loo_pass": bool(candidate.loo_pass),
                      "rank": rank_order.index(id(candidate)) if id(candidate) in rank_order else -1,
                      "correct": bool(is_correct)})
        if collect:
            features = candidate_features(task, candidate, candidates)
            rows.append({"task_id": task.task_id, "family": candidate.family,
                         "detail": candidate.detail, "label": int(is_correct),
                         "features": [round(float(v), 6) for v in features]})
    return {"result": result, "candidates": stats, "rows": rows}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _grade(task: ArcTask, result: TaskResult) -> Tuple[bool, bool]:
    """Fixed grading: pass@1 = attempt_1, pass@2 = either attempt."""
    if task.test_outputs is None or len(result.attempts) != len(task.test_outputs):
        return (False, False)
    first = second = True
    for index, gold in enumerate(task.test_outputs):
        for slot, flag in ((0, "first"), (1, "second")):
            produced = result.attempts[index][slot]
            if produced is None or produced.shape != gold.shape or not np.array_equal(produced, gold):
                if flag == "first":
                    first = False
                else:
                    second = False
    return (first, first or second)


def build_metrics(tasks: Sequence[ArcTask], outcomes: Sequence[Dict[str, Any]],
                  wall_s: float, split: str, config: SolverV2Config,
                  ranker_metrics: Optional[Dict[str, Any]] = None,
                  data_dir: str = "data/arc-agi-2/data") -> Dict[str, Any]:
    """Aggregate graded results, per-family precision and per-signature rates."""
    index = {task.task_id: task for task in tasks}
    per_task: Dict[str, Any] = {}
    by_signature: Dict[str, Dict[str, int]] = {}
    by_method: Dict[str, int] = {}
    family_stats: Dict[str, Dict[str, int]] = {}
    pass1 = pass2 = graded = solved = 0
    oracle = 0
    attempt2_distinct = 0
    ranker_agree = ranker_total = 0

    for outcome in outcomes:
        result: TaskResult = outcome["result"]
        task = index.get(result.task_id)
        g1, g2 = _grade(task, result) if task is not None else (False, False)
        graded_here = task is not None and task.test_outputs is not None
        graded += int(graded_here)
        pass1 += int(g1)
        pass2 += int(g2)
        solved += int(result.solved)
        candidates = outcome.get("candidates", [])
        oracle += int(any(c["correct"] for c in candidates))
        best_rank_correct = any(c["correct"] and c["rank"] == 0 for c in candidates)
        if candidates:
            ranker_total += 1
            ranker_agree += int(best_rank_correct)
        for candidate in candidates:
            bucket = family_stats.setdefault(candidate["family"],
                                             {"candidates": 0, "correct": 0, "loo": 0})
            bucket["candidates"] += 1
            bucket["correct"] += int(candidate["correct"])
            bucket["loo"] += int(candidate["loo_pass"])
        if result.solved:
            for name in result.method.split("+"):
                by_method[name] = by_method.get(name, 0) + 1
        signature = result.signature
        bucket = by_signature.setdefault(signature, {"tasks": 0, "solved": 0,
                                                     "pass1": 0, "pass2": 0})
        bucket["tasks"] += 1
        bucket["solved"] += int(result.solved)
        bucket["pass1"] += int(g1)
        bucket["pass2"] += int(g2)
        if len(result.attempts) and len(result.attempts[0]) > 1:
            distinct = any(not np.array_equal(pair[0], pair[1]) for pair in result.attempts)
            attempt2_distinct += int(distinct)
        per_task[result.task_id] = {
            "solved": bool(result.solved), "method": result.method,
            "runtime_s": round(result.runtime_s, 4), "signature": signature,
            "notes": result.notes, "pass1": bool(g1), "pass2": bool(g2),
            "n_candidates": len(candidates),
            "families": [c["family"] for c in candidates],
        }

    runtimes = [outcome["result"].runtime_s for outcome in outcomes] or [0.0]
    return {
        "solver": "v2-object-beam-ranked",
        "split": split,
        "data_dir": data_dir,
        "tasks": len(outcomes),
        "solved": solved,
        "solve_rate": round(solved / max(len(outcomes), 1), 4),
        "wall_seconds": round(wall_s, 2),
        "runtime_mean_s": round(statistics.fmean(runtimes), 4),
        "runtime_median_s": round(statistics.median(runtimes), 4),
        "runtime_max_s": round(max(runtimes), 4),
        "by_signature": {key: {**value,
                               "rate": round(value["solved"] / max(value["tasks"], 1), 4),
                               "pass1_rate": round(value["pass1"] / max(value["tasks"], 1), 4),
                               "pass2_rate": round(value["pass2"] / max(value["tasks"], 1), 4)}
                         for key, value in sorted(by_signature.items())},
        "by_method": dict(sorted(by_method.items(), key=lambda item: -item[1])),
        "by_family": {key: {**value,
                            "precision": round(value["correct"] / max(value["candidates"], 1), 4),
                            "loo_rate": round(value["loo"] / max(value["candidates"], 1), 4)}
                      for key, value in sorted(family_stats.items())},
        "attempt2_distinct_tasks": attempt2_distinct,
        "candidate_oracle_pass1": oracle,
        "candidate_oracle_rate": round(oracle / max(graded, 1), 4),
        "ranker_top1_correct": ranker_agree,
        "ranker_top1_rate": round(ranker_agree / max(ranker_total, 1), 4),
        "graded_tasks": graded,
        "graded_pass1": pass1,
        "graded_pass2": pass2,
        "graded_pass1_rate": round(pass1 / max(graded, 1), 4),
        "graded_pass2_rate": round(pass2 / max(graded, 1), 4),
        "ranker": ranker_metrics or {},
        "config": asdict(config),
        "per_task": per_task,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Submission / IO
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def bootstrap_submission(tasks: Sequence[ArcTask]) -> Dict[str, Any]:
    """Pre-fill every task id with a graded fallback pair."""
    submission: Dict[str, Any] = {}
    for task in tasks:
        fallbacks = [[_as_json(fallback_grid(task, i))] * 2 for i in range(len(task.test_inputs))]
        entry = {"attempt_1": fallbacks[0][0], "attempt_2": fallbacks[0][1]}
        for index, pair in enumerate(fallbacks):
            entry[f"attempt_1_{index}"] = pair[0]
            entry[f"attempt_2_{index}"] = pair[1]
        submission[task.task_id] = entry
    return submission


def entry_for(result: TaskResult) -> Dict[str, Any]:
    """Competition entry: attempt_1 / attempt_2 plus indexed extras.

    The ARC Prize submission format scores ``attempt_1`` and ``attempt_2`` for
    the task's first test input.  Tasks with several test inputs additionally
    carry ``attempt_{slot}_{index}`` keys so no prediction is lost.
    """
    first = result.attempts[0] if result.attempts else [[[0]], [[0]]]
    entry: Dict[str, Any] = {"attempt_1": _as_json(first[0]),
                             "attempt_2": _as_json(first[1])}
    for index, pair in enumerate(result.attempts):
        entry[f"attempt_1_{index}"] = _as_json(pair[0])
        entry[f"attempt_2_{index}"] = _as_json(pair[1])
    return entry


def append_log(path: Path, track: str, metric: str, value: Any, notes: str) -> None:
    """Append one TSV row to the shared experiment log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"{track}\t{metric}\t{value}\t{notes}\n")


# ---------------------------------------------------------------------------
# Ranker training
# ---------------------------------------------------------------------------


def train_ranker(dataset_path: Path, output_dir: Path, seed: int = 0) -> Dict[str, Any]:
    """Fit the ranking model on collected rows with a task-level split."""
    rows: List[Dict[str, Any]] = []
    with dataset_path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    task_ids = sorted({row["task_id"] for row in rows})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(task_ids))
    n_val = max(1, int(0.3 * len(shuffled)))
    val_ids = set(shuffled[:n_val])
    train_rows = [row for row in rows if row["task_id"] not in val_ids]
    val_rows = [row for row in rows if row["task_id"] in val_ids]

    def priors(subset: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Laplace-smoothed per-family and per-detail success rates."""
        family_counts: Dict[str, List[int]] = {}
        detail_counts: Dict[str, List[int]] = {}
        for row in subset:
            family_counts.setdefault(row["family"], [0, 0])
            family_counts[row["family"]][0] += row["label"]
            family_counts[row["family"]][1] += 1
            detail_counts.setdefault(row.get("detail", ""), [0, 0])
            detail_counts[row.get("detail", "")][0] += row["label"]
            detail_counts[row.get("detail", "")][1] += 1
        family_prior = {key: (value[0] + 1) / (value[1] + 2) for key, value in family_counts.items()}
        detail_prior = {key: (value[0] + 1) / (value[1] + 2) for key, value in detail_counts.items()
                        if value[1] >= 3}
        return family_prior, detail_prior

    family_prior, detail_prior = priors(train_rows)
    ranker = ProgramRanker(family_prior=family_prior, detail_prior=detail_prior)
    from src.program_ranker import add_prior_features

    fit_stats = ranker.fit(add_prior_features(train_rows, family_prior, detail_prior),
                           [row["label"] for row in train_rows])

    def evaluate(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Top-1 accuracy per task and pass@1 on a task subset."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in subset:
            grouped.setdefault(row["task_id"], []).append(row)
        top1 = 0
        oracle = 0
        manual_top1 = 0
        for _task_id, group in grouped.items():
            oracle += int(any(row["label"] for row in group))
            scores = ranker.predict_proba_rows(add_prior_features(group, family_prior, detail_prior))
            best = int(np.argmax(scores))
            top1 += int(group[best]["label"])
            manual = sorted(group, key=lambda row: -ranker.detail_prior.get(
                row.get("detail", ""), ranker.family_prior.get(row["family"], 0.1)))
            manual_top1 += int(manual[0]["label"])
        return {"tasks": len(grouped), "top1_correct": top1,
                "manual_top1_correct": manual_top1,
                "manual_top1_accuracy": round(manual_top1 / max(len(grouped), 1), 4),
                "top1_accuracy": round(top1 / max(len(grouped), 1), 4),
                "oracle_tasks": oracle,
                "oracle_rate": round(oracle / max(len(grouped), 1), 4),
                "rows": len(subset)}

    metrics = {"fit": fit_stats, "train": evaluate(train_rows), "validation": evaluate(val_rows),
               "seed": seed, "dataset_rows": len(rows), "tasks": len(task_ids)}
    ranker.save(output_dir / "ranker_model.joblib")
    write_json_atomic(output_dir / "ranker_metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run_split(data_dir: str, split: str, output_dir: str, config: SolverV2Config,
              workers: int, max_wall: float, limit: Optional[int], checkpoint_every: int,
              log_path: str, ranker_path: Optional[str], collect: bool,
              submission_name: str = "submission.json") -> Dict[str, Any]:
    """Solve one split, checkpoint the submission and write metrics."""
    tasks = load_tasks(data_dir, split, limit)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    submission_path = out / submission_name
    submission = bootstrap_submission(tasks)
    write_json_atomic(submission_path, submission)

    start = time.monotonic()
    deadline = start + max_wall
    outcomes: List[Dict[str, Any]] = []
    dataset_path = out / f"ranker_rows_{split}.jsonl" if collect else None
    dataset_handle = dataset_path.open("w") if dataset_path else None
    payloads = [(task, config, ranker_path, collect) for task in tasks]
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_solve_one, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:  # pragma: no cover
                    result = TaskResult(task_id=task.task_id, signature=task.signature,
                                        solved=False, method="error", runtime_s=0.0,
                                        attempts=[[fallback_grid(task, i)] * 2
                                                  for i in range(len(task.test_inputs))],
                                        notes=f"future_error:{type(exc).__name__}")
                    outcome = {"result": result, "candidates": [], "rows": []}
                outcomes.append(outcome)
                submission[task.task_id] = entry_for(outcome["result"])
                if dataset_handle is not None:
                    for row in outcome.get("rows", []):
                        dataset_handle.write(json.dumps(row) + "\n")
                completed += 1
                if completed % checkpoint_every == 0:
                    write_json_atomic(submission_path, submission)
                    print(f"[run_v2] {completed}/{len(payloads)} "
                          f"elapsed={time.monotonic() - start:.0f}s", flush=True)
                if time.monotonic() >= deadline:
                    print("[run_v2] wall clock cap hit, cancelling pending work", flush=True)
                    for pending in futures:
                        pending.cancel()
                    break
    finally:
        if dataset_handle is not None:
            dataset_handle.close()

    outcomes.sort(key=lambda outcome: outcome["result"].task_id)
    write_json_atomic(submission_path, submission)
    wall = time.monotonic() - start
    ranker_metrics = None
    metrics_path = out / "ranker_metrics.json"
    if metrics_path.exists():
        ranker_metrics = json.loads(metrics_path.read_text())
    metrics = build_metrics(tasks, outcomes, wall, split, config, ranker_metrics, data_dir)
    metrics["workers"] = workers
    metrics["unfinished"] = len(tasks) - len(outcomes)
    metrics["ranker_model"] = ranker_path
    write_json_atomic(out / f"eval_metrics_{split}_v2.json", metrics)
    write_json_atomic(out / "eval_metrics.json", metrics)

    append_log(Path(log_path), "arc-agi-2-v2", f"pass1_{split}", metrics["graded_pass1_rate"],
               f"{metrics['graded_pass1']}/{metrics['graded_tasks']} "
               f"wall={round(wall, 1)}s workers={workers} ranker={bool(ranker_path)}")
    append_log(Path(log_path), "arc-agi-2-v2", f"pass2_{split}", metrics["graded_pass2_rate"],
               f"{metrics['graded_pass2']}/{metrics['graded_tasks']} "
               f"oracle={metrics['candidate_oracle_rate']} "
               f"top1={metrics['ranker_top1_rate']}")
    print(json.dumps({key: value for key, value in metrics.items()
                      if key in ("split", "tasks", "solved", "solve_rate", "graded_pass1",
                                 "graded_pass1_rate", "graded_pass2", "graded_pass2_rate",
                                 "candidate_oracle_rate", "ranker_top1_rate",
                                 "wall_seconds", "by_method")}, indent=2), flush=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 v2 object/beam/ranked solver")
    parser.add_argument("--data-dir", default="data/arc-agi-2/data")
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--budget", type=float, default=14.0)
    parser.add_argument("--beam-budget", type=float, default=5.0)
    parser.add_argument("--object-budget", type=float, default=3.0)
    parser.add_argument("--learner-budget", type=float, default=2.5)
    parser.add_argument("--max-wall", type=float, default=3000.0)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 2))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log", default="logs/experiments.tsv")
    parser.add_argument("--ranker", default=None)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--train-ranker", action="store_true")
    parser.add_argument("--dataset", default="outputs/ranker_rows_training.jsonl")
    parser.add_argument("--no-beam", action="store_true")
    parser.add_argument("--no-objects", action="store_true")
    parser.add_argument("--no-learners", action="store_true")
    parser.add_argument("--object-tree", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    if args.train_ranker:
        metrics = train_ranker(Path(args.dataset), Path(args.output_dir))
        print(json.dumps(metrics, indent=2))
        return 0
    config = SolverV2Config(time_budget=args.budget, beam_budget=args.beam_budget,
                            object_budget=args.object_budget,
                            learner_budget=args.learner_budget,
                            enable_beam=not args.no_beam,
                            enable_objects=not args.no_objects,
                            enable_learners=not args.no_learners,
                            enable_object_tree=args.object_tree)
    run_split(args.data_dir, args.split, args.output_dir, config, args.workers,
              args.max_wall, args.limit, args.checkpoint_every, args.log, args.ranker,
              args.collect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
