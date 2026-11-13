"""CLI runner: solve an ARC-AGI-2 split, checkpoint a submission, emit metrics.

Examples
--------
    python3 -m src.dsl_search_runner --split training --limit 200
    python3 -m src.dsl_search_runner --split evaluation --max-wall 2100 --workers 8

The submission file is pre-populated with the majority-colour fallback for every
task id and rewritten atomically after each batch, so a killed run still leaves a
complete, valid ``outputs/submission.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.search import (ArcTask, DslSolver, SolverConfig, TaskResult, fallback_grid,
                        load_tasks)

Grid = np.ndarray


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------


def _solve_payload(payload: Tuple[ArcTask, SolverConfig]) -> TaskResult:
    """Worker entry point: solve one pickled task with its own solver instance."""
    task, config = payload
    try:
        return DslSolver(config).solve(task)
    except Exception as exc:  # pragma: no cover - defensive against worker crash
        return TaskResult(task_id=task.task_id, signature=task.signature, solved=False,
                          method="error", runtime_s=0.0,
                          attempts=[[fallback_grid(task, i)] * 2 for i in range(len(task.test_inputs))],
                          notes=f"worker_error:{type(exc).__name__}:{exc}"[:200])


# ---------------------------------------------------------------------------
# Submission / metrics
# ---------------------------------------------------------------------------


def grid_to_json(grid: Grid) -> List[List[int]]:
    """Convert an int8 grid to nested Python ints."""
    return np.asarray(grid, dtype=np.int8).tolist()


def attempts_to_entry(result: TaskResult, n_test: int) -> Dict[str, Any]:
    """Build the Kaggle-style attempt entry (grid for one test, list otherwise)."""
    attempt_1 = [grid_to_json(result.attempts[i][0]) for i in range(n_test)]
    attempt_2 = [grid_to_json(result.attempts[i][1]) for i in range(n_test)]
    if n_test == 1:
        return {"attempt_1": attempt_1[0], "attempt_2": attempt_2[0]}
    return {"attempt_1": attempt_1, "attempt_2": attempt_2}


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def bootstrap_submission(tasks: Sequence[ArcTask]) -> Dict[str, Any]:
    """Fallback submission covering every task id, used before any solving."""
    submission: Dict[str, Any] = {}
    for task in tasks:
        entries = []
        for index in range(len(task.test_inputs)):
            fb = fallback_grid(task, index)
            entries.append((fb, fb))
        dummy = TaskResult(task_id=task.task_id, signature=task.signature, solved=False,
                           method="fallback", runtime_s=0.0, attempts=[[a, b] for a, b in entries])
        submission[task.task_id] = attempts_to_entry(dummy, len(task.test_inputs))
    return submission


def grade_attempts(task: ArcTask, result: TaskResult) -> Tuple[bool, bool]:
    """Grade both attempts against gold test outputs when they are available."""
    if task.test_outputs is None or len(result.attempts) != len(task.test_outputs):
        return (False, False)
    exact = [True, True]
    for index, gold in enumerate(task.test_outputs):
        for slot in (0, 1):
            produced = result.attempts[index][slot]
            if produced is None or produced.shape != gold.shape or not np.array_equal(produced, gold):
                exact[slot] = False
    return (exact[0], exact[1])


def build_metrics(tasks: Sequence[ArcTask], results: List[TaskResult], wall_s: float,
                  split: str, data_dir: str) -> Dict[str, Any]:
    """Aggregate per-task results into the metrics document.

    Two families of numbers are reported: the *hypothesis* rate (a program or
    learned rule fitted every training pair, i.e. ``solved``) and, when the JSON
    carries gold test outputs, the graded pass@1 / pass@2 rate.  The solver is
    blind to the gold outputs; grading happens strictly after prediction.
    """
    task_index = {task.task_id: task for task in tasks}
    by_signature: Dict[str, Dict[str, int]] = {}
    by_method: Dict[str, int] = {}
    per_task: Dict[str, Any] = {}
    graded_pass1 = 0
    graded_pass2 = 0
    graded_tasks = 0
    for result in results:
        task = task_index.get(result.task_id)
        pass1, pass2 = grade_attempts(task, result) if task is not None else (False, False)
        if task is not None and task.test_outputs is not None:
            graded_tasks += 1
            graded_pass1 += int(pass1)
            graded_pass2 += int(pass2)
        bucket = by_signature.setdefault(result.signature, {"tasks": 0, "solved": 0})
        bucket["tasks"] += 1
        bucket["solved"] += int(result.solved)
        if result.solved:
            for name in result.method.split("+"):
                by_method[name] = by_method.get(name, 0) + 1
        per_task[result.task_id] = {
            "solved": bool(result.solved),
            "method": result.method,
            "runtime_s": round(result.runtime_s, 4),
            "signature": result.signature,
            "notes": result.notes,
            "pass1": bool(pass1),
            "pass2": bool(pass2),
        }
    solved = sum(int(r.solved) for r in results)
    runtimes = [r.runtime_s for r in results] or [0.0]
    return {
        "split": split,
        "data_dir": data_dir,
        "tasks": len(results),
        "solved": solved,
        "solve_rate": round(solved / max(len(results), 1), 4),
        "wall_seconds": round(wall_s, 2),
        "runtime_mean_s": round(statistics.fmean(runtimes), 4),
        "runtime_median_s": round(statistics.median(runtimes), 4),
        "runtime_max_s": round(max(runtimes), 4),
        "by_signature": {k: {**v, "rate": round(v["solved"] / max(v["tasks"], 1), 4)}
                         for k, v in sorted(by_signature.items())},
        "by_method": dict(sorted(by_method.items(), key=lambda kv: -kv[1])),
        "graded_tasks": graded_tasks,
        "graded_pass1": graded_pass1,
        "graded_pass2": graded_pass2,
        "graded_pass1_rate": round(graded_pass1 / max(graded_tasks, 1), 4),
        "graded_pass2_rate": round(graded_pass2 / max(graded_tasks, 1), 4),
        "per_task": per_task,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def append_experiment_log(path: Path, metric: str, value: Any, notes: str) -> None:
    """Append a tab-separated experiment row, creating the header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text("track\tmetric\tvalue\tnotes\n")
    with path.open("a") as handle:
        handle.write(f"arc-agi-2\t{metric}\t{value}\t{notes}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_split(data_dir: str, split: str, output_dir: str, config: SolverConfig,
              workers: int, max_wall: float, limit: Optional[int] = None,
              checkpoint_every: int = 5, log_path: str = "logs/experiments.tsv") -> Dict[str, Any]:
    """Solve a split end to end and return the metrics document."""
    load_start = time.monotonic()
    tasks = load_tasks(data_dir, split=split, limit=limit)
    if not tasks:
        raise SystemExit(f"no tasks found under {data_dir}/{split}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    submission = bootstrap_submission(tasks)
    # Only the evaluation split owns the canonical submission file; training
    # runs keep their own artefact so a dev run can never clobber a submission.
    submission_name = "submission.json" if split == "evaluation" else f"submission_{split}.json"
    submission_path = out / submission_name
    write_json_atomic(submission_path, submission)
    print(f"[runner] loaded {len(tasks)} {split} tasks in {time.monotonic() - load_start:.1f}s", flush=True)

    results: List[TaskResult] = []
    start = time.monotonic()
    deadline = start + max_wall
    payloads = [(task, config) for task in tasks]

    if workers <= 1:
        for index, payload in enumerate(payloads):
            if time.monotonic() >= deadline:
                print("[runner] wall clock cap hit, stopping", flush=True)
                break
            result = _solve_payload(payload)
            results.append(result)
            submission[result.task_id] = attempts_to_entry(result, len(payloads[index][0].test_inputs))
            if (index + 1) % checkpoint_every == 0:
                write_json_atomic(submission_path, submission)
                print(f"[runner] {index + 1}/{len(payloads)} solved={sum(r.solved for r in results)}", flush=True)
    else:
        solved_so_far = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_payload, payload): payload[0] for payload in payloads}
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    result = TaskResult(task_id=task.task_id, signature=task.signature, solved=False,
                                        method="error", runtime_s=0.0,
                                        attempts=[[fallback_grid(task, i)] * 2 for i in range(len(task.test_inputs))],
                                        notes=f"future_error:{type(exc).__name__}")
                results.append(result)
                solved_so_far += int(result.solved)
                submission[result.task_id] = attempts_to_entry(result, len(task.test_inputs))
                completed += 1
                if completed % checkpoint_every == 0:
                    write_json_atomic(submission_path, submission)
                    print(f"[runner] {completed}/{len(payloads)} solved={solved_so_far} "
                          f"elapsed={time.monotonic() - start:.0f}s", flush=True)
                if time.monotonic() >= deadline:
                    print("[runner] wall clock cap hit, cancelling pending work", flush=True)
                    for pending in futures:
                        pending.cancel()
                    break
        results.sort(key=lambda r: r.task_id)

    write_json_atomic(submission_path, submission)
    wall = time.monotonic() - start
    metrics = build_metrics(tasks, results, wall, split, data_dir)
    metrics["config"] = asdict(config)
    metrics["workers"] = workers
    metrics["unfinished"] = len(tasks) - len(results)
    write_json_atomic(out / f"eval_metrics_{split}.json", metrics)
    if split == "evaluation":
        write_json_atomic(out / "eval_metrics.json", metrics)
    append_experiment_log(Path(log_path), f"solve_rate_{split}", metrics["solve_rate"],
                          f"solved={metrics['solved']}/{metrics['tasks']} "
                          f"budget={config.time_budget}s workers={workers} "
                          f"wall={metrics['wall_seconds']}s methods={metrics['by_method']}")
    append_experiment_log(Path(log_path), f"graded_pass2_{split}", metrics["graded_pass2_rate"],
                          f"pass@2={metrics['graded_pass2']}/{metrics['graded_tasks']} "
                          f"pass@1={metrics['graded_pass1']}")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k in ("split", "tasks", "solved", "solve_rate", "wall_seconds",
                               "runtime_mean_s", "by_method", "unfinished", "graded_tasks",
                               "graded_pass1", "graded_pass2", "graded_pass1_rate",
                               "graded_pass2_rate")}, indent=2), flush=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 neuro-symbolic DSL solver")
    parser.add_argument("--data-dir", default="data/arc-agi-2/data")
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--budget", type=float, default=12.0, help="per-task seconds")
    parser.add_argument("--max-wall", type=float, default=2100.0, help="global seconds cap")
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log", default="logs/experiments.tsv")
    parser.add_argument("--no-depth2", action="store_true")
    parser.add_argument("--no-neuro", action="store_true")
    parser.add_argument("--no-learners", action="store_true")
    parser.add_argument("--max-hypotheses", type=int, default=2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    config = SolverConfig(time_budget=args.budget, enable_depth2=not args.no_depth2,
                          enable_neuro=not args.no_neuro,
                          enable_learners=not args.no_learners,
                          max_hypotheses=args.max_hypotheses)
    run_split(args.data_dir, args.split, args.output_dir, config, args.workers,
              args.max_wall, args.limit, args.checkpoint_every, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
