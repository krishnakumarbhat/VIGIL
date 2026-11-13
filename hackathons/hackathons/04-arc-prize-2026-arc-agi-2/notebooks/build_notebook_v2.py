"""Build and execute ``notebooks/arc_agi_2_solver.ipynb`` without Jupyter.

The container has no ``nbclient``/``jupyter``, so this script executes every
code cell in one namespace with :func:`exec`, captures stdout, and writes a
valid nbformat-4 document with the captured outputs.  The notebook is therefore
genuinely re-executed against the current ``outputs/`` artefacts.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "arc_agi_2_solver.ipynb"


CELLS: List[Tuple[str, str]] = [
    ("markdown", """# Track 04 - ARC-AGI-2 v2: object-centric induction, beam search, learned ranking

Offline reproduction notebook for the **v2** solver. No network calls, no Kaggle
submission. Every number printed below is produced by the code in this notebook
or read from `outputs/*.json` written by `src/run_v2.py`.

Pipeline: canonicalisation -> object segmentation -> object-level program
families -> heuristic beam search over composed operators -> leave-one-out
gated learned rules -> **learned ranker** that orders the candidates into
`attempt_1` / `attempt_2`."""),

    ("code", """import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.search import load_tasks
from src.solver_v2 import SolverV2, SolverV2Config
from src.object_program_inducer import ObjectProgramInducer
from src.beam_search import BeamSolver
from src.program_ranker import ProgramRanker

print("numpy", np.__version__, "| pandas", pd.__version__)
print("root", ROOT)"""),

    ("code", """DATA = ROOT / "data" / "arc-agi-2" / "data"
train_files = sorted((DATA / "training").glob("*.json"))
eval_files = sorted((DATA / "evaluation").glob("*.json"))
print("training tasks", len(train_files), "| evaluation tasks", len(eval_files))

train_tasks = load_tasks(DATA, "training")
eval_tasks = load_tasks(DATA, "evaluation")
print("train pairs", sum(len(t.train) for t in train_tasks))
print("eval median input size", int(np.median([g.size for t in eval_tasks for g in t.test_inputs])))"""),

    ("markdown", """## 1. Canonicalisation invariant

Eight D4 images plus colour relabelling must collapse to one key."""),

    ("code", """from src.canonical import D4, canonical_key

base = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
assert len({canonical_key(fn(base)) for _, fn in D4}) == 1
print("D4 orbit collapses to one key:", canonical_key(base)[:8].hex())"""),

    ("markdown", """## 2. Self-check battery (v1 DSL + v2 object families)

16 synthetic tasks: D4, colour map, upscale, tiling, gravity, crop, mirror,
object selection, deletion, hole filling, panel logic, rays."""),

    ("code", """import src.selfcheck as selfcheck

rc = selfcheck.main()
assert rc == 0, "selfcheck must pass"
print("selfcheck exit code", rc)"""),

    ("markdown", """## 3. Object-centric induction on a real task

Segmentation, per-object attributes, and the object-level programs that pass
verification on every training pair."""),

    ("code", """task = train_tasks[0]
from src.object_model import content_objects

objects = content_objects(task.train[0][0])
print(task.task_id, "| objects", len(objects))
for obj in objects[:5]:
    print("  size", obj.size, "colour", obj.colors, "box", obj.box,
          "holes", obj.holes, "sym", obj.symmetric or "-", "fill", round(obj.fill_ratio, 2))

inductions = ObjectProgramInducer().fit(task.train)
print("verified object programs:", [(p.family, p.detail, p.loo_pass) for p in inductions])"""),

    ("markdown", """## 4. Heuristic beam search

Chains of up to three operators, pruned by the shape + colour-histogram +
object-count heuristic; survivors are verified on every training pair."""),

    ("code", """beam = BeamSolver(time_budget=3.0)
start = time.monotonic()
programs = beam.solve(task.train, task.test_inputs, max_programs=2)
print("beam programs", len(programs), "in", round(time.monotonic() - start, 2), "s")
for program in programs:
    print("  ", program.name)"""),

    ("markdown", """## 5. Learned ranker

Trained on the **training split only** with a task-level 70/30 split; the
evaluation split is never used for fitting."""),

    ("code", """ranker_metrics_path = ROOT / "outputs" / "ranker_metrics.json"
if ranker_metrics_path.is_file():
    ranker_metrics = json.loads(ranker_metrics_path.read_text())
    print(json.dumps({k: v for k, v in ranker_metrics.items() if k != "fit"}, indent=2))
else:
    print("ranker model not trained yet")"""),

    ("markdown", """## 6. v2 solve of one evaluation-shaped task

The ranker orders candidates; `attempt_1` is the top-ranked prediction and
`attempt_2` the top-ranked distinct prediction."""),

    ("code", """solver = SolverV2(SolverV2Config(time_budget=12.0, beam_budget=4.0),
                 ProgramRanker(checkpoint=ROOT / "outputs" / "ranker_model.joblib"))
demo = eval_tasks[0]
result = solver.solve(demo)
print("task", demo.task_id, "| candidates", len(solver.last_candidates),
      "| method", result.method, "| runtime", round(result.runtime_s, 2))
print("attempt_1 shape", np.asarray(result.attempts[0][0]).shape,
      "| attempt_2 shape", np.asarray(result.attempts[0][1]).shape)"""),

    ("markdown", """## 7. Measured results (from `outputs/eval_metrics.json`)"""),

    ("code", """metrics = json.loads((ROOT / "outputs" / "eval_metrics.json").read_text())
summary = {k: metrics[k] for k in
           ("split", "tasks", "graded_pass1", "graded_pass1_rate", "graded_pass2",
            "graded_pass2_rate", "candidate_oracle_rate", "ranker_top1_rate",
            "wall_seconds", "runtime_mean_s")}
print(json.dumps(summary, indent=2))

family_table = pd.DataFrame(metrics["by_family"]).T
display(family_table.sort_values("candidates", ascending=False))

signature_table = pd.DataFrame(metrics["by_signature"]).T
display(signature_table[["tasks", "solved", "pass1_rate", "pass2_rate"]])"""),

    ("markdown", """## 8. Submission contract

`attempt_1` / `attempt_2` must exist for every task id, as a grid for a single
test input."""),

    ("code", """submission = json.loads((ROOT / "outputs" / "submission.json").read_text())
expected = {path.stem for path in eval_files}
print("entries", len(submission), "| expected", len(expected))
assert set(submission) == expected, "submission must cover every evaluation id"
for task_id, entry in list(submission.items())[:3]:
    grid = np.asarray(entry["attempt_1"])
    print(task_id, "attempt_1", grid.shape, "attempt_2", np.asarray(entry["attempt_2"]).shape)
assert all({"attempt_1", "attempt_2"} <= set(entry) for entry in submission.values())"""),

    ("markdown", """## 9. Honest limitations

* The object families cover selection, recolouring, deletion, movement, hole
  filling, outlining, stamping, connecting, panels, rays, windows, rescaling and
  denoising. They do **not** cover multi-step compositional rules of the kind
  that dominate ARC-AGI-2 evaluation.
* The ranker is trained on 1000 training tasks; its validation accuracy is the
  ceiling on its usefulness, and the evaluation split is harder (median grid
  18x19 versus 10x10).
* `pass@2 >= pass@1` by construction; v1 reported only the second slot, which
  understated it.
* Every acceptance requires exact reproduction of all training pairs; learned
  rules additionally report leave-one-out evidence that the ranker can use."""),

    ("code", """print("pass@1", metrics["graded_pass1"], "/", metrics["graded_tasks"],
      "| pass@2", metrics["graded_pass2"], "/", metrics["graded_tasks"],
      "| wall", metrics["wall_seconds"], "s")"""),
]


def execute_cell(source: str, namespace: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one code cell, capturing stdout as nbformat output entries."""
    buffer = io.StringIO()
    start = time.monotonic()
    with contextlib.redirect_stdout(buffer):
        exec(compile(source, "<cell>", "exec"), namespace)
    text = buffer.getvalue()
    outputs: List[Dict[str, Any]] = []
    if text:
        outputs.append({"output_type": "stream", "name": "stdout",
                        "text": text.splitlines(keepends=True)})
    outputs.append({"output_type": "execute_result",
                    "execution_count": None, "metadata": {},
                    "data": {"text/plain": [f"cell executed in {time.monotonic() - start:.2f}s"]}})
    return outputs


def build() -> Path:
    """Execute every cell and write the notebook document."""
    namespace: Dict[str, Any] = {"display": print, "__name__": "__main__"}
    cells: List[Dict[str, Any]] = []
    counter = 0
    for kind, source in CELLS:
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": source.splitlines(keepends=True)})
            continue
        counter += 1
        outputs = execute_cell(source, namespace)
        cells.append({"cell_type": "code", "execution_count": counter, "metadata": {},
                      "outputs": outputs, "source": source.splitlines(keepends=True)})
    document = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": sys_version()}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(document, indent=1))
    return NOTEBOOK


def sys_version() -> str:
    """Python version string for the notebook metadata."""
    import platform

    return platform.python_version()


if __name__ == "__main__":
    path = build()
    print(f"wrote and executed {path}")
