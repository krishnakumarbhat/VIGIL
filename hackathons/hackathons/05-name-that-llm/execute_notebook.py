"""Minimal notebook executor (no jupyter/nbconvert in this environment).

Runs every code cell of a ``.ipynb`` in a shared namespace, captures stdout and
tracebacks into the cell ``outputs`` list, and writes the executed notebook back
in place.  Non-zero exit if any cell raised.

Usage: ``python3 execute_notebook.py notebooks/inference_submission.ipynb``
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path


class NotebookExecutor:
    """Executes code cells sequentially in one globals dict."""

    def __init__(self, path: Path) -> None:
        """Load the notebook JSON."""
        self.path = Path(path)
        self.nb = json.loads(self.path.read_text())
        self.ns: dict = {"__name__": "__main__"}

    @staticmethod
    def _source(cell: dict) -> str:
        """Join a cell's source list into one string."""
        src = cell.get("source", "")
        return "".join(src) if isinstance(src, list) else src

    def run(self) -> int:
        """Execute all code cells; return the number of failures."""
        failures = 0
        counter = 0
        for cell in self.nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            counter += 1
            buf = io.StringIO()
            outputs: list[dict] = []
            code = self._source(cell)
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, f"<cell {counter}>", "exec"), self.ns)
            except Exception:
                failures += 1
                text = buf.getvalue() + traceback.format_exc()
                outputs.append({"output_type": "stream", "name": "stderr",
                                "text": text.splitlines(keepends=True)})
                print(f"cell {counter} FAILED\n{text}")
            else:
                text = buf.getvalue()
                if text:
                    outputs.append({"output_type": "stream", "name": "stdout",
                                    "text": text.splitlines(keepends=True)})
                print(f"cell {counter} ok")
            cell["execution_count"] = counter
            cell["outputs"] = outputs
        self.path.write_text(json.dumps(self.nb, indent=1))
        return failures


if __name__ == "__main__":
    target = Path(sys.argv[1])
    sys.exit(1 if NotebookExecutor(target).run() else 0)
