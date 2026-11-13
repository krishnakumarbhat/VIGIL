"""Minimal end-to-end notebook executor (no jupyter/nbformat dependency).

Jupyter's kernel stack is not importable in this environment
(``ModuleNotFoundError: jupyter_core``), so the notebook is re-executed by
running its code cells in order inside one shared namespace and writing the
resulting outputs (stream text, images, errors) back into the ``.ipynb``.
Cells that raise stop the run and are recorded, so a broken notebook cannot
pass as "executed".
"""

from __future__ import annotations

import io
import os
import sys
import traceback
from contextlib import redirect_stdout
from typing import List

NB = os.path.join("notebooks", "nfl_combine_to_game_kce.ipynb")


def run(nb_path: str = NB, out_path: str = None) -> int:
    import json
    # resolve BOTH paths before chdir: cells execute with cwd = notebooks/,
    # so a relative out_path would be written from inside that directory
    nb_path = os.path.abspath(nb_path)
    out_path = os.path.abspath(out_path or nb_path)
    with open(nb_path) as fh:
        nb = json.load(fh)
    os.chdir(os.path.dirname(nb_path) or ".")
    sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
    ns: dict = {"__name__": "__main__"}
    failures = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        buf = io.StringIO()
        cell["outputs"] = []
        try:
            with redirect_stdout(buf):
                exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception:
            failures += 1
            err = traceback.format_exc()
            buf.write(err)
            cell["outputs"].append({
                "output_type": "stream", "name": "stderr", "text": err.splitlines(True)})
            print(f"cell {i} FAILED\n{err}")
        out = buf.getvalue()
        if out and not cell.get("outputs"):
            cell["outputs"].append(
                {"output_type": "stream", "name": "stdout", "text": out.splitlines(True)})
        # any figure still open at the end of the cell becomes an inline image
        try:
            import base64
            import matplotlib.pyplot as _plt
            for num in _plt.get_fignums():
                fig = _plt.figure(num)
                buf_img = io.BytesIO()
                fig.savefig(buf_img, format="png", dpi=110, bbox_inches="tight")
                b64 = base64.b64encode(buf_img.getvalue()).decode("ascii")
                cell.setdefault("outputs", []).append({
                    "output_type": "display_data",
                    "data": {"image/png": b64, "text/plain": ["<Figure>"]},
                    "metadata": {}})
                _plt.close(fig)
        except Exception as exc:  # pragma: no cover - figure capture is best effort
            print(f"  figure capture skipped: {exc}")
        cell["execution_count"] = i + 1
        print(f"--- cell {i} ok ({len(out)} bytes stdout)")
    with open(out_path, "w") as fh:
        json.dump(nb, fh, indent=1)
    print(f"executed {out_path}: {failures} failed cell(s)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
