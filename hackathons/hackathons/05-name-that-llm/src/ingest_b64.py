"""Recover artefacts that a Colab job printed as base64 blocks in its exec log.

`colab download` depends on a local session registry that can be pruned while the remote kernel
is still healthy, which silently orphans the runtime and makes its filesystem unreachable. Every
training script therefore emits artefacts in-band:

    BEGIN_<tag> <n_bytes> <n_base64_chars>
    <chunks>
    END_<tag>

This module scans a log file, validates the decoded length, and writes each block to an output
directory. It is idempotent: rerunning rewrites the same files.
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

BEGIN = re.compile(r"^BEGIN_(\S+) (\d+) (\d+)$")


def recover(log_path: Path, out_dir: Path, prefix_filter: str | None = None) -> list[Path]:
    """Decode every BEGIN/END block in ``log_path`` into ``out_dir`` and return written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = log_path.read_text(errors="replace").splitlines()
    written: list[Path] = []
    i = 0
    while i < len(lines):
        m = BEGIN.match(lines[i].strip())
        if not m:
            i += 1
            continue
        tag, n_bytes, n_chars = m.group(1), int(m.group(2)), int(m.group(3))
        i += 1
        acc: list[str] = []
        while i < len(lines) and not lines[i].startswith(f"END_{tag}"):
            acc.append(lines[i].strip())
            i += 1
        i += 1
        blob = "".join(acc)
        if len(blob) != n_chars:
            print(f"WARN {tag}: expected {n_chars} chars, got {len(blob)}")
            continue
        try:
            raw = base64.b64decode(blob, validate=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN {tag}: bad base64 ({exc})")
            continue
        if len(raw) != n_bytes:
            print(f"WARN {tag}: expected {n_bytes} bytes, got {len(raw)}")
            continue
        name = tag if prefix_filter is None or tag.startswith(prefix_filter) else None
        if name is None:
            continue
        path = out_dir / f"{tag}.bin"
        path.write_bytes(raw)
        written.append(path)
        print(f"recovered {tag} -> {path} ({len(raw)} bytes)")
    return written


if __name__ == "__main__":
    log = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/colab_pro_artifacts")
    flt = sys.argv[3] if len(sys.argv) > 3 else None
    files = recover(log, out, flt)
    print(f"{len(files)} artefacts recovered")
