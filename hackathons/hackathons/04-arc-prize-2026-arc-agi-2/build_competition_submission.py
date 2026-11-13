"""Assemble the ARC Prize 2026 ARC-AGI-2 competition submission in the EXACT schema.

Schema (verified against data/competition/sample_submission.json):
    { task_id: [ {"attempt_1": <2D int grid>, "attempt_2": <2D int grid>} ] }
for all 240 ids in arc-agi_test_challenges.json.

Merges candidate sources in priority order:
  1. LLM test-time-compute predictions (outputs/kaggle_ttc/submission.json or outputs/pro_artifacts/submission.json)
  2. DSL/heuristic solver predictions (outputs/submission.json - v1 schema {"attempt_1":..,"attempt_2":..})
  3. Majority-colour fallback grid of the test input at the most common train output shape.

Usage:
    python3 build_competition_submission.py [--check] [--report]
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

TRACK = "/home/pope/Desktop/gpu /hackathons/04-arc-prize-2026-arc-agi-2"
COMP = f"{TRACK}/data/competition"


def load_json(path, default=None):
    if path and os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception as e:
            print(f"  WARN could not read {path}: {e!r}")
    return default


def norm(entry):
    """Normalise any candidate entry to (attempt_1, attempt_2)."""
    if entry is None:
        return None, None
    if isinstance(entry, list) and entry and isinstance(entry[0], dict):
        entry = entry[0]
    if not isinstance(entry, dict):
        return None, None
    a1 = entry.get("attempt_1")
    a2 = entry.get("attempt_2")
    return a1, a2


def valid_grid(g):
    return (isinstance(g, list) and len(g) > 0 and isinstance(g[0], list) and len(g[0]) > 0
            and all(isinstance(r, list) and len(r) == len(g[0]) for r in g)
            and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r))


def fallback(task):
    tr = task.get("train", [])
    shapes = Counter((len(e["output"]), len(e["output"][0])) for e in tr)
    shp = shapes.most_common(1)[0][0] if shapes else (3, 3)
    t = task.get("test", [])
    if not t:
        return [[0] * shp[1] for _ in range(shp[0])]
    g = np.asarray(t[0]["input"])
    bg = Counter(g.ravel().tolist()).most_common(1)[0][0]
    h, w = shp
    out = np.full((h, w), bg, dtype=int)
    out[:min(h, g.shape[0]), :min(w, g.shape[1])] = g[:min(h, g.shape[0]), :min(w, g.shape[1])]
    return out.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    sample = load_json(f"{COMP}/sample_submission.json", {})
    test_ch = load_json(f"{COMP}/arc-agi_test_challenges.json", {})
    if not test_ch:
        print("ERROR: test challenges missing")
        return 1
    print(f"test tasks: {len(test_ch)}  sample_submission keys: {len(sample)}")

    sources = [
        ("llm_ttc_kaggle", f"{TRACK}/outputs/kaggle_ttc/submission.json"),
        ("llm_ttc_pro", f"{TRACK}/outputs/pro_artifacts/submission.json"),
        ("llm_ttc_local", f"{TRACK}/outputs/llm_ttc/submission.json"),
        ("dsl_solver", f"{TRACK}/outputs/submission.json"),
    ]
    loaded = []
    for name, path in sources:
        data = load_json(path)
        if isinstance(data, dict) and data:
            loaded.append((name, data))
            print(f"source {name}: {len(data)} entries ({path})")

    merged = {}
    stats = Counter()
    for tid, task in test_ch.items():
        chosen = None
        for name, data in loaded:
            if tid in data:
                a1, a2 = norm(data[tid])
                if valid_grid(a1):
                    chosen = (a1, a2 if valid_grid(a2) else a1, name)
                    break
        if chosen:
            merged[tid] = [{"attempt_1": chosen[0], "attempt_2": chosen[1]}]
            stats[chosen[2]] += 1
        else:
            fb = fallback(task)
            merged[tid] = [{"attempt_1": fb, "attempt_2": fb}]
            stats["fallback"] += 1

    out_path = f"{TRACK}/outputs/submission_competition.json"
    with open(out_path, "w") as fh:
        json.dump(merged, fh)
    print(f"wrote {out_path} with {len(merged)} ids; sources={dict(stats)}")

    # validation against the official sample schema
    problems = []
    if set(merged.keys()) != set(sample.keys()):
        problems.append(f"id mismatch: missing={len(set(sample)-set(merged))} extra={len(set(merged)-set(sample))}")
    for tid, v in merged.items():
        if not (isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict)
                and set(v[0].keys()) == {"attempt_1", "attempt_2"}):
            problems.append(f"{tid}: bad envelope")
            break
        for k in ("attempt_1", "attempt_2"):
            if not valid_grid(v[0][k]):
                problems.append(f"{tid}.{k}: invalid grid")
                break
    print("VALIDATION:", "PASS" if not problems else f"FAIL {problems[:5]}")
    if args.report:
        # measured accuracy on the 120 evaluation tasks using the public solutions
        sol = load_json(f"{COMP}/arc-agi_evaluation_solutions.json", {})
        ch = load_json(f"{COMP}/arc-agi_evaluation_challenges.json", {})
        ev_sub = None
        for name, path in sources:
            d = load_json(path)
            if isinstance(d, dict) and any(k in sol for k in d):
                ev_sub = d
                print(f"evaluating source {name} against {len(sol)} public solutions")
                break
        if ev_sub and sol:
            hit = 0
            for tid, gold in sol.items():
                a1, _ = norm(ev_sub.get(tid))
                if a1 is not None and json.dumps(a1) == json.dumps(gold):
                    hit += 1
            print(f"MEASURED evaluation pass@1 = {hit}/{len(sol)} = {hit/max(1,len(sol)):.4f}")
            json.dump({"eval_pass1": hit / max(1, len(sol)), "n": len(sol), "hit": hit},
                      open(f"{TRACK}/outputs/eval_metrics_measured.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
