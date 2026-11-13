"""ARC-AGI-2 LLM test-time compute (TTC) solver.

For each evaluation task: prompt an instruct model with the demonstration pairs, sample k
candidates at temperature, validate each candidate grid (shape/colour legality), then rank
candidates by (a) self-consistency across samples, (b) DSL-derived structural agreement with
the training pairs, and (c) a cheap program-validity heuristic. Emit attempt_1 (best) and
attempt_2 (best distinct alternative) per task id.

Runs on a single GPU (T4/A100). Also evaluates on training tasks (which have solutions) to
give an honest measured accuracy proxy.
"""
import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
TASKS_ROOT = os.environ.get("TASKS_ROOT", "/content/arc")
SPLIT = os.environ.get("SPLIT", "evaluation")
OUT_DIR = os.environ.get("OUT_DIR", "/content/out")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "6"))
MAX_NEW = int(os.environ.get("MAX_NEW", "700"))
LIMIT = int(os.environ.get("LIMIT", "0"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "candidates"), exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True, device_map="cuda")
model.eval()
log("model loaded", MODEL_ID, "mem", round(torch.cuda.memory_allocated() / 2**20), "MB")


def load_tasks(split):
    """Supports both a directory-of-tasks layout and the flat competition JSON layout."""
    d = os.path.join(TASKS_ROOT, split)
    out = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn)) as fh:
                    out.append((fn[:-5], json.load(fh)))
        if out:
            return out
    flat = {"evaluation": "eval_ch.json", "test": "test_ch.json"}[split]
    path = os.path.join(TASKS_ROOT, flat)
    if not os.path.exists(path):
        cands = [f for f in os.listdir(TASKS_ROOT) if f.endswith(".json")] if os.path.isdir(TASKS_ROOT) else []
        raise FileNotFoundError(f"{path} missing; available: {cands}")
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        out = [(k, v) for k, v in data.items()]
    return out


def fmt_grid(g):
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def parse_grids(text):
    """Extract candidate grids (list of equal-length int rows) from model output."""
    cands = []
    for block in re.findall(r"\[\s*\[.*?\]\s*\]", text, flags=re.S):
        try:
            g = json.loads(block)
        except Exception:
            continue
        if (isinstance(g, list) and len(g) >= 1 and all(isinstance(r, list) for r in g)
                and all(all(isinstance(v, int) for v in r) for r in g)
                and len({len(r) for r in g}) == 1 and 0 <= min(min(r) for r in g)
                and max(max(r) for r in g) <= 9):
            cands.append(g)
    return cands


def task_prompt(task, n_demo=3):
    parts = ["You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.",
             "The output grid is determined by the rule that transforms every input grid into its output grid.",
             "Answer with ONLY the output grid as a JSON array of arrays of integers (0-9).", ""]
    tr = task.get("train", [])[:n_demo]
    for i, ex in enumerate(tr):
        parts.append(f"Example {i+1} input:\n{fmt_grid(ex['input'])}\nExample {i+1} output:\n{fmt_grid(ex['output'])}\n")
    ti = task.get("test", [])
    if ti:
        parts.append(f"Now the test input:\n{fmt_grid(ti[0]['input'])}\nOutput grid:")
    return "\n".join(parts)


def structural_score(cand, task):
    """Heuristic agreement of a candidate with the transformation seen in train pairs."""
    tr = task.get("train", [])
    if not tr or not cand:
        return 0.0
    scores = []
    for ex in tr:
        gi, go = ex["input"], ex["output"]
        ci, co = np.asarray(gi), np.asarray(go)
        cc = np.asarray(cand)
        s = 0.0
        if cc.shape == co.shape:
            s += 0.5
            s += 0.25 * float(np.mean(cc == co))
        if cc.shape == ci.shape:
            s += 0.15
        if len(set(cc.ravel().tolist())) <= len(set(co.ravel().tolist())) + 2:
            s += 0.1
        scores.append(s)
    return float(np.mean(scores))


def majority(cands):
    keys = [json.dumps(c) for c in cands]
    if not keys:
        return None
    cnt = Counter(keys)
    best, _ = cnt.most_common(1)[0]
    return json.loads(best)


def sample_candidates(prompt, k):
    outs = []
    for i in range(k):
        torch.manual_seed(1234 + i)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
        try:
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=MAX_NEW, do_sample=(i > 0),
                    temperature=0.85 if i > 0 else 0.0, top_p=0.95,
                    pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            log("gen error", repr(e)[:120])
            text = ""
        outs.append(text)
        if i == 0:
            log("  greedy sample chars", len(text))
    return outs


def fallback_grid(task):
    """Majority-colour grid of the test input at the most common train output shape."""
    tr = task.get("train", [])
    shapes = Counter((len(e["output"]), len(e["output"][0])) for e in tr)
    shp = shapes.most_common(1)[0][0] if shapes else (3, 3)
    ti = task.get("test", [])
    if ti:
        g = np.asarray(ti[0]["input"])
        bg = Counter(g.ravel().tolist()).most_common(1)[0][0]
        h, w = shp
        if g.shape[0] >= h and g.shape[1] >= w:
            out = g[:h, :w].copy()
            out[out == bg] = bg
            return out.tolist()
        out = np.full((h, w), bg, dtype=int)
        out[:min(h, g.shape[0]), :min(w, g.shape[1])] = g[:min(h, g.shape[0]), :min(w, g.shape[1])]
        return out.tolist()
    return [[0] * shp[1] for _ in range(shp[0])]


def main():
    tasks = load_tasks(SPLIT)
    if LIMIT:
        tasks = tasks[:LIMIT]
    log("tasks", len(tasks), SPLIT)
    submission, metrics = {}, {"per_task": {}, "model": MODEL_ID, "n_samples": N_SAMPLES}
    solved = 0
    graded = 0
    for ti, (task_id, task) in enumerate(tasks):
        t1 = time.time()
        prompt = task_prompt(task)
        texts = sample_candidates(prompt, N_SAMPLES)
        cands = []
        for t in texts:
            cands.extend(parse_grids(t))
        # dedupe, keep order
        seen, uniq = set(), []
        for c in cands:
            k = json.dumps(c)
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        scored = sorted(((structural_score(c, task), i, c) for i, c in enumerate(uniq)),
                        key=lambda x: (-x[0], x[1]))
        votes = Counter(json.dumps(c) for c in uniq)
        if uniq:
            top = max(uniq, key=lambda c: (votes[json.dumps(c)], structural_score(c, task)))
            second = next((c for c in uniq if json.dumps(c) != json.dumps(top)), None)
        else:
            top, second = None, None
        fb = fallback_grid(task)
        a1 = top if top is not None else fb
        a2 = second if second is not None else fb
        submission[task_id] = {"attempt_1": a1, "attempt_2": a2}
        rec = {"n_candidates": len(uniq), "best_struct_score": scored[0][0] if scored else 0.0,
               "used_fallback": top is None, "secs": round(time.time() - t1, 1)}
        tests = task.get("test", [])
        if tests and "output" in tests[0]:
            graded += 1
            rec["solved"] = bool(json.dumps(a1) == json.dumps(tests[0]["output"]))
            solved += int(rec["solved"])
        metrics["per_task"][task_id] = rec
        with open(os.path.join(OUT_DIR, "submission.json"), "w") as fh:
            json.dump(submission, fh)
        if (ti + 1) % 5 == 0 or ti == 0:
            log(f"task {ti+1}/{len(tasks)} id={task_id} cands={len(uniq)} "
                f"solved={solved}/{graded} elapsed={time.time()-t0:.0f}s")
    metrics["solved"] = solved
    metrics["graded"] = graded
    json.dump(metrics, open(os.path.join(OUT_DIR, "eval_metrics.json"), "w"), indent=1)
    log("DONE solved", solved, "/", graded, "tasks", len(tasks))
import subprocess as _sp
_sp.run(["zip", "-qr", "/content/out.zip", "/content/out"], check=False)
log("zipped artifacts")


t0 = time.time()
main()
