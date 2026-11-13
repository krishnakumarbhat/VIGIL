"""ARC Prize 2026 (ARC-AGI-2) - LLM test-time compute solver on Kaggle GPU.

Mounts the competition data, runs an instruct LLM with k-sample test-time compute over the
120 evaluation tasks (which have public solutions -> honest exact-match measurement) and then
over the 240 test tasks, emitting a submission.json in the EXACT competition schema:
    { task_id: [ {"attempt_1": <grid>, "attempt_2": <grid>} ] }
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

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"

def find_file(name):
    for root, dirs, files in os.walk(COMP):
        if name in files:
            return os.path.join(root, name)
    return None

eval_ch = find_file("arc-agi_evaluation_challenges.json")
eval_sol = find_file("arc-agi_evaluation_solutions.json")
test_ch = find_file("arc-agi_test_challenges.json")
log("files", eval_ch, eval_sol, test_ch)
if eval_ch is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        log("input tree", root, files[:8])

import subprocess, sys as _sys
subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=False)
try:
    import bitsandbytes as _bnb
    print("bitsandbytes", _bnb.__version__, flush=True)
except Exception as _e:
    print("bnb unavailable:", repr(_e)[:120], flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_NEW = int(os.environ.get("MAX_NEW", "900"))
os.environ.setdefault("MAX_INPUT_TOKENS", "3072")

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

qcfg = None
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes  # noqa: F401
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
except Exception as e:
    log("no 4bit path:", repr(e)[:140]); qcfg = None

kw = dict(trust_remote_code=True, device_map="cuda")
model = None
if qcfg is not None:
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=qcfg, **kw)
    except Exception as e:
        log("4bit load failed:", repr(e)[:140]); model = None
if model is None:
    MODEL_ID = os.environ.get("FALLBACK_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
model.eval()
log("model", MODEL_ID, "4bit", qcfg is not None, "mem", round(torch.cuda.memory_allocated()/2**20), "MB")


def fmt_grid(g):
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def _valid_grid(g):
    return (isinstance(g, list) and g and all(isinstance(r, list) for r in g)
            and all(all(isinstance(v, int) for v in r) for r in g)
            and len({len(r) for r in g}) == 1
            and 0 <= min(min(r) for r in g) and max(max(r) for r in g) <= 9)


def parse_grids(text):
    """Accept JSON list-of-lists AND space-separated digit rows."""
    cands = []
    for block in re.findall(r"\[\s*\[.*?\]\s*\]", text, flags=re.S):
        try:
            g = json.loads(block)
        except Exception:
            continue
        if _valid_grid(g):
            cands.append(g)
    run = []
    for ln in text.splitlines():
        toks = ln.strip().split()
        if toks and all(re.fullmatch(r"[0-9]", t) for t in toks):
            run.append([int(t) for t in toks])
        else:
            if run and _valid_grid(run):
                cands.append(run)
            run = []
    if run and _valid_grid(run):
        cands.append(run)
    return cands


def task_prompt(task, n_demo=4):
    p = ["You are solving an ARC puzzle. Infer the single rule that maps each input grid to its output grid.",
         "Reply with ONLY the output grid as a JSON array of arrays of integers 0-9. No explanation.", ""]
    for i, ex in enumerate(task.get("train", [])[:n_demo]):
        p.append(f"Example {i+1} INPUT:\n{fmt_grid(ex['input'])}\nExample {i+1} OUTPUT:\n{fmt_grid(ex['output'])}")
    t = task.get("test", [])
    if t:
        p.append(f"TEST INPUT:\n{fmt_grid(t[0]['input'])}\nOUTPUT:")
    return "\n".join(p)


def structural_score(cand, task):
    tr = task.get("train", [])
    if not tr or not cand:
        return 0.0
    cc = np.asarray(cand)
    scores = []
    for ex in tr:
        gi = np.asarray(ex["input"]); go = np.asarray(ex["output"])
        s = 0.0
        if cc.shape == go.shape:
            s += 0.45 + 0.3 * float(np.mean(cc == go))
        if cc.shape == gi.shape:
            s += 0.15
        if len(set(cc.ravel().tolist())) <= len(set(go.ravel().tolist())) + 2:
            s += 0.1
        scores.append(s)
    return float(np.mean(scores))


def fallback_grid(task):
    tr = task.get("train", [])
    shapes = Counter((len(e["output"]), len(e["output"][0])) for e in tr)
    shp = shapes.most_common(1)[0][0] if shapes else (3, 3)
    t = task.get("test", [])
    if not t:
        return [[0]*shp[1] for _ in range(shp[0])]
    g = np.asarray(t[0]["input"])
    bg = Counter(g.ravel().tolist()).most_common(1)[0][0]
    h, w = shp
    out = np.full((h, w), bg, dtype=int)
    out[:min(h, g.shape[0]), :min(w, g.shape[1])] = g[:min(h, g.shape[0]), :min(w, g.shape[1])]
    return out.tolist()


def generate(prompts, n_samples):
    """Batched sampling: prompts list -> list of candidate grids per prompt."""
    per = [[] for _ in prompts]
    for s in range(n_samples):
        greedy = (s == 0)
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=int(os.environ["MAX_INPUT_TOKENS"])).to("cuda")
        try:
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW,
                                     do_sample=not greedy,
                                     temperature=None if greedy else 0.9,
                                     top_p=None if greedy else 0.95,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            log("gen error", repr(e)[:140]); texts = [""] * len(prompts)
        for i, t in enumerate(texts):
            per[i].extend(parse_grids(t))
        log(f"  sample {s+1}/{n_samples} done; cands so far {[len(x) for x in per][:6]}")
    return per


def solve_batch(tasks, n_samples):
    prompts = [task_prompt(t) for _, t in tasks]
    cands = generate(prompts, n_samples)
    out = {}
    for (tid, task), cl in zip(tasks, cands):
        uniq, seen = [], set()
        for c in cl:
            k = json.dumps(c)
            if k not in seen:
                seen.add(k); uniq.append(c)
        votes = Counter(json.dumps(c) for c in uniq)
        if uniq:
            top = max(uniq, key=lambda c: (votes[json.dumps(c)], structural_score(c, task)))
            second = next((c for c in uniq if json.dumps(c) != json.dumps(top)), None)
        else:
            top = second = None
        fb = fallback_grid(task)
        out[tid] = {"attempt_1": top if top is not None else fb,
                    "attempt_2": second if second is not None else fb,
                    "_n_cand": len(uniq), "_used_fallback": top is None}
    return out


def main():
    global eval_ch, eval_sol, test_ch
    eval_ch = json.load(open(eval_ch))
    eval_sol = json.load(open(eval_sol)) if eval_sol else {}
    test_ch = json.load(open(test_ch))

    metrics = {"model": MODEL_ID, "n_samples": N_SAMPLES, "eval": {}, "test": {}}
    # --- evaluation (measurable) ---
    items = list(eval_ch.items())
    B = 4
    solved1 = solved2 = 0
    for i in range(0, len(items), B):
        chunk = items[i:i+B]
        res = solve_batch(chunk, N_SAMPLES)
        for tid, r in res.items():
            gold = eval_sol.get(tid)
            ok1 = bool(gold is not None and json.dumps(r["attempt_1"]) == json.dumps(gold))
            ok2 = bool(gold is not None and json.dumps(r["attempt_2"]) == json.dumps(gold))
            solved1 += ok1; solved2 += ok2
            metrics["eval"][tid] = {"solved1": ok1, "solved2": ok2, "n_cand": r["_n_cand"],
                                    "used_fallback": r["_used_fallback"]}
        log(f"eval {i+len(chunk)}/{len(items)} pass@1={solved1} pass@2={solved2}")
        json.dump(metrics, open(f"{OUT}/eval_metrics.json", "w"), indent=1)
    metrics["eval_pass1"] = solved1 / max(1, len(items))
    metrics["eval_pass2"] = (solved1 + solved2) / max(1, len(items))
    log("EVAL pass@1", metrics["eval_pass1"], "pass@2", metrics["eval_pass2"])

    # --- test set -> competition submission ---
    titems = list(test_ch.items())
    submission = {}
    for i in range(0, len(titems), B):
        chunk = titems[i:i+B]
        res = solve_batch(chunk, N_SAMPLES)
        for tid, r in res.items():
            submission[tid] = [{"attempt_1": r["attempt_1"], "attempt_2": r["attempt_2"]}]
            metrics["test"][tid] = {"n_cand": r["_n_cand"], "used_fallback": r["_used_fallback"]}
        log(f"test {i+len(chunk)}/{len(titems)}")
        with open(f"{OUT}/submission.json", "w") as fh:
            json.dump(submission, fh)
    json.dump(metrics, open(f"{OUT}/eval_metrics.json", "w"), indent=1)
    log("DONE test tasks", len(submission), "eval pass@1", metrics["eval_pass1"])


main()
