"""Track 05 - train and persist the full-data Qwen2.5-0.5B adapter used for inference.

The cross-fitted member `local_qwen05b_s1337` (OOF balanced accuracy 0.6483) is the second
strongest component of the 0.6817 blend, but cross-validation only ever produced per-fold models.
This script trains the identical architecture and recipe once on **all** label-carrying rows
(real + 6,000 mirror passages) and saves the adapter, so the ensemble recipe can be applied to the
private test set without retraining.

Recipe (mirrors the OOF run): LoRA r=16 / alpha=32 over all seven projection matrices, bf16
weights, gradient checkpointing, MAX_LEN 256, batch 8, OneCycle at 2e-4 for a single epoch.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

D = Path("/home/pope/Desktop/gpu /hackathons/05-name-that-llm")
OUT = D / "outputs" / "deploy" / "lora_qwen05b"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAXLEN = int(os.environ.get("MAX_LEN", "256"))
BS = int(os.environ.get("BS", "8"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "1.0"))
MIRROR_ROWS = int(os.environ.get("MIRROR_ROWS", "6000"))
SEED = int(os.environ.get("SEED", "1337"))

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


real = pd.read_csv(D / "data/competition/train.csv")
real["y"] = real["model"].map(lab2id)
real = real.dropna(subset=["y"]).reset_index(drop=True)
mir = pd.read_parquet(D / "data/compact/passages.parquet")
mir = mir[mir.model.isin(LABELS)].dropna(subset=["text"])
mir = mir.sample(n=min(MIRROR_ROWS, len(mir)), random_state=42).reset_index(drop=True)
mir["y"] = mir["model"].map(lab2id).astype(int)
pool = pd.concat([real[["text", "y"]], mir[["text", "y"]]], ignore_index=True)
pool["text"] = pool["text"].astype(str)
log("pool", len(pool), "real", len(real), "mirror", len(mir))

torch.manual_seed(SEED)
np.random.seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

enc = tok(pool.text.tolist(), truncation=True, max_length=MAXLEN, padding="max_length",
          return_tensors="np")
ids = enc["input_ids"].astype(np.int32)
mask = enc["attention_mask"].astype(np.int32)
y = pool.y.to_numpy().astype(np.int64)
log("tokenised", ids.shape)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=len(LABELS), dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa")
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.config.pad_token_id = tok.pad_token_id
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.cuda()
model.print_trainable_parameters()

steps = max(1, int(np.ceil(len(y) / BS) * EPOCHS))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                        weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
rng = np.random.default_rng(SEED)
model.train()
step = 0
for epoch in range(int(np.ceil(EPOCHS))):
    order = rng.permutation(len(y))
    for s in range(0, len(order), BS):
        if step >= steps:
            break
        b = order[s:s + BS]
        if len(b) < 2:
            continue
        x = torch.from_numpy(ids[b]).cuda()
        m = torch.from_numpy(mask[b]).cuda()
        t = torch.from_numpy(y[b]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=m).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), t, label_smoothing=0.05)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            log(f"step {step}/{steps} loss {loss.item():.4f} mem {torch.cuda.memory_allocated()/2**20:.0f}MB")

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
json.dump({"base_model": MODEL_ID, "labels": LABELS, "max_len": MAXLEN, "steps": step,
           "rows": int(len(pool)), "mirror_rows": int(len(mir)), "seed": SEED,
           "note": "full-data refit of the local_qwen05b_s1337 OOF member"},
          open(OUT / "deploy_meta.json", "w"), indent=1)
log("saved adapter to", OUT, "files", len(list(OUT.iterdir())))
log("DONE")
