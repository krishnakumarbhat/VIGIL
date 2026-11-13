"""Track 05 `name-that-llm` - Colab Pro burner: LoRA sequence classification, 18-way.

Trains a LoRA sequence-classification head over the 18 frontier-model labels with
5-fold StratifiedGroupKFold (group=prompt_id), so every row gets an out-of-fold
prediction that is honest under the competition's grouped validation rule.

Outputs (under /content/out):
  folds/fold{f}_oof.npy   - (n_val, 18) softmax probabilities
  folds/fold{f}_idx.npy  - row indices into the training frame
  folds/fold{f}_y.npy    - true labels
  folds/fold{f}_status.json - fold id, step, val balanced accuracy
  oof_all.npy, oof_by_source.json, run_meta.json
Everything is zipped to /content/out.zip for a single-file download.
"""
import json
import os
import subprocess
import sys
import time

CONFIG = {
    "MODEL_ID": os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B"),
    "DATA_PATH": os.environ.get("DATA_PATH", "/content/train_all_v2.parquet"),
    "LABELS_PATH": os.environ.get("LABELS_PATH", "/content/labels.json"),
    "OUT_DIR": os.environ.get("OUT_DIR", "/content/out"),
    "JOB_HOURS": os.environ.get("JOB_HOURS", "4.2"),
    "MAX_LEN": os.environ.get("MAX_LEN", "384"),
    "EPOCHS": os.environ.get("EPOCHS", "1.0"),
    "BATCH": os.environ.get("BATCH", "8"),
    "ACCUM": os.environ.get("ACCUM", "4"),
    "USE_4BIT": os.environ.get("USE_4BIT", "0"),
    "SAMPLE_PER_FOLD": os.environ.get("SAMPLE_PER_FOLD", "11000"),
    "TOKENIZERS_PARALLELISM": "false",
}
os.environ.update(CONFIG)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


log("CONFIG", CONFIG)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46.1"], check=False)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

OUT_DIR = CONFIG["OUT_DIR"]
os.makedirs(os.path.join(OUT_DIR, "folds"), exist_ok=True)
MAX_LEN = int(CONFIG["MAX_LEN"])
EPOCHS = float(CONFIG["EPOCHS"])
BATCH = int(CONFIG["BATCH"])
ACCUM = int(CONFIG["ACCUM"])
JOB_HOURS = float(CONFIG["JOB_HOURS"])
USE_4BIT = CONFIG["USE_4BIT"] == "1"
BF16 = torch.cuda.is_bf16_supported()

labels = json.load(open(CONFIG["LABELS_PATH"]))
lab2id = {name: i for i, name in enumerate(labels)}
log("labels", len(labels))

df = pd.read_parquet(CONFIG["DATA_PATH"])
df["y"] = df["model"].map(lab2id)
df = df.dropna(subset=["y"]).reset_index(drop=True)
df["y"] = df["y"].astype(int)
if "source" not in df.columns:
    df["source"] = "all"
log("rows", len(df), "prompts", df.prompt_id.nunique(), "classes", df.y.nunique())

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for f, (_, va) in enumerate(skf.split(df, df["y"], groups=df["prompt_id"])):
    df.loc[va, "fold"] = f
log("fold sizes", df.fold.value_counts().sort_index().to_dict())

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained(CONFIG["MODEL_ID"], trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
log("pad token", tok.pad_token, tok.pad_token_id)

quant_cfg = None
if USE_4BIT:
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
            bnb_4bit_use_double_quant=True)
    except Exception as exc:
        log("4bit unavailable", repr(exc)[:140])
        quant_cfg = None

model = None
if quant_cfg is not None:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["MODEL_ID"], num_labels=len(labels), quantization_config=quant_cfg,
            device_map="cuda", trust_remote_code=True)
    except Exception as exc:
        log("4bit load failed", repr(exc)[:140])
        model = None
if model is None:
    alt = os.environ.get("FALLBACK_MODEL_ID", "Qwen/Qwen2.5-1.5B")
    if alt != CONFIG["MODEL_ID"]:
        log("fallback model", alt)
        tok = AutoTokenizer.from_pretrained(alt, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        CONFIG["MODEL_ID"] = alt
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["MODEL_ID"], num_labels=len(labels),
            dtype=torch.bfloat16 if BF16 else torch.float16,
            trust_remote_code=True)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["MODEL_ID"], num_labels=len(labels),
            torch_dtype=torch.bfloat16 if BF16 else torch.float16,
            trust_remote_code=True)

lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.config.pad_token_id = tok.pad_token_id
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.cuda()
log("model ready", CONFIG["MODEL_ID"], "4bit", quant_cfg is not None,
    "mem", round(torch.cuda.memory_allocated() / 2**20), "MB")


class Rows(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        enc = tok(str(row["text"]), truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, int(row["y"])


def collate(batch):
    feats = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    out = tok.pad({"input_ids": [f["input_ids"] for f in feats],
                   "attention_mask": [f["attention_mask"] for f in feats]},
                  return_tensors="pt")
    out["labels"] = ys
    return out


def evaluate(loader):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16 if BF16 else torch.float16):
                logits = model(**batch).logits
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
            ys.append(batch["labels"].cpu().numpy())
    model.train()
    return np.concatenate(probs), np.concatenate(ys)


oof = np.zeros((len(df), len(labels)), dtype=np.float32)
covered = np.zeros(len(df), dtype=bool)
use_scaler = not BF16
scaler = torch.amp.GradScaler("cuda") if use_scaler else None

for fold in range(5):
    tr = df[df.fold != fold]
    va = df[df.fold == fold]
    cap = int(CONFIG["SAMPLE_PER_FOLD"])
    if cap and len(tr) > cap:
        counts = tr.y.value_counts()
        keep = []
        for cls, n in counts.items():
            idx = tr.index[tr.y == cls]
            k = max(1, int(round(cap * n / counts.sum())))
            keep.extend(idx.to_numpy()[:k])
        tr = tr.loc[keep]
        log(f"fold {fold}: subsampled train -> {len(tr)} rows")
    dl_tr = DataLoader(Rows(tr), batch_size=BATCH, shuffle=True, collate_fn=collate,
                       num_workers=2, drop_last=True)
    dl_va = DataLoader(Rows(va), batch_size=16, shuffle=False, collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=2e-4, weight_decay=0.01)
    total_steps = max(1, int(len(dl_tr) * EPOCHS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=total_steps,
                                                pct_start=0.06)
    model.train()
    step, micro, running, done = 0, 0, 0.0, False
    while not done:
        for batch in dl_tr:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16 if BF16 else torch.float16):
                out = model(**batch)
            loss = out.loss / ACCUM
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += out.loss.item()
            micro += 1
            if micro % ACCUM == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if step < total_steps - 1:
                    sched.step()
                step += 1
                if step % 50 == 0:
                    log(f"fold {fold} step {step}/{total_steps} loss {running / (50 * ACCUM):.4f} "
                        f"mem {round(torch.cuda.memory_allocated() / 2**20)}MB")
                    running = 0.0
                if step % 200 == 0 or step >= total_steps:
                    probs, ys = evaluate(dl_va)
                    acc = balanced_accuracy_score(ys, probs.argmax(1))
                    log(f"fold {fold} CHECKPOINT step {step} val_bal_acc {acc:.4f}")
                    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_oof.npy"), probs)
                    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_y.npy"), ys)
                    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_idx.npy"),
                            va.index.to_numpy())
                    oof[va.index.to_numpy()] = probs
                    covered[va.index.to_numpy()] = True
                    json.dump({"fold": fold, "step": step, "val_bal_acc": float(acc)},
                              open(os.path.join(OUT_DIR, "folds", f"fold{fold}_status.json"), "w"))
                    model.save_pretrained(os.path.join(OUT_DIR, "folds", f"adapter_fold{fold}"))
                if step >= total_steps or (time.time() - T0) > JOB_HOURS * 3600:
                    done = True
                    break
        if (time.time() - T0) > JOB_HOURS * 3600:
            log("TIME BUDGET reached in fold", fold)
            done = True
    if (time.time() - T0) > JOB_HOURS * 3600:
        log("stopping after fold", fold, "due to time budget")
        break

np.save(os.path.join(OUT_DIR, "oof_all.npy"), oof)
breakdown = {}
for src in sorted(df.source.unique()):
    mask = (df.source == src).to_numpy() & covered
    if mask.sum():
        breakdown[src] = {"n": int(mask.sum()),
                          "bal_acc": float(balanced_accuracy_score(df.y.to_numpy()[mask],
                                                                    oof[mask].argmax(1)))}
json.dump(breakdown, open(os.path.join(OUT_DIR, "oof_by_source.json"), "w"), indent=1)
if covered.any():
    mask = covered
    log("OVERALL OOF bal_acc",
        balanced_accuracy_score(df.y.to_numpy()[mask], oof[mask].argmax(1)),
        "coverage", int(mask.sum()), "/", len(df))
json.dump({"labels": labels, "config": CONFIG, "oof_by_source": breakdown,
           "finished_at": time.time()},
          open(os.path.join(OUT_DIR, "run_meta.json"), "w"), indent=1)
tok.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
subprocess.run(["zip", "-qr", "/content/out.zip", "/content/out"], check=False)
log("DONE - artifacts zipped at /content/out.zip")
