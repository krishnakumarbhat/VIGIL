"""Track 05 `name-that-llm` — heavy GPU burner on Colab Pro (T4).

5-fold StratifiedGroupKFold (group=prompt_id) LoRA sequence-classification over the
18 frontier-model labels, trained on competition passages + external corpus, with
per-fold OOF logits so a full OOF prediction table can be assembled offline.

Runs for a wall-clock budget (JOB_HOURS) and checkpoints the adapter every
CHECK_EVERY steps so a killed session does not lose work.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B")
DATA_PATH = os.environ.get("DATA_PATH", "/content/data/train_all.parquet")
OUT_DIR = os.environ.get("OUT_DIR", "/content/out")
LABELS_PATH = os.environ.get("LABELS_PATH", "/content/data/labels.json")
MAX_LEN = int(os.environ.get("MAX_LEN", "384"))
EPOCHS = float(os.environ.get("EPOCHS", "2.0"))
BATCH = int(os.environ.get("BATCH", "4"))
ACCUM = int(os.environ.get("ACCUM", "4"))
JOB_HOURS = float(os.environ.get("JOB_HOURS", "5.0"))
CHECK_EVERY = int(os.environ.get("CHECK_EVERY", "300"))
LR = float(os.environ.get("LR", "2e-4"))
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"

# argv overrides: MODEL_ID DATA_PATH LABELS_PATH OUT_DIR JOB_HOURS MAX_LEN EPOCHS BATCH ACCUM
import sys as _sys
_argmap = {"MODEL_ID": 0, "DATA_PATH": 1, "LABELS_PATH": 2, "OUT_DIR": 3,
           "JOB_HOURS": 4, "MAX_LEN": 5, "EPOCHS": 6, "BATCH": 7, "ACCUM": 8}
for _k, _i in _argmap.items():
    if len(_sys.argv) > _i and _sys.argv[_i] != "":
        globals()[_k] = _sys.argv[_i]


MAX_LEN=int(MAX_LEN); EPOCHS=float(EPOCHS); BATCH=int(BATCH); ACCUM=int(ACCUM); JOB_HOURS=float(JOB_HOURS)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "folds"), exist_ok=True)
T0 = time.time()

labels = json.load(open(LABELS_PATH))
lab2id = {l: i for i, l in enumerate(labels)}
print("labels", len(labels), flush=True)


def log(*a):
    print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)


df = pd.read_parquet(DATA_PATH)
df["y"] = df["model"].map(lab2id)
df = df.dropna(subset=["y"]).reset_index(drop=True)
df["y"] = df["y"].astype(int)
log("rows", len(df), "prompts", df.prompt_id.nunique(), "classes", df.y.nunique())

from sklearn.model_selection import StratifiedGroupKFold

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for f, (_, va) in enumerate(skf.split(df, df["y"], groups=df["prompt_id"])):
    df.loc[va, "fold"] = f
log("fold sizes", df.fold.value_counts().sort_index().to_dict())

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    qcfg = None
    if USE_4BIT:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    _dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(labels), quantization_config=qcfg,
            dtype=_dt, trust_remote_code=True, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(labels), quantization_config=qcfg,
            torch_dtype=_dt, trust_remote_code=True, attn_implementation="sdpa")
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.cuda()
    log("model ready", MODEL_ID, "mem", round(torch.cuda.memory_allocated() / 2**20), "MB")
except Exception as e:
    log("MODEL_INIT_ERROR", repr(e))
    raise


class DS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = tok(str(r["text"]), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, int(r["y"])


def collate(batch):
    feats = [b[0] for b in batch]
    labels_ = torch.tensor([b[1] for b in batch], dtype=torch.long)
    out = tok.pad(
        {"input_ids": [f["input_ids"] for f in feats], "attention_mask": [f["attention_mask"] for f in feats]},
        return_tensors="pt",
    )
    out["labels"] = labels_
    return out


def evaluate(model, loader, dev):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            lg = model(**b).logits.float()
            all_p.append(torch.softmax(lg, -1).cpu().numpy())
            all_y.append(b["labels"].cpu().numpy())
    model.train()
    return np.concatenate(all_p), np.concatenate(all_y)


best_fold = -1
for fold in range(5):
    tr = df[df.fold != fold]
    va = df[df.fold == fold]
    dl_tr = DataLoader(DS(tr), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    dl_va = DataLoader(DS(va), batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = max(1, int(len(dl_tr) * EPOCHS)) // ACCUM
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(1, steps), pct_start=0.06)
    USE_SCALER = (not torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler("cuda") if USE_SCALER else None
    log(f"=== fold {fold}: train {len(tr)} val {len(va)} opt_steps {steps}")

    model.train()
    step, tstep, running = 0, 0, 0.0
    done = False
    while not done:
        for b in dl_tr:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                out = model(**b)
            loss = out.loss / ACCUM
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += out.loss.item()
            tstep += 1
            if tstep % ACCUM == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if step < steps - 1:
                    sched.step()
                step += 1
                if step % 25 == 0:
                    log(f"  fold {fold} step {step}/{steps} loss {running/(25*ACCUM):.4f} mem {round(torch.cuda.memory_allocated()/2**20)}MB")
                    running = 0.0
                if step % CHECK_EVERY == 0 or step >= steps:
                    p, y = evaluate(model, dl_va, "cuda")
                    from sklearn.metrics import balanced_accuracy_score
                    acc = balanced_accuracy_score(y, p.argmax(1))
                    log(f"  fold {fold} CHECKPOINT step {step} val_bal_acc {acc:.4f}")
                    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_oof.npy"), p)
                    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_y.npy"), y)
                    va_idx = va.index.to_numpy()
                    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_idx.npy"), va_idx)
                    model.save_pretrained(os.path.join(OUT_DIR, "folds", f"adapter_fold{fold}"))
                    json.dump({"fold": fold, "step": step, "val_bal_acc": float(acc)},
                              open(os.path.join(OUT_DIR, "folds", f"fold{fold}_status.json"), "w"))
                    if acc > best_fold:
                        best_fold = acc
                if step >= steps or (time.time() - T0) > JOB_HOURS * 3600:
                    done = True
                    break
        if (time.time() - T0) > JOB_HOURS * 3600:
            log("TIME BUDGET reached during fold", fold)
            done = True
    p, y = evaluate(model, dl_va, "cuda")
    from sklearn.metrics import balanced_accuracy_score
    acc = balanced_accuracy_score(y, p.argmax(1))
    log(f"FOLD {fold} FINAL val_bal_acc {acc:.4f}")
    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_oof.npy"), p)
    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_y.npy"), y)
    np.save(os.path.join(OUT_DIR, "folds", f"fold{fold}_idx.npy"), va.index.to_numpy())
    model.save_pretrained(os.path.join(OUT_DIR, "folds", f"adapter_fold{fold}"))
    json.dump({"fold": fold, "final_bal_acc": float(acc), "steps": step},
              open(os.path.join(OUT_DIR, "folds", f"fold{fold}_status.json"), "w"))
    if (time.time() - T0) > JOB_HOURS * 3600:
        log("TIME BUDGET reached, stopping after fold", fold)
        break

# per-source OOF breakdown
from sklearn.metrics import balanced_accuracy_score as _ba
_oof = np.zeros((len(df), len(labels)), dtype=np.float32)
for _f in range(5):
    _m = df.fold == _f
    _p = np.load(os.path.join(OUT_DIR, "folds", f"fold{_f}_oof.npy"))
    _i = np.load(os.path.join(OUT_DIR, "folds", f"fold{_f}_idx.npy"))
    _oof[_i] = _p
_break = {}
for _src in sorted(df.source.unique()):
    _m = (df.source == _src).to_numpy()
    if _m.sum():
        _break[_src] = {"n": int(_m.sum()), "bal_acc": float(_ba(df.y.to_numpy()[_m], _oof[_m].argmax(1)))}
np.save(os.path.join(OUT_DIR, "oof_all.npy"), _oof)
json.dump(_break, open(os.path.join(OUT_DIR, "oof_by_source.json"), "w"), indent=1)
log("OOF by source", _break)
tok.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
json.dump({"labels": labels, "model_id": MODEL_ID, "max_len": MAX_LEN, "lr": LR,
           "job_hours": JOB_HOURS, "finished_at": time.time()},
          open(os.path.join(OUT_DIR, "run_meta.json"), "w"), indent=1)
log("DONE. artifacts in", OUT_DIR)
