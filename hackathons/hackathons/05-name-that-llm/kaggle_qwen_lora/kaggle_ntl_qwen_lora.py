"""Track 05 `name-that-llm` - Kaggle T4 recovery run of the Colab Pro recipe.

Same design as the Colab Pro A100 run that reached 0.5063 grouped balanced accuracy:
  * base model  : Qwen2.5-7B-Instruct, 4-bit NF4 (bitsandbytes works on Kaggle CUDA 12.x)
  * task        : 18-way sequence classification of the source LLM
  * validation  : 5-fold StratifiedGroupKFold(shuffle=True, random_state=42), groups=prompt_id
  * training mix: real competition rows (2,699) + public mirror passages (9,290); the external
                  corpus is subsampled when USE_EXTERNAL=1
  * validation rows are the REAL competition rows only, so the reported metric is directly
    comparable with the LB and with the A100 run
  * every fold's OOF matrix is written to /kaggle/working (Kaggle persists kernel outputs, so
    the artefacts survive even if the run is interrupted)
"""
import json
import os
import time

import subprocess as _sp
import sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "-U", "torchao"], check=False)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MAXLEN = int(os.environ.get("MAX_LEN", "384"))
EPOCHS = float(os.environ.get("EPOCHS", "1.0"))
BS = int(os.environ.get("BS", "8"))
ACCUM = int(os.environ.get("ACCUM", "4"))
LR = float(os.environ.get("LR", "2e-4"))
USE_EXTERNAL = os.environ.get("USE_EXTERNAL", "0") == "1"
EXT_ROWS = int(os.environ.get("EXT_ROWS", "12000"))
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"
FOLD = int(os.environ.get("ONLY_FOLD", "-1"))

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}

subprocess_ok = True
try:
    import subprocess
    subprocess.run(["python3", "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=False)
except Exception:
    subprocess_ok = False


def find_file(name, root="/kaggle/input"):
    for r, d, fs in os.walk(root):
        if name in fs:
            return os.path.join(r, name)
    return None


real_path = find_file("train.csv")
log("real train:", real_path)
real = pd.read_csv(real_path)
real["source"] = "real"
real["n_words"] = real["text"].astype(str).str.split().str.len()
real["domain"] = ""
real["topic"] = ""
real["split"] = "competition"

mirror = None
for r, d, fs in os.walk("/kaggle/input"):
    if "passages.csv" in fs:
        mirror = pd.read_csv(os.path.join(r, "passages.csv"),
                             usecols=["prompt_id", "model", "text", "n_words", "domain", "topic", "split"])
        break
frames = [real[["prompt_id", "model", "text", "n_words", "domain", "topic", "split", "source"]]]
if mirror is not None:
    m = mirror.dropna(subset=["text"]).copy()
    m["source"] = "mirror"
    frames.append(m[["prompt_id", "model", "text", "n_words", "domain", "topic", "split", "source"]])
    log("mirror rows", len(m))

if USE_EXTERNAL:
    for r, d, fs in os.walk("/kaggle/input"):
        if "external_all_wide.parquet" in fs:
            ext = pd.read_parquet(os.path.join(r, "external_all_wide.parquet"))
            rows = []
            for c in [c for c in ext.columns if c not in ("id", "prompt")]:
                s = ext[c]
                idx = s.notna() & (s.astype(str).str.len() > 40)
                for i in ext.index[idx]:
                    rows.append({"prompt_id": f"ext_{ext.at[i,'id']}", "model": c,
                                 "text": str(ext.at[i, c])[:2000],
                                 "n_words": len(str(ext.at[i, c]).split()), "domain": "external",
                                 "topic": "", "split": "external", "source": "external"})
            e = pd.DataFrame(rows).sample(n=min(EXT_ROWS, len(rows)), random_state=42)
            frames.append(e)
            log("external rows", len(e))
            break

df = pd.concat(frames, ignore_index=True)
df = df[df.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
df["text"] = df["text"].astype(str).str.slice(0, 2500)
df["y"] = df["model"].map(lab2id)
df = df.dropna(subset=["y"]).reset_index(drop=True)
df["y"] = df["y"].astype(int)
log("train frame", df.shape, "prompts", df.prompt_id.nunique())

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

real_mask = (df.source == "real").to_numpy()
rdf = df[real_mask].reset_index()
log("real rows", len(rdf), "prompts", rdf.prompt_id.nunique())

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(rdf, rdf["y"], groups=rdf["prompt_id"]))

from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, BitsAndBytesConfig)

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

quant = None
if USE_4BIT:
    try:
        import bitsandbytes  # noqa: F401
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
    except Exception as e:
        log("4bit unavailable", repr(e)[:140]); quant = None

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=len(LABELS), quantization_config=quant,
    device_map="cuda", trust_remote_code=True)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.config.pad_token_id = tok.pad_token_id
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.cuda()
log("model ready", MODEL_ID, "4bit", quant is not None,
    "mem", round(torch.cuda.memory_allocated() / 2**20), "MB")


class Rows(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        enc = tok(str(row["text"]), truncation=True, max_length=MAXLEN,
                  padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, int(row["y"])


def collate(batch):
    feats = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    out = tok.pad({"input_ids": [f["input_ids"] for f in feats],
                   "attention_mask": [f["attention_mask"] for f in feats]}, return_tensors="pt")
    out["labels"] = ys
    return out


def evaluate(loader):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.cuda() for k, v in b.items()}
            lg = model(**b).logits
            ps.append(torch.softmax(lg.float(), -1).cpu().numpy())
            ys.append(b["labels"].cpu().numpy())
    model.train()
    return np.concatenate(ps), np.concatenate(ys)


oof = np.zeros((len(rdf), len(LABELS)), dtype=np.float32)
BF16 = torch.cuda.is_bf16_supported()
scaler = torch.amp.GradScaler("cuda") if not BF16 else None
status = []

for fold, (ti, vi) in enumerate(folds):
    if FOLD >= 0 and fold != FOLD:
        continue
    val = rdf.iloc[vi]
    train = df[~df.index.isin(val.index)]          # mirror + external + other real folds
    dl_tr = DataLoader(Rows(train), batch_size=BS, shuffle=True, collate_fn=collate,
                       num_workers=2, drop_last=True)
    dl_va = DataLoader(Rows(val), batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    total = max(1, int(len(dl_tr) * EPOCHS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.06)
    model.train()
    step, run, done = 0, 0.0, False
    while not done:
        for b in dl_tr:
            b = {k: v.cuda() for k, v in b.items()}
            out = model(**b)
            loss = out.loss / ACCUM
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            run += out.loss.item()
            if (step + 1) % ACCUM == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if scaler is not None:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if step < total - 1:
                    sch.step()
                step += 1
                if step % 50 == 0:
                    log(f"fold {fold} step {step}/{total} loss {run/(50*ACCUM):.4f}")
                    run = 0.0
                if step >= total:
                    done = True
                    break
    p, y = evaluate(dl_va)
    oof[vi] = p
    acc = balanced_accuracy_score(y, p.argmax(1))
    status.append({"fold": fold, "bal_acc": float(acc), "n_val": int(len(vi))})
    log(f"FOLD {fold} bal_acc {acc:.4f} (n_val={len(vi)})")
    np.save(f"{OUT}/oof_real_fold{fold}.npy", p)
    np.save(f"{OUT}/oof_real_idx_fold{fold}.npy", val.index.to_numpy())
    np.save(f"{OUT}/oof_real_y_fold{fold}.npy", y)
    json.dump(status, open(f"{OUT}/fold_status.json", "w"), indent=1)
    model.save_pretrained(f"{OUT}/adapter_fold{fold}")

if status:
    covered = np.zeros(len(rdf), dtype=bool)
    for fold, (ti, vi) in enumerate(folds):
        covered[vi] = True
    meta = {"model": MODEL_ID, "folds": status,
            "mean_bal_acc": float(np.mean([s["bal_acc"] for s in status])),
            "max_len": MAXLEN, "epochs": EPOCHS, "lr": LR, "train_rows": int(len(df))}
    json.dump(meta, open(f"{OUT}/cv_report.json", "w"), indent=1)
    log("MEAN fold bal_acc", meta["mean_bal_acc"])
log("DONE")
