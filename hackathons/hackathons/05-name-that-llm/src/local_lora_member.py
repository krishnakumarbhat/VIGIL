"""Track 05 `name-that-llm` - local GPU LoRA member (RTX 3050, <3.2 GB VRAM).

5-fold StratifiedGroupKFold (group=prompt_id) over the REAL competition train rows, training on
mirror passages as augmentation (mirror prompts are disjoint from the competition prompts, so no
leakage), validating on the real rows only. Writes:
  outputs/oof_local_<tag>.npy   - (n_real, 18) out-of-fold probabilities
  outputs/test_local_<tag>.npy  - probabilities for data/competition/test.csv
  outputs/local_<tag>_status.json
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

D = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm"
OUT = f"{D}/outputs"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAXLEN = int(os.environ.get("MAX_LEN", "384"))
EPOCHS = float(os.environ.get("EPOCHS", "3.0"))
BS = int(os.environ.get("BS", "4"))
ACCUM = int(os.environ.get("ACCUM", "4"))
LR = float(os.environ.get("LR", "2e-4"))
TAG = os.environ.get("TAG", "qwen05b")
SEED = int(os.environ.get("SEED", "42"))
USE_MIRROR = os.environ.get("USE_MIRROR", "1") == "1"
MIRROR_ROWS = int(os.environ.get("MIRROR_ROWS", "9000"))

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}

real = pd.read_csv(f"{D}/data/competition/train.csv")
real["y"] = real["model"].map(lab2id)
real = real.dropna(subset=["y"]).reset_index(drop=True)
real["y"] = real["y"].astype(int)
log("real rows", len(real), "prompts", real.prompt_id.nunique())

train_pool = real.copy()
if USE_MIRROR:
    mp = f"{D}/data/compact/passages.parquet"
    if os.path.exists(mp):
        mir = pd.read_parquet(mp)
        mir = mir[mir.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
        mir = mir.sample(n=min(MIRROR_ROWS, len(mir)), random_state=42)
        mir["y"] = mir["model"].map(lab2id)
        mir = mir.dropna(subset=["y"]).reset_index(drop=True)
        mir["y"] = mir["y"].astype(int)
        train_pool = pd.concat([real, mir], ignore_index=True)
        log("mirror augmentation rows", len(mir), "pool", len(train_pool))

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
_attn = os.environ.get("ATTN_IMPL", "sdpa")
_DTYPE = torch.bfloat16 if os.environ.get("DTYPE", "bf16") == "bf16" else torch.float32
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), dtype=_DTYPE, trust_remote_code=True,
        attn_implementation=_attn)
except TypeError:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), torch_dtype=_DTYPE, trust_remote_code=True,
        attn_implementation=_attn)
except Exception as e:
    log("attn_implementation failed, retrying default:", repr(e)[:120])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), dtype=_DTYPE, trust_remote_code=True)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.config.pad_token_id = tok.pad_token_id
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.cuda()
log("model ready", MODEL_ID, "mem", round(torch.cuda.memory_allocated() / 2**20), "MB")


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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
                lg = model(**b).logits
            ps.append(torch.softmax(lg.float(), -1).cpu().numpy())
            ys.append(b["labels"].cpu().numpy())
    model.train()
    return np.concatenate(ps), np.concatenate(ys)


skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(real), len(LABELS)), dtype=np.float32)
BF16 = torch.cuda.is_bf16_supported()
scaler = torch.amp.GradScaler("cuda") if (not BF16 and _DTYPE == torch.float32) else None
accs = []

torch.manual_seed(SEED)
np.random.seed(SEED)

for fold, (ti, vi) in enumerate(skf.split(real, real["y"], groups=real["prompt_id"])):
    val = real.iloc[vi]
    train = train_pool[~train_pool.index.isin(val.index)]
    torch.manual_seed(SEED * 100 + fold)
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
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
                if step % 10 == 0:
                    log(f"fold {fold} step {step}/{total} loss {run/(10*ACCUM):.4f} "
                        f"mem {round(torch.cuda.memory_allocated()/2**20)}MB")
                    run = 0.0
                if step >= total:
                    done = True
                    break
    p, y = evaluate(dl_va)
    oof[vi] = p
    acc = balanced_accuracy_score(y, p.argmax(1))
    accs.append(float(acc))
    log(f"FOLD {fold} bal_acc {acc:.4f} (n_val={len(vi)})")
    np.save(f"{OUT}/oof_local_{TAG}_fold{fold}.npy", p)
    np.save(f"{OUT}/oof_local_{TAG}_idx_fold{fold}.npy", val.index.to_numpy())
    np.save(f"{OUT}/oof_local_{TAG}_y_fold{fold}.npy", y)
    json.dump({"folds": accs, "mean_bal_acc": float(np.mean(accs)), "model": MODEL_ID},
              open(f"{OUT}/local_{TAG}_status.json", "w"), indent=1)

np.save(f"{OUT}/oof_local_{TAG}.npy", oof)
meta = {"model": MODEL_ID, "mean_bal_acc": float(np.mean(accs)), "folds": accs,
        "max_len": MAXLEN, "epochs": EPOCHS, "rows": int(len(real)),
        "prompts": int(real.prompt_id.nunique()), "mirror_rows": int(len(train_pool) - len(real))}
json.dump(meta, open(f"{OUT}/local_{TAG}_status.json", "w"), indent=1)
log("MEAN grouped bal_acc", meta["mean_bal_acc"])
with open(f"{D}/logs/experiments.tsv", "a") as fh:
    fh.write(f"05-name-that-llm\tlocal_lora_{TAG}_bal_acc\t{meta['mean_bal_acc']:.5f}\t"
             f"{MODEL_ID} LoRA 5-fold SGKF(real-only validation, mirror-augmented training), "
             f"local RTX3050, folds={accs}\n")
log("DONE")
