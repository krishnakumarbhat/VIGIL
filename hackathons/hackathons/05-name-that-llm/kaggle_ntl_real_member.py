"""Track 05 `name-that-llm` - GPU ensemble member on the REAL competition train set.

Mounts the competition data (competition_sources) plus the public mirror corpora
(dataset_sources), then trains 5-fold StratifiedGroupKFold (group=prompt_id) sequence
classifiers over the 18 model labels. Saves per-fold OOF probability matrices (full coverage of
the real train rows), test-set probabilities, and a status file with the grouped balanced
accuracy, so the local cross-fitted stacker can blend this model with the Colab Pro LoRA.

Models are selected by MODEL_ID (env var); the kernel can be pushed once per model.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/deberta-v3-base")
MAXLEN = int(os.environ.get("MAX_LEN", "320"))
EPOCHS = float(os.environ.get("EPOCHS", "3.0"))
BS = int(os.environ.get("BS", "16"))
LR = float(os.environ.get("LR", "2e-5"))
SEED = int(os.environ.get("SEED", "42"))
USE_MIRROR = os.environ.get("USE_MIRROR", "0") == "1"

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}


def find_file(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    return None


tr_path = find_file("train.csv")
te_path = find_file("test.csv")
log("train", tr_path, "test", te_path)
tr = pd.read_csv(tr_path)
te = pd.read_csv(te_path)
log("train", tr.shape, "test", te.shape)

tr["y"] = tr["model"].map(lab2id)
tr = tr.dropna(subset=["y"]).reset_index(drop=True)
tr["y"] = tr["y"].astype(int)
log("labelled train rows", len(tr), "prompts", tr.prompt_id.nunique())

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), dtype=torch.float32)
except TypeError:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), torch_dtype=torch.float32)
model.cuda()
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()


class DS(Dataset):
    def __init__(self, texts, ys):
        self.texts = texts
        self.ys = ys

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = tok(self.texts[i], truncation=True, max_length=MAXLEN,
                  padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, int(self.ys[i])


def collate(batch):
    feats = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch])
    out = tok.pad({"input_ids": [f["input_ids"] for f in feats],
                   "attention_mask": [f["attention_mask"] for f in feats]}, return_tensors="pt")
    out["labels"] = ys
    return out


skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(tr), len(LABELS)), dtype=np.float32)
test_probs = np.zeros((len(te), len(LABELS)), dtype=np.float32)
fold_acc = []

for fold, (ti, vi) in enumerate(skf.split(tr, tr["y"], groups=tr["prompt_id"])):
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    dtr = DataLoader(DS(tr.text.iloc[ti].tolist(), tr.y.iloc[ti].tolist()),
                     batch_size=BS, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)
    dva = DataLoader(DS(tr.text.iloc[vi].tolist(), tr.y.iloc[vi].tolist()),
                     batch_size=32, shuffle=False, collate_fn=collate, num_workers=4)
    dte = DataLoader(DS(te["text"].astype(str).tolist(), [0] * len(te)),
                     batch_size=32, shuffle=False, collate_fn=collate, num_workers=4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                             weight_decay=0.01)
    steps = max(1, int(len(dtr) * EPOCHS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    step, run = 0, 0.0
    done = False
    while not done:
        for b in dtr:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            scaler.scale(loss).backward()
            run += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if step < steps - 1:
                sch.step()
            step += 1
            if step % 50 == 0:
                log(f"fold {fold} step {step}/{steps} loss {run/50:.4f}")
                run = 0.0
            if step >= steps:
                done = True
                break

    def infer(loader):
        model.eval()
        ps = []
        with torch.no_grad():
            for b in loader:
                b = {k: v.cuda() for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(**b).logits
                ps.append(torch.softmax(lg.float(), -1).cpu().numpy())
        model.train()
        return np.concatenate(ps)

    oof[vi] = infer(dva)
    test_probs += infer(dte) / 5
    acc = balanced_accuracy_score(tr.y.iloc[vi].to_numpy(), oof[vi].argmax(1))
    fold_acc.append(float(acc))
    log(f"FOLD {fold} bal_acc {acc:.4f}")
    np.save(f"{OUT}/oof_{SEED}.npy", oof)
    np.save(f"{OUT}/test_{SEED}.npy", test_probs)
    json.dump({"model": MODEL_ID, "seed": SEED, "fold": fold, "bal_acc": acc,
               "max_len": MAXLEN, "epochs": EPOCHS},
              open(f"{OUT}/status_{SEED}_fold{fold}.json", "w"))
    model.save_pretrained(f"{OUT}/model_fold{fold}")
    tok.save_pretrained(f"{OUT}/model_fold{fold}")

y_all = tr.y.to_numpy()
meta = {"model": MODEL_ID, "seed": SEED, "bal_acc_oof": float(balanced_accuracy_score(y_all, oof.argmax(1))),
        "acc_oof": float(accuracy_score(y_all, oof.argmax(1))), "fold_bal_acc": fold_acc,
        "rows": int(len(tr)), "prompts": int(tr.prompt_id.nunique())}
json.dump(meta, open(f"{OUT}/cv_report_{SEED}.json", "w"), indent=1)
log("OOF bal_acc", meta["bal_acc_oof"], "acc", meta["acc_oof"])
tr[["id", "prompt_id", "model", "y"]].to_parquet(f"{OUT}/row_index_{SEED}.parquet", index=False)
log("DONE")
