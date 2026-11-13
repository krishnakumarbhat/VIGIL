import os, json, time, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
          'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
          'glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra',
          'gpt-6-astra','kimi-k3','qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}

IN = "/kaggle/input"
def find_csv(name, must_have=None):
    for root, d, fs in os.walk(IN):
        for f in fs:
            p = os.path.join(root, f)
            if f == name or (f.endswith(".csv") and name in f):
                return p
    return None

# ---- load passages (competition-like, 18 labels) ----
passages = None
for root, d, fs in os.walk(IN):
    for f in fs:
        if f == "passages.csv":
            passages = pd.read_csv(os.path.join(root, f), usecols=["prompt_id","model","text","n_words","domain","topic","split"])
            break
    if passages is not None: break
log("passages", None if passages is None else passages.shape)

# ---- external wide corpus ----
ext = None
for root, d, fs in os.walk(IN):
    for f in fs:
        if f == "external_all_wide.parquet":
            ext = pd.read_parquet(os.path.join(root, f))
            break
    if ext is not None: break
log("ext", None if ext is None else ext.shape)

frames = []
if passages is not None:
    p = passages.dropna(subset=["text"]).copy()
    p["source"] = "comp"; p["topic"] = p.get("topic", "")
    frames.append(p[["prompt_id","model","text","n_words","domain","topic","split","source"]])
if ext is not None:
    rows = []
    for c in [c for c in ext.columns if c not in ("id","prompt")]:
        s = ext[c]
        idx = s.notna() & (s.astype(str).str.len() > 40)
        for i in ext.index[idx]:
            rows.append({"prompt_id": f"ext_{ext.at[i,'id']}", "model": c, "text": str(ext.at[i, c])[:2000],
                         "n_words": len(str(ext.at[i, c]).split()), "domain": "external", "topic": "", "split": "external", "source": "external"})
    e = pd.DataFrame(rows)
    cnt = e.groupby("prompt_id").model.nunique()
    e = e[e.prompt_id.isin(set(cnt[cnt >= 4].index))].sample(n=min(50000, len(e)), random_state=42)
    e["text"] = e["text"].str.slice(0, 2000)
    frames.append(e)
df = pd.concat(frames, ignore_index=True)
df = df[df.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
df["text"] = df["text"].astype(str).str.slice(0, 2000)
df["y"] = df.model.map(lab2id)
log("train rows", len(df), "prompts", df.prompt_id.nunique())

from sklearn.model_selection import StratifiedGroupKFold
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for f, (_, va) in enumerate(skf.split(df, df.y, groups=df.prompt_id)):
    df.loc[va, "fold"] = f
log("folds", df.fold.value_counts().sort_index().to_dict())

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import balanced_accuracy_score, accuracy_score

MODEL = os.environ.get("MODEL_ID", "microsoft/deberta-v3-base")
MAXLEN = int(os.environ.get("MAX_LEN", "320"))
EPOCHS = float(os.environ.get("EPOCHS", "2.0"))
BS = int(os.environ.get("BS", "16"))
LR = float(os.environ.get("LR", "2e-5"))
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS), dtype=torch.float32)
except TypeError:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS), torch_dtype=torch.float32)
model.cuda()
if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()

class DS(Dataset):
    def __init__(self, d): self.d = d.reset_index(drop=True)
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        e = tok(str(r["text"]), truncation=True, max_length=MAXLEN, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, int(r["y"])
def collate(b):
    feats = [x[0] for x in b]; ys = torch.tensor([x[1] for x in b])
    out = tok.pad({"input_ids":[f["input_ids"] for f in feats], "attention_mask":[f["attention_mask"] for f in feats]}, return_tensors="pt")
    out["labels"] = ys
    return out

oof = np.zeros((len(df), len(LABELS)), dtype=np.float32)
for fold in range(5):
    tr, va = df[df.fold != fold], df[df.fold == fold]
    dtr = DataLoader(DS(tr), batch_size=BS, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)
    dva = DataLoader(DS(va), batch_size=32, shuffle=False, collate_fn=collate, num_workers=4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = max(1, int(len(dtr) * EPOCHS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda")
    model.train(); step = 0; run = 0.0
    done = False
    while not done:
        for b in dtr:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            scaler.scale(loss).backward(); run += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if step < steps - 1: sch.step()
            step += 1
            if step % 100 == 0:
                log(f"fold{fold} step {step}/{steps} loss {run/100:.4f}"); run = 0.0
            if step >= steps: done = True; break
    model.eval(); ps = []
    with torch.no_grad():
        for b in dva:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**b).logits.float()
            ps.append(torch.softmax(lg, -1).cpu().numpy())
    P = np.concatenate(ps); y = va.y.to_numpy()
    oof[va.index.to_numpy()] = P
    acc = balanced_accuracy_score(y, P.argmax(1))
    log(f"FOLD {fold} bal_acc {acc:.4f} plain {accuracy_score(y, P.argmax(1)):.4f}")
    np.save(f"{OUT}/oof_deberta.npy", oof)
    df.iloc[va.index].assign(fold=fold, pred=P.argmax(1), y=y).to_parquet(f"{OUT}/oof_deberta_fold{fold}.parquet", index=False)
    model.save_pretrained(f"{OUT}/deberta_fold{fold}")
    tok.save_pretrained(f"{OUT}/deberta_fold{fold}")

y_all = df.y.to_numpy()
meta = {"model": MODEL, "max_len": MAXLEN, "epochs": EPOCHS, "lr": LR, "rows": len(df),
        "bal_acc_oof": float(balanced_accuracy_score(y_all, oof.argmax(1))),
        "acc_oof": float(accuracy_score(y_all, oof.argmax(1))), "labels": LABELS}
json.dump(meta, open(f"{OUT}/cv_report.json", "w"), indent=1)
log("OOF bal_acc", meta["bal_acc_oof"], "acc", meta["acc_oof"])
df[["prompt_id","model","y","fold"]].to_parquet(f"{OUT}/oof_index.parquet", index=False)
log("DONE")
