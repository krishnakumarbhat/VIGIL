"""Track 05 - DeBERTa-v3-large 5-fold member (A100 second-stage job).

A different architecture from the Qwen members is the cheapest way to add real ensemble diversity:
the linear members proved that decorrelated errors matter more than individual strength (a 0.36
TF-IDF model lifted the blend by four points).

Protocol matches every other Track 05 member: 5-fold StratifiedGroupKFold(shuffle, seed 42,
groups=prompt_id) over the 2,699 real rows, the held-out fold's real rows removed from training,
mirror passages (disjoint `g###` prompt namespace) always available.

Artefacts are written per fold as `oof_dl3l_fold{i}.npy` / `oof_dl3l_idx_fold{i}.npy`, which the
generic member loader in `src/stack_v2.py` picks up automatically once copied into
`outputs/`.
"""
import json
import math
import os
import shutil
import subprocess as _sp
import sys as _sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

_sp.run([_sys.executable, "-m", "pip", "install", "-q", "-U", "sentencepiece"], check=False)

OUT = os.environ.get("OUT_DIR", "/content/out_dl")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()



def emit_b64(path, tag, chunk=3500):
    """Print a file as base64 with BEGIN/END markers so stdout capture is lossless.

    `colab download` proved unreliable (the local session registry can be pruned mid-run,
    orphaning an otherwise healthy runtime), so artefacts are shipped in-band over the exec
    stdout stream instead. Chunk width stays well below any line-length limit.
    """
    import base64
    raw = open(path, "rb").read()
    blob = base64.b64encode(raw).decode("ascii")
    print(f"BEGIN_{tag} {len(raw)} {len(blob)}", flush=True)
    for i in range(0, len(blob), chunk):
        print(blob[i:i + chunk], flush=True)
    print(f"END_{tag}", flush=True)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


MODEL_ID = os.environ.get("MODEL_ID", "microsoft/deberta-v3-large")
MAXLEN = int(os.environ.get("MAX_LEN", "256"))
BS = int(os.environ.get("BS", "16"))
ACCUM = int(os.environ.get("ACCUM", "1"))
LR = float(os.environ.get("LR", "8e-5"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
DATA = os.environ.get("DATA_PATH", "/content/payload.parquet")
PREFIX = os.environ.get("PREFIX", "dl3l")

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}

payload = pd.read_parquet(DATA)
payload = payload[payload.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
payload["y"] = payload["model"].map(lab2id).astype(int)
real = payload[payload.source == "real"].reset_index(drop=True)
aux = payload[payload.source != "real"].reset_index(drop=True)
log("real", len(real), "aux", len(aux))

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"


def encode(texts):
    enc = tok(texts, truncation=True, max_length=MAXLEN, padding="max_length",
              return_tensors="np")
    ids = enc["input_ids"].astype(np.int32)
    mask = enc["attention_mask"].astype(np.int32)
    return ids, mask


real_ids, real_mask = encode(real.text.tolist())
aux_ids, aux_mask = encode(aux.text.tolist())
log("tokenised", real_ids.shape, aux_ids.shape)

# DeBERTa-v2 (the backbone of v3) has no SDPA path and raises if asked for one.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=len(LABELS), dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
model.cuda()
log("model ready", MODEL_ID, "mem", round(torch.cuda.memory_allocated() / 2**30, 1), "GB")

from sklearn.metrics import accuracy_score, balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
             .split(real, real.y, groups=real.prompt_id))
y_real = real.y.to_numpy()
rng = np.random.default_rng(42)


@torch.no_grad()
def infer(ids, mask, bs=128):
    model.eval()
    out = []
    for s in range(0, len(ids), bs):
        x = torch.from_numpy(ids[s:s + bs]).cuda()
        m = torch.from_numpy(mask[s:s + bs]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(input_ids=x, attention_mask=m).logits.float().cpu().numpy())
    return np.concatenate(out)


oof = np.zeros((len(real), len(LABELS)), dtype=np.float32)
meta_folds = []
for fold, (ti, vi) in enumerate(folds):
    tr_ids = np.concatenate([aux_ids, real_ids[ti]])
    tr_mask = np.concatenate([aux_mask, real_mask[ti]])
    tr_y = np.concatenate([aux.y.to_numpy(), y_real[ti]])
    steps_per_epoch = math.ceil(len(tr_y) / BS) // ACCUM
    total = max(1, steps_per_epoch * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, int(total * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    model.train()
    step = micro = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(len(tr_y))
        for s in range(0, len(order), BS):
            b = order[s:s + BS]
            if len(b) < 2:
                continue
            x = torch.from_numpy(tr_ids[b]).cuda()
            m = torch.from_numpy(tr_mask[b]).cuda()
            y = torch.from_numpy(tr_y[b]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=m).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y,
                                                         label_smoothing=SMOOTH) / ACCUM
            loss.backward()
            micro += 1
            if micro % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step < total:
                    sched.step()
                if step % 100 == 0:
                    log(f"fold{fold} ep{epoch} step {step}/{total} loss {loss.item() * ACCUM:.4f}")
    logits = infer(real_ids[vi], real_mask[vi])
    probs = torch.softmax(torch.from_numpy(logits), -1).numpy()
    oof[vi] = probs
    bal = float(balanced_accuracy_score(y_real[vi], probs.argmax(1)))
    meta_folds.append({"fold": fold, "bal_acc": bal,
                       "acc": float(accuracy_score(y_real[vi], probs.argmax(1)))})
    log(f"FOLD {fold} bal_acc {bal:.4f} acc {meta_folds[-1]['acc']:.4f}")
    np.save(f"{OUT}/oof_{PREFIX}_fold{fold}.npy", probs.astype(np.float32))
    np.save(f"{OUT}/oof_{PREFIX}_idx_fold{fold}.npy", vi)
    emit_b64(f"{OUT}/oof_{PREFIX}_fold{fold}.npy", f"{PREFIX}_probs_{fold}")
    emit_b64(f"{OUT}/oof_{PREFIX}_idx_fold{fold}.npy", f"{PREFIX}_idx_{fold}")
    json.dump({"model": MODEL_ID, "folds": meta_folds, "max_len": MAXLEN},
              open(f"{OUT}/{PREFIX}_status.json", "w"), indent=1)

meta = {"model": MODEL_ID, "folds": meta_folds,
        "mean_bal_acc": float(np.mean([f["bal_acc"] for f in meta_folds])),
        "oof_bal_acc": float(balanced_accuracy_score(y_real, oof.argmax(1)))}
json.dump(meta, open(f"{OUT}/{PREFIX}_cv_report.json", "w"), indent=1)
log("MEAN", round(meta["mean_bal_acc"], 4), "OOF", round(meta["oof_bal_acc"], 4))

shutil.make_archive(f"/content/{PREFIX}_bundle", "zip", OUT)
log("bundle", round(os.path.getsize(f'/content/{PREFIX}_bundle.zip') / 2**20, 2), "MB")
emit_b64(f"/content/{PREFIX}_bundle.zip", f"{PREFIX}_bundle")
log("DONE")
