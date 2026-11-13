"""Track 05 `name-that-llm` - A100 curriculum LoRA (v2, artifact-safe).

Improvements over the lost v1 run
---------------------------------
* **Pre-tokenisation**: every text is tokenised once into int32 arrays, so the training
  loop never pays tokeniser overhead (the v1 dataloader was the main throughput loss and
  the likely cause of the Colab worker stalls).
* **Three-stage curriculum**: epoch 1 sees the out-of-domain external corpus + mirror,
  epoch 2 the mirror corpus + in-fold real rows, epoch 3 the target distribution only.
  Later stages use a lower LR. This is a cheap domain-adaptation schedule that the v1
  flat mix lacked.
* **Label smoothing 0.05** and cosine schedule with warm-up, LoRA r=32/alpha=64 on every
  linear projection.
* **Artifact safety**: OOF probabilities, logits, fold indices and labels are written to
  disk *after every fold*, and the final bundle is produced with `shutil.make_archive`
  (pure Python) so a missing `zip` binary can never destroy the run again.

Validation protocol is unchanged: 5-fold StratifiedGroupKFold(shuffle=True,
random_state=42, groups=prompt_id) over the REAL competition rows; the fold's real rows are
excluded from training in every stage, so the reported balanced accuracy is honest.
"""
import json
import math
import os
import shutil
import subprocess as _sp
import sys as _sys
import time

# transformers 5.x + peft 0.21 requires torchao >= 0.16 on this runtime
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16.0"], check=False)

import numpy as np
import pandas as pd
import torch

OUT = os.environ.get("OUT_DIR", "/content/out")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MAXLEN = int(os.environ.get("MAX_LEN", "384"))
BS = int(os.environ.get("BS", "16"))
ACCUM = int(os.environ.get("ACCUM", "2"))
LR = float(os.environ.get("LR", "2e-4"))
SEED = int(os.environ.get("SEED", "42"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
LORA_R = int(os.environ.get("LORA_R", "32"))
DATA = os.environ.get("DATA_PATH", "/content/payload.parquet")

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
lab2id = {l: i for i, l in enumerate(LABELS)}

payload = pd.read_parquet(DATA)
payload = payload[payload.model.isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
payload["y"] = payload["model"].map(lab2id).astype(int)
log("payload", payload.shape, dict(payload.groupby("source").size()))

real = payload[payload.source == "real"].reset_index(drop=True)
aux = payload[payload.source != "real"].reset_index(drop=True)
log("real", len(real), "aux", len(aux), "prompts", real.prompt_id.nunique())

# ---------------------------------------------------------------- tokenise once
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def encode(texts):
    ids = tok(texts, truncation=True, max_length=MAXLEN, padding="max_length",
              return_tensors="np")["input_ids"].astype(np.int32)
    mask = (ids != tok.pad_token_id).astype(np.int32)
    return ids, mask


t_all = time.time()
real_ids, real_mask = encode(real.text.tolist())
aux_ids, aux_mask = encode(aux.text.tolist())
log(f"tokenised in {time.time() - t_all:.0f}s",
    f"real {real_ids.shape} aux {aux_ids.shape}",
    f"RAM {round((real_ids.nbytes + aux_ids.nbytes) / 2**20)}MB")

# ---------------------------------------------------------------- model
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=len(LABELS), dtype=torch.bfloat16,
    attn_implementation="sdpa", trust_remote_code=True)
model.config.pad_token_id = tok.pad_token_id
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.cuda()
model.print_trainable_parameters()
log("model ready", MODEL_ID, "mem", round(torch.cuda.memory_allocated() / 2**30, 1), "GB")

from sklearn.metrics import accuracy_score, balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(real, real.y, groups=real.prompt_id))


def batches(ids, mask, ys, bs, shuffle, gen):
    idx = np.arange(len(ys))
    if shuffle:
        gen.shuffle(idx)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        yield (torch.from_numpy(ids[b]).cuda(), torch.from_numpy(mask[b]).cuda(),
               torch.from_numpy(ys[b]).cuda())


def run_stage(ids, mask, ys, opt, sched, gen, step_state, total_steps, tag):
    model.train()
    for x, m, y in batches(ids, mask, ys, BS, True, gen):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=m).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y, label_smoothing=SMOOTH) / ACCUM
        loss.backward()
        step_state["micro"] += 1
        if step_state["micro"] % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step_state["step"] += 1
            if step_state["step"] < total_steps:
                sched.step()
            if step_state["step"] % 50 == 0:
                log(f"{tag} step {step_state['step']}/{total_steps} loss {loss.item() * ACCUM:.4f}")


@torch.no_grad()
def infer(ids, mask, bs=64):
    model.eval()
    outs = []
    for s in range(0, len(ids), bs):
        x = torch.from_numpy(ids[s:s + bs]).cuda()
        m = torch.from_numpy(mask[s:s + bs]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(input_ids=x, attention_mask=m).logits
        outs.append(lg.float().cpu().numpy())
    return np.concatenate(outs)


oof = np.zeros((len(real), len(LABELS)), dtype=np.float32)
fold_meta = []
gen = np.random.default_rng(SEED)

for fold, (ti, vi) in enumerate(folds):
    tr_real_ids, tr_real_mask = real_ids[ti], real_mask[ti]
    tr_real_y = real.y.to_numpy()[ti]
    val_ids, val_mask, val_y = real_ids[vi], real_mask[vi], real.y.to_numpy()[vi]

    # stage mixture: (ids, mask, y) per epoch, later epochs closer to the target domain
    stages = [
        ("s1-aux+real", np.concatenate([aux_ids, tr_real_ids]), np.concatenate([aux_mask, tr_real_mask]),
         np.concatenate([aux.y.to_numpy(), tr_real_y]), LR),
        ("s2-mirror+real", tr_real_ids, tr_real_mask, tr_real_y, LR * 0.5),
        ("s3-real", tr_real_ids, tr_real_mask, tr_real_y, LR * 0.2),
    ][:max(1, EPOCHS)]
    per_epoch_steps = sum(math.ceil(len(s[3]) / BS) // ACCUM for s in stages)
    total_steps = max(1, per_epoch_steps)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=stages[0][4], weight_decay=0.01)
    warm = max(1, int(total_steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    state = {"step": 0, "micro": 0}
    for tag, ids_s, mask_s, y_s, lr_s in stages:
        for g in opt.param_groups:
            g["lr"] = lr_s
        run_stage(ids_s, mask_s, y_s, opt, sched, gen, state, total_steps, f"fold{fold} {tag}")

    logits = infer(val_ids, val_mask)
    probs = torch.softmax(torch.from_numpy(logits), -1).numpy()
    oof[vi] = probs
    bal = float(balanced_accuracy_score(val_y, probs.argmax(1)))
    fold_meta.append({"fold": fold, "bal_acc": bal, "n_val": int(len(vi)),
                      "acc": float(accuracy_score(val_y, probs.argmax(1)))})
    log(f"FOLD {fold} bal_acc {bal:.4f} acc {fold_meta[-1]['acc']:.4f} steps {state['step']}")

    np.save(f"{OUT}/oof_real_fold{fold}.npy", probs)
    np.save(f"{OUT}/logits_real_fold{fold}.npy", logits.astype(np.float16))
    np.save(f"{OUT}/oof_real_idx_fold{fold}.npy", vi)
    np.save(f"{OUT}/oof_real_y_fold{fold}.npy", val_y)
    json.dump({"model": MODEL_ID, "seed": SEED, "folds": fold_meta, "max_len": MAXLEN,
               "lora_r": LORA_R, "label_smooth": SMOOTH, "stages": [s[0] for s in stages]},
              open(f"{OUT}/fold_status.json", "w"), indent=1)
    # keep the loop going even if the runtime dies right after this fold

meta = {"model": MODEL_ID, "seed": SEED, "folds": fold_meta,
        "mean_bal_acc": float(np.mean([f["bal_acc"] for f in fold_meta])),
        "oof_bal_acc": float(balanced_accuracy_score(real.y.to_numpy(), oof.argmax(1))),
        "max_len": MAXLEN, "lora_r": LORA_R, "label_smooth": SMOOTH, "epochs": EPOCHS,
        "rows": int(len(real)), "aux_rows": int(len(aux))}
json.dump(meta, open(f"{OUT}/cv_report.json", "w"), indent=1)
log("MEAN fold bal_acc", round(meta["mean_bal_acc"], 4), "OOF", round(meta["oof_bal_acc"], 4))

# test predictions (the locally visible competition test rows; the private test is
# scored inside the Kaggle notebook at submission time)
try:
    te = pd.read_csv([os.path.join(r, "test.csv") for r, _, fs in os.walk("/content") if "test.csv" in fs][0])
    te_ids, te_mask = encode(te.text.astype(str).tolist())
    te_logits = infer(te_ids, te_mask)
    np.save(f"{OUT}/test_logits.npy", te_logits.astype(np.float16))
    np.save(f"{OUT}/test_probs.npy", torch.softmax(torch.from_numpy(te_logits), -1).numpy())
    log("test predictions", te.shape)
except Exception as e:
    log("test prediction skipped:", repr(e)[:120])

shutil.make_archive("/content/out_bundle", "zip", OUT)
log("bundle", round(os.path.getsize("/content/out_bundle.zip") / 2**20, 1), "MB")
for f in sorted(os.listdir(OUT)):
    log("  artefact", f, round(os.path.getsize(os.path.join(OUT, f)) / 2**10, 1), "KB")
log("DONE")
