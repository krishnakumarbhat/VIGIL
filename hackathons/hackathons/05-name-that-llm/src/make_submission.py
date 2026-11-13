"""Track 05 - build the hidden-test submission from the exported ensemble recipe.

The blend that scores 0.61 cross-fitted lives in `outputs/ensemble_recipe.json` as
temperatures + weights + per-class biases. This script turns that recipe into predictions for an
arbitrary test frame by

1. computing each *available* member's probabilities from persisted artefacts
   (`outputs/kaggle_deberta/model_fold*` for DeBERTa, `outputs/deploy/*.joblib` for the sparse
   members),
2. dropping members with no artefact and renormalising the remaining weights (the recipe then
   degrades gracefully instead of failing), and
3. applying `z = sum_i w_i * log(p_i) / T_i + b`, `pred = argmax z`.

Usage
  python3 src/make_submission.py --test data/competition/test.csv --out outputs/submission.csv
  python3 src/make_submission.py --list-members
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

D = Path("/home/pope/Desktop/gpu /hackathons/05-name-that-llm")
OUT = D / "outputs"
DEP = OUT / "deploy"
EPS = 1e-9


def load_recipe() -> dict:
    """Read the exported blend recipe (v2 weights/temps/biases form)."""
    recipe = json.loads((OUT / "ensemble_recipe.json").read_text())
    if recipe.get("type") == "linear_meta":
        raise SystemExit("recipe is a linear_meta model; use stack_lr's coefficients directly")
    return recipe


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_deberta(texts: list[str], max_len: int = 256, batch: int = 64) -> np.ndarray | None:
    """Average the five fold checkpoints' softmax outputs (test-time fold ensembling)."""
    paths = sorted(OUT.glob("kaggle_deberta/model_fold*"))
    if not paths:
        return None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    probs = []
    for path in paths:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(
            path, dtype=torch.bfloat16).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), batch):
                enc = tok(texts[s:s + batch], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to(device)
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        probs.append(np.concatenate(out))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(probs, axis=0)


def predict_sparse(name: str, texts: list[str]) -> np.ndarray | None:
    """Recreate a persisted sparse member's probabilities for the given texts."""
    if name not in {"svm", "svmext", "char", "word", "style", "knn"}:
        return None
    vec_path = DEP / "vectorisers.joblib"
    if not vec_path.exists():
        return None
    vecs = joblib.load(vec_path)
    texts = [str(t)[:6000] for t in texts]
    if name == "knn":
        bundle = joblib.load(DEP / "knn.joblib")
        X = bundle["vectoriser"].transform(texts)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1))).ravel()
        X = X.multiply(1.0 / np.maximum(norms, 1e-9)[:, None]).tocsr()
        sims = (X @ bundle["matrix"].T).toarray()
        k = bundle["k"]
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        probs = np.full((len(texts), len(bundle["labels"])), 1e-3)
        for r in range(len(texts)):
            cols = idx[r]
            np.add.at(probs[r], bundle["y"][cols], np.clip(sims[r, cols], 0, None) ** 3)
        return probs / probs.sum(axis=1, keepdims=True)
    if name == "style":
        bundle = joblib.load(DEP / "style.joblib")
        from ntl_style import style_matrix
        X = bundle["scaler"].transform(style_matrix(texts))
        p = bundle["model"].predict_proba(X)
        full = np.zeros((len(texts), len(bundle["labels"])))
        for j, c in enumerate(bundle["classes"]):
            full[:, c] = p[:, j]
        return full
    bundle = joblib.load(DEP / f"{name}.joblib")
    model = bundle["model"]
    if name == "svmext":
        # svmext carries its own vectorisers: they were fitted on real+mirror+external and have
        # a different vocabulary/dimension than the mirror-only `svm` pair.
        w12, ch = bundle["word12"], bundle["char"]
    else:
        w12, ch = vecs["word12"], vecs["char"]
    if name in {"svm", "svmext"}:
        from scipy.sparse import hstack
        X = hstack([w12.transform(texts), ch.transform(texts)]).tocsr()
    elif name == "char":
        X = ch.transform(texts)
    else:
        X = vecs["word13"].transform(texts)
    p = model.predict_proba(X)
    full = np.zeros((len(texts), len(bundle["labels"])))
    for j, c in enumerate(bundle["classes"]):
        full[:, c] = p[:, j]
    return full


def predict_lora(texts: list[str], batch: int = 32) -> np.ndarray | None:
    """Score texts with the full-data LoRA adapter (stands in for the s1337 OOF member)."""
    adapter = DEP / "lora_qwen05b"
    if not (DEP / "lora_qwen05b" / "deploy_meta.json").exists():
        return None
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = json.loads((adapter / "deploy_meta.json").read_text())
    labels = meta["labels"]
    tok = AutoTokenizer.from_pretrained(adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # Qwen ships no pad token; batch inference needs one
    model = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=len(labels), dtype=torch.bfloat16).eval()
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, adapter).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], truncation=True, max_length=meta["max_len"],
                      padding="max_length", return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def available_members(recipe_members: list[str]) -> list[str]:
    """Report which recipe members can be recomputed from artefacts on this machine."""
    ok = []
    for m in recipe_members:
        if m == "deberta_base" and list(OUT.glob("kaggle_deberta/model_fold*")):
            ok.append(m)
        elif m in {"svm", "svmext", "char", "word", "style", "knn"} and (DEP / f"{m}.joblib").exists():
            ok.append(m)
        elif m == "local_qwen05b_s1337" and (DEP / "lora_qwen05b" / "deploy_meta.json").exists():
            ok.append(m)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(D / "data/competition/test.csv"))
    ap.add_argument("--out", default=str(OUT / "submission.csv"))
    ap.add_argument("--list-members", action="store_true")
    args = ap.parse_args()

    recipe = load_recipe()
    members = recipe["members"]
    have = available_members(members)
    if args.list_members:
        print("recipe members:", members)
        print("reproducible now:", have)
        print("missing (weights dropped + renormalised):", [m for m in members if m not in have])
        return
    if not have:
        raise SystemExit("no member artefacts available; run src/fit_deploy_sparse.py first")

    test = pd.read_csv(args.test)
    texts = test["text"].astype(str).tolist()
    print(f"test rows: {len(texts)} | members: {have}")
    dropped = [m for m in members if m not in have]
    if dropped:
        print(f"dropped (no artefact): {dropped}")

    z = np.zeros((len(texts), len(recipe["labels"])))
    total_w = 0.0
    for m in have:
        if m == "deberta_base":
            p = predict_deberta(texts)
        elif m == "local_qwen05b_s1337":
            p = predict_lora(texts)
        else:
            p = predict_sparse(m, texts)
        if p is None:
            print(f"  {m}: unavailable, skipped")
            continue
        w = float(recipe["weights"][m])
        t = float(recipe["temperatures"][m])
        z += w * np.log(np.clip(p, EPS, None)) / t
        total_w += w
        print(f"  {m}: weight {w:.3f} temp {t:.3f} ok")
    z = z / max(total_w, EPS) + np.asarray(recipe["biases"])

    labels = recipe["labels"]
    pred = [labels[i] for i in z.argmax(1)]
    sub = pd.DataFrame({"id": test["id"], "model": pred})
    allowed = set(labels)
    assert sub["model"].isin(allowed).all(), "prediction outside the label set"
    assert not sub["model"].isna().any() and len(sub) == len(test)
    sub.to_csv(args.out, index=False)
    sub.to_parquet(Path(args.out).with_suffix(".parquet"), index=False)
    print(f"wrote {args.out} ({len(sub)} rows)")
    print(sub["model"].value_counts().head(6).to_string())


if __name__ == "__main__":
    sys.path.insert(0, str(D / "src"))
    main()
