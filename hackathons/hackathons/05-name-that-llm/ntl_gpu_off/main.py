"""Name That LLM - GPU encoder + TF-IDF submission kernel with internet disabled.

The competition rejects notebook submissions whose kernel enables internet, so every
model must load from attached Kaggle inputs.  DeBERTa-v3-base is mounted through
``model_sources``; the TF-IDF branch is trained from ``train.csv`` only.

Both branches produce 5-fold grouped out-of-fold probabilities (group = ``prompt_id``);
the blend weight is picked on those OOF scores with balanced accuracy, which is the
competition metric.  A TF-IDF-only fallback guarantees a valid submission file even if
the encoder cannot be loaded.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 42
OUT = "/kaggle/working"
START = time.time()


def log(*args: object) -> None:
    """Timestamped progress line, flushed so the kernel log stays readable."""
    print(f"[{time.time() - START:7.1f}s]", *args, flush=True)


def find_file(name: str) -> str:
    """Locate a file anywhere under /kaggle/input."""
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def model_dir() -> Optional[str]:
    """Directory holding the attached PyTorch encoder weights, if any."""
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "config.json" in files and any(
            f in files for f in ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
        ):
            return root
    return None


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Per-class recall averaged over the classes present in ``y``."""
    recalls = []
    for label in np.unique(y):
        mask = y == label
        recalls.append(float((pred[mask] == label).mean()))
    return float(np.mean(recalls))


train = pd.read_csv(find_file("train.csv"))
test = pd.read_csv(find_file("test.csv"))
log("train", train.shape, "test", test.shape)

labels = sorted(train["model"].unique())
label_to_id = {label: index for index, label in enumerate(labels)}
y = train["model"].map(label_to_id).to_numpy()
groups = train["prompt_id"].to_numpy()
n_train, n_test, n_class = len(train), len(test), len(labels)

from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(train["text"], y, groups))
log("folds", len(splits))


def tfidf_branch() -> Tuple[np.ndarray, np.ndarray]:
    """Word + character TF-IDF logistic regression, cross-fitted on the grouped folds."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                           max_features=250_000)
    x_all = hstack([word.fit_transform(train["text"]), char.fit_transform(train["text"])]).tocsr()
    x_test = hstack([word.transform(test["text"]), char.transform(test["text"])]).tocsr()
    oof = np.zeros((n_train, n_class))
    test_prob = np.zeros((n_test, n_class))
    for fold, (fit_idx, val_idx) in enumerate(splits):
        model = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced")
        model.fit(x_all[fit_idx], y[fit_idx])
        cols = model.classes_.astype(int)
        oof[val_idx[:, None], cols] = model.predict_proba(x_all[val_idx])
        test_prob[:, cols] += model.predict_proba(x_test) / len(splits)
        log("tfidf fold", fold, "done")
    return oof, test_prob


def encoder_branch(epochs: int = 3, max_len: int = 256, batch_size: int = 8,
                   lr: float = 2e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-tune the attached DeBERTa-v3 encoder per fold and average test probabilities."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = model_dir()
    if path is None:
        raise RuntimeError("no encoder weights under /kaggle/input")
    log("encoder", path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class TextDataset(Dataset):
        """Tokenised texts with optional integer labels."""

        def __init__(self, texts: List[str], targets: Optional[np.ndarray] = None) -> None:
            self.encoded = tokenizer(list(texts), truncation=True, max_length=max_len, padding=True)
            self.targets = targets

        def __len__(self) -> int:
            return len(self.encoded["input_ids"])

        def __getitem__(self, index: int) -> Dict[str, object]:
            item = {key: torch.tensor(value[index]) for key, value in self.encoded.items()}
            if self.targets is not None:
                item["labels"] = torch.tensor(int(self.targets[index]))
            return item

    test_loader = DataLoader(TextDataset(test["text"]), batch_size=32)
    oof = np.zeros((n_train, n_class))
    test_prob = np.zeros((n_test, n_class))
    for fold, (fit_idx, val_idx) in enumerate(splits):
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=n_class, local_files_only=True).to(device)
        loader = DataLoader(TextDataset(train["text"].iloc[fit_idx], y[fit_idx]),
                            batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=lr, total_steps=max(1, len(loader)) * epochs, pct_start=0.15)
        model.train()
        for epoch in range(epochs):
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()
            log("encoder fold", fold, "epoch", epoch, "done")
        model.eval()
        chunks = []
        with torch.no_grad():
            for batch in DataLoader(TextDataset(train["text"].iloc[val_idx]), batch_size=32):
                batch = {key: value.to(device) for key, value in batch.items()}
                chunks.append(torch.softmax(model(**batch).logits, -1).cpu().numpy())
        oof[val_idx] = np.vstack(chunks)
        chunks = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                chunks.append(torch.softmax(model(**batch).logits, -1).cpu().numpy())
        test_prob += np.vstack(chunks) / len(splits)
        del model
        torch.cuda.empty_cache()
    return oof, test_prob


branches: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
try:
    branches["tfidf"] = tfidf_branch()
    log("tfidf OOF balanced accuracy", round(balanced_accuracy(y, branches["tfidf"][0].argmax(1)), 4))
except Exception as exc:  # noqa: BLE001 - the kernel must still emit a submission
    log("tfidf failed", repr(exc))

try:
    branches["encoder"] = encoder_branch()
    log("encoder OOF balanced accuracy",
        round(balanced_accuracy(y, branches["encoder"][0].argmax(1)), 4))
except Exception as exc:  # noqa: BLE001
    log("encoder failed", repr(exc))

if not branches:
    raise RuntimeError("no branch produced predictions")

if len(branches) == 2:
    best = (-1.0, 0.5)
    for weight in np.linspace(0.0, 1.0, 21):
        oof = weight * branches["tfidf"][0] + (1.0 - weight) * branches["encoder"][0]
        score = balanced_accuracy(y, oof.argmax(1))
        if score > best[0]:
            best = (score, float(weight))
    score, weight = best
    log("blend weight tfidf", weight, "OOF balanced accuracy", round(score, 4))
else:
    weight = 1.0
    name = next(iter(branches))
    score = balanced_accuracy(y, branches[name][0].argmax(1))
    log("single branch", name, "OOF balanced accuracy", round(score, 4))

if len(branches) == 2:
    test_prob = weight * branches["tfidf"][1] + (1.0 - weight) * branches["encoder"][1]
else:
    test_prob = next(iter(branches.values()))[1]

prediction = pd.DataFrame({"id": test["id"], "model": [labels[i] for i in test_prob.argmax(1)]})
prediction.to_csv(f"{OUT}/submission.csv", index=False)
prediction.to_parquet(f"{OUT}/submission.parquet", index=False)
log("wrote submission", prediction.shape)
print(prediction.to_string(index=False), flush=True)
