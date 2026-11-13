"""Name That LLM submission kernel: DeBERTa-v3-base + TF-IDF, 5-fold grouped blend.

Runs inside the competition sandbox, so `test.csv` here is the real hidden scoring set.
Writes both submission.csv and submission.parquet with the sample columns (id, model).
"""
import os
import numpy as np
import pandas as pd

SEED = 42
OUT = "/kaggle/working"


def find(fn):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if fn in files:
            return os.path.join(root, fn)
    raise FileNotFoundError(fn)


tr = pd.read_csv(find("train.csv"))
te = pd.read_csv(find("test.csv"))
print("train", tr.shape, "test", te.shape, flush=True)

LABELS = sorted(tr["model"].unique())
lab2id = {l: i for i, l in enumerate(LABELS)}
y = tr["model"].map(lab2id).values
groups = tr["prompt_id"].values
n_tr, n_te, K = len(tr), len(te), len(LABELS)

probs = []


def tfidf_block():
    """Word + char TF-IDF logistic regression fitted on the full labelled set."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=120_000)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        sublinear_tf=True, max_features=200_000)
    Xtr = hstack([w.fit_transform(tr["text"]), c.fit_transform(tr["text"])]).tocsr()
    Xte = hstack([w.transform(te["text"]), c.transform(te["text"])]).tocsr()
    m = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced").fit(Xtr, y)
    p = np.zeros((n_te, K))
    p[:, [lab2id[k] for k in m.classes_]] = m.predict_proba(Xte)
    return p


try:
    probs.append(("tfidf", tfidf_block()))
    print("tfidf done", flush=True)
except Exception as exc:  # keep the kernel alive
    print("tfidf failed:", exc, flush=True)


def deberta_block(epochs=3, maxlen=320, bs=16, lr=2e-5):
    """Five-fold grouped fine-tune of a small pretrained encoder, averaged over test."""
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = os.environ.get("MODEL_ID", "microsoft/deberta-v3-base")
    tok = AutoTokenizer.from_pretrained(name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class DS(Dataset):
        def __init__(self, texts, labels=None):
            self.enc = tok(list(texts), truncation=True, max_length=maxlen, padding=True)
            self.y = labels

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            if self.y is not None:
                item["labels"] = torch.tensor(int(self.y[i]))
            return item

    te_dl = DataLoader(DS(te["text"].values), batch_size=32)
    cfg_path = name + "_cfg"
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    total = np.zeros((n_te, K))
    for fold, (a, b) in enumerate(sgkf.split(tr["text"], y, groups)):
        cfg = AutoModelForSequenceClassification.from_pretrained(name, num_labels=K,
                                                                 cache_dir=cfg_path).to(dev)
        dl = DataLoader(DS(tr["text"].values[a], y[a]), batch_size=bs, shuffle=True)
        opt = torch.optim.AdamW(cfg.parameters(), lr=lr, weight_decay=0.01)
        n_steps = max(1, len(dl)) * epochs
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=int(n_steps),
                                                 pct_start=0.1)
        cfg.train()
        for ep in range(int(epochs)):
            for batch in dl:
                batch = {k: v.to(dev) for k, v in batch.items()}
                out = cfg(**batch)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(cfg.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
        cfg.eval()
        fold_p = []
        with torch.no_grad():
            for batch in te_dl:
                batch = {k: v.to(dev) for k, v in batch.items()}
                fold_p.append(torch.softmax(cfg(**batch).logits, -1).cpu().numpy())
        total += np.vstack(fold_p)
        print(f"fold {fold} done", flush=True)
        del cfg
        torch.cuda.empty_cache()
    return total / 5


try:
    probs.append(("deberta", deberta_block()))
    print("deberta done", flush=True)
except Exception as exc:
    print("deberta failed:", exc, flush=True)

if not probs:
    raise RuntimeError("no model produced predictions")

weights = {"tfidf": 0.35, "deberta": 0.65}
mix = np.zeros((n_te, K))
wsum = 0.0
for name, p in probs:
    w = weights.get(name, 0.5)
    mix += w * p
    wsum += w
mix /= wsum

sub = pd.DataFrame({"id": te["id"], "model": [LABELS[i] for i in mix.argmax(1)]})
sub.to_csv(f"{OUT}/submission.csv", index=False)
sub.to_parquet(f"{OUT}/submission.parquet", index=False)
print(sub.head(), sub.shape, flush=True)
print("written", flush=True)
