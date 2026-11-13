"""Feature extraction for Track 05 (Name That LLM).

Three sparse views plus one dense stylometric block:

* ``char``  - sublinear TF-IDF over character n-grams (typography / spacing tells)
* ``word``  - sublinear TF-IDF over word n-grams, ``lowercase=False`` so that
              casing tells survive (the task requires keeping this)
* ``shape`` - TF-IDF over synthetic word-shape / punctuation / unicode tokens
* ``sty``   - dense, standardised count and burstiness statistics

All vectorisers are fitted on a corpus supplied by the caller.  In the CV
pipeline that corpus is the *external* corpus only, whose prompt ids are
disjoint from every competition-style row, so no validation text ever touches
the vocabulary or the IDF weights.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
LAB2ID = {label: i for i, label in enumerate(LABELS)}

BASE = "/home/pope/Desktop/gpu /hackathons/05-name-that-llm/data/compact"
EXTRA_CLAUDE = ("/home/pope/Desktop/gpu /hackathons/05-name-that-llm/data/"
                "ntl-extra-api-passages/ntl-extra-api-passages/ntl_extra_passages_claude.csv")


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_competition_frame() -> pd.DataFrame:
    """Competition-style passages (9,290 rows / 510 prompts / 18 labels).

    ``pilot_range`` rows are generated with a *shared* length range so the label
    explains most of their within-prompt length variance; they are off-protocol
    for the real test set (which used per-generation targets).  They stay in the
    frame (the OOF deliverable covers every row) but every headline metric is
    also reported on the clean subset that excludes them.
    """
    df = pd.read_parquet(f"{BASE}/passages.parquet")
    df = df[df["model"].isin(LABELS)].dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df["y"] = df["model"].map(LAB2ID).astype(int)
    df["source"] = "comp"
    df["group"] = df["prompt_id"].astype(str)
    return df.reset_index(drop=True)


def load_external_frame(max_per_class: int = 1200, min_words: int = 40,
                        max_words: int = 250, seed: int = 42) -> pd.DataFrame:
    """External 18-label corpus, length-matched to the competition register.

    Rows are filtered to the competition word range, deduplicated on normalised
    text, then capped per class so the extra data cannot swamp the 9.3k
    competition rows.  Prompt ids are prefixed ``ext_``/``exc_`` so they can
    never collide with competition prompt groups.
    """
    ext = pd.read_parquet(f"{BASE}/external.parquet")
    ext = ext[ext["model"].isin(LABELS)].dropna(subset=["text"]).copy()
    ext["text"] = ext["text"].astype(str)
    ext["nw"] = ext["text"].str.split().str.len()
    ext = ext[(ext["nw"] >= min_words) & (ext["nw"] <= max_words)]
    ext["h"] = ext["text"].str.strip().str.lower().map(
        lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    ext = ext.drop_duplicates("h")
    ext["source"] = "external"

    extra = pd.read_csv(EXTRA_CLAUDE)
    extra = extra[extra["model"].isin(LABELS)].dropna(subset=["text"]).copy()
    extra["text"] = extra["text"].astype(str)
    extra["nw"] = extra["text"].str.split().str.len()
    extra = extra[(extra["nw"] >= min_words) & (extra["nw"] <= max_words)]
    extra["h"] = extra["text"].str.strip().str.lower().map(
        lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    extra = extra.drop_duplicates("h")
    extra["source"] = "extra_claude"
    extra["prompt_id"] = "exc_" + extra["prompt_id"].astype(str)

    frames = []
    for name, part in (("external", ext), ("extra_claude", extra)):
        keep = []
        for _, g in part.groupby("model"):
            keep.append(g.sample(n=min(len(g), max_per_class), random_state=seed))
        part = pd.concat(keep, ignore_index=True)
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["y"] = out["model"].map(LAB2ID).astype(int)
    out["group"] = out["prompt_id"].astype(str)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# stylometry
# --------------------------------------------------------------------------- #
_PUNCT = {
    "period": ".", "comma": ",", "semicolon": ";", "colon": ":", "bang": "!",
    "question": "?", "emdash": "\u2014", "endash": "\u2013", "hyphen": "-",
    "paren_open": "(", "paren_close": ")", "bracket_open": "[", "bracket_close": "]",
    "brace_open": "{", "brace_close": "}", "dquote": '"', "squote": "'",
    "slash": "/", "backslash": "\\", "pipe": "|", "at": "@", "dollar": "$",
    "percent": "%", "caret": "^", "amp": "&", "plus": "+", "equals": "=",
    "lt": "<", "gt": ">", "asterisk": "*", "hash": "#", "tilde": "~",
    "backtick": "`", "underscore": "_",
}
_TYPO = {
    "curly_squote": "\u2018", "curly_apos": "\u2019", "curly_dopen": "\u201c",
    "curly_dclose": "\u201d", "ellipsis_char": "\u2026", "bullet_char": "\u2022",
    "middot": "\u00b7", "nbsp": "\u00a0", "arrow_r": "\u2192", "degree": "\u00b0",
    "times": "\u00d7", "section": "\u00a7", "prime": "\u2032", "thinspace": "\u2009",
    "zwsp": "\u200b", "minus_sign": "\u2212",
}
_HEDGES = ("delve", "crucial", "moreover", "furthermore", "notably", "essentially",
           "actually", "really", "just", "quite", "arguably", "overall", "ultimately",
           "importantly", "indeed", "however", "therefore", "thus", "hence", "also",
           "basically", "simply", "literally", "genuinely", "fascinating", "intricate")


def style_vector(text: str) -> list[float]:
    """Return ~100 interpretable style statistics for one passage.

    Counts are expressed as densities (per 1k chars or per 100 words) so the
    vector is comparable across the 35-word pilot passages and the 320-word
    long ones.  Raw length is included separately and separately scaled.
    """
    n = max(len(text), 1)
    words = text.split()
    nw = max(len(words), 1)
    lower = [w.lower() for w in words]
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    slens = [len(s.split()) for s in sents] or [0]
    mean_s = float(np.mean(slens))
    sd_s = float(np.std(slens))
    ttr = len(set(lower)) / nw
    counts = pd.Series(lower).value_counts()

    f: list[float] = [
        np.log1p(n) / 6.0,
        np.log1p(nw) / 5.0,
        mean_s / 40.0,
        sd_s / 25.0,
        sd_s / max(mean_s, 1.0),                      # sentence-length CV
        float(np.max(slens)) / max(mean_s, 1.0),
        ttr,
        float((counts == 1).sum()) / nw,              # hapax rate
        float((counts >= 2).sum()) / nw,              # repeated-type rate
        float(-(counts / nw * np.log(counts / nw + 1e-9)).sum()) / 6.0,  # word entropy
        sum(len(w) for w in words) / nw / 6.0,        # mean token length
        float(np.std([len(w) for w in words])) / 4.0,  # token-length sd
        sum(1 for w in words if len(w) <= 2) / nw,
        sum(1 for w in words if len(w) >= 8) / nw,
        sum(1 for w in words if len(w) >= 12) / nw,
        text.count("\n") / nw * 100,
        text.count("\n\n") / nw * 100,
        text.count("  ") / nw * 100,
        text.count("\t") / nw * 100,
    ]
    for ch in _PUNCT.values():
        f.append(text.count(ch) / n * 1000)
    for ch in _TYPO.values():
        f.append(text.count(ch) / n * 1000)
    f.append(sum(1 for c in text if ord(c) > 127) / n * 1000)  # non-ascii density

    # casing / digit shapes
    upper = sum(1 for c in text if c.isupper())
    digit = sum(1 for c in text if c.isdigit())
    f += [
        upper / n,
        digit / n,
        sum(1 for w in words if w.isupper() and len(w) > 1) / nw,
        sum(1 for w in words if w[:1].isupper()) / nw,
        sum(1 for w in words if w.islower()) / nw,
        sum(1 for w in words if any(c.isdigit() for c in w)) / nw,
        sum(1 for w in words if re.fullmatch(r"[A-Za-z]+", w) is None) / nw,
        sum(1 for w in words if w.endswith(".")) / nw,
    ]
    # markdown / structure
    f += [
        len(re.findall(r"^\s{0,3}#{1,6}\s", text, re.M)) / nw * 100,
        len(re.findall(r"^\s*[-*+]\s", text, re.M)) / nw * 100,
        len(re.findall(r"^\s*\d+[.)]\s", text, re.M)) / nw * 100,
        len(re.findall(r"^\s*>\s", text, re.M)) / nw * 100,
        len(re.findall(r"```", text)) / nw * 100,
        len(re.findall(r"`[^`\n]+`", text)) / nw * 100,
        len(re.findall(r"\*\*[^*]+\*\*", text)) / nw * 100,
        len(re.findall(r"(?<!\*)\*[^*\n]+\*(?!\*)", text)) / nw * 100,
        len(re.findall(r"\[[^\]]*\]\([^)]*\)", text)) / nw * 100,
        text.count("|") / nw * 100,
        len(re.findall(r"\b[A-Z]{2,}\b", text)) / nw * 100,
        len(re.findall(r"\b\d{4}\b", text)) / nw * 100,
        len(re.findall(r"\d+[%°]", text)) / nw * 100,
        len(re.findall(r"[.!?]\s+[a-z]", text)) / max(len(sents), 1),   # lowercase after stop
        len(re.findall(r",\s*,", text)) / nw * 100,
        len(re.findall(r"\b(\w+)\s+\1\b", text, re.I)) / nw * 100,      # immediate repetition
    ]
    # pronoun / hedge rate
    f += [
        lower.count("i") / nw, lower.count("you") / nw, lower.count("we") / nw,
        lower.count("they") / nw, lower.count("it") / nw,
        sum(1 for w in lower if w in _HEDGES) / nw,
        sum(1 for w in lower if w.startswith("un")) / nw,
        sum(1 for w in lower if w.endswith("ly")) / nw,
    ]
    # token-length histogram
    hist = np.zeros(13)
    for w in words:
        hist[min(len(w), 12)] += 1
    f += list(hist / nw)
    return f


STYLE_DIM = len(style_vector("Hello world. This is a test."))


# --------------------------------------------------------------------------- #
# word-shape token projection
# --------------------------------------------------------------------------- #
_PUNCT_RUN = re.compile(r"[^\w\s]+")


def _shape(word: str) -> str:
    """Collapse a token to a character-class pattern (``Apple`` -> ``Aa``)."""
    out = []
    for c in word:
        if c.isupper():
            c2 = "A"
        elif c.islower():
            c2 = "a"
        elif c.isdigit():
            c2 = "9"
        else:
            c2 = "p"
        if not out or out[-1] != c2:
            out.append(c2)
    return "S" + "".join(out)[:6]


def shape_tokens(text: str) -> str:
    """Build the token string consumed by the ``shape`` TF-IDF view.

    Emits word-shape unigrams, punctuation bigrams/trigrams, and named unicode
    typography tokens, e.g. ``SH:SH:'' PU:, PU:,. TY:emdash``.
    """
    toks: list[str] = []
    for w in text.split()[:400]:
        toks.append("W:" + _shape(w))
    for m in _PUNCT_RUN.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            toks.append("P:" + run)
        else:
            for i in range(min(len(run) - 1, 6)):
                toks.append("P2:" + run[i:i + 2])
    for name, ch in _TYPO.items():
        c = text.count(ch)
        if c:
            toks.append(f"T:{name}:{min(c, 5)}")
    for marker, tok in (("\n\n", "N:par"), ("\n", "N:nl"), ("  ", "N:dspace"),
                        ("**", "M:bold"), ("##", "M:head"), ("- ", "M:bullet"),
                        ("```", "M:fence"), ("`", "M:code"), ("|", "M:pipe")):
        c = text.count(marker)
        if c:
            toks.append(f"{tok}:{min(c, 5)}")
    return " ".join(toks)


# --------------------------------------------------------------------------- #
# feature stack
# --------------------------------------------------------------------------- #
class FeatureStack:
    """Fit the three TF-IDF views + stylometry scaler on one corpus, then transform.

    ``views`` selects which blocks are emitted.  ``char_ngram``/``word_ngram``
    are exposed so the ablation grid can vary them without new code.
    """

    def __init__(self, views: Sequence[str] = ("char", "word", "shape", "sty"),
                 char_ngram: tuple[int, int] = (2, 5), word_ngram: tuple[int, int] = (1, 2),
                 char_max: int = 150_000, word_max: int = 120_000,
                 shape_max: int = 60_000, min_df: int = 4, seed: int = 0) -> None:
        self.views = tuple(views)
        self.char_ngram = char_ngram
        self.word_ngram = word_ngram
        self.char_max, self.word_max, self.shape_max = char_max, word_max, shape_max
        self.min_df = min_df
        self.seed = seed
        self._vecs: dict[str, TfidfVectorizer] = {}
        self._scaler: StandardScaler | None = None

    def fit(self, texts: Iterable[str]) -> "FeatureStack":
        texts = [str(t) for t in texts]
        if "char" in self.views:
            self._vecs["char"] = TfidfVectorizer(
                analyzer="char", ngram_range=self.char_ngram, sublinear_tf=True,
                lowercase=False, min_df=self.min_df, max_features=self.char_max,
                dtype=np.float32).fit(texts)
        if "word" in self.views:
            self._vecs["word"] = TfidfVectorizer(
                analyzer="word", ngram_range=self.word_ngram, sublinear_tf=True,
                lowercase=False, min_df=self.min_df, max_features=self.word_max,
                token_pattern=r"(?u)\b\w+\b", dtype=np.float32).fit(texts)
        if "shape" in self.views:
            self._vecs["shape"] = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                token_pattern=r"\S+", min_df=3, max_features=self.shape_max,
                dtype=np.float32).fit([shape_tokens(t) for t in texts])
        if "sty" in self.views:
            sty = self._style_matrix(texts)
            self._scaler = StandardScaler().fit(sty)
        return self

    @staticmethod
    def _style_matrix(texts: Sequence[str]) -> np.ndarray:
        return np.asarray([style_vector(t) for t in texts], dtype=np.float64)

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        texts = [str(t) for t in texts]
        blocks = []
        if "char" in self.views:
            blocks.append(self._vecs["char"].transform(texts))
        if "word" in self.views:
            blocks.append(self._vecs["word"].transform(texts))
        if "shape" in self.views:
            blocks.append(self._vecs["shape"].transform([shape_tokens(t) for t in texts]))
        if "sty" in self.views:
            z = self._scaler.transform(self._style_matrix(texts))
            z = np.clip(z, -6, 6) * 0.35          # keep the dense block from dominating
            blocks.append(csr_matrix(z.astype(np.float32)))
        return hstack(blocks, format="csr").astype(np.float32)

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        texts = [str(t) for t in texts]
        return self.fit(texts).transform(texts)
