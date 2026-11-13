"""Corpus loaders for Track 05 (Name That LLM) - REAL competition protocol.

The competition `train.csv` (150 prompts x 18 models = 2,699 rows) is the only
source of validation rows.  Everything else in `data/compact/` is a scraped
mirror whose prompt ids and passage texts are disjoint from the competition
file (verified by hash), so it may be used as *training augmentation* only.

One class per file, fully typed, no cross-module state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS: tuple[str, ...] = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
LAB2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}

TRACK_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_TRAIN = TRACK_ROOT / "data" / "competition" / "train.csv"
COMPETITION_TEST = TRACK_ROOT / "data" / "competition" / "test.csv"
COMPACT_DIR = TRACK_ROOT / "data" / "compact"
EXTRA_CLAUDE = (TRACK_ROOT / "data" / "ntl-extra-api-passages" / "ntl-extra-api-passages"
                / "ntl_extra_passages_claude.csv")

WORD_COL = "n_words"
LENGTH_BINS: tuple[int, ...] = (0, 60, 120, 180, 1000)


@dataclass(frozen=True)
class Corpus:
    """A labelled passage corpus with prompt-level grouping metadata."""

    texts: tuple[str, ...]
    y: np.ndarray
    groups: np.ndarray
    n_words: np.ndarray
    source: str

    def __len__(self) -> int:
        """Number of passages."""
        return len(self.texts)

    def frame(self) -> pd.DataFrame:
        """Materialise a tidy frame (used for diagnostics only)."""
        return pd.DataFrame({
            "text": list(self.texts),
            "y": self.y,
            "model": [LABELS[i] for i in self.y],
            "prompt_id": self.groups,
            WORD_COL: self.n_words,
        })


def _finalise(texts: Any, models: Any, groups: Any, source: str) -> Corpus:
    """Drop blank/invalid rows, map labels to ids and compute word counts."""
    frame = pd.DataFrame({"text": list(map(str, texts)),
                          "model": list(models),
                          "prompt_id": list(map(str, groups))})
    frame = frame[frame["model"].isin(LABELS)]
    frame = frame[frame["text"].str.strip().str.len() > 0]
    frame = frame.drop_duplicates(subset=["text", "model"])
    n_words = frame["text"].str.split().str.len().to_numpy(dtype=np.int32)
    return Corpus(
        texts=tuple(frame["text"].tolist()),
        y=frame["model"].map(LAB2ID).to_numpy(dtype=np.int64),
        groups=frame["prompt_id"].to_numpy(dtype=object),
        n_words=n_words,
        source=source,
    )


class CompetitionLoader:
    """Loads the REAL `data/competition/train.csv` (never any mirror row)."""

    def __init__(self, path: Path = COMPETITION_TRAIN) -> None:
        self.path = Path(path)

    def load(self) -> Corpus:
        """Return the competition training corpus, grouped by ``prompt_id``."""
        df = pd.read_csv(self.path)
        missing = {"id", "prompt_id", "text", "model"} - set(df.columns)
        if missing:
            raise ValueError(f"{self.path} missing columns: {sorted(missing)}")
        return _finalise(df["text"], df["model"], df["prompt_id"], "competition")


class AugmentationLoader:
    """Loads disjoint mirror passages suitable for extra *training* rows only.

    Selection is length-matched to the competition register, deduplicated on a
    normalised-text hash, capped per class, and hard-filtered against every
    competition text so leakage is impossible even if a mirror row were a copy.
    """

    def __init__(self, min_words: int = 40, max_words: int = 400,
                 max_per_class: int = 900, seed: int = 42,
                 include_extra_claude: bool = True) -> None:
        self.min_words = int(min_words)
        self.max_words = int(max_words)
        self.max_per_class = int(max_per_class)
        self.seed = int(seed)
        self.include_extra_claude = bool(include_extra_claude)

    @staticmethod
    def _hash(text: str) -> str:
        """Stable hash of the whitespace-normalised lowercase text."""
        norm = " ".join(text.split()).lower()
        return hashlib.md5(norm.encode("utf-8")).hexdigest()

    def _read_mirror(self) -> pd.DataFrame:
        """Concatenate every mirror frame available on disk."""
        parts: list[pd.DataFrame] = [pd.read_parquet(COMPACT_DIR / "external.parquet")]
        if self.include_extra_claude and EXTRA_CLAUDE.exists():
            extra = pd.read_csv(EXTRA_CLAUDE)
            extra = extra[["prompt_id", "model", "text"]].copy()
            extra["prompt_id"] = "exc_" + extra["prompt_id"].astype(str)
            parts.append(extra)
        df = pd.concat(parts, ignore_index=True)
        return df[["prompt_id", "model", "text"]]

    def load(self, exclude_hashes: set[str] | None = None) -> Corpus:
        """Return the capped augmentation corpus."""
        df = self._read_mirror()
        df["text"] = df["text"].astype(str)
        df = df[df["model"].isin(LABELS)]
        n_words = df["text"].str.split().str.len()
        df = df[(n_words >= self.min_words) & (n_words <= self.max_words)].copy()
        df["h"] = df["text"].map(self._hash)
        df = df.drop_duplicates("h")
        if exclude_hashes:
            df = df[~df["h"].isin(exclude_hashes)]
        keep = [g.sample(n=min(len(g), self.max_per_class), random_state=self.seed)
                for _, g in df.groupby("model", sort=True)]
        df = pd.concat(keep, ignore_index=True)
        return _finalise(df["text"], df["model"], df["prompt_id"], "mirror")


def text_hashes(corpus: Corpus) -> set[str]:
    """Hash set for leakage-proof exclusion across corpora."""
    return {AugmentationLoader._hash(t) for t in corpus.texts}
