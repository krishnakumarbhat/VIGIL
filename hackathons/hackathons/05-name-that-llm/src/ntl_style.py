"""Stylometric feature extractor shared by the OOF member and the deployable artefact.

Extracted verbatim from `sparse_members.py` so the deployable member and the OOF matrix it was
calibrated on use byte-identical features.
"""
import re

import numpy as np

PATTERNS = {
    "comma": r",", "period": r"\.", "semicolon": r";", "colon": r":", "emdash": "\u2014",
    "endash": "\u2013", "exclam": r"!", "question": r"\?", "paren": r"\(", "bracket": r"\[",
    "quot": r'"', "apos": r"'", "bullet": r"(?m)^\s*[-*+]\s", "numlist": r"(?m)^\s*\d+[.)]\s",
    "header": r"(?m)^#{1,6}\s", "bold": r"\*\*", "italic": r"(?<!\*)\*(?!\*)", "code": r"`",
    "table": r"(?m)^\s*\|", "arrow": "\u2192", "slash": r"/", "tilde": r"~", "at": r"@",
}
RE = {k: re.compile(v) for k, v in PATTERNS.items()}
SENT = re.compile(r"[.!?]+\s")


def style_matrix(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        lines = t.split("\n")
        line_lens = [len(x) for x in lines] or [0]
        sents = [s for s in SENT.split(t) if s.strip()]
        sent_lens = [len(s) for s in sents] or [0]
        words = t.split()
        wl = [len(w) for w in words] or [0]
        feats = [
            np.log1p(n), np.log1p(len(lines)), np.log1p(len(words)), np.log1p(len(sents)),
            float(np.mean(line_lens)), float(np.max(line_lens)), float(np.std(line_lens)),
            float(np.mean(sent_lens)), float(np.max(sent_lens)), float(np.std(sent_lens)),
            float(np.mean(wl)), float(np.max(wl)), float(np.std(wl)),
            sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
            sum(c.isspace() for c in t) / n,
            sum(not c.isascii() for c in t) / n,
            t.count("\n\n") / max(len(lines), 1),
            sum(1 for x in line_lens if x < 40) / len(line_lens),
            sum(1 for x in line_lens if x > 300) / len(line_lens),
        ]
        for rx in RE.values():
            feats.append(len(rx.findall(t)) / n * 1000.0)
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)



