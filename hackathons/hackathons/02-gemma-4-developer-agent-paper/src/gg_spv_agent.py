"""GG-SPV: Graph-Guided Hierarchical Fault Localization & Speculative Patch Verification.

This module implements a research agent stack for the Kaggle "Gemma 4 Developer
Agent" track.  It is built around the *real* competition artifacts:

* ``tasks.jsonl``            - 129 SWE-style tasks (gold patches included),
* ``graphs/<repo>_<sha>.json``   - NetworkX node-link code graphs (symbol nodes with
  source text and ``calls`` edges),
* ``embeddings/<repo>_<sha>.npz`` - per-symbol 256-d embeddings.

The three research components are:

``GraphIndex``
    Deterministic offline re-implementation of the harness code-intelligence
    tools (``search_similar_code``, ``get_code_neighbors``, ``get_code_subgraph``,
    ``read_file``) plus a BM25 index.  Every response is token-accounted.
``Localizer``
    Graph-Guided (GG) localization: identifier-aware hybrid retrieval
    (BM25 + real embedding cosine), seed selection, 2-hop call-graph expansion
    with score decay and pruning.  ``localize_flat`` is the control condition:
    flat, file-at-a-time reading with no graph.
``PatchVerifier``
    Speculative patch verification: syntax check, graph-consistency of the
    edited symbol, caller-compatibility analysis over the 2-hop neighbourhood,
    and scope discipline.  Produces a calibrated confidence in [0, 1].
``AgentLoop``
    Full simulated agent trajectory (localize -> hypothesise -> verify ->
    resample -> submit) with structured JSON telemetry.

Everything in this file is deterministic given ``AgentConfig.seed``.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import hashlib
import heapq
import json
import math
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants mirroring the real harness limits (see HARNESS_README.md §7.1).
# ---------------------------------------------------------------------------

MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10_000
MAX_STDOUT_CHARS = 5_000
CHARS_PER_TOKEN = 4.0  # deterministic sub-word proxy used for token accounting

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in into is it its of on
    or that the this to was were will with you your please should would could
    we i not no yes when where which who what how use using used can may new
    add added make made get set do does did done being been over under each
    all any some more most other than then there these those our out up down
    off again further once here why how all both between during before after
    above below between".split()
    """.replace('"', "").split()
)


def tokens_of(text: str) -> List[str]:
    """Lower-case word/sub-word tokens of ``text`` (identifier-aware)."""
    parts: List[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for piece in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9_]+", raw):
            piece = piece.lower()
            if len(piece) >= 2:
                parts.append(piece)
    return parts


def estimate_tokens(text: str) -> int:
    """Deterministic token proxy: characters / 4 (rounded up)."""
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Dataclasses: configuration, results, telemetry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    """Static configuration for one benchmark run of the GG-SPV agent."""

    model_id: str = "gemma-4-12b-it-qat-w4a16-ct"
    max_tool_calls: int = 40
    max_attempts: int = 4
    seed_top_m: int = 8
    expand_hops: int = 2
    hop_decay: float = 0.55
    max_expanded: int = 60
    retrieval_pool: int = 50
    bm25_weight: float = 0.5
    accept_threshold: float = 0.5
    flat_read_files: int = 12
    fidelity: float = 0.80  # simulated edit-fidelity of the model (explicit knob)
    seed: int = 20261022
    simulated: bool = True  # True: simulated model; tools are always simulated stubs


@dataclass
class ToolCall:
    """One accounted tool invocation (structured telemetry record)."""

    name: str
    args: Dict[str, Any]
    tokens: int
    ok: bool


@dataclass
class LocalizationResult:
    """Output of a localizer: ranked symbols/files plus token accounting."""

    ranked_symbols: List[str]
    ranked_files: List[str]
    scores: Dict[str, float]
    tokens_to_rank: Dict[int, int]  # cumulative tokens spent to produce top-k
    tool_calls: List[ToolCall] = field(default_factory=list)
    strategy: str = "gg"

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.tool_calls)


@dataclass
class EditBlock:
    """Symbol-level edit materialized from a gold-patch hunk."""

    old_text: str
    new_text: str
    added_text: str
    removed_text: str


@dataclass
class CandidatePatch:
    """A speculative patch: edit one graph symbol's source text."""

    symbol: str
    old_text: str
    new_text: str
    origin: str = "model"  # model | oracle | distractor
    added_text: str = ""
    removed_text: str = ""


@dataclass
class VerificationResult:
    """Outcome of speculative patch verification."""

    accepted: bool
    confidence: float
    checks: Dict[str, Any]


@dataclass
class TaskTruth:
    """Ground truth derived from the gold patch (see ``GoldOracle``)."""

    instance_id: str
    target_symbols: List[str]
    target_files: List[str]
    gt_mode: str  # symbol | file
    edits: Dict[str, "EditBlock"]  # symbol -> edit materialized from gold hunks


# ---------------------------------------------------------------------------
# Gold-patch oracle: build ground truth and oracle materializations.
# ---------------------------------------------------------------------------


def file_to_symbol_prefix(filepath: str) -> Optional[str]:
    """``fastapi/routing.py`` -> ``fastapi.routing`` (mirrors graph node ids)."""
    if not filepath.endswith(".py"):
        return None
    return filepath[: -len(".py")].replace("/", ".")


class GoldOracle:
    """Derives ground-truth fault locations and oracle edits from gold patches.

    The harness never exposes ``patch`` to the agent; this class is used *only*
    by the benchmark to (a) score localization, (b) materialize simulated model
    outputs.  Attribution uses, in order of reliability: git hunk context (the
    enclosing function name emitted after ``@@``), exact changed-line matches
    inside a symbol body, and finally a whole-file fallback.
    """

    @staticmethod
    def parse_patch(patch: str) -> Dict[str, Dict[str, Any]]:
        """Parse a unified diff into per-file hunks with line-type provenance."""
        files: Dict[str, Dict[str, Any]] = {}
        current: Optional[str] = None
        for line in patch.splitlines():
            if line.startswith("+++ b/"):
                current = line[len("+++ b/") :].strip()
                files.setdefault(current, {"hunks": [], "context_names": []})
                continue
            if current is None:
                continue
            if line.startswith("@@"):
                m = re.match(r"^@@ [^@]* @@ ?(.*)$", line)
                if m and m.group(1).strip():
                    files[current]["context_names"].append(m.group(1).strip())
                files[current]["hunks"].append({"lines": []})
                continue
            if line.startswith(("--- a/", "diff ", "index ", "new file", "deleted file")):
                continue
            if line[:1] in "+- " and files[current]["hunks"]:
                files[current]["hunks"][-1]["lines"].append(
                    (line[:1], line[1:].rstrip("\n"))
                )
        return files

    @staticmethod
    def attribute(
        nodes: Sequence[Dict[str, str]], filepath: str, hunk_info: Dict[str, Any]
    ) -> Tuple[List[str], str]:
        """Map one patched file to its faulty graph symbols.

        Returns ``(symbol_ids, mode)`` where mode is ``hctx`` (hunk context),
        ``linehit`` (changed-line match), or ``file`` (fallback).
        """
        prefix = file_to_symbol_prefix(filepath)
        if prefix is None:
            return [], "file"
        cands = [n for n in nodes if n["id"] == prefix or n["id"].startswith(prefix + ".")]
        if not cands:
            return [], "file"
        for ctx in hunk_info.get("context_names", []):
            names = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", ctx)
            for name in reversed(names):
                hits = [
                    n
                    for n in cands
                    if n["id"].split(".")[-1] == name
                    or n["text"].lstrip().startswith(
                        (f"def {name}", f"async def {name}", f"class {name}")
                    )
                ]
                if hits:
                    return [n["id"] for n in hits], "hctx"
        changed = {
            t[1].strip()
            for h in hunk_info.get("hunks", [])
            for t in h["lines"]
            if t[0] in "+-" and len(t[1].strip()) >= 8
        }
        scored: collections.Counter = collections.Counter()
        for n in cands:
            for line in changed:
                if line in n["text"]:
                    scored[n["id"]] += 1
        if scored:
            best = max(scored.values())
            return [k for k, v in scored.items() if v == best], "linehit"
        return [n["id"] for n in cands], "file"

    @classmethod
    def build(cls, task: Dict[str, Any], nodes: Sequence[Dict[str, str]]) -> TaskTruth:
        """Construct the ``TaskTruth`` for one benchmark task."""
        parsed = cls.parse_patch(task["patch"])
        node_text = {n["id"]: n["text"] for n in nodes}
        targets: List[str] = []
        target_files: List[str] = []
        modes: List[str] = []
        edits: Dict[str, EditBlock] = {}
        for filepath, info in parsed.items():
            syms, mode = cls.attribute(nodes, filepath, info)
            modes.append(mode)
            target_files.append(filepath)
            targets.extend(syms)
            blocks = cls._edit_blocks(info["hunks"])
            for sym in syms:
                if mode == "file" or not blocks:
                    continue
                text = node_text.get(sym, "")
                chosen: Optional[EditBlock] = None
                for block in blocks:
                    old_lines = {
                        l.strip() for l in block.old_text.splitlines() if len(l.strip()) >= 6
                    }
                    if old_lines and sum(1 for l in old_lines if l in text) / len(old_lines) >= 0.7:
                        chosen = block
                        break
                edits[sym] = chosen or blocks[0]
        gt_mode = "symbol" if any(m in ("hctx", "linehit") for m in modes) else "file"
        return TaskTruth(
            instance_id=task["instance_id"],
            target_symbols=sorted(set(targets)) if gt_mode == "symbol" else [],
            target_files=sorted(set(target_files)),
            gt_mode=gt_mode,
            edits=edits,
        )

    @staticmethod
    def _dedent_block(block: str) -> str:
        """Remove the common leading indentation (keeps relative structure)."""
        lines = block.splitlines()
        indents = [
            len(l) - len(l.lstrip())
            for l in lines
            if l.strip()
        ]
        cut = min(indents) if indents else 0
        return "\n".join(l[cut:] if len(l) >= cut else l for l in lines).strip("\n")

    @classmethod
    def _edit_blocks(cls, hunks: Sequence[Dict[str, Any]]) -> List[EditBlock]:
        """Build one :class:`EditBlock` per informative hunk of a file."""
        blocks: List[EditBlock] = []
        for hunk in hunks:
            old_lines = [t[1] for t in hunk["lines"] if t[0] in " -"]
            new_lines = [t[1] for t in hunk["lines"] if t[0] in " +"]
            added = [t[1] for t in hunk["lines"] if t[0] == "+"]
            removed = [t[1] for t in hunk["lines"] if t[0] == "-"]
            if not any(len(l.strip()) >= 8 for l in old_lines + new_lines):
                continue
            blocks.append(
                EditBlock(
                    old_text=cls._dedent_block("\n".join(old_lines)),
                    new_text=cls._dedent_block("\n".join(new_lines)),
                    added_text=cls._dedent_block("\n".join(added)),
                    removed_text=cls._dedent_block("\n".join(removed)),
                )
            )
        return blocks


# ---------------------------------------------------------------------------
# GraphIndex: deterministic offline implementation of the harness tools.
# ---------------------------------------------------------------------------


class BM25Index:
    """Minimal BM25 over symbol documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: Sequence[List[str]]) -> None:
        self.docs = docs
        self.doclen = np.array([len(d) + 1 for d in docs], dtype=np.float64)
        self.avgdl = float(self.doclen.mean())
        df: collections.Counter = collections.Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [collections.Counter(d) for d in docs]

    def search(self, query: Sequence[str], k: int, allowed: Optional[set] = None) -> List[Tuple[int, float]]:
        scores = np.zeros(len(self.docs))
        for term in query:
            w = self.idf.get(term)
            if w is None:
                continue
            for idx, tf in enumerate(self.tf):
                if allowed is not None and idx not in allowed:
                    continue
                f = tf.get(term, 0)
                if f:
                    denom = f + 1.5 * (1 - 0.75 + 0.75 * self.doclen[idx] / self.avgdl)
                    scores[idx] += w * f * 2.5 / denom
        top = np.argsort(-scores)[: max(k * 3, k)]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0][:k]


class GraphIndex:
    """In-memory code graph + embedding index with harness-faithful tools.

    All tool methods return ``(response_text, ok)`` and register a
    :class:`ToolCall` on the ledger so token efficiency is measurable.
    """

    def __init__(self, graph_path: str, embedding_path: Optional[str] = None) -> None:
        with open(graph_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.repo_name: str = payload.get("graph", {}).get("repo_name", "")
        self.nodes: List[Dict[str, str]] = payload["nodes"]
        self.ids: List[str] = [n["id"] for n in self.nodes]
        self.text: Dict[str, str] = {n["id"]: n["text"] for n in self.nodes}
        self.index_of: Dict[str, int] = {n["id"]: i for i, n in enumerate(self.nodes)}
        self.out_edges: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
        self.in_edges: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
        self._edge_set: set = set()
        for e in payload["edges"]:
            self.out_edges[e["source"]].append((e["target"], e.get("type", "calls")))
            self.in_edges[e["target"]].append((e["source"], e.get("type", "calls")))
            self._edge_set.add((e["source"], e["target"]))
        # Symbol id -> source file (dotted prefix path).
        self.file_of: Dict[str, str] = {}
        self.file_nodes: Dict[str, List[str]] = collections.defaultdict(list)
        for nid in self.ids:
            filepath = nid.replace(".", "/") + ".py"
            self.file_of[nid] = filepath
            self.file_nodes[filepath].append(nid)
        # Shipped-library prior: the harness strips every patch hunk that
        # touches tests/, docs, scripts or runner config (HARNESS_README §8.1),
        # so fault localization searches library code only - for ALL strategies.
        self.non_library_prefixes = (
            "tests.", "test.", "scripts.", "docs.", "docs_src.",
            "benchmarks.", "examples.", "conftest",
        )
        self.library_set = {
            nid for nid in self.ids if not nid.startswith(self.non_library_prefixes)
        }
        self.library_index_set = {
            i for i, nid in enumerate(self.ids) if nid in self.library_set
        }
        self.library_files = sorted(
            {self.file_of[nid] for nid in self.library_set}
        )
        # Embeddings (real, 256-d) aligned with node ids.
        self.embeddings: Optional[np.ndarray] = None
        self.embedded_ids: Optional[List[str]] = None
        if embedding_path and os.path.exists(embedding_path):
            with np.load(embedding_path, allow_pickle=False) as npz:
                keys = [k for k in npz.files if k in self.index_of]
                if keys:
                    mat = np.stack([npz[k].astype(np.float32) for k in keys])
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    self.embeddings = mat / np.maximum(norms, 1e-8)
                    self.embedded_ids = keys
        # Lexical indexes.
        self.symbol_docs = [
            tokens_of(nid.replace(".", " ")) + tokens_of(self.text[nid]) for nid in self.ids
        ]
        self.symbol_bm25 = BM25Index(self.symbol_docs)
        # Generic-name prior: how frequent a symbol tail name is among
        # library nodes (routes like ``patch``/``get`` and dunders dominate
        # graphs but are rarely the root cause of a bug report).
        lib_ids = [nid for nid in self.ids if nid in self.library_set]
        tail_counts: collections.Counter = collections.Counter(
            nid.split(".")[-1] for nid in lib_ids
        )
        n_lib = max(1, len(lib_ids))
        self.tail_freq = {t: c / n_lib for t, c in tail_counts.items()}
        file_docs: Dict[str, List[str]] = {}
        for nid in self.ids:
            file_docs.setdefault(self.file_of[nid], []).extend(self.symbol_docs[self.index_of[nid]])
        self.file_order = sorted(file_docs)
        self.file_bm25 = BM25Index([file_docs[f] for f in self.file_order])
        self.ledger: List[ToolCall] = []

    # -- accounting ---------------------------------------------------------
    def _record(self, name: str, args: Dict[str, Any], response: str, ok: bool) -> None:
        # The harness truncates every tool response at max_stdout_chars=5,000;
        # token accounting uses the same cap so both strategies are comparable.
        visible = response[:MAX_STDOUT_CHARS]
        self.ledger.append(
            ToolCall(name=name, args=args, tokens=estimate_tokens(visible), ok=ok)
        )

    def _err(self, name: str, args: Dict[str, Any], message: str) -> Tuple[str, bool]:
        payload = json.dumps(
            {"status": "error", "error_type": "ToolError", "error_message": message}
        )
        self._record(name, args, payload, False)
        return payload, False

    # -- resolution (4-tier, per HARNESS_README §6.3) -----------------------
    def resolve_node(self, query: str) -> Optional[str]:
        """Resolve a short symbol name to a graph node id (4 tiers)."""
        q = query.strip()
        if q in self.index_of:
            return q
        tails: Dict[str, List[str]] = collections.defaultdict(list)
        for nid in self.ids:
            tails[nid.split(".")[-1]].append(nid)
        for name in (q, q.split(".")[-1]):
            if tails.get(name):
                return sorted(tails[name])[0]
        lowered = {k.lower(): v for k, v in tails.items()}
        for name in (q.lower(), q.split(".")[-1].lower()):
            if lowered.get(name):
                return sorted(lowered[name])[0]
        for name in (q.lower(), q.split(".")[-1].lower()):
            for key in sorted(lowered):
                if name and name in key:
                    return sorted(lowered[key])[0]
        return None

    # -- the four simulated harness tools ----------------------------------
    def search_similar_code(self, query: str, k: int = 10) -> Tuple[str, bool]:
        """Hybrid retrieval: BM25 + real-embedding cosine (RRF fused)."""
        args = {"query": query, "k": k}
        q_tokens = [t for t in tokens_of(query) if t not in _STOPWORDS]
        bm25_hits = self.symbol_bm25.search(
            q_tokens, self.retrieval_pool(), allowed=self.library_index_set
        )
        cos_hits: List[Tuple[int, float]] = []
        if self.embeddings is not None:
            # Pseudo query vector: centroid of embeddings of top BM25 nodes
            # whose symbol id contains a query token (Rocchio-style).
            seed_idx = [
                i
                for i, _ in bm25_hits[:12]
                if any(t in self.ids[i].lower() for t in q_tokens)
            ][:5]
            if not seed_idx:
                seed_idx = [i for i, _ in bm25_hits[:3]]
            if seed_idx:
                qvec = self.embeddings[seed_idx].mean(axis=0)
                qvec /= max(float(np.linalg.norm(qvec)), 1e-8)
                sims = self.embeddings @ qvec
                cand = [i for i in range(len(self.ids)) if i in self.library_index_set]
                cand.sort(key=lambda i: -float(sims[i]))
                cos_hits = [(i, float(sims[i])) for i in cand[: self.retrieval_pool()]]
        fused = self._rrf(
            [[i for i, _ in bm25_hits]], [[i for i, _ in cos_hits]], k
        )
        results = [
            {
                "symbol": self.ids[i],
                "file": self.file_of[self.ids[i]],
                "score": round(float(s), 4),
            }
            for i, s in fused
        ]
        response = json.dumps({"status": "ok", "results": results})
        self._record("search_similar_code", args, response, True)
        return response, True

    def retrieval_pool(self) -> int:
        return 50

    @staticmethod
    def _rrf(lists: Sequence[Sequence[int]], k_list: Sequence[Sequence[int]], k: int) -> List[Tuple[int, float]]:
        """Reciprocal-rank fusion of two ranked lists."""
        scores: Dict[int, float] = collections.defaultdict(float)
        for lst in lists:
            for rank, idx in enumerate(lst, start=1):
                scores[idx] += 1.0 / (60 + rank)
        for lst in k_list:
            for rank, idx in enumerate(lst, start=1):
                scores[idx] += 1.0 / (60 + rank)
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return top

    def get_code_neighbors(
        self, node: str, edge_type: Optional[str] = None, max_neighbors: int = 50
    ) -> Tuple[str, bool]:
        """Callers/callees of a symbol (harness-faithful neighbour tool)."""
        args = {"node": node, "edge_type": edge_type, "max_neighbors": max_neighbors}
        resolved = self.resolve_node(node)
        if resolved is None:
            return self._err("get_code_neighbors", args, f"cannot resolve node {node!r}")

        def filt(pairs: Iterable[Tuple[str, str]]) -> List[str]:
            out = [
                src
                for src, et in pairs
                if edge_type is None or et.lower() == edge_type.lower()
            ]
            return sorted(set(out))[:max_neighbors]

        payload = {
            "status": "ok",
            "node": resolved,
            "callers": filt(self.in_edges.get(resolved, [])),
            "callees": filt(self.out_edges.get(resolved, [])),
        }
        response = json.dumps(payload)
        self._record("get_code_neighbors", args, response, True)
        return response, True

    def get_code_subgraph(self, nodes: Sequence[str], hops: int = 2) -> Tuple[str, bool]:
        """Induced call subgraph connecting ``nodes``, expanded ``hops`` outward."""
        args = {"nodes": list(nodes), "hops": hops}
        resolved = [self.resolve_node(n) for n in nodes]
        resolved = [r for r in resolved if r is not None]
        if not resolved:
            return self._err("get_code_subgraph", args, "no node resolved")
        frontier = set(resolved)
        seen = set(frontier)
        layers: List[set] = []
        for _ in range(hops):
            nxt = set()
            for nid in frontier:
                for tgt, _ in self.out_edges.get(nid, []):
                    nxt.add(tgt)
                for src, _ in self.in_edges.get(nid, []):
                    nxt.add(src)
            nxt -= seen
            if not nxt:
                break
            layers.append(set(nxt))
            seen |= nxt
            frontier = nxt
        def degree(nid: str) -> int:
            return len(self.out_edges.get(nid, [])) + len(self.in_edges.get(nid, []))

        # Deterministic truncation, mirroring the harness' 5,000-char stdout cap:
        # keep the most highly-connected symbols per hop layer.
        capped_layers: List[List[str]] = []
        for layer in layers:
            ranked = sorted(layer, key=lambda n: (-degree(n), n))[:60]
            capped_layers.append(sorted(ranked))
        allowed = set(resolved) | {n for layer in capped_layers for n in layer}
        edges = [
            {"source": s, "target": t, "type": et}
            for s in sorted(allowed)
            for t, et in self.out_edges.get(s, [])
            if t in allowed
        ][:100]
        payload = {
            "status": "ok",
            "seeds": resolved,
            "layers": capped_layers,
            "nodes": sorted(allowed),
            "edges": edges,
        }
        response = json.dumps(payload)
        self._record("get_code_subgraph", args, response, True)
        return response, True

    def read_file(self, filepath: str) -> Tuple[str, bool]:
        """Virtual ``read_file`` over the graph (symbol bodies concatenated)."""
        args = {"filepath": filepath}
        nodes = self.file_nodes.get(filepath, [])
        if not nodes:
            return self._err("read_file", args, f"no graph content for {filepath}")
        lines: List[str] = []
        for nid in sorted(nodes):
            lines.extend(self.text[nid].splitlines())
            lines.append("")
        truncated = lines[:MAX_FILE_LINES]
        content = "\n".join(truncated)[:MAX_FILE_CHARS]
        payload = {
            "status": "ok",
            "filepath": filepath,
            "content": content,
            "start_line": 1,
            "end_line": len(truncated),
            "total_lines": len(lines),
            "is_truncated": len(lines) > MAX_FILE_LINES,
        }
        response = json.dumps(payload)
        self._record("read_file", args, response, True)
        return response, True

    def term_document_frequency(
        self, terms: Sequence[str], library_only: bool = True
    ) -> Dict[str, int]:
        """Document frequency of each term over symbol documents."""
        pool = self.library_index_set if library_only else set(range(len(self.ids)))
        out: Dict[str, int] = {}
        for t in terms:
            out[t] = sum(1 for i in pool if t in self.symbol_docs[i])
        return out

    # -- convenience used by the verifier -----------------------------------
    def callers_of(self, symbol: str) -> List[str]:
        return sorted({s for s, _ in self.in_edges.get(symbol, [])})


# ---------------------------------------------------------------------------
# Localizers.
# ---------------------------------------------------------------------------


class Localizer:
    """Graph-Guided (GG) and flat-file fault localizers with token ledgers."""

    def __init__(self, index: GraphIndex, config: AgentConfig) -> None:
        self.index = index
        self.config = config

    @staticmethod
    def query_terms(problem_statement: str, limit: int = 40) -> List[str]:
        """IDF-free distinctive terms of the problem statement (stopwords removed)."""
        counts = collections.Counter(
            t for t in tokens_of(problem_statement) if t not in _STOPWORDS
        )
        return [t for t, _ in counts.most_common(limit)]

    _HTTP_VERBS = frozenset(
        "get post put delete patch options head trace websocket api_route router".split()
    )

    # -- Graph-Guided ------------------------------------------------------
    def localize_gg(self, problem_statement: str) -> LocalizationResult:
        """Multi-query hybrid retrieval -> seeds -> 2-hop decayed expansion.

        Ranking stages: (1) RRF-fused hybrid retrieval (BM25 + real embedding
        cosine per query), (2) graph-informed re-ranking - call-graph
        centrality boost, generic-name / HTTP-verb penalties, code-identifier
        boost, (3) 2-hop call-graph expansion where expanded symbols inherit a
        decayed seed score plus a link-count bonus.
        """
        cfg = self.config
        terms = self._terms(problem_statement)
        idents = [x.lower() for x in self._identifier_terms(problem_statement)]
        queries = [" ".join(terms)]
        if len(idents) >= 3:
            queries.append(" ".join(idents[:10]))
        fused_scores: Dict[str, float] = {}
        for query_text in queries:
            response, _ = self.index.search_similar_code(
                query_text, k=cfg.retrieval_pool
            )
            for rank, hit in enumerate(json.loads(response)["results"], start=1):
                fused_scores[hit["symbol"]] = fused_scores.get(hit["symbol"], 0.0) + (
                    1.0 / (60.0 + rank)
                )
        deg = lambda s: len(self.index.out_edges.get(s, [])) + len(self.index.in_edges.get(s, []))  # noqa: E731
        deg_max = max([1] + [deg(s) for s in fused_scores])
        scores: Dict[str, float] = {}
        for symbol, score in fused_scores.items():
            tail = symbol.split(".")[-1]
            score *= 1 + 0.5 * math.log1p(deg(symbol)) / math.log1p(deg_max)
            if self.index.tail_freq.get(tail, 0.0) > 0.02:
                score *= 0.55  # very common name in this repo (routes, helpers)
            if tail.startswith("__") or tail in self._HTTP_VERBS:
                score *= 0.45  # dunder / HTTP-verb nodes are rarely root causes
            if idents and any(i in symbol.lower() for i in idents):
                score *= 1.3  # symbol path matches a code identifier in the issue
            scores[symbol] = score
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        seeds = [symbol for symbol, _score in ranked[: cfg.seed_top_m]]
        if seeds:
            response2, _ = self.index.get_code_subgraph(seeds, hops=cfg.expand_hops)
            sub = json.loads(response2)
            seed_min = min(scores.get(s, 0.0) for s in seeds)
            link_count: collections.Counter = collections.Counter()
            hop_of: Dict[str, int] = {}
            for hop, layer in enumerate(sub["layers"], start=1):
                for nid in layer:
                    hop_of.setdefault(nid, hop)
                    for seed in seeds:
                        if (seed, nid) in self.index._edge_set or (
                            nid, seed
                        ) in self.index._edge_set:
                            link_count[nid] += 1
            for nid, hop in hop_of.items():
                if nid in scores:
                    continue
                tail = nid.split(".")[-1]
                decay = cfg.hop_decay**hop
                bonus = 0.25 * decay * int(any(t in tail for t in terms))
                link_bonus = 0.15 * (link_count.get(nid, 0) - 1)
                scores[nid] = max(0.0, seed_min * decay + bonus + max(0.0, link_bonus))
        # Two-tier ranking: retrieval seeds always precede expanded symbols
        # (expansion adds neighbourhood context for verification; it never
        # displaces the retrieval evidence at the top of the list).
        seed_ranked = sorted(seeds, key=lambda s: (-scores.get(s, 0.0), s))
        expanded = sorted(
            (s for s in scores if s not in set(seeds)),
            key=lambda s: (-scores.get(s, 0.0), s),
        )
        order = seed_ranked + expanded
        # Token-to-rank accounting: the ranking is materialized after the
        # retrieval + subgraph responses are in context.
        tokens_to_rank: Dict[int, int] = {}
        total = sum(c.tokens for c in self.index.ledger)
        for k in (1, 3, 5):
            tokens_to_rank[k] = total
        files = self._rank_files(order)
        calls = list(self.index.ledger)
        self.index.ledger = []
        return LocalizationResult(
            ranked_symbols=order,
            ranked_files=files,
            scores=scores,
            tokens_to_rank=tokens_to_rank,
            tool_calls=calls,
            strategy="gg",
        )

    # -- Flat baseline ------------------------------------------------------
    def localize_flat(self, problem_statement: str) -> LocalizationResult:
        """No-graph control: BM25 file ranking + whole-file reads + scan."""
        cfg = self.config
        terms = self._terms(problem_statement)
        query = " ".join(terms)
        lib_files = set(self.index.library_files)
        file_hits = [
            (i, s)
            for i, s in self.index.file_bm25.search(terms, cfg.flat_read_files * 4)
            if self.index.file_order[i] in lib_files
        ][: cfg.flat_read_files]
        ranked_files = [self.index.file_order[i] for i, _ in file_hits]
        order: List[str] = []
        scores: Dict[str, float] = {}
        cumulative = 0
        tokens_to_rank: Dict[int, int] = {}
        calls: List[ToolCall] = []
        for fr_rank, filepath in enumerate(ranked_files, start=1):
            response, _ = self.index.read_file(filepath)
            call = self.index.ledger[-1]
            calls.append(call)
            cumulative += call.tokens
            content = json.loads(response)["content"].lower()
            match_score = sum(content.count(t) for t in terms)
            file_score = 1.0 / (60.0 + fr_rank) * (1 + math.log1p(match_score))
            # Rank symbols inside the file by textual match to the query.
            sym_scores: List[Tuple[float, str]] = []
            for nid in sorted(self.index.file_nodes[filepath]):
                text_l = self.index.text[nid].lower()
                s = sum(text_l.count(t) for t in terms) + (
                    0.5 if any(t in nid.lower() for t in terms) else 0.0
                )
                sym_scores.append((float(s), nid))
            sym_scores.sort(reverse=True)
            for s, nid in sym_scores:
                if nid not in scores:
                    scores[nid] = file_score * (1 + s)
                    order.append(nid)
            for k in (1, 3, 5):
                if k not in tokens_to_rank and len(order) >= k:
                    tokens_to_rank[k] = cumulative
        total = sum(c.tokens for c in calls)
        for k in (1, 3, 5):
            tokens_to_rank.setdefault(k, total)
        self.index.ledger = []
        return LocalizationResult(
            ranked_symbols=order,
            ranked_files=ranked_files,
            scores=scores,
            tokens_to_rank=tokens_to_rank,
            tool_calls=calls,
            strategy="flat",
        )

    # -- helpers ------------------------------------------------------------
    def _terms(self, problem_statement: str) -> List[str]:
        """TF-IDF query terms: ubiquitous tokens are dropped, rare ones kept."""
        counts = collections.Counter(
            t for t in tokens_of(problem_statement) if t not in _STOPWORDS and len(t) > 2
        )
        if not counts:
            return []
        df = self.index.term_document_frequency(list(counts))
        n = max(1, len(self.index.library_index_set))
        scored = []
        for t, c in counts.items():
            ratio = df[t] / n
            if ratio > 0.05 or df[t] == 0:
                continue
            idf = math.log(n / (1 + df[t]))
            scored.append((c * idf * idf, t))
        scored.sort(reverse=True)
        if len(scored) < 6:
            ranked = sorted(counts, key=lambda t: df[t])
            scored = [(1.0, t) for t in ranked[:10]]
        return [t for _, t in scored[:20]]

    @staticmethod
    def _identifier_terms(problem_statement: str) -> List[str]:
        """Code-like tokens (CamelCase / snake_case) in the problem statement."""
        out: List[str] = []
        for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", problem_statement):
            if re.search(r"[_A-Z]", raw[1:]) and raw.lower() not in _STOPWORDS:
                out.append(raw)
        seen: set = set()
        uniq = []
        for t in out:
            if t.lower() not in seen:
                seen.add(t.lower())
                uniq.append(t)
        return uniq

    def _rank_files(self, symbol_order: Sequence[str]) -> List[str]:
        files: List[str] = []
        seen = set()
        for nid in symbol_order:
            fp = self.index.file_of[nid]
            if fp not in seen:
                seen.add(fp)
                files.append(fp)
        return files


# ---------------------------------------------------------------------------
# Patch verifier.
# ---------------------------------------------------------------------------


class PatchVerifier:
    """Speculative verification of a candidate symbol edit.

    Checks: (1) syntax of the added core (structural fragments allowed),
    (2) definition preserved, (3) applicability - the diff context must exist
    at the edit site, (4) graph consistency - the edited symbol must belong to
    the localized sub-graph, (5) evidence margin - a patch on a weakly ranked
    symbol is speculative, (6) caller compatibility - names removed by the
    edit must not be referenced by callers, (7) scope discipline.  Confidence
    is the product of per-check factors, thresholded by ``accept_threshold``.
    """

    def __init__(self, index: GraphIndex, config: AgentConfig) -> None:
        self.index = index
        self.config = config

    @staticmethod
    def _syntax_status(text: str) -> Tuple[bool, Optional[str]]:
        """Parse a hunk excerpt with indentation normalization + tail trimming.

        Patch excerpts start at arbitrary nesting depth and may cut across
        block boundaries (``for ...:`` whose body never appears).  We re-base
        the indentation on the first line, then drop dangling tail lines until
        the fragment parses.  The check only passes if at least half of the
        original lines survive the trim, so degenerate fragments are rejected.
        """
        raw_lines = text.splitlines()
        total = max(1, sum(1 for l in raw_lines if l.strip()))
        base = next((len(l) - len(l.lstrip()) for l in raw_lines if l.strip()), 0)
        rebased = [
            (" " * max(0, len(l) - len(l.lstrip()) - base) + l.lstrip())
            if l.strip()
            else ""
            for l in raw_lines
        ]
        error: Optional[str] = None
        for head in (rebased, textwrap.dedent(text).splitlines()):
            lines2 = list(head)
            for _ in range(len(lines2) + 1):
                kept = sum(1 for l in lines2 if l.strip())
                if kept < math.ceil(0.5 * total):
                    break
                try:
                    ast.parse("\n".join(lines2) or "pass")
                    return True, None
                except SyntaxError as exc:
                    error = str(exc)
                if lines2:
                    lines2.pop()
        return False, error or "empty excerpt"

    @staticmethod
    def _is_benign_fragment(added: str) -> bool:
        """True for patch excerpts that legitimately fail ``ast.parse``.

        These are docstring prose lines, comment lines, continuation lines
        ending in a bracket/comma, and single-line signature/parameter edits
        whose surrounding statement is outside the excerpt window.
        """
        lines = [l for l in added.splitlines() if l.strip()]
        if not lines:
            return False
        # comment-only additions
        if all(l.lstrip().startswith("#") for l in lines):
            return True
        # structural continuation lines (old behaviour)
        if all(
            l.rstrip().endswith((",", "(", "[", "{", "%", "+")) or "(" in l
            for l in lines
        ):
            return True
        # docstring body: quotes are balanced *somewhere* in the symbol, but
        # an excerpt of the prose alone is not a statement — accept lines
        # that look like natural-language documentation.
        proseish = sum(
            1
            for l in lines
            if not re.match(
                r"\s*(?:return|raise|yield|assert|if|for|while|with|try|except|"
                r"elif|else|def|class|import|from)\b",
                l,
            )
            and not l.rstrip().endswith(":")
        )
        if proseish == len(lines):
            return True
        # single signature/parameter line, e.g. ``x: int = 1,`` or ``def f():``
        if len(lines) == 1 and re.match(
            r"\s*(?:def\s+|@)?[A-Za-z_][\w.:]*(?:\s*[:=].*)?,?\s*$", lines[0]
        ):
            return True
        return False

    def verify(
        self, candidate: CandidatePatch, localization: LocalizationResult
    ) -> VerificationResult:
        checks: Dict[str, Any] = {}
        confidence = 1.0
        # 1. syntax of the added core (hunk excerpts may be structural
        #    fragments, e.g. a new argument in a call's parameter list)
        added = candidate.added_text or candidate.new_text
        ok_syntax, syntax_error = self._syntax_status(added)
        structural = False
        if not ok_syntax and added.strip():
            structural = self._is_benign_fragment(added)
            ok_syntax = structural
        if not added.strip():
            # pure-deletion edits add nothing; syntax is vacuously fine
            ok_syntax, structural = True, False
            checks["pure_deletion"] = True
        checks["syntax_ok"] = ok_syntax
        checks["structural_fragment"] = structural
        if not ok_syntax:
            checks["syntax_error"] = syntax_error
            confidence *= 0.0
        # 2. definition preserved
        old_names = set(re.findall(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", candidate.old_text))
        new_names = set(re.findall(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", candidate.new_text))
        checks["definition_preserved"] = not old_names or old_names <= new_names
        if not checks["definition_preserved"]:
            confidence *= 0.35
        # 3. applicability: the diff context must exist at the edit site
        source = self.index.text.get(candidate.symbol, "")
        src_lines = {l.strip() for l in source.splitlines()}
        old_lines = {
            l.strip() for l in candidate.old_text.splitlines() if len(l.strip()) >= 6
        }
        stripped_old = [l.strip() for l in candidate.old_text.splitlines() if l.strip()]
        import_hunk = bool(stripped_old) and all(
            re.match(r"(?:from\s+\S+(?:\.\S+)*\s+)?import\s", l) or l.startswith("#")
            for l in stripped_old
        )
        if import_hunk:
            # module-level import hunks sit outside any symbol body; the
            # symbol-excerpt search cannot confirm them, so stay neutral
            checks["module_level_edit"] = True
            checks["applicability"] = None
        elif old_lines:
            frac = sum(1 for l in old_lines if l in src_lines) / len(old_lines)
            checks["applicability"] = round(frac, 3)
            # continuous factor: context match at the edit site is the single
            # strongest discriminator between true and distractor patches
            confidence *= 0.15 + 0.85 * min(1.0, frac / 0.7)
        else:
            checks["applicability"] = None
        # 4. graph consistency: symbol inside localized subgraph
        top = set(localization.ranked_symbols[: max(20, self.config.seed_top_m * 3)])
        in_graph = candidate.symbol in self.index.index_of
        checks["symbol_in_graph"] = in_graph
        if not in_graph:
            confidence *= 0.1
        in_localization = candidate.symbol in top or any(
            self.index.file_of.get(candidate.symbol, "") == f
            for f in localization.ranked_files[:3]
        )
        checks["graph_consistent"] = bool(in_localization)
        if not in_localization:
            confidence *= 0.4
        # 5. evidence margin: patching a weakly ranked symbol is speculative
        all_scores = list(localization.scores.values())
        best = max(all_scores) if all_scores else 0.0
        ratio = (
            localization.scores.get(candidate.symbol, 0.0) / best if best > 0 else 0.0
        )
        checks["evidence_ratio"] = round(float(ratio), 4)
        confidence *= min(1.0, 0.5 + 5.0 * ratio)
        # 6. caller compatibility over the 2-hop neighbourhood
        callers = self.index.callers_of(candidate.symbol)
        removed = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate.old_text)) - set(
            re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate.new_text)
        )
        removed -= {"self", "cls", "none", "true", "false", "not", "in", "is", "and", "or"}
        risky = [
            c
            for c in callers
            if any(re.search(rf"\b{re.escape(name)}\b", self.index.text.get(c, "")) for name in removed)
        ]
        checks["callers_total"] = len(callers)
        checks["risky_callers"] = risky[:10]
        caller_risk = min(1.0, len(risky) / max(1, min(len(callers), 10)))
        checks["caller_risk"] = round(caller_risk, 4)
        # soft penalty: removed identifiers still visible in callers are a
        # warning, not a veto (gold docstring/param edits trip it too)
        confidence *= 1.0 - 0.5 * caller_risk
        # 7. scope discipline: keep edits near the localized neighbourhood
        neighbourhood = set(localization.ranked_symbols[:30])
        scope_ok = candidate.symbol in neighbourhood
        checks["scope_ok"] = bool(scope_ok)
        if not scope_ok:
            confidence *= 0.5
        confidence = float(max(0.0, min(1.0, confidence)))
        accepted = confidence >= self.config.accept_threshold
        checks["decision"] = "accept" if accepted else "reject"
        return VerificationResult(accepted=accepted, confidence=confidence, checks=checks)


# ---------------------------------------------------------------------------
# Simulated model + agent loop.
# ---------------------------------------------------------------------------


@dataclass
class TaskOutcome:
    """Per-task structured telemetry returned by :meth:`AgentLoop.run_task`."""

    instance_id: str
    config_name: str
    strategy: str
    model_id: str
    localized: bool
    recall_at: Dict[str, bool]
    tokens_read: int
    tool_calls: int
    attempts: int
    patch_submitted: bool
    patch_correct: bool
    verifier_accepted: bool
    wall_time_s: float
    telemetry: Dict[str, Any] = field(default_factory=dict)


class SimulatedModel:
    """Deterministic stand-in for a quantized Gemma-4 coder.

    The model is grounded in the benchmark in two ways: (a) it can only
    "propose" an edit whose *content* is materialized from the gold patch (so
    correctness is scored against real artifacts, not imagination), and (b) its
    success probability is an explicit, declared fidelity knob ``f`` swept in
    the ablation.  Sampling is seeded per task, so runs are reproducible.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def rng_for(self, instance_id: str, attempt: int) -> np.random.Generator:
        digest = hashlib.sha256(f"{instance_id}:{attempt}:{self.config.seed}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        return np.random.default_rng(seed)

    def propose(
        self,
        instance_id: str,
        attempt: int,
        ranked_symbols: Sequence[str],
        truth: TaskTruth,
    ) -> Tuple[Optional[CandidatePatch], bool]:
        """Propose a speculative edit; returns ``(candidate, is_correct)``."""
        if not ranked_symbols:
            return None, False
        pick = ranked_symbols[0]
        rng = self.rng_for(instance_id, attempt)
        edit = truth.edits.get(pick)
        hit = pick in truth.target_symbols and edit is not None
        faithful = bool(rng.random() < self.config.fidelity)
        if hit and edit is not None:
            if faithful:
                return (
                    CandidatePatch(
                        pick, edit.old_text, edit.new_text, origin="oracle",
                        added_text=edit.added_text, removed_text=edit.removed_text,
                    ),
                    True,
                )
            # Correct locus, unfaithful edit: corrupt the gold block so the
            # simulation's fidelity knob actually changes the artifact.
            rng2 = self.rng_for(instance_id, attempt * 7919)
            mode = int(rng2.integers(0, 3))
            if mode == 0 and "=" in edit.new_text:
                broken = edit.new_text.replace("=", "==", 1)
            elif mode == 1 and len(edit.new_text.splitlines()) > 1:
                # drop the last non-empty line: often a syntax/structure error
                lines = edit.new_text.splitlines()
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip():
                        del lines[i]
                        break
                broken = "\n".join(lines)
            else:
                # semantic corruption: reference a nonexistent identifier
                ids = sorted(
                    set(re.findall(r"\b[a-z_][a-z0-9_]{3,}\b", edit.new_text))
                )
                if ids:
                    victim = ids[int(rng2.integers(0, len(ids)))]
                    broken = edit.new_text.replace(victim, f"undefined_{victim}", 1)
                else:
                    broken = edit.new_text + "\n    pass  # corrupted\n"
            if broken == edit.new_text:
                broken += "\n    pass  # corrupted (no-op guard)\n"
            # the verifier must see the *broken* text, not the gold excerpt
            return (
                CandidatePatch(
                    pick, edit.old_text, broken, origin="model",
                    added_text="", removed_text=edit.removed_text,
                ),
                False,
            )
        # Wrong locus: apply the first gold edit block to the wrong symbol.
        src = next(iter(truth.edits.values()), None)
        if src is None:
            return None, False
        return (
            CandidatePatch(
                pick, src.old_text or "pass", src.new_text or "pass", origin="model",
                added_text=src.added_text, removed_text=src.removed_text,
            ),
            False,
        )


class AgentLoop:
    """Localize -> hypothesize -> verify -> (resample | submit)."""

    def __init__(
        self,
        index: GraphIndex,
        config: AgentConfig,
        use_verifier: bool = True,
        hops: Optional[int] = None,
    ) -> None:
        self.index = index
        self.config = dataclasses.replace(
            config, expand_hops=hops if hops is not None else config.expand_hops
        )
        self.localizer = Localizer(index, self.config)
        self.verifier = PatchVerifier(index, self.config)
        self.model = SimulatedModel(self.config)
        self.use_verifier = use_verifier

    def run_task(
        self,
        task: Dict[str, Any],
        truth: TaskTruth,
        strategy: str = "gg",
        config_name: str = "gg_spv",
    ) -> TaskOutcome:
        start = time.perf_counter()
        statement = task["problem_statement"]
        if strategy == "gg":
            localization = self.localizer.localize_gg(statement)
        else:
            localization = self.localizer.localize_flat(statement)
        recall_at = {
            "1": bool(localization.ranked_symbols[:1] and set(localization.ranked_symbols[:1]) & set(truth.target_symbols)),
            "3": bool(set(localization.ranked_symbols[:3]) & set(truth.target_symbols)),
            "5": bool(set(localization.ranked_symbols[:5]) & set(truth.target_symbols)),
        }
        tool_calls = len(localization.tool_calls)
        attempts = 0
        accepted = False
        correct = False
        submitted = False
        trace: List[Dict[str, Any]] = []
        pool = list(localization.ranked_symbols)
        for attempt in range(1, self.config.max_attempts + 1):
            if tool_calls >= self.config.max_tool_calls:
                break
            attempts = attempt
            candidate, is_correct = self.model.propose(
                task["instance_id"], attempt, pool, truth
            )
            if candidate is None:
                break
            if self.use_verifier:
                vres = self.verifier.verify(candidate, localization)
                trace.append(
                    {
                        "attempt": attempt,
                        "symbol": candidate.symbol,
                        "origin": candidate.origin,
                        "correct": is_correct,
                        "confidence": round(vres.confidence, 4),
                        "decision": vres.checks["decision"],
                        "checks": vres.checks,
                    }
                )
                if vres.accepted:
                    accepted = True
                    correct = is_correct
                    submitted = True
                    break
                # speculative rejection: re-rank and retry
                pool = [s for s in pool if s != candidate.symbol]
            else:
                accepted = True
                correct = is_correct
                submitted = True
                trace.append(
                    {
                        "attempt": attempt,
                        "symbol": candidate.symbol,
                        "origin": candidate.origin,
                        "correct": is_correct,
                        "decision": "accept_unverified",
                    }
                )
                break
        outcome = TaskOutcome(
            instance_id=task["instance_id"],
            config_name=config_name,
            strategy=strategy,
            model_id=self.config.model_id,
            localized=recall_at["3"],
            recall_at=recall_at,
            tokens_read=localization.total_tokens,
            tool_calls=tool_calls + attempts,
            attempts=attempts,
            patch_submitted=submitted,
            patch_correct=correct,
            verifier_accepted=accepted,
            wall_time_s=time.perf_counter() - start,
            telemetry={
                "strategy": strategy,
                "fidelity": self.config.fidelity,
                "hops": self.config.expand_hops,
                "use_verifier": self.use_verifier,
                "top5": localization.ranked_symbols[:5],
                "targets": truth.target_symbols[:5],
                "gold_files": truth.target_files,
                "gt_mode": truth.gt_mode,
                "trace": trace,
                "tokens_to_rank": localization.tokens_to_rank,
            },
        )
        return outcome


# ---------------------------------------------------------------------------
# Dataset loading helpers.
# ---------------------------------------------------------------------------


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """Load the benchmark ``tasks.jsonl``."""
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def index_artifacts(data_dir: str) -> Dict[str, Dict[str, str]]:
    """Map ``base_commit -> {"graph": ..., "embedding": ...}`` for local artifacts."""
    out: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    pattern = re.compile(r"^([A-Za-z]+)_([0-9a-f]{40})\.(json|npz)$")
    for name in sorted(os.listdir(data_dir)):
        m = pattern.match(name)
        if not m:
            continue
        kind = "graph" if m.group(3) == "json" else "embedding"
        out[m.group(2)][kind] = os.path.join(data_dir, name)
    return out


def load_index_for_task(
    task: Dict[str, Any], artifacts: Dict[str, Dict[str, str]]
) -> Optional[GraphIndex]:
    """Build a :class:`GraphIndex` for a task's ``base_commit`` if available."""
    entry = artifacts.get(task["base_commit"], {})
    if "graph" not in entry:
        return None
    return GraphIndex(entry["graph"], entry.get("embedding"))
