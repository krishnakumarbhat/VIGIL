# GG-SPV: Graph-Guided Hierarchical Fault Localization and Speculative Patch Verification for LLM Coding Agents

**Track:** Google — The Gemma 4 Developer Agent Paper Track

> **Measurement note.** All localization, verification, and token-accounting numbers in this paper are measured on the competition's real repository artifacts (code-property graphs and 256-d node embeddings for 40 tasks across `fastapi`, `requests`, `rich`, and `httpx`, plus gold patches used as ground truth). The Gemma-4 model weights are not runnable in the evaluation environment, so **patch generation is simulated** with a declared fidelity knob; every simulated quantity is labeled as such. Real vs. simulated quantities are separated in every table.

## Abstract

LLM coding agents waste context reading code: a flat file-listing baseline spends a median of 3,168 tokens per task before proposing a patch, and retrieves the true faulting symbol in the top 5 only 27.3% of the time. We propose **GG-SPV**, a two-stage agent architecture that (1) localizes faults with a graph-guided ranker fusing BM25, node embeddings, call-graph centrality, and a shipped-library prior, then (2) filters speculative patches with a seven-check verifier (syntax, definition preservation, edit-site applicability, graph consistency, evidence margin, caller compatibility, scope). On 22 tasks with symbol-level ground truth over the competition's real repository artifacts, graph guidance raises recall@1 from 4.5% to 18.2% (4.0×) while cutting mean tokens read by 43% and tool calls from 12.2 to 3.5. Conditional on successful localization, the verifier separates gold patches from distractors at AUROC 0.875 (same-file) / 0.905 (cross-file) with TPR 0.90 at a 3.8%/0.0% false-positive rate. We report an honest negative result — call-graph expansion beyond the seed set does not improve top-k recall — and an explicitly labeled simulation of patch synthesis whose fidelity knob monotonically controls the pass rate.

## 1. Introduction

Agentic bug fixing decomposes into fault localization (which symbols to edit?) and patch synthesis (what edit?). Most agent frameworks collapse the two into an open-ended explore-edit loop whose cost grows with repository size. Two observations motivate a structured alternative. First, repositories ship *static structure* — call graphs, inheritance, embedding-similar code — that agents typically ignore, re-reading files the graph could have pruned. Second, LLM patch proposals are *speculative*: small models frequently emit syntactically invalid or mis-targeted edits, yet agent loops submit them without a cheap mechanical check.

GG-SPV addresses both: a graph-guided localizer that spends tool budget only on a pruned subgraph, and a patch verifier that gates submission on deterministic checks before any execution is attempted.

**Contributions.**
1. A graph-guided localizer (BM25 ⊕ embedding cosine ⊕ RRF fusion ⊕ call-centrality boost ⊕ generic-name penalty) evaluated on real competition graphs (§2.1).
2. A seven-check speculative-patch verifier with continuous confidence factors, evaluated as a gold-vs-distractor discriminator (§2.2).
3. A reproducible ablation suite (12 configurations) with all numbers measured on real artifacts and simulated patch synthesis explicitly labeled (§3).

## 2. Methodology

### 2.1 Graph-Guided Hierarchical Localization

The localizer ranks library symbols (excluding `tests./scripts./docs./docs_src./benchmarks./examples.` — the shipped-library prior, mirroring the harness's test-stripping behavior) using four signal families:

- **Lexical BM25** over identifier-split symbol text, seeded by tf-idf terms with document frequency ≤ 5% (rare terms are discriminative), plus an optional second query of code identifiers extracted from the problem statement.
- **Dense retrieval** with the competition's real 256-d node embeddings: a Rocchio pseudo-query (centroid of top lexical hits moved toward the query) scores every node by cosine similarity.
- **Fusion** by Reciprocal Rank Fusion (RRF), which is scale-free and needs no score calibration between BM25 and cosine.
- **Graph structure**: a call-graph centrality boost ×(1 + 0.5·log1p(deg)/log1p(deg_max)), an identifier-path boost ×1.3 when a symbol's dotted name appears verbatim in the statement, and a generic-name penalty (tail-frequency names ×0.55; dunder and HTTP-verb names ×0.45).

The top-8 fused symbols become *seeds*; a two-tier ranking then appends 2-hop call-graph expansions (verification context only — expansions never displace seeds, §3.3).

### 2.2 Speculative Patch Verification

Submitted patches pass seven checks whose factors multiply into a confidence in [0, 1]; confidence ≥ 0.5 accepts:

1. **Syntax** of the added core via `ast.parse` with indentation re-basing, with a benign-fragment allowance for docstring prose, comments, and continuation lines that legitimately fail standalone parsing.
2. **Definition preservation** — every `def`/`class` name in the old text survives in the new text.
3. **Edit-site applicability** — the fraction of the diff's context lines present at the edit site (module-level import hunks are exempted as unverifiable-from-excerpt); continuous factor 0.15 + 0.85·min(1, frac/0.7). This is the strongest single discriminator (§3.2).
4. **Graph consistency** — the patched symbol appears in the localized top-k or top-3 files.
5. **Evidence margin** — min(1, 0.5 + 5·relative score), punishing patches to weakly ranked symbols.
6. **Caller compatibility** — soft penalty (×(1 − 0.5·risk)) when identifiers removed by the edit remain visible in 2-hop callers.
7. **Scope discipline** — the symbol lies in the localized neighbourhood.

### 2.3 Simulated Patch Synthesis (labeled)

Gemma-4 weights are unavailable offline. Candidates are materialized from gold edits with a declared fidelity knob *f* (per-model: e2b 0.45, e4b 0.62, 12b 0.80, 31b 0.90; swept over f ∈ {0.3, 0.5, 0.7, 1.0}) that corrupts edits at rate (1 − f). All localization and verifier numbers are measured on real artifacts; only the pass-rate-vs-fidelity curve is a function of the simulation.

## 3. Empirical Ablations

All numbers are from the executed run (`python3 run_benchmark.py`, 40 tasks with local graph+embedding artifacts; symbol-ground-truth subset n = 22; token counts are the chars/4 proxy over deterministic offline tool stubs). Patch synthesis is simulated; see §2.3.

### 3.1 Localization: graph guidance vs. flat reading

| Configuration | R@1 | R@3 | R@5 | file R@5 | mean tokens | median tokens | tool calls |
|---|---|---|---|---|---|---|---|
| flat baseline (12b) | 0.045 | 0.227 | 0.273 | 0.375 | 3,691 | 3,168 | 12.18 |
| GG, hops=0 (seed ranking only) | **0.182** | 0.273 | 0.318 | 0.350 | **2,102** | **1,760** | 3.48 |
| GG, hops=1 | 0.182 | 0.273 | 0.318 | 0.350 | 3,023 | 2,500 | 3.48 |
| GG-SPV, hops=2 | 0.182 | 0.273 | 0.318 | 0.350 | 3,032 | 2,500 | 4.70 |

Graph guidance raises recall@1 4.0× (0.045 → 0.182, 1/22 → 4/22 tasks) while cutting mean tokens 43% (3,691 → 2,102 for the seed-only variant) and tool calls 3.5×. **Honest negative result:** 1- and 2-hop call-graph expansions do not improve top-k recall at all (0.318 at every hop setting) and cost ~920 extra tokens per task; on this benchmark, better ranking of the seed set — not deeper expansion — is what moves recall, and hops=0 is the token-optimal configuration. Expansions still serve as verification context (§2.2).

### 3.2 Verifier discrimination (conditional on localization)

The verifier's contract is to gate speculative patches *over the localized neighbourhood*, so we score gold edits whose symbol reached the localizer's top-20 (n = 10 of 69 gold edits; the other 59 are outside the verifier's scope) against same-file distractors (n = 26) and cross-file distractors (n = 30), using the 12b configuration (f = 0.80):

| Metric | Value |
|---|---|
| gold mean confidence | 0.774 |
| same-file distractor mean confidence | 0.103 |
| cross-file distractor mean confidence | 0.031 |
| TPR @ 0.5 | **0.90** |
| FPR @ 0.5 (same-file / cross-file) | 0.038 / 0.000 |
| AUROC (same-file / cross-file) | **0.875 / 0.905** |

The strongest single factor is edit-site applicability: the fraction of the diff's context lines present at the edit site averages 0.46 for golds vs. 0.06 for hard distractors. End-to-end, the verifier cuts the speculative submission rate from 0.55 (ungated) to 0.125–0.15 while *raising* the correct-patch rate (below), i.e. it preferentially rejects unfaithful edits.

### 3.3 Simulated model fidelity sweep (labeled: simulated synthesis)

Pass rate over all 40 tasks, and conditioned on successful localization (7/40 tasks have the gold symbol in the top-5):

| Configuration | fidelity f | pass (all 40) | pass \| localized (7) | verifier accept rate |
|---|---|---|---|---|
| gg_spv_e2b | 0.45 | 0.100 | 4/7 | 0.125 |
| gg_spv_e4b | 0.62 | 0.100 | 4/7 | 0.125 |
| gg_spv_12b | 0.80 | 0.125 | 5/7 | 0.150 |
| gg_spv_31b | 0.90 | 0.125 | 5/7 | 0.150 |
| flat baseline (no verifier) | 0.80 | 0.025 | 1/6 | 0.550 |
| gg, no verifier | 0.80 | 0.100 | 4/7 | 0.550 |

The sweep is monotone (4/7 → 5/7 conditioned on localization) but the aggregate pass rate is dominated by localization recall, not by fidelity: at n = 7 localized tasks a one-task difference is within noise. The defensible claims are (a) graph guidance quadruples the end-to-end pass rate over the flat baseline (0.025 → 0.10–0.125) by localizing more faults, and (b) the verifier raises pass|localized from 4/7 (ungated, f = 0.80) to 5/7 while accepting only 15% of speculative submissions. Absolute pass rates characterize the simulator, not Gemma-4.

## 4. Limitations

- 40 of 129 tasks have local graph+embedding artifacts; symbol-level ground truth exists for 22. Confidence intervals are wide (recall@1 differences of ±0.09 are within noise at n=22).
- Patch synthesis is simulated; absolute pass rates characterize the simulator, not Gemma-4. The fidelity sweep demonstrates monotonicity, not accuracy.
- The verifier's strongest check (edit-site applicability) is evaluated with gold diffs as context carriers; real model diffs may carry noisier context.
- Offline tool stubs simulate the harness's tool contract (deterministic, 5,000-char responses, chars/4 token proxy); sandbox execution is not exercised.

## 5. Related Work

Spectrum-based fault localization (Abreu et al., *J. Syst. Softw.* 82(5), 2009, Ochiai) scores statements from test spectra; we replace spectra with LLM query relevance plus graph structure. Agentless (Xia et al., FSE 2025) and RepoGraph (Ouyang et al., ICLR 2025) show static structure narrows the localization search space; GG-SPV adds a hierarchical seed/expansion split and a verification gate. SWE-agent (Yang et al., NeurIPS 2024) and AutoCodeRover (Zhang et al., ISSTA 2024) motivate agent-computer interfaces over repositories; our contribution is orthogonal — a measurable localization-and-verification layer compatible with any proposer. Recent LLM-FL studies (arXiv:2510.20521; 2502.13966; 2509.20552) evaluate LLMs as localizers; we isolate the token/recall frontier and the verifier's discriminative power.

## References

1. Abreu, R., Zoeteweij, P., van Gemund, A.J. (2009). On the coupling of testing and debugging... *J. Syst. Softw.* 82(5), 786–802. DOI: 10.1016/j.jss.2009.06.035.
2. Xia, C.S., Deng, Y., Dunn, S., Zhang, L. (2025). Agentless: Demystifying LLM-based Software Engineering Agents. *Proc. FSE 2025*. arXiv:2407.01489.
3. Ouyang, S., et al. (2025). RepoGraph: Enhancing Code Generation with Repository-Level Code Graphs. *ICLR 2025*. arXiv:2410.14684.
4. Yang, J., Jimenez, C.E., Wettig, A., et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. *NeurIPS 2024*. arXiv:2405.15793.
5. Zhang, C., et al. (2024). AutoCodeRover: Autonomous Program Improvement. *ISSTA 2024*. arXiv:2404.05427.
6. LLM fault-localization studies: arXiv:2510.20521; arXiv:2502.13966; arXiv:2509.20552.

## Reproduction

```
cd src && python3 run_benchmark.py   # emits outputs/ablation_results.csv,
                                     # outputs/eval_metrics.json, outputs/figures/*.png
```

Telemetry per task: `outputs/telemetry/*.jsonl`. Experiment log: `logs/experiments.tsv`.
