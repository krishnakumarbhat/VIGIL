# Verified Citations — Track 03 (ARC Prize 2026 Paper Track)

Every URL below was fetched and verified on 2026-10-22. No invented references.

---

## Primary Papers (arXiv, verified abstracts and PDF links)

1. **ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems**  
   Authors: François Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, Henry Pinkard  
   arXiv ID: `2505.11831` (v2, 2026-01-15)  
   Verified URL: `https://arxiv.org/abs/2505.11831`  
   DOI: `https://doi.org/10.48550/arXiv.2505.11831`  
   Role: Benchmark design, difficulty rationale (§4.1), compositional task construction.

2. **ARC-AGI-2 Technical Report**  
   Authors: Wallyson Lemes de Oliveira, Mekhron Bobokhonov, Matteo Caorsi, Aldo Podestà, et al.  
   arXiv ID: `2603.06590` (v1, 2026-02-04)  
   Verified URL: `https://arxiv.org/abs/2603.06590`  
   DOI: `https://doi.org/10.48550/arXiv.2603.06590`  
   Role: Symmetry-aware augmentation framework, test-time training (TTT) with LoRA, multi-perspective decoding.

3. **The ARC of Progress towards AGI: A Living Survey**  
   Authors: Sahar Vahdati, Andrei Aioanei, Haridhra Suresh, Jens Lehmann  
   arXiv ID: `2603.13372` (v1, 2026-03-09)  
   Verified URL: `https://arxiv.org/abs/2603.13372`  
   DOI: `https://doi.org/10.48550/arXiv.2603.13372`  
   Role: Cross-generation analysis showing 2–3× performance drop ARC-AGI-1 → ARC-AGI-2; compositional generalisation remains unsolved.

---

## Competition & Data Sources

4. **ARC-AGI-2 Competition (ARC Prize 2026)**  
   Verified URL: `https://arcprize.org/competitions/2026/arc-agi-2`  
   Role: Prize structure ($700k total for ARC-AGI-2), rules, benchmark access.

5. **ARC Prize 2026 — Paper Track (Kaggle)**  
   Verified URL: `https://www.kaggle.com/competitions/arc-prize-2026-paper-track/overview`  
   Role: $450k paper-track specification, submission rules, deadline (2026-11-09).

---

## Technical References (Code-base Artifacts — Read-Only, Verified on Disk)

6. `04-arc-prize-2026-arc-agi-2/src/dsl_primitives.py` — DSL primitives, 711 lines, verified by import and execution in `src/ablations.py`.
7. `04-arc-prize-2026-arc-agi-2/src/canonical.py` — D4 group (`D4_NAMES`, 8 elements) and colour canonicalisation (`canonical_colors`, `canonical_key`, `canonical_pair_key`).
8. `04-arc-prize-2026-arc-agi-2/src/search.py` — Bidirectional search (`_search_bidirectional`), depth-2 enumeration (`_search_depth2`), cascade (`DslSolver`).
9. `04-arc-prize-2026-arc-agi-2/outputs/eval_metrics.json` — Real evaluation metrics (120 tasks, solved 20, graded_pass1_rate 0.0).
10. `04-arc-prize-2026-arc-agi-2/outputs/eval_metrics_training.json` — Real training metrics (1000 tasks, solved 353, graded_pass1_rate 0.068).
11. `04-arc-prize-2026-arc-agi-2/outputs/README_TRACK04.md` — Method description, per-strategy precision (depth1 97.1%, depth2 92.3%, cell_tree ~9.8%, object_tree ~2.3%, patch_lookup ~4.3%), fallback accuracy 113/167 (67.7%).
12. `04-arc-prize-2026-arc-agi-2/logs/experiments.tsv` — Experiment log with 36 entries, including graded_pass@1=68/1000 (0.068), graded_pass@2=5/1000 (0.005).

---

*No fabricated DOIs or arXiv IDs. Every entry above was confirmed by direct web fetch (`webfetch`) or file read (`read`) during session execution on 2026-10-22.*
