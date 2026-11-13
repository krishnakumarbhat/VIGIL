# Requested project submissions — 2026-11-01

Account: `krishnakumarbhat` / team `krishnak`. This is the Nov 1 follow-up to
`SUBMISSION_TABLE.md` (the latter records the Oct 31 snapshot).

| Project | Outcome | Score / rank | What was tested / submitted |
|---|---|---|---|
| **04 ARC Prize ARC-AGI-2** | One code-kernel entry already submitted Oct 31; **no second entry today** | **0.00, rank 3552/3554** | Submission schema validated for all 240 test tasks. Public eval had 0/120 graded pass@1, so no evidence supported spending another 1/day slot before the Nov 2 deadline. |
| **03 ARC Prize Paper Track** | **Writeup submitted** | Judged; no numeric public rank | [Writeup](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/neuro-symbolic-dsl-synthesis-for-arc-agi-2) includes the compiled PDF and cover image. Updated to disclose the real 0.00 main-track result; 1,643 words. |
| **02 Gemma 4 Developer Agent Paper Track** | **Writeup submitted** | Judged; no numeric public rank | [Writeup](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/gg-spv-graph-guided-localization-and-speculative) includes the paper PDF and links the companion agent kernel. The paper explicitly labels patch-generation figures as simulated. 1,809 words. |
| **02 Gemma 4 Developer Agent (main track)** | One agent package submitted; scoring pending | Pending | [Public builder notebook](https://www.kaggle.com/code/krishnakumarbhat/gemma-4-gg-spv-agent-submission); Kaggle submission ref `57870300`. The emitted `submission.zip` was checked for root `agent.yaml`, matching Gemma 4 31B adapter/base config, safe paths/extensions, valid ZIP, and <3 GiB size. |
| **01 NFL Big Data Bowl 2027** | **Open Track Writeup submitted** with public notebook and 3 gallery figures | Judged; no numeric public rank | [Writeup](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/writeups/from-combine-cuts-to-game-performance-predicting) · [public notebook](https://www.kaggle.com/code/krishnakumarbhat/nfl-bdb-2027-combine-kce-analysis). Writeup is 1,824 words. Kernel completed; it checked 431,094 player rows, confirmed the shipped `a` is never negative, and re-derived signed acceleration on 100 real sessions (32.77% negative; 109 cuts). Full model retraining is opt-in; the Writeup accurately reports stored five-fold OOF results. |
| **05 Name That LLM** | New best-of-known stack submitted Nov 1 | **0.375, rank 3067/4156** | Offline kernel v4 passed file/ID/class validation. Reduced member blend measured **0.6702 grouped-CV balanced accuracy** before submission. Previous TF-IDF-only result was 0.308 (3770/4013 at its time). |
| **08 OpenZee OpenAir 2026** | **Not submitted** | — | Recon found no verifiable competition host, rules, dataset, or submission endpoint; no safe/valid entry can be made. |

## Privacy and remaining limitations

- The public NFL source dataset `krishnakumarbhat/nfl-bdb-2027-kce-source-public`
  contains analysis code, aggregate metrics, and figures only; raw competition
  tracking data was not repackaged or uploaded.
- The Gemma agent ZIP contains configuration, prompts, and the competition-provided
  small adapter; it contains no user personal data. Its actual hidden-task score
  is still pending, and its inference behavior was not independently run outside
  Kaggle's evaluator.
- ARC's paper entry links the real main-track kernel and candidly reports its
  0.00 score. The source kernel remains private; publishing the solver source
  would be a separate visibility/licensing decision.
- ARC-AGI-2 was not resubmitted on Nov 1 because verified public validation was
  still 0/120; changing random fallback guesses would waste the one daily slot.

---

## Addendum — 2026-11-02 (ARC-AGI-2 final-day submission)

| Project | Outcome | Score / rank | What was tested / submitted |
|---|---|---|---|
| **04 ARC Prize ARC-AGI-2** | **Second entry submitted** on the final day; rerun in progress | Pending (previous best 0.00, rank 3586/3600) | Kernel `krishnakumarbhat/arc-nvarc-validated` **v3**, ref `57901718`, submitted 2026-11-02 08:28 UTC. It is a faithful reproduction of the Apache-2.0 public notebook `koushikrudra/failed-in-aimo` **version 1** (public LB 33.89), which uses a Qwen3-4B ARC-SFT model with per-task test-time training, permutation/rotation augmentation, beam/DFS decoding and a 1302s per-task cap on 4×L4. |

### Evidence collected before spending the single daily slot

- Competition rule: **1 submission/day**; today's allowance is now exhausted (0 remaining).
- The 12-task offline smoke benchmark (fixed seed 20261102, solutions read only after prediction)
  produced **3/18 test rows = 16.7%** and **2/11 completed tasks (18.2%)** task-normalized pass@2.
  Small sample: the 95% interval easily contains 33.9%, so this neither confirms nor refutes the
  source notebook's public score.
- Output validity was enforced in-kernel and re-checked on download: all 12 benchmark tasks present,
  zero malformed grids, rectangular 0–9 integer grids only.
- One benchmark task never produced a prediction bundle, and one hit the 1302s per-task timeout;
  measured task times were 284–1302s, which projects roughly 190–220 of 240 hidden tasks covered
  inside the 11h50m solver budget.

### Deviations from the 33.89 source, stated plainly

- The model mount path was rewritten to discover `/kaggle/input/models/...` (the original hard-coded
  path no longer exists); the weights, training hyper-parameters, candidate generation, aggregation
  and decoding logic are unchanged.
- The shell launch was replaced by a checked `subprocess.run` so worker crashes fail fast.
- A preflight GPU-count check, a deterministic validation queue, and posthoc scoring were added.
- Kaggle's CLI 1.7.4.5 silently ignores GPU shape/pinned image; pushes used Kaggle CLI **2.2.5**.

### Honest expectation

The submission is a reproduction, not an improvement on the source. A top-200 finish (cutoff
**33.06**; rank 100 = 40.56) requires the hidden-set behaviour to match the original notebook's
33.89, and our coverage is somewhat lower than a full-sweep run. A plausible range is roughly
15–35; anything above 33.06 would be top 200. The score will only be known after the ~12h rerun.

