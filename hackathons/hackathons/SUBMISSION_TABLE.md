# Kaggle submission status — all entered competitions (2026-10-31)

**Updated follow-up:** see [`SUBMISSION_FOLLOWUP_2026-11-01.md`](SUBMISSION_FOLLOWUP_2026-11-01.md)
for the requested ARC / Gemma / NFL / Name That LLM / OpenZee outcomes, including
the writeups and Nov 1 public score/rank.

Account: `krishnakumarbhat` (team `krishnak`). Source: Kaggle API
`competitions/submissions/list` + downloaded full public leaderboards.

## Open competitions — what can still take a submission

| Competition | Deadline | Type | CSV upload? | Subs made | Best public | Public rank |
|---|---|---|---|---|---|---|
| playground-series-s6e10 | 2026-10-31 23:59 | Playground, ROC-AUC | yes | 15 | **0.96227** | **19 / 3654** (quota exhausted; no more slots before 23:59 UTC) |
| arc-prize-2026-arc-agi-2 | 2026-11-02 23:59 | code kernel only, 1/day | no | **2**: Oct 31 v2 DSL kernel `krishnakumarbhat/arc-agi2-dsl-submit` (0.00); **Nov 2** NVARC+Qwen3-4B TTT/TTA kernel `krishnakumarbhat/arc-nvarc-validated` v3, ref `57901718`, PENDING | **0.00** (best registered; `57901718` awaiting score) | **3586 / 3600** (2026-11-02 07:14 board) |
| nfl-big-data-bowl-2027 | 2027-01-06 23:59 | judged notebook (no metric) | n/a | 0 | – | – |
| name-that-llm | 2027-01-14 23:59 | code kernel only, 5/day | no | **2** (`ntl-tfidf-only` v2 public 0.308 rank 3770/4013; `ntl-deberta-offline` v1 completed 0.0556 due to encoder mismatch; `ntl-recipe-offline` v2 pushed — targets full blend ~0.67 grouped CV; dataset upload complete but index pending) | **0.308** | **3770 / 4013** |
| gemma-4-developer-agent | 2026-12-02 23:59 | hackathon judging (YAML agent + inference server) | n/a | 0 | – | – |
| arc-prize-2026-paper-track | 2026-11-09 23:59 | paper submission | n/a | 0 | – | – |
| gemma-4-developer-agent-paper | 2026-11-12 23:59 | paper submission | n/a | 0 | – | – |

## Closed competitions — submission window already shut

| Competition | Closed on | Subs made | Best score | Rank |
|---|---|---|---|---|
| google-code-golf-2025 | 2025-10-30 | 4 | 912493 | **383 / 1142** |
| maze-crawler | 2026-06-30 | 1 | 600 (unscored) | not ranked |
| nvidia-nemotron-model-reasoning | 2026-06-15 | 1 | unscored | not ranked |
| playground-series-s5e10 | 2025-10-31 | 0 | – | no entry |
| make-data-count-finding-data-references | 2025-09-09 | 0 | – | no entry |
| MABe-mouse-behavior-detection | 2025-12-15 | 0 | – | no entry |
| 2025-uml-501-lab-3q14 | 2025-08-29 | 0 | – | no entry |
| ai-mathematical-olympiad-progress-prize-3 | 2026-04-15 | 0 | – | no entry |
| recodai-luc-scientific-image-forgery | 2026-04-22 | 0 | – | no entry |
| rogii-wellbore-geology-prediction | 2026-08-05 | 0 | – | no entry |

## Why some open competitions still have no submission

- **arc-prize-2026-arc-agi-2** — `isKernelsSubmissionsOnly = true`, 1 submission/day,
  0/1 used. The existing solver kernel `arc-agi2-llm-ttc` exits with
  `KernelWorkerStatus.ERROR`, and a file upload is rejected (`BlobFileTokens must be
  specified`). A working rewrite is required before the 2026-11-02 deadline.
- **nfl-big-data-bowl-2027** — no `evaluationMetric` and no `sample_submission`; the
  competition is judged on a notebook + writeup, so there is no leaderboard rank to gain
  from a CSV.
- **gemma-4 / paper tracks** — judged submissions (writeup/paper), not prediction files.

## Submissions made (numeric leaderboard)

- **s6e10** — 15 submissions, best public 0.96227, rank 19/3654 (quota exhausted; no slots before 23:59 UTC).
- **google-code-golf-2025** — 4 submissions, score 912493, rank 383/1142 (closed 2025-10-30).
- **arc-prize-2026-arc-agi-2** — 2 kernel submissions: v2 DSL/beam/object solver (graded 0.0, public 0.00) and the Nov 2 NVARC+Qwen3-4B reproduction (ref `57901718`, pending); deadline 2026-11-02 23:59 UTC.
- **name-that-llm** — 2 submissions: `ntl-tfidf-only` v2 (public 0.308, rank 3770/4013, 1 of 5/day used); `ntl-deberta-offline` v1 (OOF BA 0.0556, encoder feature space mismatched); `ntl-recipe-offline` v2 pushed (targets blended ~0.67 grouped CV, dataset `ntl-deploy-artifacts` upload finished, index pending).
- **maze-crawler** — 1 unscored submission; **nvidia-nemotron** — 1 unscored submission.

## Why some competitions still have no submission

- **nfl-big-data-bowl-2027** — no `evaluationMetric` and no `sample_submission`; judged on notebook + writeup only.
- **gemma-4-developer-agent / paper-track / paper** — judged/paper format; `gemma-4` requires a `submission.zip` of `agent.yaml` + ADK agent config (not a CSV), which is a separate engineering track from prediction-file competitions.
- **s6e10** — quota fully used (10/10); no more slots before 23:59 UTC deadline.
