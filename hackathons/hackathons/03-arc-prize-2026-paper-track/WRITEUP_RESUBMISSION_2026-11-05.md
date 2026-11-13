# ARC Prize 2026 Paper Track — Writeup Resubmission

**Date:** 2026-11-05 (UTC)
**Competition:** `arc-prize-2026-paper-track` (deadline 2026-11-09 23:59 UTC)
**URL:** https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/neuro-symbolic-dsl-synthesis-for-arc-agi-2

## What changed

| Field | Before | After |
|---|---|---|
| Title | Neuro-Symbolic DSL Synthesis for ARC-AGI-2 | Two ARC-AGI-2 Systems Under a 12-Hour Budget |
| Subtitle | "…solver's 0.00 public leaderboard result" | 30.56 LLM reproduction, late-submission cause, coverage framing |
| Body | System A only, claim of 0.00 | Two-system study: System A (0.00) + System B reproduction (30.56), coverage model S = C x p, protocol failure |
| PDF | `paper.pdf` (old, 0.00 only) | `paper_v2.pdf` (6 pages, both systems, measured figures) |

## Verification

- Page renders title/subtitle/body live; `30.56` appears 5x, `late submission` and `coverage` present.
- Attachments listed: `paper_v2.pdf` (new), `cover_image.png`, plus legacy `paper.pdf` which Kaggle does not expose a remove control for.
- Status: **Submitted**; editable until the deadline.

## Honest-disclosure notes kept in the text

- System B is a fork of Apache-2.0 `koushikrudra/failed-in-aimo` v1, authors credited.
- The 30.56 submission completed after the close and was excluded as late; ranked entry remains 0.00 (rank 3599/3627).
- Validation is a 12-task smoke benchmark, not an unbiased estimate (public eval may be contaminated).

## Final revision, 2026-11-06

**Live title:** When Training Fit Is Not Reasoning: An Executable ARC-AGI-2 Audit
**Paper:** `outputs/paper_final.pdf` (5 pages) — canonicalisation counterexamples + proof + 34,992 exhaustive assertions + 5,000 seeded scheduling simulations.
**Attachments:** `paper_final.pdf`, `arc_audit_final.zip` (13 competition-only files, 149 KB, re-extracted and re-run cleanly), `fig09_audit_sensitivity.png`. `paper_v2.pdf` removed; legacy `paper.pdf` is not removable by Kaggle.
**Status:** Submitted, editable until 2026-11-09 23:59 UTC.
**Honesty deltas vs prior version:** withdrawn "16.7% evaluation grids solved" and inferred hidden coverage/top-200 task-count claims; metric field clarified as candidate discovery; simulation labelled conditional, not confidence bounds.
**Budget guard:** `copilot-budget.json` set to soft $2.20 / hard $2.50 with providers github-copilot, opencode, openrouter, google. Requires OpenCode restart to reload; free-model sessions report $0 cost so the guard cannot trip on them.
