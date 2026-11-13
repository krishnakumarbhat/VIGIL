# Worklog - Multi-Track Marathon

## 2026-10-21 ~23:40 UTC - bootstrap
- Verified Kaggle API (krishnakumarbhat), Colab old+pro grants, local RTX 3050, Playwright+Chrome.
- Created 8 track folders under `/home/pope/Desktop/gpu /hackathons/`.
- Patched a DNS wait-loop into a private copy of run_gpu.sh (`_gpu_probe/run_gpu.sh`) because
  `*.prod.colab.dev` publishes its A record 10-45s after `colab new` -> NameResolutionError.

## 2026-10-21 ~00:20 UTC - data unblock
- Kaggle CLI download returned 403 (rules not accepted) for 5 competitions.
- Discovered competition data IS mountable inside Kaggle kernels via competition_sources
  (NFL 2027 extracted: combine_tracking 84MB, game_tracking_2023 335MB, player_play 147MB).
- Kaggle API endpoints for join/accept-rules are not available in CLI 1.7.4.5.
- Playwright + system Chrome 154 (agent-browser npm symlink is broken) -> rules accepted for all 6
  competitions; download-all now returns HTTP 302.

## 2026-10-21 ~00:40-01:40 UTC - parallel track execution
- T04 ARC solver: DSL + D4/colour canonicalisation + bidirectional search; train pass@1 6.8%, eval 0/120.
- T07 OpenCV: See-Reason-Act-Verify agent, AWS suite, 55 tests, ROC-AUC 0.9917, 35 FPS.
- T01 NFL: KCE/DBR on real data, 2,652 cuts (915 INSIDE / 1,737 OUTSIDE), OOF R2 0.0355.
- T02 GG-SPV: recall@5 0.318 vs flat 0.273, tokens -43%, verifier AUROC 0.875/0.905.
- T03 paper: 1330-word writeup + cover image + 6 figures + 5 verified citations.
- T05 stage1: grouped CV balanced accuracy 0.3906.

## GPU lane status
- Kaggle: ntl-stage2-deberta (deberta-v3-base 5-fold) running; ps6e10-stack v4 queued (2-GPU-slot limit).
- Colab Pro: Qwen2.5-3B LoRA (bf16, no bitsandbytes - bnb/CUDA13 mismatch on Colab), 5.2h budget.
- Local: CPU-heavy feature work; RTX 3050 for small neural probes.

## 2026-10-22 ~06:20 UTC - iteration: T05 n0 -> n9 (stack_probs_c8) [KEEP]
- Lanes at start: Kaggle 2 GPU slots RUNNING (ntl-stage2-deberta, ps6e10-stack); Colab
  idle/none active. Chose CPU-lane node n0 (`I1 NB-SVM+probe stack`) on Track 05.
- Implemented `src/stack_probe.py` + `src/stack_addendum.py` + `src/stack_model.py` +
  `src/stack_finalize.py`. Rebuilt the cleared feature cache (`all` + new `all35` views)
  and member OOFs (svc_extra_claude_w50, nbsvm_all, svc35_w0).
- Measurements (SGKF5 groups=prompt_id, clean subset n=9120):
  - svc35_w0 alone 0.330 (fold log ~0.332-0.352); 3-member simplex blend best 0.39062 (tie)
  - LR cross-fitted stacker (probs only, class_weight=balanced): C sweep
    0.5/1/2/3/4/8/16/32/1000 -> 0.3962/0.4021/0.4034/0.4062/0.4063/**0.40863**/0.40858/0.40834/0.4013
  - REJECTED: LGBM stack (0.383), LR+kNN vote feats (0.365), LR+margins/knn (0.345),
    rank-mean simplex blends (<0.379), logit-adjustment alpha>0 (hurts at every C).
- KEEP: `stack_probs_c8` = 0.40863 (+0.018 over 0.39061). Deliverables refit:
  `outputs/final_model_stacked.joblib`, submission regenerated, recon sanity
  0.167 -> 0.1867, `cv_stack_report.json`/`final_model_config.json`/notes updated.
- Signal: prob-level ensembling of 3 diverse linear members beats any hand-blend;
  probe next: inject the DeBERTa stage-2 member, then a 4-member iteration (n9).

## 2026-10-22 ~08:05-08:15 UTC - iteration: T06 n5a (nnprior_inject_lr) [DISCARD], n5b rank-stack [DISCARD]
- Lanes at start: Kaggle GPU x2 RUNNING (ntl-stage2-deberta, ps6e10-stack); Colab: no active
  session; local RTX 3050 already carrying the PREVIOUS iteration's `realmlp_stack.py` job
  (PID 3167832, started ~07:18Z, stdout /tmp/opencode/t06full2.log) -> took the complementary CPU lane.
- Node: n5 CPU leg (I6): inject banked oof_nnprior.npy (kNN k=5 inv-dist, fp16 cdist,
  leakage-safe fold-safe references) as 4th member into cross-fitted LR stacker.
- Implemented `src/stack_nn_lr.py` (strict OOF stack weights via the banked members'
  StratifiedKFold(10,42); math numerically verified per equations.md: clip is rank-preserving;
  equal-weight prob-vs-logit pooling AUC differs (synthetic +0.0051); unequal-weight pool scale
  gap up to 0.033 AUC).
- Measurements (StratifiedKFold10 rs42, n=699,636):
  - members: lgbm 0.957583 / cat 0.959046 / xgb 0.956385 / nnprior 0.943050 / baseline blend 0.959265
  - lr_logits C{.25..1e6}: all 0.959235; lr_logits+probs: best 0.959242 (C=0.5)
  - lr_rank (per-col global ranks): 0.958444..0.958447
  - 4-member coord-ascent reference: 0.959263 = cat-dominant, drops nnprior
- DISCARD: every stack config <= baseline 0.959265. nnprior member is too weak (0.943) to help at
  this operating point; T06 baseline stands at 0.959265.
- Literature (worker_1 ses_e9e8c3c4): kNN-as-tabular (TabR, ICLR-2025 kNN-tabular) = textbook;
  wiring-level twist only, no novelty claim attached to this reject.
- HANDOFF: `realmlp_stack.py` (n5 GPU leg) still running; when it completes (~09:30Z):
  artifacts `outputs/oof_realmlp.npy`, `oof_blend_v2.npy`, `cv_scores.json`, + its own TSV row.
  NOTE: its final stack uses IN-SAMPLE coord-ascent weights -> next iteration should re-verify the
  5-member stack strictly (cross-fitted LR, same protocol as stack_nn_lr.py) before est. keep.

## 2026-10-22 ~08:15-08:55 UTC - iteration: T05 n9b leg-2 (lenprior_binartifact) [DISCARED/OVERTURNED]
- Lanes at start: Kaggle GPU x2 RUNNING (ntl-stage2-deberta, ps6e10-stack); Colab RUNNING
  (Track-05 Qwen2.5-3B LoRA staging, auto-started by the run_track05_colab supervisor);
  local RTX 3050 carrying realmlp_stack.py (T06 n5 leg, finished ~08:55 with blend_v2
  0.959628 KEEP per its own stdout) -> took the CPU lane on Track 05 again.
- Node: n9 leg-2 "prompt-conditional length priors" -> implemented as length-bin
  conditional class-prior offsets (Dirichlet-shrunk log-ratio tables + bin one-hot)
  appended to the 54-dim 3-member OOF-prob stacker. Pre-registered audits: literature
  (worker_1 ses_e9e7c65a1ffe: wiring-level variant of MRP/multicalibration/label-shift,
  no direct prior art) + protocol audit (worker_2 ses_e9e7c65a0ffe: fold-leakage traps +
  in-session anchor re-run demand).
- Implementation `src/stack_lenpriors.py`. Anchor re-run reproduced the recorded C=8
  stack EXACTLY (0.40863, drift 0.0). Full grid kept in cv_lenprior_report.json.
- RED FLAG sequence (all in src/stack_lenpriors_{check,drill,drill2,final,verdict}.py):
  1. V1 best c32_m10 = 0.51191 (+0.103) - monotone to 0.60417 at C=1000, fold BAs
     [0.547,0.551,0.542,0.527,0.393] - fold-4 anomaly.
  2. Negative control (bin permutation) did NOT collapse (0.50889/0.51691).
  3. Gaussian-noise extra blocks ~ anchor (0.3967/0.3949) so not a generic-geometry
     channel; one-hot block alone ~ nothing; analytic post-hoc alpha<=3 ~ nothing.
  4. r-block-only: real 0.49992 ~= permutted 0.48836; dies at m=1e6 (0.40815).
  5. Prompt-split tune(255p)/est(255p) single-peek still +0.099 on EST rows.
  6. FULLY DECOUPLED tables (val tables from fold half A, stacker fit on half B):
     real 0.17321 / perm 0.19333 @C8; 0.10525/0.11173 @C32 vs 0.40084/0.40081 ->
     catastrophic collapse, real~=perm. VERDICT: same-fold LO/full table-coupling
     channel artifact; zero deployable length-prior signal. DISCARD. Baseline
     0.40863 STANDS.
- Note: SGKF fold index arrays have long same-class runs -> stride halving (0::2) is
  NOT a random half (class 7 was absent from tiA); always seed-shuffle before halving.
- Artifacts: outputs/cv_lenprior_report.json (full forensic pack);
  oof_stack_lenprior.npy DELETED; corrective rows in logs/experiments.tsv;
  jsonl discard entry + graph n11/edge n9->n11; ideas.md I1 block updated;
  equations.md gained the COUPLING TEST protocol.
- HANDOFF (next iteration, high priority, CPU lane): T06 n5b strict re-verify of the
  5-member stack incl. oof_realmlp.npy (realmlp_stack.py finished at ~08:55Z with
  blend_v2 0.959628, in-sample coord-ascent weights) using the cross-fitted LR protocol
  of stack_nn_lr.py; and T05 n9 primary leg when ntl-stage2-deberta lands (inject
  MiniLM/DeBERTa OOF probs into the C=8 stack, THEN re-apply the coupling test to any
  new table-derived blocks).

## 2026-10-22 ~08:30 UTC - iteration: T01 n4 translation-gap family (REJECT) + T06 kernel repair (KEEP infra)
- Lanes at start: Kaggle both GPU slots (ntl-stage2-deberta RUNNING; ps6e10-stack ERROR);
  Colab idle; local GPU idle -> chose CPU node n4 (Track 01 NFL, I5 extension).
- Implemented `src/translation_features.py` (combine<->game translation deltas +
  within-athlete combine session variability; 39 features; defender-relative
  kinematics NOT buildable - game_tracking samples 1.6 athletes/play, documented reject)
  and `src/run_translation_eval.py` + `src/run_translation_attribution.py`.
- Measured (identical GroupKFold folds, same harness): first run headline
  0.2820 -> 0.2825 (+0.0005) suggested KEEP, but attribution revealed the
  nominal baseline drifts 0.2820 -> 0.2847 across processes. Corrected
  within-run control: baseline 0.2847 vs both-families 0.2825 -> REJECT.
  by_player +0.0034 only; sx_only -0.0041; translate_only -0.0058. First
  attribution attempt was invalid (numeric-only filter silently dropped
  categorical route/formation/alignment dummies) -> corrected and re-run.
- Protocol rule added to equations.md: Track 01 keep/reject decided within-run.
- Artifacts: outputs/translation_metrics.json, outputs/translation_attribution.json,
  src/translation_features.py (kept for reference, not in delivery path).
- T06 n5: ps6e10-stack kernel v4 diagnosed (str category columns reaching LR
  SimpleImputer) and relaunched with the fixed repo version (kernels push, quota OK:
  5.11h/45h). v4 COMPLETE: FINAL OOF AUC 0.959272 (flat vs 0.959265); logreg+TE
  member is dead weight (w=0, 0.9406) -> replace it for a real metric move.
  Test/OOF npy artifacts copied to 06-.../outputs.
- n4a (per-route-ran conditional residual model) added as the next T01 frontier node.

## 2026-10-22 ~09:15-10:05 UTC - iteration: T06 n5->n11 strict re-stack [KEEP]
- Lanes at start: local GPU idle (verbalinux only), Kaggle ntl-stage2-deberta RUNNING /
  ps6e10-stack COMPLETE, Colab track05 Qwen job RUNNING. A CONCURRENT loop iteration was
  live (fixed cat2 dtype bug, relaunched ps6e10-stack kernel v4 -> fresh OOF baseline
  0.959272 and test_*.npy members) - my iteration took the complementary strict-verify leg.
- Node: n5 closing leg (I6): STRICT cross-fitted re-verification of realmlp_stack.py's
  in-sample coord-ascent blend (0.959629), per handoff instruction.
- Implemented 06-.../src/stack_strict_v2.py: meta-weights refit per outer fold of the SAME
  StratifiedKFold(10,42) panel; 24 LR configs (logits4 / logits+probs4 / logits5 with
  nnprior, C sweep 0.25..1000) + strict coordinate ascent (weights fit on meta-train rows).
- Measurements (n=699,635):
  - in-sample coord ascent (reference): 0.959629 weights {lgbm .0795, cat .2727, xgb .0114, realmlp .6364}
  - strict LR logits4: 0.959601-0.959602 flat across C; logits+probs4: 0.959605-0.959607
  - strict LR logits5 (+nnprior): 0.959591 at EVERY C -> nnprior strictly hurts (2nd rejection)
  - BEST strict coord ascent4: 0.959622 mean weights {lgbm .0885, cat .2695, xgb .0071, realmlp .6350}
  - stack-weight optimism Delta = in-sample - strict = +0.000006 AUC (NOT overfitting at n~630k)
- KEEP: ps6e10_roc_auc 0.959622 vs kernel-v4 baseline 0.959272 (+0.00035); vs the live
  iteration's hand weights (0.959354) +0.00027. Test-side refreshed with strict mean
  weights over banked kernel-v4 test logits: submission.csv (mean 0.4471 vs prevalence
  0.4436 - sane), cv_scores.json (strict fields merged), test_blend_prob_strict.npy.
- Novelty check (worker_1): OOF LR meta-stacker textbook (Wolpert92 / Ting&Witten99 /
  SuperLearner07); Purucker&Beel23 documents validation-weight-search overfit; found NO
  published Delta-optimism quantification for greedy coord-ascent on OOF AUC at fixed n -
  our +0.000006 datapoint is logged in equations.md. No novelty claim; measurement node.
- Artifacts: outputs/cv_stack_strict.json, outputs/submission.csv, outputs/cv_scores.json,
  outputs/test_blend_prob_strict.npy, /tmp/opencode/t06_strict.log, experiments.tsv row.
- NEXT (n11): replace dead logreg+TE member; RealMLP diversity members; target 0.9613.
