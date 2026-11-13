# Ideas frontier

- I1 Track 05: replace hand stylometry with a learned linear probe over character 3-6 grams with
  NB log-count ratios; add external corpus as training-only augmentation; calibrate per-class priors.
  -> DONE (2026-10-22): cross-fitted LR(C=8, balanced) meta-stacker over OOF probs of
  svc_extra_claude_w50 + nbsvm_all + new svc35_w0 char3-5/word1-3 diversity member.
  Grouped CV bal_acc 0.39061 -> **0.40863** (KEEP). Rejected: kNN-TFIDF vote features
  (0.365), margin/entropy stack features (0.383), rank-mean blends (0.379).
  Leg 2 (n9b, DISCARD, same day): length-bin conditional prior offsets
  (LO-prompt shrunk tables + bin one-hot into the LR stacker) measured
  0.51191 at C=32 but OVERTURNED by the decoupled-table forensic test
  (real 0.173 ~= permuted 0.193 vs 0.4008; same-fold LO/full table-coupling
  artifact; zero bin-prior information). Baseline 0.40863 stands.
  Next: feed the running Kaggle DeBERTa stage-2 OOF into the same meta stack.
- I2 Track 05: ensemble Stage-1 (LinearSVC+NB-SVM) with DeBERTa (Kaggle) and Qwen LoRA (Colab) logits,
  temperature-scaled; expect >0.46 balanced accuracy.
- I5 Track 01 -> REJECTED (2026-10-22): combine<->game translation-gap deltas +
  within-athlete combine session-variability features do NOT robustly move the
  headline. Identical-folds within-run: by_game_play 0.2847 -> 0.2825 (both
  families), sx_only -0.0041, translate_only -0.0058, by_player +0.0034 only.
  Harness baseline drifts +-0.003 between runs -> keep/reject must be decided
  within-run. Next (I5a): per-route-ran conditional residual model.
- I3 Track 04: LLM test-time compute over ARC tasks (sample k programs, self-verify) on Kaggle T4 with
  an instruction model; combine with DSL prior for candidate generation.
- I4 Track 04: object-centric induction (per-object transformation + position grammar) instead of cell rules.
- I5 Track 01: link combine KCE/DBR to rookie separation via player-level mixed model with game-level
  random effects; add position-specific normalisation and route efficiency features.
- I6 Track 06: add TabPFN-style nearest-neighbour prior + RealMLP, stack with rank averaging.
- I7 Track 07: run the agent on Graviton via QEMU emulation to produce COOL arm64 evidence locally.
- I8 Track 02: ship the agent as a Kaggle dataset + harness so judges can reproduce (Best New Resource prize).
- I6 Track 06 leg-1 (2026-10-22): nnprior injection REJECTED (0.959242 best vs 0.959265 baseline,
  18 strict cross-fitted LR stack configs incl. rank features all <= baseline). kNN k=5 inv-dist
  member alone 0.943 — too weak at this operating point. RealMLP-TD 10-fold GPU leg still running
  (started 07:18Z); strict 5-member LR re-stack of whichever members bank, before keep.
- I6 Track 06 leg-2 (2026-10-22): RealMLP 10-fold leg banked (OOF 0.959001) and STRICTLY
  re-verified via cross-fitted coordinate ascent (src/stack_strict_v2.py): OOF 0.959622
  vs kernel-v4 baseline 0.959272 (KEEP). In-sample-vs-strict optimism Delta only
  +0.000006 AUC; hand-set weights (0.959354) dominated by fitted ones. nnprior as 5th
  member hurts at every C (0.959591) -> strictly rejected twice now. Test-side refreshed
  with strict mean weights {lgbm .0885, cat .2695, xgb .0071, realmlp .6350}.
  NEXT: replacement for the dead logreg+TE member; RealMLP diversity members
  (different seed/epochs, fold-ensembled test scores); target 0.9613.
