# Equations

## Velocity cross-product (Track 01)
cross_t = v_pre_x * v_post_y - v_pre_y * v_post_x
turn_angle = atan2(|cross_t|, dot(v_pre, v_post))
break_class = INSIDE if cross_t > 0 else OUTSIDE   # only when turn_angle > 35 deg

## Kinematic Cut Efficiency (KCE)
KCE = ||p_end - p_start||_2 / (sum_i ||p_{i+1} - p_i||_2 + eps)

## Deceleration-to-Burst Ratio (DBR)
DBR = max_t ||a_t||_decel / (max_t ||a_t||_burst + eps)

## Balanced accuracy (Track 05)
BA = (1/K) sum_k recall_k,  recall_k = TP_k / (TP_k + FN_k)

## ROC-AUC (Track 06)
AUC = P(s(x+) > s(x-)) over all positive/negative pairs.

## Dihedral canonicalisation (Track 04)
key(g) = min over d in D4, pi in S_colours of  hash(d(pi(g)))
Search-space reduction factor = 8 * |S_colours| = 16x for 2-colour cases.

## GroupKFold reproducibility floor (Track 01, MEASURED 2026-10-22)
Same-harness same-folds runs of the nominal baseline (sep_at_pass_forward,
lightgbm, by_game_play) drift by +-0.003 R2 across processes (0.2820 vs
0.2847); by_player reproduced exactly (0.2625). Therefore keep/reject on
Track 01 must always be decided against a within-run control, not against a
stored baseline value: keep iff aug > control in the SAME run.

## Stack-weight optimism (Track 06 strict-fold rule)
Every stacked-blend measurement must report
    Delta_optimism = AUC(weights fitted in-sample) - AUC(weights fitted out-of-fold)
where out-of-fold means the meta-learner itself is refit per outer fold and its
predictions concatenated (rolling cross-fit on the SAME StratifiedKFold(10,42)
panel as the members; each row's member score never involved that row).
Measured 2026-10-22 (n=699,635, 4 members, greedy coordinate ascent):
Delta_optimism = 0.959629 - 0.959622 = +0.000006 AUC -> in-sample greedy
weights were NOT overfitting at this n; hand-set weights (0.15/0.55/0.08/0.02/0.20)
measured 0.959354, i.e. dominated by fitted weights by -0.00027 AUC.
Deploy only weights whose OOF refit protocol is logged; hand weights never staged.

## Score-scale invariance / pooling scale (Track 06 stacks)
AUC(s) = P(s(x+) > s(x-)) is a function of score ORDERING only, hence
AUC(g(s)) = AUC(s) for any strictly increasing g  (e.g. prob->logit clip @1e-6; verified numerically).
Equal-weight mean on prob scale != equal-weight mean on logit scale (different orderings;
synthetic check +0.0051 AUC), and for unequal weights the gap can reach ~0.033 AUC
-> both pooling scales (and per-column rank features) are distinct stack hypotheses,
each needing its own OOF measurement.

## Stacker feature-block coupling test (Track 05, mandatory gate, added 2026-10-22)
Let T_f = fold-f train label count table over (bin b, class c) and r[b,c] its
shrunk log-prior-ratio. Coupled regime: fit rows' r-features come from LO-tables
(T_f minus own prompt) while validation rows get r from T_f (full train fold).
Because LO(T_f) and T_f share ~all rows, any fold-level noise vector in T_f is
reproduced on BOTH sides of the fit/predict boundary; LR(C>=16) then fits
per-bin offsets that improve the stacker's own validation rows by up to +0.10 BA
while carrying no generalizable label information.

Gate (decoupled-table test): recompute validation tables from an independent
row half A of T_f, fit the stacker only on half B with LO-within-B tables.
If the measured delta survives (real >> permuted control) the block is real;
if delta collapses or inverts, the value is an artifact and the block is void.
Measured 2026-10-22 for the length-bin block: coupled +0.103; decoupled
real 0.173 / permuted 0.193 vs 0.401 anchor -> VOID (discarded).

Corollary: stratified fold index arrays can carry long same-class runs; any
"honest half" of a fold must be a seeded RANDOM permutation split, not a stride.
