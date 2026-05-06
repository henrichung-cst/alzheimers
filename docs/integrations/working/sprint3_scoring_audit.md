# Sprint 3 — Scoring formula audit (Section C, group 1)

Per audit-plan §5 Sprint 3: justify or revert items that modify SigProb / TPDS /
PDS formulas. Sign-off gate: each item has a verdict row; reverted code lives
behind a default-OFF flag (parking-in-place pattern, consistent with the
`ENABLE_KINASE_AUGMENTATION` precedent set in Sprint 5 scoping).

## Items audited

### 1. EM promiscuity weight (`abde752` / INC-25) — REVERT

`abde752` introduced `em_promiscuity_weight = TRUE` as the default in
`Cal_SigProb` / `Cal_SigProb_animal`, applying `1/log2(1+degree)` to the
EM-Target Hill term. Sprint 1 attributed the legacy `evaluation.PDS` drift
(max\|diff\|≈0.089 vs native `93b9881`) to this commit.

**No empirical or theoretical justification is recorded** in
`docs/integrations/` or in the integration code (only a feature proposal under
`../incytr/docs/incytr_proposals/EM_TARGET_SPECIFICITY_PROPOSAL.md`).

**Verdict: revert + park behind flag.**

Two code locations apply the weight:

a. **incytr package** — `R/analysis.R::Cal_SigProb` and
   `R/factorial.R::Cal_SigProb_animal` / `run_factorial`: flag-controlled
   already, just flip defaults `TRUE` → `FALSE`. All factorial tests already
   pass `em_promiscuity_weight = FALSE` explicitly, so no test changes needed.

b. **alzheimers wrapper** —
   `code/integration/wrappers/run_incytr_factorial_all_pairs.R` line 1035–1041:
   hard-codes `em_w_vec <- em_weight_log(...)` and folds it into the third
   Hill component. Gate this with an env var (default off) so the factorial
   all-pairs scoring matches the package default.

After flipping defaults, `code/integration/tests/run_degenerate_2cond.sh`
shows `sigprob` as a clean must-match slot (already was), confirming the
revert is in effect. **`evaluation.PDS` continues to drift by ~0.089** — but
inspection of the Sprint 0 synthetic fixture reveals why: all three EM nodes
(G5, G6, G11) have `em_degree = 1` in the synthetic Layer-3 DB, so
`em_weight_log(1) = 1/log2(2) = 1` is a no-op even when the flag was on.
INC-25 has **near-zero observable effect on this fixture**.

This means **Sprint 1's PDS-drift attribution was incomplete**: the residual
0.089 PDS drift flows through `kl.pathways → Cal_PDS` from INC-28
(`6858063` memory permutation pass) — already routed to Sprint 5. The
synthetic fixture cannot empirically distinguish INC-25 from INC-28 because
the `em_degree=1` no-op masks INC-25 entirely. On real data (e.g.,
intermediate `em_degree` in the thousands for pleiotropic kinases like
Ep300), the INC-25 weight `1/log2(1+16000) ≈ 0.07` produces a substantial
SigProb shift; the revert is therefore a meaningful change for production
runs even though the synthetic test cannot show it.

**Re-attribution**: the Sprint 1 ledger entry for INC-25 ("prime suspect for
the legacy `evaluation.PDS` drift") should be downgraded to "contributor;
empirical effect on synthetic fixture is zero due to em_degree=1; revert as
parking-in-place". The dominant PDS-drift driver on this fixture is INC-28.

### 2. `logi()` k parameter & scoring weights — SIGN OFF (no drift)

`logi()` body in `R/math.R` is byte-identical to native `93b9881`:
`return(2/(1+exp(-k*x))-1)`. No drift in the function itself.

Call-site `k` parameter:

- **Legacy two-condition path** — `R/evaluation.R::Pathway_evaluation` and
  `R/analysis.R::Cal_PDS_with_Cal_TPDS` use `k_logi = 2` (native default).
- **Factorial path** — `R/factorial.R::Pathway_evaluation_factorial` uses
  `k_logi = 2 / log(2)`. This is documented (line 167–168) as a deliberate
  scale calibration: factorial OLS fits on log-FC inputs whereas native
  `logi(aFC, k=2)` consumes aFC; `k = 2/log(2)` calibrates the logistic
  steepness so the two scales agree. Routed through INC-DESIGN-3 (Sprint 1)
  and verified in `test-Pathway_evaluation_factorial.R`. Not drift.

Native scoring weights (`score.weight` in `Pathway_evaluation`) — wrapper
calls in `run_incytr.R` use either the native default or `rep(0, 6)` for the
expression-only baseline; no drift from native semantics.

### 3. `Find_highexp_gene` replacement — SIGN OFF as wrapper-config divergence

Native `Find_highexp_gene` (in `R/utils.R`, `cutoff_percentile = 0.5`) is
byte-identical to the inherited definition. The function is not modified.

The all-pairs wrappers (`run_incytr_all_pairs.R`,
`run_incytr_factorial_all_pairs.R`) **bypass** `Find_highexp_gene` by passing
explicit `gene.use_Sender` / `gene.use_Receiver` whitelists computed by the
Python adapter side, where `EXPR_DETECTION_THRESHOLD = 0.10` selects genes
detected in ≥10 % of cells (snRNA-seq detection rate).

This is a wrapper-level config decision (snRNA-seq sparsity differs from
scRNA-seq, where native's 50th-percentile-of-expressed cutoff was originally
designed). Justification: snRNA-seq nuclei have 5–10× lower UMI counts than
scRNA-seq cells; using the % cells with nonzero UMI is the standard
single-nucleus expression-admission rule. The package itself is unmodified.

New ledger row `ALZ-22` (wrapper config) records the divergence; no code
change.

## Verification gate

1. Edit `R/analysis.R` and `R/factorial.R` to flip three `em_promiscuity_weight = TRUE` defaults to `FALSE` (and update `@param` doc strings).
2. Edit `run_incytr_factorial_all_pairs.R` to gate `em_w_vec` application behind `Sys.getenv("ENABLE_EM_PROMISCUITY_WEIGHT", "0") == "1"`.
3. `bash code/integration/tests/run_degenerate_2cond.sh` — `evaluation.PDS` must move from a drift slot to a must-match slot.
4. `Rscript -e 'pkgload::load_all("../incytr"); testthat::test_dir("../incytr/tests/testthat")'` — must remain 593/593 (or higher).
5. `bash code/integration/tests/run_duckdb_enumeration_equiv.sh` — must continue to pass.
6. `bash code/integration/tests/run_verify_phase2.sh` — must continue to SKIP gracefully or pass.
