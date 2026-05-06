# Incytr pre-audit divergence report

Comparison of slot-by-slot output between **upstream-native** Incytr at commit `93b9881`
(tree-identical to current `upstream/master` HEAD `2a94051`) and **current `incytr-audit` HEAD**,
both run on the synthetic harness defined in `tests/testthat/helper-golden.R`
(seed=42; 12 genes × 40 cells; 2 cell types × 2 conditions).

Source script: `tests/testthat/compare_golden_outputs.R`. Native run used a
parallel `helper-golden.R` adapted for native API (explicit `gene.use_Sender`/
`gene.use_Receiver`, `slot_or_null` for slots not present in native).

## Summary

| Slot | Status |
| --- | --- |
| `expr_bygroup` | structural diff (list inequality, both length 2) |
| `sigprob` | **match** |
| `evaluation` | dim 3×7 → 3×10; new cols `Ack_score`, `KGG_score`, `Rme1_score`; `PDS` max\|diff\|=8.87e-02 |
| `kl_pathways` | `SiK_*_EI_*` non-numeric inequality (6 cols); `SiK_score_*` max\|diff\|≈1.78e-01 |
| `kl_evidence` | added in current HEAD (absent in native) |
| `EI` | structural diff (list inequality, both length 2) |
| `sc_FC` | **match** |
| `p_value` | **match** |
| `pr_FC` | **match** |
| `ps_FC` | **match** |
| `py_FC` | **match** |
| `Ack_FC` | added in current HEAD |
| `KGG_FC` | added in current HEAD |
| `Rme1_FC` | added in current HEAD |

## Per-slot detail

### `expr_bygroup` — structural diff
List of length 2 in both, but element-level inequality (likely column ordering or
factor-level changes between native and refactored `Expr_bygroup`). To resolve in
Sprint 1: dump both lists, diff column names + factor levels per condition.

Candidate commits (from `incytr_audit_commit_list.md`):
- `4a9423c` phase 0.4: relocate expression and EM helpers
- `c4698ef` phase 0.2: generalize integrate_omics
- `719c2b1` phase 1.1: generalize barcodes_bycondition

### `sigprob` — match
Native and current produce bitwise-identical `SigProb` data frames on the synthetic
two-condition input. This anchors Sprint 1: the per-animal/`Contrast_SigProb`
extension is invoked only when `mode == "factorial"`; the legacy two-condition path
is preserved without numerical drift.

### `evaluation` — added scoring columns + non-zero `PDS` drift
Three new columns (`Ack_score`, `KGG_score`, `Rme1_score`) — corresponds to the
`Ack`/`KGG`/`Rme1` PTM tracks and pY parallel work. `PDS` differs by up to
~0.089, which under a `[0,1]`-bounded score is material.

Candidate commits:
- `e09105a` phase 3: generalize TPDS/PDS pipeline
- `c8e764e` phase 1.6: add N-condition integration
- `abde752` add EM promiscuity weighting and edge … _(likely the PDS drift driver)_
- `c3580fc` apply cutoff to all omics slots

### `kl_pathways` — `SiK_*` columns drifted
Both `SiK_score_condA/B` differ by ~0.18 (large fraction of native magnitude), and
six `SiK_*_EI_*` columns are non-numeric-inequal — indicating type or category
changes. Most likely driven by the kinase-evidence refactor in
`build_kinase_evidence` / `attach_kinase_evidence`.

Candidate commits:
- `ca6a96e` phase 1.3: generalize kinase scoring
- `d6d5c8c` kinase library adaptor
- `4e2e672` phase 0.1: refactor Integr_multiomics

### `kl_evidence` — new slot in current HEAD
Native has no `kl.evidence` slot; current HEAD attaches per-(Path, Kinase) activity
rows including external `kinase_library_score`. This is the home of the kinase
augmentation channel that is currently conflated with the native SiK channel via
`ENABLE_KINASE_AUGMENTATION`. Sprint 5 territory.

### `EI` — structural diff
Native and current both produce length-2 lists, but element contents differ.
Likely tied to `Cal_EI` rewrite + condition-naming standardization (`3954fef`).

### `Ack_FC`, `KGG_FC`, `Rme1_FC` — new slots
Acetylation/glycation/methylation tracks added past the upstream baseline — these
exist independently of the factorial extension and need explicit C-bucket
justification (or revert).

Candidate commits:
- `c3580fc` apply cutoff to all omics slots and f…
- `4e2e672` phase 0.1: refactor Integr_multiomics
- `0adcca8` (alzheimers) feat: add tyrosine phospho (pY) track parallel

## Implications for Sprint 1 (factorial-extension audit)

1. `sigprob`, `sc_FC`, `p_value`, `pr_FC`, `ps_FC`, `py_FC` all match bitwise on the
   two-condition synthetic input — Sprint 1 can lean on these as a known-good
   regression anchor while introducing factorial-mode equivalents.
2. `evaluation.PDS` drift (~0.089) on a two-condition input means at least one of
   the **scoring formula** changes (EM promiscuity weight, `logi` parameters,
   `KPDS.weight` defaults) is active even outside factorial mode. This is the
   first concrete C-bucket candidate — Sprint 3 territory.
3. The new `Ack_FC`/`KGG_FC`/`Rme1_FC` slots, plus three new evaluation columns,
   indicate scope expansion outside the factorial mandate. Each gets its own
   ledger row.
4. `kl_evidence` introduces the external kinase-library augmentation channel
   alongside native SiK. Sprint 5 must split the `ENABLE_KINASE_AUGMENTATION`
   flag to control these independently.

## Files

- Native fixture: `../incytr/tests/testthat/fixtures/golden_native_93b9881.rds`
- Current fixture: `../incytr/tests/testthat/fixtures/golden_current_head.rds`
- Test fixture (alias): `../incytr/tests/testthat/fixtures/golden_output_v1.rds`
- Comparison script: `../incytr/tests/testthat/compare_golden_outputs.R`
- Native generator (worktree): `/home/hchung/Projects/work/incytr-93b9881/tests/testthat/generate_golden_native.R`

## Sprint 1 verdict addendum (2026-05-05)

Sprint 1 closed with the following outcomes against the implications listed above:

1. **`sigprob`/`sc_FC`/`p_value`/`pr_FC`/`ps_FC`/`py_FC` bitwise match — confirmed and signed off.**
   The wrapper-side degenerate 2-condition runner at
   `code/integration/tests/run_degenerate_2cond.sh` re-asserts this each run
   and hard-fails on any drift in those slots. Used as the regression
   anchor for Sprints 2–5.

2. **`evaluation.PDS` drift (~0.089) — attributed to `abde752` (INC-25)** "add EM
   promiscuity weighting and edge confidence scoring to SigProb." The EM
   weight modifies `Cal_SigProb` upstream of PDS, so the legacy two-condition
   path inherits the drift. Routed to **Sprint 3** in the ledger.

3. **`Ack_FC`/`KGG_FC`/`Rme1_FC` slots + cutoff change — attributed to
   `c3580fc` (INC-30)**. Split into INC-30 (PTM-track scope expansion) and
   INC-30.b (filter change). Both routed to **Sprint 4**. Pairs with ALZ-2
   `0adcca8` for the pY parallel.

4. **`kl_evidence` slot + `SiK_score_*` drift (~0.18) — attributed to
   `6858063` (INC-28)** "memory permutation pass" with co-suspect `ca6a96e`
   (INC-13) "phase 1.3 generalize kinase scoring." Routed to **Sprint 5**
   alongside the kinase augmentation flag split.

5. **`EI` structural diff** — attributed to `3954fef` (INC-18) "phase 0.5
   standardize condition naming." Structural-only (cond1/cond2 → condA/condB
   labels), no numerical drift. Signed off as A-bucket.

The four §7 open design decisions are recorded as `INC-DESIGN-1..4` in the
ledger with `disposition = signed-off` and `verdict_date = 2026-05-05`.

Sprint 1 introduced no code changes to the `incytr` package (read-only
sprint). Wrapper-side artifact: `code/integration/tests/run_degenerate_2cond.sh`.
Working note: `docs/integrations/working/sprint1_factorial_surface.md`.

## Sprint 2 verdict addendum (2026-05-05)

Sprint 2 audited the **performance bucket** — rewrites that should not change
scientific output relative to native Incytr on a matched two-condition input.
None of the Sprint-2 commits introduce numerical drift; the equivalence
claims they assert in their own commit messages were reproduced with explicit
runners.

1. **`c1b6fb9` split — `INC-37` (perf) + `INC-37.b` (correctness, fix-forward).**
   The bundled commit's three correctness hunks are all numerically neutral on
   the in-use code path:
   - `R/analysis.R` `pathway_inference` null-gene fallback fixes a **native
     `93b9881` bug** (the message claimed it was setting `gene.use_Sender`/
     `gene.use_Receiver` to all genes but only updated the unused `gene_use`
     variable; the buggy branch is unreachable in our wrapper, which always
     passes both args).
   - `R/math.R` `setDT(df)` → `as.data.table(df)` removes an in-place caller
     mutation (also a native bug); return-value semantics are identical.
   - 7× `|` → `||` in scalar `is.null(x) | foo(x)` patterns is lazy-eval
     defense; no behavior change.

   Neither hunk was re-routed to Sprint 3 (formula) or Sprint 4 (filter).
   Per-fix verdict in `docs/integrations/working/sprint2_perf_split.md`. The
   two native-bug fixes are **flagged for an upstream PR follow-up**, not
   blocking.

2. **DuckDB enumeration (ALZ-18) bitwise-equivalent to native
   `pathway_inference()`.** Verified by
   `code/integration/tests/run_duckdb_enumeration_equiv.sh` on the synthetic
   12-gene × 40-cell × 2-condition fixture with `cutoff_SigProb = 0` and
   `em_promiscuity_weight = FALSE`. The runner sorts both pathway sets and
   compares (Ligand, Receptor, EM, Target) tuples bitwise. Pre-prune cutoffs
   `Hill < 0.01` and `SigProb >= 0.01 OR` are bucket C and remain Sprint 4
   territory; Sprint 2 only verifies that the enumerator is equivalence-
   preserving when those filters are off.

3. **Vectorized receiver scoring (ALZ-15) promoted to a one-command runner.**
   `verify_phase2.R` is now invoked via
   `code/integration/tests/run_verify_phase2.sh` at `tol=1e-10`. The runner
   gracefully SKIPs when Phase 1 per-pair CSVs and Phase 2 receiver Parquets
   aren't both present (i.e., when `code/integration/intermediates/all_pairs/`
   doesn't have a fully-run pipeline output); when they exist, mismatch is a
   hard failure.

4. **Source-split / dependency-removal (INC-35, INC-36, INC-38) signed off.**
   Each commit's own equivalence claim ("13/13 snapshots exact match",
   "all 148 tests pass", "+23 tests, 13/13 snapshots unchanged") was
   reproduced via `run_degenerate_2cond.sh` (must-match slots clean) and the
   593/593 testthat run.

5. **No new package-level commits.** Sprint 2 was read-only for the `incytr`
   package, same as Sprint 1. All artifacts land in `alzheimers`:
   `code/integration/tests/run_duckdb_enumeration_equiv.sh`,
   `code/integration/tests/run_verify_phase2.sh`, ledger updates, and
   `docs/integrations/working/sprint2_perf_split.md`.

## Sprint 3 verdict addendum (2026-05-05)

Sprint 3 audited the **scoring-formula bucket** (Section C, group 1): items
that modify the SigProb / TPDS / PDS formulas relative to native Incytr.

1. **INC-25 (`abde752` EM promiscuity weight) — reverted to default-off,
   parking-in-place.** Three package-side default flips (`em_promiscuity_weight = TRUE → FALSE` in `Cal_SigProb`, `Cal_SigProb_animal`,
   `run_factorial`) plus an env-var gate in
   `code/integration/wrappers/run_incytr_factorial_all_pairs.R`
   (`ENABLE_EM_PROMISCUITY_WEIGHT=1` to opt back in). Pattern matches
   the `ENABLE_KINASE_AUGMENTATION` precedent — code preserved behind
   a default-off flag rather than physically removed. No empirical or
   theoretical justification for the original default-on choice was
   recorded in `docs/integrations/` or in the integration code.

2. **Sprint 1 attribution amended.** Sprint 1 named INC-25 the prime
   `evaluation.PDS` drift driver, with magnitude ≈0.089 on the Sprint 0
   synthetic fixture. Inspection during Sprint 3 showed the synthetic
   Layer-3 DB has `em_degree = 1` for every EM node, so
   `em_weight_log(1) = 1/log2(2) = 1` is a no-op even when the flag was on
   — INC-25 has zero observable effect on this fixture. The residual
   ~0.089 PDS drift therefore flows through `kl.pathways → Cal_PDS` from
   INC-28 (`6858063` memory permutation pass), which Sprint 1 had already
   routed to Sprint 5. The INC-25 revert remains substantive on production
   data where `em_degree` is non-trivial (e.g., Ep300 at ~16k yields
   weight ≈0.07), but the Sprint 0 fixture cannot empirically distinguish
   INC-25 from INC-28.

3. **`logi()` k parameter audit (INC-DESIGN-5) — no drift.** `logi()`
   body byte-identical to native (`2/(1+exp(-k*x))-1`). Legacy two-
   condition path uses `k_logi = 2` (native default). Factorial path uses
   `k_logi = 2/log(2)` as a deliberate scale calibration (factorial OLS
   fits log-FC; native consumes aFC), already covered by INC-DESIGN-3
   and `test-Pathway_evaluation_factorial.R`. Signed off as A-bucket.

4. **`Find_highexp_gene` audit (ALZ-22) — wrapper-config divergence,
   justified.** Native `Find_highexp_gene` (`R/utils.R`,
   `cutoff_percentile = 0.5`) is byte-identical to the inherited
   definition; the package itself is unmodified. The all-pairs wrappers
   bypass it entirely by passing explicit `gene.use_Sender` /
   `gene.use_Receiver` whitelists computed Python-side using
   `EXPR_DETECTION_THRESHOLD = 0.10` (10% of cells with nonzero UMI). This
   is the standard snRNA-seq detection-rate rule, appropriate for sparse
   single-nucleus data (5–10× lower UMI than scRNA-seq); native's
   50th-percentile-of-expressed cutoff was designed for scRNA-seq density.
   Signed off without code change.

5. **Sprint 3 deliverables.**
   - Package edits: 3 default-flip lines plus `@param` docstring updates
     in `R/analysis.R` and `R/factorial.R`.
   - Wrapper edits: env-var gate around `em_w_vec` application in
     `code/integration/wrappers/run_incytr_factorial_all_pairs.R`.
   - Working note: `docs/integrations/working/sprint3_scoring_audit.md`.
   - Verification: `run_degenerate_2cond.sh` (must-match slots clean),
     `run_duckdb_enumeration_equiv.sh` (still equivalent), 593/593
     testthat across 216 files.
