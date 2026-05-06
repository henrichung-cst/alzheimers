# Sprint 1 working note — factorial code surface map

Working scratchpad for Sprint 1 D1. May be deleted after Sprint 1 sign-off.

## Factorial-only S4 slots in `R/Incytr_class.R`

Added by `4a05aae` "phase 2.1: add factorial S4 slots and create_Incytr validation":
- `animal_id` (character) — column in meta identifying biological replicate
- `expr.byanimal` (list) — per-animal expression
- `sigprob.byanimal` (data.frame) — per-animal SigProb
- `design` (ANY) — n_animals × p design matrix
- `contrasts` (list) — named numeric contrast vectors

All five slots are `NULL`/empty by default. Legacy two-condition path never reads them.

## Factorial-only functions in `R/factorial.R`

| Function | Introducing commit | Covering testthat file |
|---|---|---|
| `Expr_bygroup_animal()` | `84f258f` (phase 2.2-2.3) | `test-Expr_bygroup_animal.R` |
| `Cal_SigProb_animal()` | `84f258f` | `test-Cal_SigProb_animal.R` |
| `compute_em_target_weight()` | `84f258f` (helper for animal SigProb) | indirectly `test-Cal_SigProb_animal.R` |
| `Contrast_SigProb()` | `ad4042e` (phase 2.4) | `test-Contrast_SigProb.R` |
| `contrast_to_conditions()` | `ad4042e` | `test-Contrast_SigProb.R` |
| `classify_regime()` | `ad4042e` | `test-Contrast_SigProb.R` |
| `run_factorial()` | `ff123e0` (phase 2.5) | `test-factorial_integration.R` |

Subsequent simplification commits `9d888ea`, `44d3008` reduce repetition / remove dead test scaffolding; do not change semantics.

## Factorial branches in legacy files

Pattern: `if (identical(object@options$mode, "factorial")) { ... }`

| File | Line | Function | Purpose | Introducing commit |
|---|---|---|---|---|
| `R/analysis.R` | 310 | `Cal_scFC` | dispatches to factorial scFC path | `e09105a` (phase 3) |
| `R/evaluation.R` | 77 | `Pathway_evaluation` | dispatches to factorial pathway evaluation | `e09105a` |
| `R/evaluation.R` | 251 | `Export_results` | per-contrast export | `e09105a` |
| `R/evaluation.R` | 494 | `Cal_PDS` | per-contrast PDS | `e09105a` |
| `R/Incytr_class.R` | 188 | `create_Incytr` | sets `mode <- "factorial"` when design is supplied | `4a05aae` |

## Slot/function additions that are NOT factorial-extension

Routed to other sprints. None of these belong to A-bucket.

| Slot or function | Introducing commit | Bucket / Sprint |
|---|---|---|
| `Ack_FC`, `KGG_FC`, `Rme1_FC` slots | `c3580fc` "apply cutoff to all omics slots" | C / Sprint 4 (new omics tracks; ALZ-side `0adcca8` adds the pY parallel) |
| `kl.evidence`, `kl.activity` slots | `6858063` "memory permutation pass" | C / Sprint 5 (kinase augmentation channel) |
| EM promiscuity weight `1/log2(1+degree)` | `abde752` | C / Sprint 3 (scoring formula change; **prime PDS-drift suspect**) |
| `as_kldata()` / `as_kl_evidence()` adaptors | `d6d5c8c` | C / Sprint 5 |
| `pathways_5steps`, condition-label permutation | `f922c3a`, `b654fdd` | C / Sprint 5 |
| Pre-prune cutoffs in `compute_sigprob` (`Hill < 0.01`) | `c3580fc` (mixed) | C / Sprint 4 (filter, not factorial) |

## Drift attribution leads (for Sprint 3/5 hand-off)

- **`evaluation.PDS` drift max\|diff\|≈0.089** on two-condition input. Pre-diff candidates: `e09105a`, `c8e764e`, `abde752`, `c3580fc`. Strongest single suspect: `abde752` "add EM promiscuity weighting and edge confidence scoring to SigProb" — it modifies SigProb upstream of PDS, so even the legacy two-condition path inherits the change. Verify in Sprint 3 by reverting `abde752` on a worktree and re-running goldens.
- **`kl_pathways.SiK_score_*` drift max\|diff\|≈0.18** on two-condition input. Candidates: `ca6a96e`, `d6d5c8c`, `4e2e672`, `6858063`. Strongest single suspect: `6858063` "memory permutation pass" — touches `R/kinases.R` for 651 lines and introduces the `kl.evidence`/`kl.activity` slots that are the home of the augmented-kinase channel. Verify in Sprint 5.

## Test coverage map (factorial-named files, all currently passing per Sprint 0)

- `helper-factorial.R` — shared synthetic factorial fixture
- `test-create_Incytr_factorial.R` — `4a05aae`
- `test-Expr_bygroup_animal.R`, `test-Cal_SigProb_animal.R` — `84f258f`
- `test-Contrast_SigProb.R` — `ad4042e`
- `test-factorial_integration.R` — `ff123e0`
- `test-Cal_scFC_factorial.R`, `test-Cal_PDS_factorial.R`, `test-Pathway_evaluation_factorial.R`, `test-Export_results_factorial.R` — `e09105a`
- `test-n_condition_integration.R` — `c8e764e`
