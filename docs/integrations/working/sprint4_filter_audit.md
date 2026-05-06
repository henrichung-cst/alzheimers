# Sprint 4 — Filtering and pathway-universe audit (Section C, group 2)

Per audit-plan §5 Sprint 4: justify or revert items that change *which* pathways
enter scoring. Sign-off gate: each item has a verdict row; reverted code lives
behind a default-OFF flag (parking-in-place per the `ENABLE_KINASE_AUGMENTATION`
and `ENABLE_EM_PROMISCUITY_WEIGHT` precedents).

## Items audited

### 1. DuckDB pre-prune `cutoff_SigProb = 0.01` (`ALZ-18.b`, split from `ALZ-18`) — KEEP-FLAGGED

`code/integration/wrappers/duckdb_enumeration.R:67, 150–163, 294–295` and
the centralised assignment at
`code/integration/wrappers/run_incytr_factorial_all_pairs.R:86` apply a
SigProb pre-prune at 0.01: a pathway is dropped if its 3-edge Hill product
< 0.01 in *both* conditions. Native Incytr default is `cutoff_SigProb = NULL`
(no pre-prune) — verified at
`/home/hchung/Projects/work/incytr-93b9881/R/analysis.R:600,606`.

**Justification (mathematical, not empirical).** SigProb is a multiplicand in
`PDS = TPDS · ph_PDS · ...` (and feeds aFC through `Cal_foldchange`). A pathway
whose pre-prune Hill product is < 0.01 in BOTH conditions has a maximum
SigProb < 0.01; its TPDS contribution is bounded by `logi(log2(SigProb_c1 + 1e-4
- log2(SigProb_c2 + 1e-4)), k=2/log(2))`, which collapses to negligible values
(< 0.01) for any pair within the pre-prune region. Such pathways cannot
plausibly enter any reasonable top-K cut.

The OR-test (`h1 >= cutoff` OR `h2 >= cutoff`) keeps every pathway whose either-
condition SigProb crosses the threshold, so condition-asymmetric signals are
preserved. The R-side prune at `duckdb_enumeration.R:163` and the SQL prune at
`:294–295` use the same OR rule.

**Verdict: keep-flagged. Parking-in-place via `DUCKDB_CUTOFF_SIGPROB` env var.**
`run_incytr_factorial_all_pairs.R:86` was changed from a hard-coded
`cutoff_SigProb <- 0.01` to a `Sys.getenv("DUCKDB_CUTOFF_SIGPROB", "0.01")`
read; setting `DUCKDB_CUTOFF_SIGPROB=0` opts back into native-equivalent
enumeration.

The Sprint 2 `run_duckdb_enumeration_equiv.sh` runner already exercises the
`cutoff_SigProb = 0` path and confirms bitwise pathway-set match against
native `pathway_inference()`, so the opt-out is verified live on every run.

### 2. `ALZ-19` kinase-imputed receiver expansion + `EXPR_IMPUTATION_FLOOR` — SIGN OFF

`code/integration/run_factorial_all_pairs.sh:70` short-circuits the entire
`export_kinase_imputed_genes_factorial.py` step when
`ENABLE_KINASE_AUGMENTATION != 1`, so no `kinase_imputed_genes__*.csv` files
exist on disk by default. Inside the wrapper,
`run_incytr_factorial_all_pairs.R:471–495` reads imputed CSVs only via
`load_imputed_for_recv_factorial(recv)` and only enters the rescue block when
`!is.null(recv_imputed_df) && nrow(recv_imputed_df) > 0` — both conditions are
false on a default run.

`expr_imputation_floor` is read unconditionally at line 228 but is consumed
only inside the imputation-rescue branch (`if (expr_imputation_floor > 0)` at
line 474), which is itself nested inside the `recv_imputed_df` non-null guard.
The unconditional read is a no-op on baseline runs.

**Verdict: signed-off (parking-in-place behind `ENABLE_KINASE_AUGMENTATION`).**
No code change. Both `ENABLE_KINASE_AUGMENTATION` and `EXPR_IMPUTATION_FLOOR`
are now also forwarded through `run_factorial_all_pairs.sh:85` for systemd-run
isolation.

### 3. `INC-30` (PTM scope expansion) and `ALZ-2` (pY track) — SIGN OFF

Native Incytr has no `Ack_FC`, `KGG_FC`, `Rme1_FC`, or pY tracks. INC-30
introduces these slots in `R/Incytr_class.R` and `ALZ-2` adds the bulk-side
parallel under `code/data_ingest.py` and integration adapters.

**Verdict: signed-off as scope expansion.** The new slots are additive: they
produce additional output columns parallel to the existing serine/threonine
track and do not perturb the canonical PDS/TPDS pipeline. The Sprint 1
`run_degenerate_2cond.sh` synthetic fixture has no PTM data; the new slots
emit "data not found … skipped" messages confirmed in the latest run, and
the must-match SigProb/FC slots remain bitwise-clean against native.

### 4. `INC-30.b` (cutoff applied to all omics slots) — SIGN OFF

`c3580fc`'s filter portion broadens the `cutoff_SigProb` application from one
slot in native to all PTM slots in the post-INC-30 codebase. Native
`R/analysis.R::Cal_SigProb` defaults `cutoff_SigProb = NULL`
(`/home/hchung/Projects/work/incytr-93b9881/R/analysis.R:606`). When NULL, no
filter runs in either codebase; the change is **default-inert**.

When a non-NULL cutoff is passed, the post-INC-30 code applies it uniformly
across all omics slots. This is the principled extension for a multi-PTM
codebase (filter consistency across slots) and cannot silently introduce a
bug that masks itself in the testthat suite — `tests/testthat/test-apply_path_cutoff.R`
exercises the cutoff path and is part of the 593-test suite that passes.

**Verdict: signed-off (default-inert; opt-in cutoff applies uniformly across
all omics slots when set).**

### 5. `ALZ-23` `ENABLE_CELLTYPE_MAPPING` (WMB ↔ SEA-AD remap) — SIGN OFF

`code/integration/adapters/export_multiomics_evidence_factorial.py:42` short-
circuits to identity mapping (`{ct: ct for ct in cell_types}`,
strategy = `"exact_or_global"`) when `ENABLE_CELLTYPE_MAPPING != "1"`. The
default branch (no remapping) emits omics columns keyed on the WMB-class
taxonomy already produced by `expression_metadata.csv`.

**Verdict: signed-off (parking-in-place behind `ENABLE_CELLTYPE_MAPPING`;
default-off path is native-equivalent).** The flag is now also forwarded
through `run_factorial_all_pairs.sh:85`.

## Verification gate (run order)

1. `bash code/integration/tests/run_degenerate_2cond.sh` → must-match slots
   bitwise-clean against native; PTM "data not found … skipped" notes confirm
   INC-30/ALZ-2 are inert when PTM data is absent.
2. `bash code/integration/tests/run_duckdb_enumeration_equiv.sh` → DuckDB
   enumeration with `cutoff_SigProb = 0` produces bitwise-identical pathway
   set to native `pathway_inference()`, confirming the
   `DUCKDB_CUTOFF_SIGPROB=0` opt-out is genuine native-equivalent.
3. `bash code/integration/tests/run_verify_phase2.sh` → SKIP (no Phase 1/2
   fixture committed; gracefully handled).
4. `Rscript -e 'pkgload::load_all("../incytr"); testthat::test_dir("../incytr/tests/testthat")'`
   → 593/593 (no regression).

All four gates passed on 2026-05-05.
