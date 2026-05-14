# Incytr pair-mode benchmark plan

**Goal**: Verify that the upstream `~/Projects/work/incytr/` R package's pair-mode produces numerically-identical outputs to the legacy `data/incytr/incytr_commandline.R` driver, while running substantially faster — using the v1 (8-cluster) inputs we just measured.

**Baseline (already measured, 2026-05-12, ma_4mo_AppP vs ma_2mo_AppP)**:

| Metric | Value |
|---|---|
| Steady-state per-pair time | ~94 s (range 30–150 s, bigger DE sets → slower) |
| Projected per-comparison (64 pairs, `nboot=100`) | ~1.7 h |
| Projected full 16-comparison v1 sweep | ~26–28 h |
| Peak RSS | 8.9 GB (single R process) |
| Init overhead (R + library + data load) | ~3:53 on first pair |

**Hypothesis**: With matched knob settings, upstream produces bit-for-bit identical scoring columns (modulo permutation-test RNG) and meaningfully faster wall-clock — driven by data.table/matrixStats vectorization in `Expr_bygroup`, `Cal_SigProb`, and especially `Permutation_test`.

## Why this matters

If parity + speedup hold → retire `incytr_commandline.R`, run the 16-comparison sweep through upstream in a fraction of 28 h, and unblock similar consolidation for v2 (46 clusters, larger Type vocabulary will exaggerate the upstream speedup).

If parity fails → identify which step's snapshot drifted and decide whether to port legacy semantics into upstream or freeze legacy as the v1 reference.

## Known API differences to reconcile

The two pipelines run the same 12-step sequence but the upstream package has cleaner defaults and a few arg renames. To make outputs comparable we must match knobs explicitly, not rely on defaults.

| Step | Legacy driver value | Upstream default | What to set in benchmark |
|---|---|---|---|
| `create_Incytr` arg name | `condition = c(...)` | `conditions = c(...)` | use `conditions = ...` |
| `Expr_bygroup`, `Permutation_test`, `Integr_kinasedata`, `Kinase_exploration` | `mean_method = "mean"` | `mean_method = NULL` (trimean) | force `"mean"` — most likely numeric drift source |
| `Cal_SigProb(cutoff_SigProb=…)` | `0.25` | `NULL` (no filter) | `0.25` |
| `Integr_multiomics(*.correction=…)` | `0.001` | `NULL` | `0.001` |
| `Pathway_evaluation` extras | not passed | `cutoff_TPDS=NULL, abs.value=NULL, style=NULL` | pass `NULL` explicitly |
| `Cal_PDS(cutoff_PDS=…)` | `0.1` | `0.0` | `0.1` |
| `Permutation_test(seed.use=…)` | `1L` | `1L` | `1L` |
| `Permutation_test(nboot=…)` | `100` | `100` | `100` |
| kldata format | runs `homologene::human2mouse()` | expects pre-mapped mouse symbols | already patched in legacy; upstream needs raw mouse-format kldata too |
| `cell_group` for `Integr_kinasedata` | `levels(Xobject@meta$Type)` | `levels(factor(meta$Type))` | match exact order via `sort(unique(Type))` |

## Phase A — Set up the upstream comparison runner

**A.1** Confirm upstream `Incytr` is loadable from the alzheimers pixi env.

```fish
cd /home/hchung/Projects/work/alzheimers
pixi run -- Rscript -e 'library(Incytr); cat(as.character(packageVersion("Incytr")), "\n"); cat("score_factorial_paths exported:", "score_factorial_paths" %in% getNamespaceExports("Incytr"), "\n")'
```

If not installed: install from the local path (`devtools::install("~/Projects/work/incytr")`). Do not pull from CRAN — this is the in-development copy.

**A.2** Write `data/incytr/v1_8clusters/provenance/run_upstream_pair.R`.

Single-pair pair-mode runner. CLI args: `condition1 condition2 sender receiver`. Mirrors the legacy 12-step sequence but uses upstream-exported functions with explicitly matched knobs (table above).

Per-step wall-clock instrumentation: `t <- Sys.time()` before each step, log `difftime` after, accumulate to a CSV sidecar so we can see exactly which step the upstream accelerates. Output:

- `output/upstream_<cond1>_<cond2>_<sender>_<receiver>.csv` — pair output table (matches legacy `Export_results` schema)
- `output/upstream_<cond1>_<cond2>_<sender>_<receiver>.timing.csv` — per-step wall-clock + peak RSS

**A.3** Write `data/incytr/v1_8clusters/provenance/run_legacy_single_pair.sh`.

Short-circuit `incytr_commandline.R` to one pair by replacing the outer `for (i in groups) for (j in groups)` loop with a single `Sender.group/Receiver.group` from CLI args. Otherwise identical logic and knobs. Output: `output/legacy_<cond1>_<cond2>_<sender>_<receiver>.csv` + `.timing.csv`.

## Phase B — Generate matched legacy reference for one pair

Pick one pair from the measured run. **Recommendation: `Astrocytes → Microglia`** — it was a 2:00 pair (mid-range), big enough that any per-step speedup is visible, small enough that re-running both pipelines fits in <10 minutes.

Run both runners on the same input set:

```fish
cd data/incytr/v1_8clusters
./provenance/run_legacy_single_pair.sh ma_4mo_AppP ma_2mo_AppP Astrocytes Microglia
pixi run -- Rscript provenance/run_upstream_pair.R ma_4mo_AppP ma_2mo_AppP Astrocytes Microglia
```

Both should write to `output/`. The legacy run should complete in ~2 min; the upstream target is meaningfully under that.

## Phase C — Compare outputs

Write `data/incytr/v1_8clusters/provenance/diff_upstream_vs_legacy.R`. Takes two CSV paths, reports:

1. **Schema parity**: same column names (order-normalized), same column count, same row count.
2. **Identifier alignment**: `ID_1` is the legacy primary key — must join 1:1 with no orphans either side.
3. **Numeric tolerance**:
   - Scoring columns (`SigProb`, `TPDS`, `PDS`, `*_FC`, `*_log2FC`): `all.equal(legacy, upstream, tolerance = 1e-8)`. Looser tol (`1e-4`) acceptable only if upstream documents the diff (e.g. trimean vs mean — but we're matching `mean_method="mean"` so this shouldn't fire).
   - Integer/factor columns: exact equality.
4. **Permutation p-values**: structural check first (range `[0,1]`, no NaNs). Exact equality only if upstream's `Permutation_test` uses the same RNG path with `seed.use=1L`. If exact fails: report the drift, isolate to step 10, decide whether to keep upstream's vectorized perm or port legacy's loop-based perm.

Pass criteria: schema + IDs + numeric tolerance pass with at most a documented permutation-RNG diff.

## Phase D — Wall-clock + memory comparison

Per-step table from the timing sidecars:

| Step | Legacy (s) | Upstream (s) | Speedup |
|---|---|---|---|
| 1 create_Incytr | | | |
| 2 pathway_inference | | | |
| 3 Expr_bygroup | | | |
| 4 Cal_SigProb | | | |
| 5 Cal_scFC | | | |
| 6 Integr_multiomics | | | |
| 7 Pathway_evaluation | | | |
| 8 Integr_kinasedata | | | |
| 9 Cal_PDS | | | |
| 10 Permutation_test | | | |
| 11 Kinase_exploration | | | |
| 12 Export_results | | | |
| **Total per pair** | **~120** | **?** | **?** |

Plus `/usr/bin/time -v` peak RSS for both. Legacy peaked at 8.9 GB; upstream is expected lower due to data.table memory hygiene.

## Phase E — Decision matrix

| Parity result | Speedup | Action |
|---|---|---|
| Bit-identical (mod perm-RNG) | ≥3× | Adopt upstream as v1 driver. Kick off full 16-comparison sweep on upstream. |
| Bit-identical | <3× | Adopt upstream for maintainability; budget the legacy-equivalent ~28 h sweep on upstream. |
| Drift confined to perm-RNG | any | Treat as expected; document and adopt. |
| Drift in deterministic columns | any | Stop. Isolate to one of the 12 steps. Decide whether to port legacy semantics into upstream (preferred — upstream is the future) or freeze legacy as the v1 reference (only if upstream's defaults are scientifically wrong for this dataset). |

## Phase F — Stretch: v2 (46 clusters) on the same upstream driver

If A–E pass on v1, the same upstream runner handles v2 with zero code changes (the 46-type vocabulary is just a different `meta$Type` factor). One smoke-test pair on v2 (e.g., a mid-size sender/receiver) confirms before triggering the full v2 sweep — the 46² = 2,116 pair count makes upstream's speedup essential there, not optional.

## Deliverables

- `data/incytr/v1_8clusters/provenance/run_upstream_pair.R`
- `data/incytr/v1_8clusters/provenance/run_legacy_single_pair.sh`
- `data/incytr/v1_8clusters/provenance/diff_upstream_vs_legacy.R`
- `data/incytr/v1_8clusters/output/{legacy,upstream}_ma_4mo_AppP_ma_2mo_AppP_Astrocytes_Microglia.csv`
- `data/incytr/v1_8clusters/output/{legacy,upstream}_...timing.csv`
- This document, updated with the timing table and parity verdict in Phase D/E.

## Non-goals

- We are **not** extending the closed direct-deconvolution path. This benchmark only validates that upstream can serve as a faster drop-in for the legacy driver on the v1 inputs we already have.
- We are **not** rewriting any pipeline logic. If parity fails we document it; we do not silently change upstream's behavior.
- We are **not** running the full 16-comparison sweep as part of this diagnostic. That decision lives in Phase E.
