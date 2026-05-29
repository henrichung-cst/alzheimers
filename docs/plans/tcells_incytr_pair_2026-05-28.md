# T-cell Incytr pair-mode (D5)

**Date:** 2026-05-28
**Status:** proposed
**Depends on:** `tcells_percell_aggregation_2026-05-28.md` (state-keyed substrate, complete)
**Stream:** D5 of `meeting_notes_triage_2026-05-27.md`

## Goal

Run Incytr pair-mode on the per-donor T-cell deconvoluted substrate, scored
per-donor independently, across (later day) − (day-2 baseline) contrasts.
Reuse the sce4-parity scoring path (`incytr_commandline.R` + the six overrides
in `CLAUDE.md`); do **not** reuse the mouse cohort inputs, contrasts, or output
directory.

Cohort = the T-cell exhaustion experiment only. Not mouse, not human.

## Constraints (carry from cohort + repo invariants)

- **donor1 = 3 channels** (pr + py + ps). **donor2 = 2 channels** (pr + py, no IMAC).
  Driver currently reads `ps_yuyu_deconvoluted.csv` unconditionally — must be
  parameterized.
- **Per-donor independence** (meeting notes): timepoints not comparable across
  donors. No cross-donor pooling, no joint Seurat object.
- **Per-cell ProjecTILs state as ident** (not Seurat cluster). Names already
  alphanumeric-sanitized by the extract step. The `<condition>_<state>` split in
  the driver still works.
- **Contrasts:**
  - donor1 shared days = {2, 13, 17, 20} → 3 contrasts: `d13_vs_d2`, `d17_vs_d2`, `d20_vs_d2`
  - donor2 shared days = {2, 5, 7, 9, 11} → 4 contrasts: `d5_vs_d2`, `d7_vs_d2`, `d9_vs_d2`, `d11_vs_d2`
  - Condition vocabulary: `d{day}` (e.g. `d13`). Substrate columns `d{day}_{state}` already match `<condition>_<cluster>` split semantics.
- **sce4-parity overrides apply unchanged** (DG no-cap, `pmax(pr_, 1)` floor,
  trimean, `correction=0.01`, `cutoff_SigProb=0`/`cutoff_PDS=0`, `Cal_scFC ε=0.01`).
  These are method, not cohort.
- **Significance filter retained downstream**: `(SigProb_<later> > 0.1 OR SigProb_d2 > 0.1) AND |PDS| >= 0.2` via `filter_significant_paths.py`.

## Files

### New: `alz/incytr_pair/build_tcells_seurat.R`

Per-donor T-cell Seurat object for the input-gene-list step.

- Read raw `data/datasets/tcells/<donor>/scrna/...rds` (memory-safe: DietSeurat → RNA-only, drop scale.data) — the same input the extract step reads.
- Join per-cell ProjecTILs state from `data/derived/tcells_incytr_inputs/<donor>/scrna/projectils_predictions.csv` via the same 14-entry `LABEL_MAP` as the extract step. Drop NA-state cells.
- Set `obj$Type <- state`, `obj$condition <- sprintf("d%d", day)`, `obj$Type_condition <- paste0(Type, "_", condition)`.
- `Idents(obj) <- factor(obj$Type)` (state-level for HEG; later overridden for FindAllMarkers).
- Alphanumeric guard on `Type` (carried from extract — defensive double-check).
- Write `data/derived/tcells_incytr_inputs/<donor>/incytr_obj.rds`.

No spine CSV (every observed state is in scope; no levy_t5 analog exists).

### New: `alz/incytr_pair/build_tcells_input_gene_list.R`

Per-donor `input_gene_list.csv` = `DEG ∪ prG` per state, per the sce4 receiver-gene rule.

- Read `incytr_obj.rds` (per donor).
- `Idents(obj) <- "Type_condition"`. `FindAllMarkers(only.pos=TRUE, logfc.threshold=1.2)`. Filter `avg_log2FC > 1.5 & p_val < 1e-4`. Strip `_<condition>` suffix to recover state.
- `prG` computed on the **donor's `pr_deconvoluted.csv`**, columns `d{day}_{state}`, exactly as `incytr_commandline.R:161-172`: quantile-normalize the floored `[d_baseline, d_later]` columns across all genes via `limma::normalizeBetweenArrays`, then `log2`, then keep genes with `|pr_log2FC| > 1`. Per contrast × state.
- Union (gene, state) across DEG ∪ prG. Write `data/derived/tcells_incytr_inputs/<donor>/input_gene_list.csv`.

DG step here is `logfc>1.5 & p<1e-4` (no `top_n(500)` cap — sce4-parity override #1).

### Edit: `alz/incytr_pair/incytr_commandline.R`

Parameterize channels + inputs dir to permit T-cell donor2 (no IMAC) without
forking the driver.

- Add env vars: `INPUTS_DIR` (existing — confirm honored everywhere), `CHANNELS` (default `"pr,py,ps"` for mouse), `PR_FILE` / `PY_FILE` / `PS_FILE` (default `pr_yuyu_deconvoluted.csv` etc.).
- Conditional read: if `"ps"` not in `CHANNELS`, skip the `ps <- read_csv(...)` line and pass `ps.data_condition1 = NULL, ps.data_condition2 = NULL` into the scoring call. Verify `Cal_pairwise_grid` accepts NULL ps (the mouse path uses ps; if it errors, fall back to a synthetic site-keyed CSV with one row of NA — preferred is honest NULL).
- Update the numeric regex `^(...|pr_|ps_|py_)` so the suffix set adapts to channels actually present (cosmetic — drops downstream column-trim breakage).
- Verify the `<condition>_<cluster>` parser handles `d13_CD4Th17`-style names (single `_`, alphanumeric on both sides — confirms with extract assertions).

Mouse path is unchanged: defaulted env vars reproduce the existing
`run_pair_mode.sh` behavior byte-exactly. Verify with `verify-incytr-sce4` after
the edit.

### New: `alz/incytr_pair/run_pair_mode_tcells.sh`

Per-donor runner. Pattern follows `run_pair_mode.sh` but with T-cell paths.

- Pre-flight per donor: `incytr_obj.rds`, `pr_deconvoluted.csv`, `py_deconvoluted.csv`, (donor1 only) `ps_deconvoluted.csv`, `input_gene_list.csv`.
- Per-donor contrast loop. Baseline is `d2` for both donors:
  ```bash
  declare -A DONOR_DAYS=( [donor1]="13 17 20" [donor2]="5 7 9 11" )
  declare -A DONOR_CHANNELS=( [donor1]="pr,py,ps" [donor2]="pr,py" )
  ```
- Output: `outputs/reports/incytr_pair_mode_tcells/<donor>/d<later>_d2_incytr_output.parquet`.
- Resumable (skip if parquet exists and non-empty).
- Smoke mode: `--smoke <donor>` → one contrast at `NBOOT=2`.
- Apply `filter_significant_paths.py --dir outputs/reports/incytr_pair_mode_tcells/<donor>/` after the contrast loop completes.
- Memory pattern carried from mouse runner (`NPAIR_WORKERS=1 N_CHUNK_MULT=8`). T-cell pair counts are smaller per contrast (14² = 196 donor1, 11² = 121 donor2) vs mouse 961 — should fit comfortably; do not pre-tune N_CHUNK_MULT.

### Edit: `pixi.toml`

Add tasks:
```toml
tcells-build-incytr-seurat = "pixi run Rscript alz/incytr_pair/build_tcells_seurat.R"
tcells-build-input-gene-list = "pixi run Rscript alz/incytr_pair/build_tcells_input_gene_list.R"
tcells-incytr = "bash alz/incytr_pair/run_pair_mode_tcells.sh"
```

D6 bundle (out of scope for this plan, but the wiring point):
```toml
tcells = { depends-on = [
  "ingest-tcells-scrna", "install-projectils",
  "tcells-projectils-map", "tcells-scrna-extract", "tcells-export-bulk",
  "tcells-decompose",
  "tcells-build-incytr-seurat", "tcells-build-input-gene-list",
  "tcells-incytr"
] }
```

## Pipeline order

```
tcells-decompose                       # already complete — substrate exists
tcells-build-incytr-seurat             # per-donor incytr_obj.rds
tcells-build-input-gene-list           # per-donor input_gene_list.csv
tcells-incytr                          # 3 contrasts donor1 + 4 contrasts donor2
filter_significant_paths.py            # applied inside the runner
```

## Verification

1. `verify-incytr-sce4` still passes after `incytr_commandline.R` parameterization. Hard gate — channel parameterization must not regress mouse parity.
2. donor1: 3 parquet files under `outputs/reports/incytr_pair_mode_tcells/donor1/`, each with `14² = 196` sender × receiver pairs.
3. donor2: 4 parquet files under `outputs/reports/incytr_pair_mode_tcells/donor2/`, each with `11² = 121` pairs.
4. Numeric column suffix set in donor2 outputs contains only `pr_*`, `py_*` (no `ps_*`). Confirms NULL-ps path actually skipped ps scoring.
5. PDS distribution sanity per donor: not all zero, not all NA. Spot-check a Tex vs Naive pair where biology predicts a non-trivial PDS shift between baseline and late day.
6. Mass-identity in the substrate is already verified upstream (≤ 2e-15) — no need to re-check here.

## Open / deferred

- **`Cal_pairwise_grid` NULL-ps handling:** verify by inspection of `~/Projects/work/incytr/R/grid.R` before running the build. If the scorer hard-requires ps (e.g. multiplies by ps as a factor), the donor2 path needs a stub matching pr's row count with NA values plus a small driver branch that excludes ps from PDS aggregation. Either way, donor2 runs.
- **No per-donor kinase MEA** in this plan. D3's kinase MEA on donor1 IMAC is independent of Incytr and is tracked separately.
- **D6 main runner** (`alz/runners/main/run_tcells.sh`) is the wrap-up after D5 verifies. Out of scope here.
