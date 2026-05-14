# incytr-cleanup branch audit

**Date:** 2026-05-12
**Branch:** `incytr-cleanup`
**Scope:** Modified + added files since `main`

## Scoped files

**Modified:**
- `alz/build_unified_viewer.py`
- `alz/integration/export_factorial_inputs.py`
- `alz/integration/factorial.R`
- `alz/integration/load.R`
- `alz/integration/omics_loaders.py`
- `alz/integration/views.sql`
- `alz/viewer/paths.py`
- `alz/viewer/template/body.html`
- `alz/viewer/template/js/tabs/incytr_heatmap.js`
- `alz/viewer/template/js/tabs/incytr_pathways.js`
- `alz/viewer/template/js/tabs/incytr_state.js`
- `alz/viewer/template/js/tabs/temporal_v2.js`

**Added (untracked):**
- `alz/integration/factorial_reshard.R`
- `alz/integration/factorial_subset_perm_bench.R`
- `docs/celltype_taxonomy_consolidation.md`
- `docs/incytr_deconvolution_pivot.md`

## Blockers (must fix before merge)

1. **`receiver_cache` schema mismatch.** On-disk parquets under `outputs/reports/incytr_factorial/receiver_cache/` are wide-format (columns like `SiK_score_AppP_2mo`, `SigProb_AppP_2mo`), but the rewritten `load.R` + `build_unified_viewer.py` now query long-format columns (`contrast`, `pvalue`, `PDS`). Next `pixi run viewer` will throw a DuckDB parse error.
   - **Fix:** Regenerate the cache by re-running `pixi run incytr-factorial` (or `factorial_reshard.R` once it is wired up).

2. **`alz/integration/factorial_subset_perm_bench.R:45,63`.** Calls bare `construct_factorial_paths()` / `score_factorial_paths()` without the `factorial_engine$construct` / `factorial_engine$score` dispatch and without `require_production_engine()`. Will crash if the installed `Incytr` package only exports those names through the dispatch shim.

3. **`alz/integration/factorial_reshard.R:65`.** Calls `Incytr::factorial_results_long()` directly with no presence guard equivalent to `require_production_engine()`. Same failure mode as above if the function is not exported from the installed package.

## Warnings

- **`alz/build_unified_viewer.py:84–98`.** `TISSUE_CATEGORIES` is a local hardcoded copy of `config.WMB_CLASSES`. The 22-non-neuronal assumption is also embedded in the heatmap height formula (`build_unified_viewer.py:237`: `22 * Math.max(nS, nR)`). Should import from `config`.
- **Threshold drift in viewer JS.** Heatmap reset (`alz/viewer/template/js/tabs/incytr_heatmap.js:136`) and temporal defaults (`alz/viewer/template/js/tabs/temporal_v2.js:64`) hardcode `pvalue=0.05`, `absPds=0.01`. The matching grids in `build_unified_viewer.py:1011,1017` (`_INCYTR_PATHWAY_PVALUES`, `_INCYTR_PATHWAY_ABS_PDS`) are not imported from `config_integration.py`. A grid change would silently desync the reset values.
- **`alz/integration/factorial_reshard.R:85–86`.** Desanitization uses `gsub("_", " ", ...)` only, but `sanitize_celltype()` (`load.R:6`) also maps `/`→`-`. Fragile if the taxonomy ever introduces a `/`.
- **`alz/integration/factorial.R:193–194` and `factorial_reshard.R:87–88`.** `n_pre` is set equal to `n_post`. The `pair_metadata.parquet` schema carries them as conceptually distinct (pre- vs post-filter path counts) but both scripts collapse them. Provenance loss, not a crash.
- **`docs/celltype_taxonomy_consolidation.md:38`.** Line-number citation for `export_factorial_inputs.py` is off by ~15 lines (claims 537–572, actual ~553–569).

## Nits

- `alz/integration/factorial_reshard.R` and `alz/integration/factorial_subset_perm_bench.R` are orphan scripts — no pixi task, not referenced from `run_factorial.sh`. Either wire them into pixi tasks / the runner shell scripts, or add a docstring marking them as manual diagnostics.
- `alz/viewer/paths.py:50`. `REPORT_MD` points to a non-existent `pipeline_notes/phase2_payload_report.md` (silent miss, consumed optionally).
- Deployed `outputs/reports/incytr_factorial/views.sql` predates the 2026-05-12 `load.R` rewrite. Semantics unchanged; will auto-refresh on next factorial run.
- `alz/viewer/template/js/tabs/incytr_pathways.js:18`. `_IP_SCORE_COLS_FALLBACK` is not imported from a shared source. Currently correct, but duplicated.
- `docs/incytr_deconvolution_pivot.md` is marked "proposal" but the bulk-omics pass-through it describes is already implemented in `load.R:108–111`. Update status or move to retrospective.

## Top priority

Regenerate `receiver_cache` before the next viewer build — that is the only finding that will actively break the pipeline.
