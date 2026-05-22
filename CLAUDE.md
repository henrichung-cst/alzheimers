# CLAUDE.md

Project context, commands, architecture, and outputs: see [`README.md`](README.md) and [`docs/foundation/`](docs/foundation/). This file holds agent-specific operating overrides, correctness invariants, and traps.

## Project-specific behavior overrides

- **No unit test suite.** The global "run tests after impl phase" rule is satisfied here by `python alz/bulk_mea/summary.py` and the verification harness (`alz/decomposition_mea/verify_decomposition.py --all`).
- **DuckDB spill directory** is `~/.cache/duckdb` via `.envrc` to avoid OOM on tmpfs `/tmp`. If DuckDB hits disk-full, verify with `echo $DUCKDB_TEMP_DIR`.

## Drive naming — do not rename either side

Drive-side is `alzheimers_hc/`, local is `alzheimers/`. High-priority collaborators have links to `alzheimers_hc` on the drive; renaming either side breaks their access. Under `~/mnt/gdrive/` the directory appears as `alzheimers_hc`; the local project root and pixi env are `alzheimers`. Scripts mapping local → drive must not assume identity.

## Closed paths — do not reopen

(See `docs/foundation/analysis_charter.md` and the global Anti-shim rule.)

- **Closed:** direct (statistical) deconvolution, per-cluster stoichiometry, factor model, two-compartment, transcript-only rescue, factorial Incytr (archived 2026-05-18), WMB-34 spine, Levy-19 spine.
- **Active:** stoichiometry + MEA on bulk; levy_t5 forward projection (`P_c = f_c × bulk`); pair-mode Incytr on the 31-cluster spine.

Do not reintroduce closed paths as flags, fallbacks, or sensitivity toggles.

## Pair-mode Incytr correctness invariants

- Production entry: `Incytr::Cal_pairwise_grid` (`~/Projects/work/incytr/R/grid.R`). Factorial APIs `construct_factorial_paths` / `score_factorial_paths` were deleted at commit `424119f` — do not reintroduce them.
- Output shape: **31² = 961 sender × receiver pairs per contrast, 9 contrasts.**
- **Pair pvalue is untrustworthy — filter/rank pathways on `|PDS|`.**
- Integration config lives in `alz/integration/config_integration.py`, not `alz/shared/config.py`.
- Required R deps: `Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`.

## Mass identity (decomposition verification)

`Σ_c [P_c × (N_c / N_total)] ≈ bulk`, **not** `Σ_c P_c = bulk`. The `f_c` weights are per-cell rates × N_total/N_c; literal summation overshoots.

## Schema & data conventions

- When adding provenance/metadata columns, match the existing schema's type exactly (e.g., if single-contrast uses string format for `imputed_nodes`, factorial must too).
- Aggregation queries: verify whether stats (std, consistency) should be computed over raw route-level rows or pre-aggregated sender-level values.

## Pipeline-specific gotchas

- **`analysis_mode` flow** — Kedro parameter in `conf/base/parameters.yml`, default `males_only`. `KEDRO_ENV=full_cohort` overlays the sensitivity config. Affects `enrich.py`, `attribute.py`, `mechanism.py` — **not** `normalize.py` (always uses all 72 samples). Legacy `ANALYSIS_MODE` env var was retired and is silently ignored.
- **Outlier detection requires stoichiometry** — `alz/ingest/song.py --outliers` reads `stoichiometry_matrix.csv`, so `normalize.py` must run first. Falls back to total proteome if unavailable.
- **WMB prerequisite** — `run_live_pipeline.sh` gates on `wmb_kinase_expression.csv` / `wmb_proteome_expression.csv`; run `run_wmb_expression.sh` first.
- **WMB region scope** — `WMB_REGION_SCOPE` defaults to `whole_brain` (correct for the specificity score's brain-wide denominator). `cortex_hpf` is a sensitivity toggle only. Active scope is stamped to `wmb_kinase_expression.scope.json`; a scope mismatch forces recompute.
- **Atlas cache compressed** — raw h5ads under `data/external/allen_abc/` are zstd-compressed (~115 GB → ~26 GB). Decompress with `bash alz/runners/supporting/decompress_atlas_cache.sh` before re-running `wmb_expression.py`. Provenance in `data/external/allen_abc/MANIFEST.json`.
- **WMB expression memory** — `wmb_expression.py --proteome` processes 6,308 genes × 13 regions; use `skip_regional=True`, `chunk_size=2000` to stay under ~30 GB RAM.
- **Stage 6 pY track gating** — `build_celltype_decomposition.py --track py` (or `both`) requires `raw_phospho_normalized_pY.csv` from Stage 1. Smoke runner tolerates missing pY; re-run `pixi run normalize` if you need it.
- **Unified-viewer hard refresh** — `build_unified_viewer.py` inlines PAYLOAD as `<script type="application/json" id="payload-data">` into `index.html`. After `pixi run viewer`, hard-refresh (Ctrl+Shift+R / Cmd+Shift+R) or the cached HTML serves stale data. Quick check: `PAYLOAD.meta.generated_at` in DevTools.
- **API caching** — delete `data/derived/caches/` to force re-fetch from MyGene / Allen Brain Atlas.
