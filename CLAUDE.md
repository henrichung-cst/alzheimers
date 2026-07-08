# CLAUDE.md

Project context, commands, architecture, and outputs: see [`README.md`](README.md) and [`docs/foundation/`](docs/foundation/). This file holds agent-specific operating overrides, correctness invariants, and traps.

## Project-specific behavior overrides

- **No unit test suite.** The global "run tests after impl phase" rule is satisfied here by `python alz/bulk_mea/summary.py` and the verification harness (`alz/decomposition_mea/verify_decomposition.py` — runs hard checks by default: mass identity + spine coverage; diagnostics require `--include-diagnostics` or explicit `--checks`).
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
- **Path-filter production default** (`alz/incytr_pair/filter_significant_paths.py`): `SigProb > 0.1 (either condition) AND |PDS| >= 0.2`, uncapped. No FDR/p_adj arm — the `nboot=100` `p_value_*` columns in `wide/` are informational only. The driver emits all paths (`cutoff_SigProb = cutoff_PDS = 0`); filtering is downstream, not in the driver.
- **gene.use selection is cohort-dependent.** AD reads frozen per-pair node sets via `SCE4_GENEUSE_DIR` (set only by the AD runner); when that dir is unset (t-cells), the driver derives `DEG ∪ prG` per contrast — DEG = this contrast's two conditions' markers at `avg_log2FC > 1 & p_val < 1e-4`, prG = `Incytr::proteomics_gene(style="aFC", cutoff=1, strict=TRUE)`. Human transgenes (App/Psen1/Mapt) are force-included into prG where present.
- **Scorer parameters are `Incytr` package defaults** (`correction = 0.01` on `Cal_SigProb`/`Cal_scFC`, `cutoff_SigProb = cutoff_PDS = 0`, trimean `mean_method = NULL`) — the app inherits them, no call-site overrides. Roxygen at each definition records the calibration. Do not re-inline these as app overrides.
- **`floor_pr` (`pmax(pr_*, 1)`) is app-side and load-bearing** (`incytr_commandline.R`, inline comment ~L310). Deconvolved `pr_yuyu` carries ~1e-5 residuals; flooring `< 1` to `1` stops `Cal_foldchange`'s zero-correction branch from amplifying them into spurious fold-changes. Not a package default (only correct for deconvolved input) — do not remove.

## Mass identity (decomposition verification)

`Σ_c [P_c × (N_c / N_total)] ≈ bulk`, **not** `Σ_c P_c = bulk`. The `f_c` weights are per-cell rates × N_total/N_c; literal summation overshoots.

## Project rules

- **Gated plans: stop at each phase boundary.** When the user defines a checkpoint ("run one, then show me, then run the rest"), stop and report before launching the next phase even when the outcome looks settled. Never fold sequential-but-approved steps into one irreversible action.
- **Intermediates are derivable.** Missing `*.rds` / derived CSVs / cached matrices are not blockers — reconstructing them from raw inputs + method IS the assignment. "We can't reproduce because X.rds isn't on disk" is a restatement of the problem, not a finding. Do not propose accepting stale artifacts as authoritative.
- **No collaborator contact — ever.** Do not propose emailing, asking, or requesting anything from external collaborators (e.g. Yuyu Song, CST team). Not as an option, not as a dismissed alternative. All investigation proceeds from artifacts already on disk + what is derivable from them. If those routes are exhausted, report that honestly — do not escalate externally.
- **Viewer ports = lift, not rewrite.** When porting a viewer to a new cohort, reuse `alz/viewer/template/js/tabs/*.js` / `js/widgets/*.js` / `js/01_state.js` verbatim — reshape the new cohort's payload to fit the unified viewer's `PAYLOAD.*` contract. Greenfielding a new app.js ships a permanently inferior feature surface. Genuinely-N/A features get dropped at `TAB_MANIFEST` level, not silently reimplemented.
- **Multi-session plans go in `docs/plans/`.** Write final plans to `docs/plans/<descriptive-noun-phrase>.md` in the repo, not `~/.claude/plans/`. The staging path is ephemeral; the repo is the durable review location.
- **Use `pixi run` for all verification.** Bare `python` is the system interpreter (no pyarrow/duckdb/project-deps). Always invoke `pixi run python …` or the relevant `pixi run <task>`. If a dep import fails unexpectedly, confirm the interpreter with `pixi run python -c "import sys; print(sys.executable)"` before concluding a real failure.

## Schema & data conventions

- When adding provenance/metadata columns, match the existing schema's type exactly (e.g., if single-contrast uses string format for `imputed_nodes`, factorial must too).
- Aggregation queries: verify whether stats (std, consistency) should be computed over raw route-level rows or pre-aggregated sender-level values.

## Pipeline-specific gotchas

- **`analysis_mode` flow** — Kedro parameter in `conf/base/parameters.yml`, default `males_only`. `KEDRO_ENV=full_cohort` overlays the sensitivity config. Affects `enrich.py`, `attribute.py`, `mechanism.py` — **not** `normalize.py` (always uses all 72 samples). Legacy `ANALYSIS_MODE` env var was retired and is silently ignored.
- **Outlier detection requires stoichiometry** — `alz/ingest/song.py --outliers` reads `stoichiometry_matrix.csv`, so `normalize.py` must run first. Falls back to total proteome if unavailable.
- **WMB prerequisite** — the bulk pipeline needs `wmb_kinase_expression.csv` / `wmb_proteome_expression.csv`; run `run_wmb_expression.sh` (or `pixi run wmb-export`) first. `run_all.sh` auto-resolves the upstream WMB h5ad + SEA-AD downloads unless `--skip-atlas`.
- **WMB region scope** — `WMB_REGION_SCOPE` defaults to `whole_brain` (correct for the specificity score's brain-wide denominator). `cortex_hpf` is a sensitivity toggle only. Active scope is stamped to `wmb_kinase_expression.scope.json`; a scope mismatch forces recompute.
- **Atlas cache compressed** — raw h5ads under `data/external/allen_abc/` are zstd-compressed (~115 GB → ~26 GB). Decompress with `bash alz/runners/supporting/decompress_atlas_cache.sh` before re-running `wmb_expression.py`. Provenance in `data/external/allen_abc/MANIFEST.json`.
- **WMB expression memory** — `wmb_expression.py --proteome` processes 6,308 genes × 13 regions; use `skip_regional=True`, `chunk_size=2000` to stay under ~30 GB RAM.
- **Stage 6 pY track gating** — `build_celltype_decomposition.py --track py` (or `both`) requires `raw_phospho_normalized_pY.csv` from Stage 1. Smoke runner tolerates missing pY; re-run `pixi run normalize` if you need it.
- **Unified-viewer hard refresh** — `build_unified_viewer.py` inlines PAYLOAD as `<script type="application/json" id="payload-data">` into `index.html`. After `pixi run viewer`, hard-refresh (Ctrl+Shift+R / Cmd+Shift+R) or the cached HTML serves stale data. Quick check: `PAYLOAD.meta.generated_at` in DevTools.
- **API caching** — delete `data/derived/caches/` to force re-fetch from MyGene / Allen Brain Atlas.

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in `henrichung-cst/alzheimers` via the `gh` CLI. External PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, label strings equal to their role names (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
