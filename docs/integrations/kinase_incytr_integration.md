# Kinase ↔ Incytr Integration

**Status (2026-05-19):** Pair-mode on the **levy_t5** spine (31 clusters, `min_cells = 5`, no rank gate) is the active integration path. The factorial Incytr engine was archived on 2026-05-18 (`archive/incytr_factorial_2026-05-18/`) — its plumbing is preserved on disk but is no longer reachable from any pixi task or runner. Upstream `Incytr::construct_factorial_paths` / `score_factorial_paths` were deleted at commit `424119f`; the current production entry point is `Incytr::Cal_pairwise_grid` (`~/Projects/work/incytr/R/grid.R`).

The factorial-era version of this document is preserved at `docs/archive/kinase_incytr_integration_factorial_era.md`. The plan that called for its rebuild is at `docs/archive/incytr_remediation_plan.md` (superseded — factorial was deleted, not rebuilt).

## Architecture

```
            kinase pipeline                          snRNA pipeline
                  │                                        │
    alz/decomposition/build_celltype_decomposition.py      │
                  │  (per-cluster phospho + protein         │
                  │   projected onto levy_t5 spine)         │
                  ▼                                        │
    alz/decomposition/enrich_celltype.py                    │
                  │  (per-cluster factorial OLS +           │
                  │   MEA, 9 contrasts, NaN where           │
                  │   rank-deficient)                       │
                  ▼                                        ▼
    bench/incytr_pair_levy_t5/incytr_input/  ◄── alz/integration/build_yuyu_kldata.py
                  │                          ◄── alz/integration/build_cluster_spine.py
                  ▼
    Incytr::Cal_pairwise_grid   (upstream, ~/Projects/work/incytr/R/grid.R)
                  │
                  ▼
    bench/incytr_pair_levy_t5/output/contrast=<C>/data.parquet
                  │  (9 wide parquets, 31² = 961 rows each)
                  ▼
    alz/integration/pair_to_receiver_cache.py
                  │
                  ▼
    outputs/reports/unified_viewer/  (consumed by alz/build_unified_viewer.py)
```

## In-tree files (`alz/integration/`)

| File | Role |
|---|---|
| `config_integration.py` | Filter values, design columns, contrast vectors, paths. `load_cluster_spine()` is the single source of truth for the 31-cluster levy_t5 spine; consumed by `alz/snrna_proportions.py`, `alz/decomposition/verify_decomposition.py`, and the viewer. |
| `build_cluster_spine.py` | Run-once generator: builds the levy_t5 31-cluster spine CSV from the Levy lab cluster key + barcode-to-cluster table. Outputs to `data/incytr_frozen/v2_46clusters/spines/<name>/cluster_spine.csv`. |
| `extract_cluster_assignments.R` | Run-once generator: emits `barcode_to_cluster.csv` and `cell_metadata.csv` from the legacy `incytr_obj.rds`. |
| `plot_cluster_spine.py` | Diagnostic plots over `cluster_spine.csv`. |
| `build_seaad_bridge.py` | One-shot: hand-curated `cluster_to_seaad_supertype.csv` (levy_t5 cluster → SEA-AD supertype). Direct crosswalks only — no chained mappings through intermediate vocabularies. |
| `build_yuyu_kldata.py` | Builds the `kldata_pspy.csv` consumed by pair-mode (`bench/incytr_pair_levy_t5/incytr_input/kldata.csv`). |
| `pair_to_receiver_cache.py` | Reshapes pair-mode wide outputs (9 contrasts × 961 rows) into the long-form `receiver_cache/` layout the unified viewer consumes. Invoked by `alz/runners/main/run_pair_mode_viewer_build.sh`. |

## Entry points

| Command | Purpose |
|---|---|
| `bash alz/runners/main/run_pair_mode_pipeline.sh` | End-to-end: per-cluster decomposition → enrich → pair-mode inputs → `Cal_pairwise_grid` → reshape. |
| `bash alz/runners/main/run_pair_mode_viewer_build.sh` | Reshape only: pair-mode parquet → `receiver_cache/`. |
| `bash bench/run_pair_mode.sh` | Bench wrapper for `bench/incytr_pair_levy_t5/` (Incytr invocation in isolation). |
| `pixi run install-incytr` | Reinstall upstream `Incytr` after changes in `~/Projects/work/incytr/`. |

R dependencies still required: `Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`. Integration config lives in `alz/integration/config_integration.py`, not `alz/config.py`.

## Invariants

- **31² = 961 sender × receiver pairs per contrast** — must hold for every one of the 9 contrasts. A drop indicates either a spine-rebuild bug or a silent filter in the upstream call.
- **9 contrasts** — disease × timepoint (3 diseases × 3 timepoints; const + App + Tau + Int + time_4mo + time_6mo + App×time4 + App×time6 + Tau×time4 + Tau×time6, contrast list in `config_integration.py`).
- **Rank-deficient clusters emit NaN, not silent drops** — preserves the 31-cluster spine in every output. Verify with `python alz/decomposition/verify_decomposition.py --spine levy_t5 --all`.
- **Pair-mode pvalue is untrustworthy** — filter / rank pathways on `|PDS|`, not pvalue. (Reason: pair-mode permutation pvalues conflate enumeration biases with signal; see memory note `project_incytr_pair_pvalue_untrustworthy`.)
- **Direct levy_t5 crosswalks only** — reference annotations (WMB, SEA-AD, HBCA, Song) map *directly* to levy_t5 clusters. No chained mappings through intermediate vocabularies.

## Upstream surface

- `~/Projects/work/incytr/R/grid.R` — `Cal_pairwise_grid` (production entry point), shared path + expression cache, optional DuckDB engine.
- `~/Projects/work/incytr/R/kinases.R` — `Integr_kinasedata`, `Kinase_exploration`, `Integr_multiomics`. Existing integration hooks for kinase/multi-omics evidence (the natural surface for MEA enrichment imputation — design TBD).
- `~/Projects/work/incytr/R/evaluation.R` — `Cal_SigProb`, `Cal_PDS`, `Cal_scFC`, `Pathway_evaluation`.

## Pointers

- `alz/integration/README.md` — file-by-file layout (this doc is the architectural view).
- `README.md` — repo-level overview and the 2026-05-18 archival note.
- `CLAUDE.md` — "Integration Code (Incytr)" section + Gotchas.
- `docs/incytr_deconvolution_pivot.md` — rationale for the levy_t5 spine and the move to forward projection (P_c = f_c × bulk).
- `docs/plans/change_request_02_spine_rethreshold.md` — current spine parameters (`min_cells = 5`, no rank gate).
