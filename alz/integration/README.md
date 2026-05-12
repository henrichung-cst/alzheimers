# Incytr Integration

A thin AD-specific wrapper around the upstream `incytr` R package
(`~/Projects/work/incytr/`). All math, scoring, and pathway construction
lives in the package; this directory only loads AD inputs, persists
results to parquet, and defines aggregation views.

## Files

| File | Role |
|---|---|
| `export_factorial_inputs.py` | Reads the Song h5ad, filters to males x 4 genotypes x 3 timepoints, writes the on-disk contract `load.R` consumes. Also writes pseudobulk raw counts (`pseudobulk_counts.mtx` + sidecars) for DESeq2 seed-list DE. |
| `compute_seed_lists.R` | Per-cell-type DESeq2 on pseudobulk + limma on `pr_matrix.csv` → `deg_lists.json` + `prg_list.csv`. Loaded by `load.R` when present; consumed by upstream `construct_factorial_paths` for DEG/prG `.label` assignment. See `docs/integrations/seed_list_labels.md`. |
| `factorial.R` | Entry point. Asserts the production engine is present (`Incytr::construct_factorial_paths`, `Incytr::score_factorial_paths`), then loads inputs, runs the engine, writes per-receiver parquet. |
| `load.R` | AD data loader. Reads `expression_*.csv`/`.mtx`, `animal_metadata.csv`, and (when present) `deg_lists.json`/`prg_list.csv` produced by the exporter and seed-list step. |
| `persist.R` | Hive-partitioned parquet writers + `views.sql` copy. |
| `views.sql` | DuckDB views over `receiver_cache/` for backbone provenance, contrast comparison, temporal dynamics, hub matrix, backbone recurrence, target convergence. |
| `run_factorial.sh` | One-line `Rscript` shim invoked by `pixi run incytr-factorial`. |
| `config_integration.py` | Filter values, design columns, contrast vectors, paths. Read by `export_factorial_inputs.py`. |

## How to run

```bash
pixi run install-incytr             # builds + installs ../incytr
pixi run export-factorial-inputs    # h5ad -> data/incytr_factorial_inputs/
pixi run compute-seed-lists         # DESeq2 DEGs + limma prGs (optional, gates on pseudobulk)
pixi run incytr-factorial           # runs the factorial engine (uses seed lists if present)
```

The R wrapper hard-fails before loading any data if the upstream package
does not export `construct_factorial_paths()` and `score_factorial_paths()`.
Until those land in `../incytr` with the constrained pathway construction
and scale guardrails described in `docs/incytr_remediation_plan.md`,
`pixi run incytr-factorial` will refuse to start.

## What's not here

The legacy R wrappers, Python adapters, sidecars, and orchestrator
shell scripts that previously lived under `alz/integration/wrappers/`
and `alz/integration/adapters/` were retired during the rewrite.
Recover them from git history if needed; the current architecture
treats them as the bug, not the baseline. See
`docs/incytr_remediation_plan.md` for the rationale.
