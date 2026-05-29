# `alz/decomposition_mea/` — per-cluster proportional decomposition + MEA

Stage 6–7 of the main pipeline. Converts bulk phospho/proteome into per-cluster
substrates via a forward projection onto snRNA-derived proportions
(`P_c = f_c × bulk`), then runs per-cluster factorial OLS + MEA.

Does NOT perform statistical (inverse-problem) deconvolution — that path is
closed; see `archive/deconvolution/docs/deconvolution_infeasibility.md` and
`docs/foundation/analysis_charter.md`. The WMB-class statistical-deconvolution +
factorial-OLS cluster that used to live here was deleted 2026-05-29 (closed path).

## File inventory

| File | Role | Entry point |
|---|---|---|
| `build_celltype_decomposition.py` | Stage 6: project bulk phospho (pS/pT, pY) and protein onto `f_c` per-cell-rate weights; writes `protein_per_cluster.parquet`, `phospho_per_cluster{,_pY}.parquet`, `decomposition_audit.json` | `python alz/decomposition_mea/build_celltype_decomposition.py --spine levy_t5 --track both` |
| `enrich_celltype.py` | Stage 7: per-cluster factorial OLS + GSEA MEA on the projected phospho cube; writes `mea_per_cluster{,_pY}.parquet`, `site_level_ols_per_cluster{,_pY}.parquet`, and CSV sidecars | `python -m alz.decomposition_mea.enrich_celltype --spine levy_t5 --track st` |
| `build_per_animal_site_ols.py` | Publisher: unions st + py OLS into `per_animal/site_level_ols.parquet`; renames `cluster` → `cell_type`; consumed by the unified viewer | `python alz/decomposition_mea/build_per_animal_site_ols.py --spine levy_t5` |
| `verify_decomposition.py` | Verification harness (4 contracts: mass identity, coverage, per-cluster vs bulk MEA, Incytr pair count); exits non-zero on failure | `python alz/decomposition_mea/verify_decomposition.py --spine levy_t5` |

## Key invariants

**Mass identity** — `Σ_c [P_c × (N_c / N_total)] ≈ bulk`, not `Σ_c P_c = bulk`.
The `f_c` weight is a per-cell-rate (`share_c × N_total / N_c`), so literal
summation overshoots. Verified by the `mass` check (threshold `max_rel_err < 1e-6`).

**Sign convention** — `+` = up in disease, matching `bulk_mea` NES/β and Incytr PDS/sclog2FC.

**Spine** — `levy_t5` (31 clusters). Spine definition lives in
`alz/integration/config_integration.py` (`load_cluster_spine()`).

**Track vocabulary** — `st` = IMAC pS/pT (suffix `""`), `py` = pY (suffix `"_pY"`).
pY requires `raw_phospho_normalized_pY.csv` from Stage 1; tolerates missing pY gracefully.

## Upstream prerequisites

1. `alz/reference/snrna_proportions.py --run --spine levy_t5` — produces `proportions.parquet`
2. `alz/bulk_mea/normalize.py` — produces `total_proteome_normalized.csv` and `raw_phospho_normalized*.csv`

## Downstream consumers

- `alz/integration/build_normalized_substrate.py` — reads `protein_per_cluster.parquet`, `phospho_per_cluster*.parquet`
- `alz/integration/build_omics_trace.py` — reads same
- `alz/build_unified_viewer.py` — reads `mea_per_cluster.parquet`, `per_animal/site_level_ols.parquet`
- `alz/viewer/paths.py` — path constant for `per_animal/site_level_ols.parquet`

## Runner scripts

| Script | Covers |
|---|---|
| `alz/runners/main/rerun_decomposition_chain.sh` | Full chain: pseudobulk → proportions → Stage 6 → Stage 7 (st+py) → publisher → verify |
| `alz/runners/main/run_all.sh` (steps D-decomp … D-verify) | Same chain inside the full pipeline |
| `alz/runners/main/run_pair_mode_pipeline.sh` | Runs Stage 6–7 as part of the pair-mode pipeline |
