# Incytr per-cluster omics integration — plan

Pre-Step-8 revision: switch the Incytr factorial wrapper from bulk-tissue
phospho/proteomics inputs to **cell-type-attributed per-cluster** inputs,
using the 19-cluster Levy spine and the kinase-attribution evidence mask.

## Background

`alz/kinase_attribute.py` now produces a per-cluster evidence table
(`unified_attribution.csv`, `n_kinases × 9 contrasts × 19 clusters`)
keyed on `config.CLUSTER_SPINE`. The Incytr integration still ships
flat bulk omics matrices (one row per gene, one column per animal)
that are used identically regardless of which cluster acts as sender or
receiver. The instruction now is that phospho + proteomics must enter
Incytr **already attributed to cell types**, not bulk.

The upstream `Incytr` R package already supports per-cluster omics:
`integrate_omics_layer_factorial` (in `~/Projects/work/incytr/R/analysis.R`
~L519–L545) calls `resolve_wide(data_wide, role)`, which dispatches on
`is.list(data_wide)` and looks up by cluster name. **No upstream change
is required.** The work is entirely in `alz/integration/`.

## Approach: attribution-as-mask (option b)

Direct deconvolution remains closed (`archive/deconvolution/docs/`).
We do not reconstruct per-cluster intensities. Instead, for each
(omics layer, cluster) pair we produce a genes × animals matrix where:

- Rows whose attribution evidence supports the cluster keep the **bulk
  LFC / intensity** value.
- Rows where the cluster is not supported are set to **NaN**
  (R `NA_real_`), so Incytr's path-scoring skips them rather than
  treating them as zero signal.

"Supported" = the gene (or its kinase substrate context) appears in
`unified_attribution.csv` for that cluster with a confidence tier of
`moderate` or `high`. (Low-confidence rows are masked out.)
The mask is built per omics layer:

- **pr (total proteome)**: per-protein gene symbol. Evidence: WMB
  specificity on the gene itself for the cluster's WMB parent class
  (no kinase context needed).
- **ps / py (phospho)**: per-protein gene symbol (sites already
  collapsed to gene upstream). Evidence: any kinase whose substrate
  set includes a site on this protein and whose unified-attribution
  row for this cluster passes the tier filter.

The mask is a single `gene × cluster` boolean table, materialized once
and applied to all three omics matrices.

## Files to change

| File | Change |
|---|---|
| `alz/integration/export_factorial_inputs.py` | Add `_build_attribution_mask(unified_attribution, kldata)` → `DataFrame[gene × cluster bool]`. Replace `write_omics_bundle` flat-CSV write with `write_per_cluster_omics_bundle` that emits parquet per (layer, cluster) under `data/incytr_factorial_inputs/per_cluster/{pr,ps,py}/{cluster}.parquet`. Drop the flat `{pr,ps,py}_matrix.csv` writes. Bump h5ad cell labels from WMB-34 to Levy-19 (use the barcode-keyed `incytr_cluster_assignments.csv` bridge added in Step 5, not `obs["subclass_name"]`). |
| `alz/integration/omics_loaders.py` | Add `apply_cluster_mask(matrix_df, mask_df, cluster)` helper returning a NaN-masked copy. Keep the existing bulk loaders — the bulk matrix is the input to masking. |
| `alz/integration/load.R` | Replace `pr_mat <- read.csv(...)` etc. with a loop over `per_cluster/{layer}/*.parquet` (use `arrow::read_parquet`), build `list(cluster_name = matrix, ...)` per layer, and return `list(pr = list(data_wide = <list>), ps = ..., py = ...)`. |
| `alz/integration/factorial.R` | No change — already wraps `inputs$pr_mat` as `list(data_wide = ...)`. The `data_wide` value just becomes a list instead of a matrix. |
| `alz/integration/config_integration.py` | Add `PER_CLUSTER_OMICS_DIR` path constant. Switch the cell-type label set to `config.CLUSTER_SPINE` (and remove any WMB-34 lookups). |
| `alz/integration/views.sql` | Audit aggregation views for hard-coded WMB-34 labels; rewrite cluster filters to use the 19-cluster spine. |
| `pixi.toml` | No change (pyarrow already pinned). |

Concurrent rename in `export_factorial_inputs.py` (Step 8 proper):
the h5ad → metadata export already needs to switch from `subclass_name`
to the barcode-keyed Levy assignment file; bundling this with the
per-cluster omics change keeps the cell-label and omics-label spines
in sync in one commit.

## Verification

1. `export_factorial_inputs.py` writes `per_cluster/{pr,ps,py}/` with
   exactly 19 parquet files per layer, named after `CLUSTER_SPINE` entries.
2. Spot-check: for cluster `Microglia`, the `pr` matrix has finite values
   on classical microglial markers (Csf1r, Tyrobp, Trem2) and `NaN` on
   neuronal-restricted genes (Snap25, Slc17a7) absent from the attribution.
3. `load.R` reads all three layers into `list(data_wide = list(<19
   clusters>))` shape; no missing clusters.
4. `pixi run incytr-factorial` runs end-to-end on the new inputs; the
   number of scored (sender, receiver) cluster pairs == 19 × 19 = 361.
5. Compare a sample of pathway scores to the previous bulk-input run and
   confirm the per-cluster outputs differ where the mask zeroed evidence
   (i.e., the mask is doing real work and isn't a no-op).

## Out of scope

- Reopening direct deconvolution (option a)
- Changing the OLS design matrix or animal intersection
- Modifying upstream `Incytr` R sources (none required)
- Pseudobulk / DEG list construction (unchanged; that layer remains
  transcript-only and was rewired to Levy-19 in Step 5/6)

## Sequencing

This plan replaces what was Step 8 ("rewire export_factorial_inputs.py
for Levy labels") with a wider Step 8:

- **8a** Switch h5ad cell labels in `export_factorial_inputs.py` from
  `subclass_name` to barcode-keyed Levy cluster (already-planned work).
- **8b** Build the gene×cluster attribution mask from
  `unified_attribution.csv`.
- **8c** Replace flat omics writes with per-cluster parquet writes.
- **8d** Update `load.R` reader.
- **8e** Update `config_integration.py` and `views.sql` to Levy-19.
- **8f** End-to-end smoke run; verify the 361 pair count.

Step 9+ (downstream consumer validation, factorial re-run, docs refresh)
proceeds as previously sequenced.
