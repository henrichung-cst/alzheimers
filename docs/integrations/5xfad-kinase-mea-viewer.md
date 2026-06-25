# 5xFAD Kinase MEA and Viewer Notes

> For the live Kinase Explorer Attribution tab contract, see
> [`docs/foundation/kinase_explorer_attribution.md`](../foundation/kinase_explorer_attribution.md).
> This note remains the 5xFAD integration/provenance reference.

This note records the 5xFAD kinase enrichment conventions used by
`alz/cohorts/fivexfad/ingest.py` and the unified viewer 5xFAD kinase tab.

## Scope

5xFAD is treated as a supporting AD cohort in the unified viewer. Cortex and
hippocampus are modeled independently, then exposed as viewer filter dimensions
inside the Mouse (5xFAD) kinase surface.

Primary kinase MEA tracks are:

- IMAC/ST
- pY

Stoichiometry is the primary analysis track. Raw phospho is retained as the
sensitivity track.

## Sample Handling

The ingest manifest records tissue, assay, raw run, age, genotype, biological
sample ID, pool status, duplicate group, analysis action, and sensitivity flag.

Primary MEA uses samples with `analysis_action == "primary"`. Explicit pool runs
and sensitivity-only runs are retained in provenance but not used in primary
TG-vs-WT contrasts. Technical duplicate runs are averaged after log2 transform so
they contribute one biological sample column.

The primary sample counts are retained in contrast QC and viewer payload
provenance. They do not gate MEA contrasts.

## Log Transform Convention

Nonpositive intensity values are set to missing before log2 transform. This
matches the Song and Mukesh code paths and avoids creating artificial extreme
negative log2 values through pseudocount substitution.

The same convention is used for total proteome, IMAC/ST, and pY inputs.

## MEA Contrast Policy

The previous 5xFAD-only `MIN_REPLICATED_GROUP_N` mask has been removed. 5xFAD
now follows the Song/Mukesh convention: if a contrast has observed data and the
shared MEA path can build a ranked site vector, MEA is run. Low sample counts
remain visible as raw `n_wt` / `n_tg` evidence, not as a viewer-facing
under-replication gate.

The shared MEA behavior remains:

- drop missing LFC rows before ranking
- drop sites without kinase-library-compatible motifs
- median-center each contrast's LFC vector
- winsorize at the 1st and 99th percentiles
- skip contrasts only when the ranked site count is below `MEA_MIN_SITES`

These operations are inherited from `alz.bulk_mea.enrich._run_mea`, which is also
used by the Song/Mukesh kinase workflows.

## Viewer Behavior

The Mouse (5xFAD) kinase tab should display the same kinase-viewer structure as
the Song/Mukesh tabs where the data support it. The viewer does not display an
`under_replicated` status or gray out 6-month cells. For the current rebuilt MEA
outputs, all 5xFAD tissue/assay/track combinations contain 3, 6, 9, and 12 month
TG-vs-WT contrasts.

Viewer-facing site labels are compact gene-site labels such as `Atp1a3_S456`
while preserving the original `site_id` in hover/title metadata and detail
sidecars.

5xFAD bulk audit detail sidecars are packaged as compressed per-kinase bundles
under `outputs/reports/unified_viewer/edge_slices/fivexfad_detail/`. The
`supporting_5xfad.detail_shards` map is keyed by kinase, and each
`*.json.gz` bundle contains detail records keyed by
`kinase|tissue|assay|analysis_track`. This keeps the 5xFAD lazy-loading layout
parallel to Song/Mukesh per-kinase sidecars while avoiding thousands of broad
uncompressed JSON files.

## Native snRNA Attribution

5xFAD attribution is now backed by matched 5xFAD snRNA data, not by Song
attribution rows. The attribution builder reads the canonical Seurat object:

`data/datasets/5xFAD/primary/scrna/reclustering/fivex_renamed_from_merged.RDS`

The cell-type identity column is `new_clusters`. `fine_cluster` is not used as
the modeling or viewer spine.

The builder uses `data/datasets/5xFAD/metadata/omics_join_manifest.csv` and keeps
only rows with `per_animal_integration_action == "use"`. The pooled-only
`WildT_06mo_C_11` cortex sample remains excluded from matched per-animal
integration.

Generated attribution artifacts:

- `outputs/reports/kinase_attribution_5xfad/fivexfad_snrna_attribution.csv` (direction: LFC + cell support)
- `outputs/reports/kinase_attribution_5xfad/fivexfad_snrna_expression.csv` (detection + mean log2 expression)
- `outputs/reports/kinase_attribution_5xfad/fivexfad_expression_specificity.csv` (standard detection metric)
- `outputs/reports/kinase_attribution_5xfad/fivexfad_snrna_cell_counts.csv`

The viewer payload exposes 5xFAD-native fields from the repo-wide standard
detection metric (`alz/cross_reference/specificity.py`): `fivexfad_detected`,
`fivexfad_fraction_cells_expressing`, `fivexfad_concentration`,
`fivexfad_concentration_of_total`, `fivexfad_concentration_tier`,
`fivexfad_effective_n`, `fivexfad_top_celltype`, plus `fivexfad_lfc`, snRNA
sample/cell counts, and `cluster_source == "new_clusters"`. The WMB cross-check
likewise uses detection (`wmb_detected`, `wmb_concentration`,
`wmb_concentration_tier`); WMB and SEA-AD remain reference layers joined by
kinase and shared 46-cluster label, not the primary 5xFAD attribution source.

The native 5xFAD snRNA location is the standard detection metric, computed per
tissue. For each kinase, `specificity.compute` runs separately within cortex and
hippocampus over the 46-cluster `new_clusters` spine, pooling ages, genotypes,
and samples inside the tissue: a cell type is detected when the kinase transcript
is present in ≥10% of its cells, and concentration / effective-N are computed on
linear expression over the detected set. TG-vs-WT LFC, p-values, sample counts,
and cell counts remain scoped to tissue x age x cell type. If a tissue x age x
cell-type row has fewer than 3 local snRNA cells across the contrast, the
categorical location confidence is set to `none` for that row. The same local
cell-count gate is applied to
packaged per-cell-type decomposition MEA rows, so sparse cell-type rows do not
drive agreement calls, decomp bars, or `very_high` promotion.

For first-load performance, the compact per-kinase/tissue/age attribution
summaries (table filters and badges) are written to a single whole-list gzipped
sidecar, `edge_slices/fivexfad_attribution_summary.json.gz`, referenced from the
payload as `celltype_attribution_summary_shard` (audit P2). The viewer fetches it
once on first 5xFAD/Crosstable render via `_f5EnsureShardData()` rather than
parsing it inline at startup. Full attribution rows, including long
confidence-basis strings and snRNA sample/cell counts for detail drawers, remain
in per-kinase JSON sidecars under
`outputs/reports/unified_viewer/edge_slices/fivexfad_attribution/`, loaded only
when the Attribution detail tab is opened.

## Per-Cell-Type Decomposition MEA

The 5xFAD attribution tab now includes a native per-cell-type kinase MEA
cross-check, analogous to the Song/Mukesh decomposition layer. This is not a
transcript-only enrichment. It projects 5xFAD raw phosphosite signal with matched
5xFAD snRNA `new_clusters` weights, estimates TG-vs-WT phosphosite effects per
tissue, age, track, and cell type, then runs the same
`alz.bulk_mea.enrich._run_mea` path used by the bulk analyses.

Generated artifacts live under:

- `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_snrna_pseudobulk_linear.csv.gz`
- `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_celltype_mea.parquet`
- `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_celltype_site_level_ols.parquet`
- `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_celltype_substrate_sets.csv`
- `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_celltype_mea_audit.json`

The viewer fields `Decomp NES`, `Decomp FDR`, and `Bulk match` come from this
per-cell-type phosphosite MEA layer. The bulk stoichiometry MEA remains the
anchor for direction and significance; the decomposition layer is a cell-type
cross-check, not a replacement for the bulk 5xFAD result.

The 5xFAD confidence label mirrors the Song mouse convention at the top tier:
native tissue-specific 5xFAD snRNA location evidence is not sufficient by
itself for a confidence pill. During viewer packaging, a 5xFAD row must have
significant bulk MEA and matching snRNA TG-vs-WT direction under the Song LFC
gate before it can receive `moderate` or `high`. Tissue-specific location at
least 2x uniform gives `high`; sub-high location with direction support gives
`moderate`. A `high` row is promoted to `very_high` only when the matched
per-cell-type MEA row agrees in sign with the bulk kinase MEA under the same
decomposition-agreement FDR gate used by the Song attribution model.

For first-load performance, full per-cell-type MEA rows are not embedded in the
payload. The initial block carries `celltype_agreement_index`, a compact
categorical bulk-vs-decomposition agreement index with raw NES/FDR and count
evidence. The compact decomp-bar index (kinase, tissue, track, cell type, age,
NES, FDR, substrate counts) is written to a single whole-list gzipped sidecar,
`edge_slices/fivexfad_celltype_mea_index.json.gz`, referenced as
`celltype_mea_plot_index_shard` (audit P1) and fetched once on first
5xFAD/Crosstable render via `_f5EnsureShardData()` rather than parsed inline at
startup. Full per-kinase decomposition rows are written under
`outputs/reports/unified_viewer/edge_slices/fivexfad_celltype_mea/` and fetched
only for detail fields outside that compact index. Large bulk `leading_substrates`
strings remain in compressed `fivexfad_detail` sidecars rather than
`supporting_5xfad.rows`.

Run commands:

- `pixi run 5xfad-snrna-decomp-pseudobulk` exports the matched 5xFAD snRNA
  pseudobulk matrix used for decomposition weights.
- `pixi run 5xfad-celltype-mea` runs the full tissue x track x cell-type MEA.
- `pixi run 5xfad-celltype-mea-smoke` runs one bounded cortex/pY cell-type
  batch for code-path validation only.
