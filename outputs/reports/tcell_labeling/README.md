# T-cell lineage and exhaustion-state labeling

Authoritative folder for the T-cell state-labeling analysis. Everything here is
produced by scripts under `alz/analysis/`; the report is the single narrative.

## Report
- `tcell_state_labeling_evidence.qmd` / `.html` — the evidence walkthrough.
  Render from this directory: `pixi run quarto render tcell_state_labeling_evidence.qmd`.

## `adt/` — raw CITE-seq evidence and QC
Producer: `tcell_export_marker_cells.R` (same single RDS load as RNA evidence).
- `{donor}_adt_evidence.csv` — CD3/CD4/CD8/TCF1/Ki-67/NCAM1 and isotype
  raw antibody UMI counts, plus RNA/Protein QC metadata.
- donor2 additionally carries raw TOX, BATF, PRDM1/Blimp-1, and GZMB antibody
  UMIs because those antibodies are absent from the donor1 panel.

## `cells/` — authoritative per-cell labels
Producer: `tcell_state_labels.py`.
- `{donor}_state_labels.csv` — CITE-seq CD4/CD8 lineage, definitive biological
  state, separate checkpoint/memory/cytotoxic/cell-cycle categories, raw evidence,
  and ProjecTILs reference corroboration.
- `cluster_context.csv` — native-cluster context and contaminant definitions.

The operational labels include `CD8 exhausted` and `CD8 precursor exhausted` for
this chronic-stimulation experiment. They deliberately do not claim terminal
exhaustion or directly measured functional dysfunction.

## `auroc/` — legacy ProjecTILs audit and RNA evidence input
Producer: `tcell_percell_auroc.py` (+ `tcell_export_marker_cells.R`, runner
`run_tcell_percell_auroc.sh`). Marker sets: `tcell_marker_sets.py`.
- `{donor}_percell_panel_auroc.csv` — historical marker-panel audit
- `{donor}_percell_marker_auroc.csv` — per-gene AUROC
- `marker_genes.txt`, `extract.log`
- `{donor}_marker_cell_expr.csv` — regenerable intermediate (rebuilt by the runner)

## `clusters/` — Seurat clustering (report Steps 3, 5, 6)
Producers: `tcell_cluster_findallmarkers.R`, `tcell_cellcycle_recluster.R`.
- `{donor}_cluster_allmarkers.csv`, `{donor}_cc_recluster_cells.csv` (read by the report)
- `{donor}_cc_recluster_allmarkers.csv`, `{donor}_cluster_annotation.csv`, logs

## `umap/` — native-UMAP views
Producer: `tcell_native_umap_plots.py` (from the coords CSV + embeddings; no `.rds`).
- `{donor}_native_umap_coords.csv` — per-cell UMAP coords + cluster + day (input)
- `{donor}_umap_by_cluster`, `{donor}_umap_by_state_label`
- `{donor}_umap_faceted_by_day`, `{donor}_cluster_day_composition`,
  `{donor}_state_label_by_day_stacked`
- `umap_label_comparison` — original / ProjecTILs / current evidence labels (2×3)

Note: `*_native_umap_coords.csv` derives from the native UMAP in the multi-GB
`.rds`; its exporter is no longer in the tree, so it is kept as an on-disk input.
