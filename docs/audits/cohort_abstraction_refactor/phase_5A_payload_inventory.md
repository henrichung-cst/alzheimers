# Phase 5A — Payload Contract Inventory

**Date:** 2026-06-17
**Branch:** `refactor/cohort-namespaces`
**Scope:** Read-only survey of `build_unified_viewer.py`, `build_tcell_viewer.py`,
the viewer/tcell_viewer frontend templates, existing validators, and
`docs/foundation/viewer_payload_contract.md`. No source edits.

---

## 1. Unified Viewer — Top-Level Payload Keys

All keys emitted by `build_payload()` (`build_unified_viewer.py:4654`). The
`payload` dict is assembled at lines 4920–4960.

### Master Table

| Top-level key | Sub-keys (representative) | Builder function(s) + line range | Cohort owner | Frontend consumers |
|---|---|---|---|---|
| `meta` | `schema_version`, `viewer_payload_schema_version`, `generated_at`, `cohort`, `default_context`, `contexts[]`, `capabilities{}`, `contrasts`, `diseaseGroups`, `timepoints`, `diseaseColors`, `familyMap`, `wmb_uniform`, `song_uniform`, `transcript_trace`, `omics_trace`, `omics_trace_normalized` | `build_payload()` 4693–4758 (inline) + `ensure_transcript_trace_sources()` 483, `ensure_omics_trace_sources()` 676, `ensure_omics_trace_normalized_sources()` 707 | shared/composer | `01_state.js:49` (`META = PAYLOAD.meta`); `ViewerPayload._meta()`; `evidence_row.js:120,251,687`; `transcript_trace.js:42,114`; `kinase_crosstable.js:47,52`; `kinase_audit.js:561` |
| `kinases` | `by_context.song_ad.{id, name, gene_symbol, residue_type, trajectory, peak_contrast, peak_NES, n_sig_contrasts, top_celltype_1..3, NES_<contrast>, FDR_<contrast>, ...}` | `_build_kinases_slice()` 1016–1068; wrapped by `_as_single_context_block()` 3000 | **song** (bulk MEA kinase activity) | `ViewerPayload.kinases()`; `01_state.js:51`; `kinase_explorer.js`; `kinase_audit.js`; `kinase_wiring.js:47`; `temporal_v2.js` |
| `kinase_motifs` | `{<name>: {kin_type, positions, amino_acids, matrix, st_fav}}` — one entry per kinase name | `_build_kinase_motifs()` 4608–4651 | shared/cross-cutting (union of song + human + 5xFAD kinase names) | `kinase_audit.js:1293`; `kinase_fivexfad.js:1478`; `kinase_human.js:559` |
| `celltypes` | `by_context.song_ad.{id, name, tissue_category}` | `_build_celltypes_slice()` 2991–2997; wrapped by `_as_single_context_block()` 3000 | **song** (Levy-t5 31-cluster spine) | `ViewerPayload.celltypes()`; `01_state.js:51` |
| `kinase_celltype_evidence` | columnar: `kinase_id, cell_type, confidence_tier, song_specificity, song_tau, wmb_fold, sea_ad_lfc, song_lfc, decomp_nes, decomp_fdr, ...` (~20 columns) | `build_payload()` 4760–4806 (inline from `data.celltype_evidence`) | **song** (from `celltype_evidence_table.csv`) | `kinase_explorer.js:293`; `kinase_crosstable.js:473` |
| `attribution_index` | columnar: `kinase_id, contrast_id, cell_type, confidence_tier, wmb_specificity, song_specificity, sea_ad_lfc, decomp_nes, nes, fdr, ...` (~30 columns) | `build_payload()` 4808–4880 (inline from `data.unified_attribution_full`) | **song** (from `unified_attribution_full.csv` or `unified_attribution.csv`) | `kinase_explorer.js:39,136,232`; `kinase_crosstable.js:184,957`; `kinase_wiring.js:47`; `temporal_v2.js:15`; `04_slice_cache.js:815` |
| `decomposition_index` | columnar: `kinase_id, contrast_id, cell_type, decomp_nes, decomp_fdr` | `build_payload()` 4885–4895 (inline from `data.decomposition`) | **song** (from `levy_t5/mea_per_cluster.parquet`) | `kinase_explorer.js:304`; `kinase_crosstable.js:171`; `temporal_v2.js:48` |
| `agreement_index` | columnar: `kinase_id, contrast_id, state, bulk_nes, bulk_fdr, top_cell, top_cell_nes, top_cell_fdr, n_cells_match, n_cells_oppose, _state_codes` | `_build_agreement_index()` 3064–3163 | **song** (bulk MEA vs decomp MEA) | `kinase_explorer.js:319` |
| `subclass_breakdown` | `{<kinase_id>: {<wmb_class>: <tooltip_string>}}` | `_build_subclass_breakdown()` 3008–3051 | **song** (from `wmb_kinase_expression_subclass.csv`) | `kinase_audit.js:737` |
| `audit_tables` | `{preview_rows, tables: {<key>: {key, label, type, row_count, columns, preview, relative_path, source_path}}, measurement_trace}` | `build_audit_manifest()` 808–845 (using `_audit_specs()` 161–200; `ensure_measurement_trace_sources()` 483) | **song** (AD pipeline outputs; 5xFAD manifest items added at lines 178–184) | `01_state.js:309,313`; `kinase_detail.js:2` |
| `edge_slice_ref` | `{schema_version, decomp_ols_url, decomp_ols_index, n_decomp_ols_slices, present_decomp_ols_kinase_ids, incytr_pathways_url, incytr_pathways_index, song_concordance_url, song_concordance_index, present_song_concordance_genes, human_perdonor_url, human_perdonor_index, present_human_perdonor_kinase_ids}` | `build_payload()` 4930–4948 (inline); `present_human_perdonor_kinase_ids` updated at 4954 | shared (lazy-shard URL map) | `04_slice_cache.js:15` (`ESR = PAYLOAD.edge_slice_ref`) |
| `incytr_pathways` | `by_context.song_ad.{schema_version, version, source_mode, contrasts, senders, receivers, celltype_qc, low_signal_celltypes, heatmap_counts, heatmap_counts_signed, pathway_counts, pathway_counts_low_signal_excluded, slice_index, score_columns, label_columns, label_nodes, label_vocab, direction_flag_columns, path_metric_columns, global_index, gene_node_index, trajectory_summary, empty_deg_celltypes}` | `_write_incytr_pair_pathways()` 3850–4574; wrapped by `_as_single_context_block()` 3000 | **song** (pair-mode Incytr outputs) | `ViewerPayload.incytr()`; `ViewerPayload.incytrSliceIndex()`; `incytr_heatmap.js`; `incytr_pathways.js`; `incytr_state.js` |
| `human` | `{schema_version, kinases: {id, name, gene_symbol, residue_type, n_donors_sig, n_donors_up, n_donors_down, n_donors_tested, median_nes, sea_ad_lfc, NES_<donor>_vs_CTRLmean, FDR_<donor>_vs_CTRLmean, ...}, donors, ctrl_donors, contrasts, perdonor_index, global_shift, winsor, sites, donors_all, case_donors, recurrence_ctrl, stoich_by_site, raw_phospho_by_site, [celltype_specificity]}` | `build_human_slice()` 1188–1554 (reads from `HUMAN_PERDONOR_DIR`; optional `build_celltype_specificity_payload()`) | **mukesh** (human NBB per-donor MEA) | `01_state.js:53,59` (`HAS_HUMAN`, `_KH`); `kinase_crosstable.js:220` (`PAYLOAD.human`); `kinase_human.js:165,559` |
| `supporting_5xfad` | `{schema_version, cohort, role, filters, rows, celltype_attribution_summary_index, celltype_attribution_shards, celltype_agreement_index, celltype_mea_plot_index, celltype_mea_shards, contrast_qc, sample_counts, detail_shards, celltype_ols_shards, source_files}` | `build_supporting_5xfad_slice()` 2812–2983 | **fivexfad** | `01_state.js:54` (`HAS_FIVEXFAD`); `kinase_crosstable.js:288` (`PAYLOAD.supporting_5xfad`); `kinase_fivexfad.js:49` |

**Total top-level keys emitted by the unified viewer:** `meta`, `kinases`, `kinase_motifs`, `celltypes`, `kinase_celltype_evidence`, `attribution_index`, `decomposition_index`, `agreement_index`, `subclass_breakdown`, `audit_tables`, `edge_slice_ref`, `incytr_pathways`, and optionally `human` and `supporting_5xfad`.

---

## 2. Lazy Shard Families — Unified Viewer

### 2A. `edge_slices/decomp_ols/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/decomp_ols/{kinase_id:03d}.parquet` |
| Index file | `edge_slices/decomp_ols/index.json` |
| Writer function | `_write_decomp_ols_slices()` lines 3184–3322 |
| Index entry shape | `{schema_version, slice_count, present_kinase_ids: [int, ...], filename_template: "{kinase_id:03d}.parquet", n_total_rows}` |
| Payload pointer | `edge_slice_ref.decomp_ols_url` + `edge_slice_ref.present_decomp_ols_kinase_ids` |
| Shard columns | `contrast_id, cell_type, site_id, gene_symbol, motif, lfc, se, pval, track` (float32 via `_to_float32_estimable`) |
| Source | `outputs/reports/decomposition/levy_t5/per_animal/site_level_ols.parquet` |
| Cohort owner | **song** |
| Frontend fetch site | `04_slice_cache.js` (`loadDecompOls`); consumed by Attribution drawer |

### 2B. `edge_slices/song_concordance/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/song_concordance/{GENE}.parquet` |
| Index file | `edge_slices/song_concordance/index.json` |
| Writer function | `_write_song_concordance_slices()` lines 3325–3417 |
| Index entry shape | `{schema_version, slice_count, present_genes: [str, ...], filename_template: "{gene}.parquet", n_total_rows}` |
| Payload pointer | `edge_slice_ref.song_concordance_url` + `edge_slice_ref.present_song_concordance_genes` |
| Shard columns | `gene_symbol, cell_type, contrast, song_lfc, song_se, song_pval, song_fdr, n_animals` |
| Source | `outputs/reports/kinase_attribution/song_concordance.csv` (~210 MB; not loaded into memory — read via pandas usecols+gene filter) |
| Cohort owner | **song** |
| Frontend fetch site | `04_slice_cache.js` (`loadSongConcordance`); consumed by Attribution drawer |

### 2C. `edge_slices/incytr_pathways/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/incytr_pathways/{sanitized_sender}__{sanitized_receiver}.parquet` |
| Index file | `edge_slices/incytr_pathways/index.json` |
| Global index binary | `edge_slices/incytr_pathways/incytr_index.bin.gz` |
| Writer function | `_write_incytr_pair_pathways()` lines 3850–4574 |
| Index entry shape | `{schema_version, filename_template: "{sender}__{receiver}.parquet", sanitize_rule, present: [[sender,receiver],...], n_total_rows, pair_row_counts}` |
| Payload pointer | `incytr_pathways.by_context.song_ad.slice_index`; `incytr_pathways.by_context.song_ad.global_index` |
| Shard columns | `Ligand, Receptor, EM, Target, contrast, pvalue(f32), PDS(f16), TPDS, PPDS, PhPDS_ps, PhPDS_py, SiK_score, dir_flag_cols(pr_up etc.), log2FC, Ligand_sclog2FC..., Ligand_label..., traj_labels, sign_vec` |
| Global index columns | `PDS(f4), pvalue(f4), TPDS(u2), PPDS(u2), PhPDS_ps(u2), PhPDS_py(u2), SiK_score(u2), ligandId(u2), receptorId(u2), emId(u2), targetId(u2), senderId(u1), receiverId(u1), contrastId(u1), labelBits(u1), trajBits(u1)` |
| Source | `outputs/reports/incytr_pair_mode/wide/*_incytr_output.parquet` |
| Cohort owner | **song** |
| Frontend fetch site | `04_slice_cache.js` (`loadIncytrPair`); `incytr_pathways.js`; `incytr_global_index.js` reads the `.bin.gz` |

### 2D. `edge_slices/human_perdonor/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/human_perdonor/{kinase_id:03d}.parquet` |
| Index file | `edge_slices/human_perdonor/index.json` |
| Writer function | `_write_human_perdonor_substrate_slices()` lines 1078–1134 |
| Index entry shape | `{schema_version, slice_count, present_kinase_ids: [int,...], filename_template: "{kinase_id:03d}.parquet", n_total_rows}` |
| Payload pointer | `edge_slice_ref.human_perdonor_url` + `edge_slice_ref.present_human_perdonor_kinase_ids` |
| Shard columns | `donor, leading_substrates, substrate_motifs, substrate_kl_percentiles` |
| Source | `HUMAN_PERDONOR_DIR` CSVs (`ingest_mukesh_perdonor.py` output) |
| Cohort owner | **mukesh** |
| Frontend fetch site | `04_slice_cache.js` (`loadHumanPerdonorSubstrate`); consumed by human kinase Audit drawer (Trace + Running Enrichment sub-tabs) |

### 2E. `edge_slices/fivexfad_detail/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/fivexfad_detail/{kinase}.json.gz` |
| Writer function | `_write_fivexfad_detail_shards()` lines 2521–2811 |
| Payload pointer | `supporting_5xfad.detail_shards` (`{kinase: "edge_slices/fivexfad_detail/<kinase>.json.gz"}`) |
| Shard contents | gzip JSON bundle keyed by `"<kinase>|<tissue>|<assay>|<analysis_track>"` — full per-row detail including `leading_substrates`, running enrichment arrays, site-level scores |
| Cohort owner | **fivexfad** |
| Frontend fetch site | `kinase_fivexfad.js` (detail drawer on kinase row click) |

### 2F. `edge_slices/fivexfad_celltype_mea/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/fivexfad_celltype_mea/{kinase}.json` |
| Writer function | `_write_fivexfad_celltype_mea_shards()` lines 2244–2283 |
| Payload pointer | `supporting_5xfad.celltype_mea_shards` |
| Shard contents | Full per-cell-type MEA rows (NES, FDR, ES, substrates, running enrichment, site-level OLS) for one kinase across tissues/tracks/ages |
| Cohort owner | **fivexfad** |
| Frontend fetch site | `kinase_fivexfad.js` (cell-type detail view) |

### 2G. `edge_slices/fivexfad_celltype_ols/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/fivexfad_celltype_ols/{kinase}.json` |
| Writer function | `_write_fivexfad_celltype_ols_shards()` lines 2284–2376 |
| Payload pointer | `supporting_5xfad.celltype_ols_shards` |
| Cohort owner | **fivexfad** |
| Frontend fetch site | `kinase_fivexfad.js` (OLS detail panel) |

### 2H. `edge_slices/fivexfad_attribution/`

| Field | Value |
|---|---|
| Path pattern | `outputs/reports/unified_viewer/edge_slices/fivexfad_attribution/{kinase}.json` |
| Writer function | `_write_fivexfad_attribution_shards()` lines 2038–2076 |
| Payload pointer | `supporting_5xfad.celltype_attribution_shards` |
| Shard contents | Full attribution rows including long evidence-basis strings, cell/sample counts |
| Cohort owner | **fivexfad** |
| Frontend fetch site | `kinase_fivexfad.js` (Attribution detail tab) |

### 2I. Measurement-trace / transcript-trace / omics-trace (within audit_sources/)

These are **not** edge_slices but are lazy-loaded sidecar trees. They are
referenced by `meta.transcript_trace`, `meta.omics_trace`, and
`meta.omics_trace_normalized` (not by `edge_slice_ref`).

| Family | Directory | Writer function | Payload pointer | Cohort |
|---|---|---|---|---|
| measurement_trace | `audit_sources/measurement_trace/` | `ensure_measurement_trace_sources()` 483–645 | `meta.transcript_trace` (also `audit_tables.measurement_trace`) | **song** |
| transcript_trace | `audit_sources/transcript_trace/` | `ensure_transcript_trace_sources()` 646–675 | `meta.transcript_trace` | **song** (Incytr pseudobulk substrate) |
| omics_trace | `audit_sources/omics_trace/` | `ensure_omics_trace_sources()` 676–706 | `meta.omics_trace` | **song** |
| omics_trace_normalized | `audit_sources/omics_trace_normalized/` | `ensure_omics_trace_normalized_sources()` 707–735 | `meta.omics_trace_normalized` | **song** |

Frontend consumers: `transcript_trace.js:41-42`; `evidence_row.js:52-53`.

---

## 3. T-Cell Viewer Payload

`build_tcell_payload()` at `build_tcell_viewer.py:2266`. The T-cell payload is a
**fully independent deliverable** — separate builder, separate output dir
(`outputs/reports/tcell_viewer/`), separate pixi task.

### 3A. Top-Level Keys

| Top-level key | Sub-keys | Builder function + line range | Cohort owner | Notes |
|---|---|---|---|---|
| `meta` | `schema_version=2`, `cohort="tcell"`, `default_context="donor1"`, `contexts[donor1,donor2]`, `capabilities`, `donors`, `donors_with_mea`, `contrasts`, `timepoints`, `timepoint_color_map`, `familyMap`, `fdr_threshold`, `mea_kinase_donor`, `tcell_attribution_uniform`, `tcell_attribution_caveat`, `transcript_trace`, `omics_trace`, `notes` | `build_tcell_payload()` 2430–2467 (inline) | **tcell** | Two contexts; differs from unified: `timepoint_color_map`, `mea_kinase_donor`, `tcell_attribution_*` keys |
| `kinases` | `by_context.{donor1,donor2}.{id, name, gene_symbol, residue_type, trajectory, peak_contrast, NES_<contrast>, FDR_<contrast>, ...}` | `_build_donor_kinases_slice()` 610–753 (per donor) | **tcell** | donor2 gets an empty-slice placeholder (same column set, zero rows) |
| `kinase_motifs` | `{<name>: {...}}` (union of both donors' kinase names) | `_build_kinase_motifs()` 442–500 | shared (tcell-scoped) | Same structure as unified viewer |
| `celltypes` | `{id, name, by_context: {donor1: {id,name,tissue_category,available_donors}, donor2: {...}}}` | `_build_celltypes_slice()` 769–787 + inline in `build_tcell_payload()` 2315–2324 | **tcell** | `available_donors` field unique to T-cell (not in unified viewer) |
| `celltype_assignment` | `{states:[...], by_context: {donor1: {n_kept, state_totals, confidence_by_state, state_by_day, embedding}, donor2: {...}}}` | `_build_celltype_assignment()` 789–903 | **tcell** | T-cell specific (ProjecTILs); no equivalent in unified viewer |
| `audit_tables` | Same schema as unified viewer | `build_tcell_audit_manifest()` 1945–2001 | **tcell** | T-cell specific audit sources; uses `_tcell_audit_specs()` 1528 |
| `edge_slice_ref` | `{schema_version, incytr_pathways_url, incytr_pathways_index}` | `build_tcell_payload()` 2475–2479 (inline) | **tcell** | Slimmer than unified (no decomp_ols, no human_perdonor, no song_concordance) |
| `incytr_pathways` | `{schema_version, version, source_mode="pair_mode_tcells", donors, contexts, by_context.{donor1,donor2}: {slice_index, senders, receivers, contrasts, heatmap_counts, pathway_counts, global_index, ...}, senders, receivers, contrasts, score_columns, label_columns, ...}` | `_write_tcell_pair_pathways()` 1449–1497 (outer); `_write_donor_pair_pathways()` 910–1448 (per donor) | **tcell** | Key diff: shard filename template is `{context}__{sender}__{receiver}.parquet` (3-part vs 2-part for AD) |
| `attribution_index` | columnar: `kinase_id, contrast_id, cell_type, tcell_specificity, tcell_tier, tcell_lfc, tcell_concordance, tcell_concordant, tcell_consistency, nes, fdr` | `_build_tcell_attribution_index()` 573–607; wired in `_build_donor_kinases_slice()` | **tcell** | Emitted only when donor1 MEA is present; different schema from unified viewer's `attribution_index` (tcell-specific columns) |

### 3B. T-Cell Shard Family

| Family | Path pattern | Writer | Payload pointer | Shard filename template |
|---|---|---|---|---|
| `edge_slices/incytr_pathways/` | `outputs/reports/tcell_viewer/edge_slices/incytr_pathways/{donor}__{sender}__{receiver}.parquet` | `_write_donor_pair_pathways()` 910–1448 | `incytr_pathways.by_context.<donor>.slice_index` | `{context}__{sender}__{receiver}.parquet` |

No decomp_ols, song_concordance, or human_perdonor shard families exist for T-cell.

### 3C. Shared-Machinery Candidates

The following are conceptually identical between the two builders and are
candidates for extraction into shared helpers:

| Concern | Unified function | T-cell function | Notes |
|---|---|---|---|
| Kinase motifs (PSSM) | `_build_kinase_motifs()` 4608 | `_build_kinase_motifs()` 442 | Byte-identical logic; should become one shared function |
| Incytr shard writer (pair-mode) | `_write_incytr_pair_pathways()` 3850 | `_write_donor_pair_pathways()` 910 | Same DuckDB streaming pipeline; differs only in: (a) multi-donor loop, (b) 3-part filename template, (c) per-donor context scoping. Core logic is ~70% common |
| Incytr gene-node index | `_build_incytr_gene_node_index()` 3752 | `_build_incytr_gene_node_index()` 282 | Identical logic |
| Payload adapter / sanitize | `_sanitize()` 988 | `_sanitize()` 369 | Identical |
| DuckDB temp-dir config | `_configure_duckdb_tempdir()` 466 | `_configure_duckdb_tempdir()` 406 | Identical |
| Build cache (decomp_ols/song_concordance) | `_load_build_cache()` 382, `_write_build_cache()` 415 | not present | T-cell has no equivalent caching; would become shared infra |
| Context block wrapper | `_as_single_context_block()` 3000 | not needed (multi-context native) | T-cell's `by_context` is built directly |

Genuinely T-cell-specific (no shared-machinery candidates):
- `_build_celltype_assignment()` — ProjecTILs state counts + embeddings
- `_build_tcell_attribution_index()` — T-cell specificity + concordance schema
- `_write_tcell_transcript_trace()` 2002 — T-cell pseudobulk per cluster
- `_write_tcell_omics_trace()` 2139 — T-cell protein/phospho per cluster
- `build_tcell_audit_manifest()` 1945 — T-cell specific audit tables

---

## 4. Frontend Consumer Map

### 4A. Shared adapter layer (`viewer_shared/template/js/00_payload_adapter.js`)

`ViewerPayload` exposes: `schemaVersion()`, `contexts()`, `defaultContext()`,
`activeContext()`, `contextRecord()`, `contextCapabilities()`, `contrastAxis()`,
`kinases()`, `celltypes()`, `incytr()`, `incytrSliceIndex()`, `edgeUrl()`,
`incytrShardFilename()`. Both viewers include this file via shared template
fallback.

### 4B. Payload-key → frontend file map

| Payload key | Frontend files that access it | Access pattern |
|---|---|---|
| `meta` | `01_state.js:49` | `PAYLOAD.meta` → `META` module-level var |
| `meta.song_uniform` | `kinase_audit.js:561`, `kinase_crosstable.js:47` | `PAYLOAD.meta.song_uniform` |
| `meta.wmb_uniform` | `kinase_explorer.js:54`, `kinase_crosstable.js:52` | `PAYLOAD.meta.wmb_uniform` |
| `meta.cohort` | `evidence_row.js:120,251,687`, `transcript_trace.js:114` | `PAYLOAD.meta.cohort` |
| `meta.transcript_trace` | `transcript_trace.js:42` | `PAYLOAD.meta.transcript_trace` |
| `meta.omics_trace` | `evidence_row.js:53` | `PAYLOAD.meta.omics_trace` |
| `kinases` | `ViewerPayload.kinases()` (via `00_payload_adapter.js`); all kinase tabs | `ViewerPayload.kinases(contextId)` → context-scoped slice |
| `kinase_motifs` | `kinase_audit.js:1293`, `kinase_fivexfad.js:1478`, `kinase_human.js:559` | `PAYLOAD.kinase_motifs[<name>]` (direct, not via adapter) |
| `celltypes` | `ViewerPayload.celltypes()` | `ViewerPayload.celltypes(contextId)` |
| `kinase_celltype_evidence` | `kinase_explorer.js:293`, `kinase_crosstable.js:473` | `PAYLOAD.kinase_celltype_evidence` |
| `attribution_index` | `kinase_explorer.js:39,136,232`, `kinase_crosstable.js:184,957`, `kinase_wiring.js:47`, `temporal_v2.js:15`, `04_slice_cache.js:815` | `PAYLOAD.attribution_index` |
| `decomposition_index` | `kinase_explorer.js:304`, `kinase_crosstable.js:171`, `temporal_v2.js:48` | `PAYLOAD.decomposition_index` |
| `agreement_index` | `kinase_explorer.js:319` | `PAYLOAD.agreement_index` |
| `subclass_breakdown` | `kinase_audit.js:737` | `PAYLOAD.subclass_breakdown` |
| `audit_tables` | `01_state.js:309,313`, `kinase_detail.js:2` | `PAYLOAD.audit_tables.tables`, `PAYLOAD.audit_tables.measurement_trace` |
| `edge_slice_ref` | `04_slice_cache.js:15` | `PAYLOAD.edge_slice_ref` → `ESR` module var |
| `incytr_pathways` | `ViewerPayload.incytr()`, `ViewerPayload.incytrSliceIndex()`; `incytr_heatmap.js`, `incytr_pathways.js`, `incytr_global_index.js` | Via adapter; shard URLs from `slice_index.filename_template` |
| `human` | `01_state.js:53,59` (`HAS_HUMAN`, `_KH`); `kinase_crosstable.js:220`; `kinase_human.js:165,559` | `PAYLOAD.human` (direct); tested via `HAS_HUMAN` gate |
| `supporting_5xfad` | `01_state.js:54` (`HAS_FIVEXFAD`); `kinase_crosstable.js:288`; `kinase_fivexfad.js:49` | `PAYLOAD.supporting_5xfad` (direct); tested via `HAS_FIVEXFAD` gate |

**Note:** `kinase_motifs`, `kinase_celltype_evidence`, `attribution_index`, `decomposition_index`, `agreement_index`, `subclass_breakdown`, `human`, and `supporting_5xfad` are accessed via direct `PAYLOAD.<key>` rather than through `ViewerPayload`. These are not context-routed by the adapter — they are flat top-level objects. Slice adapters in 5B/5C–E must ensure these keys remain at the same path.

### 4C. T-Cell Specific Frontend Consumers

T-cell viewer (`alz/tcell_viewer/template/`) adds one viewer-specific tab:
- `kinase_explorer.js` (T-cell version) — reads `PAYLOAD.attribution_index` (T-cell schema), `ViewerPayload.kinases()`
- `celltype_assignment.js` — reads `PAYLOAD.celltype_assignment` (T-cell only)
- `kinase_audit.js` (T-cell version) — reads `PAYLOAD.audit_tables`, `kinase_motifs`

The T-cell viewer's `incytr_pathways.by_context.<donor>.slice_index.filename_template`
is `"{context}__{sender}__{receiver}.parquet"` — the adapter's
`incytrShardFilename()` correctly substitutes `{context}` using the active
context id, so no hard-coded donor prefix appears in shared JS.

---

## 5. Validators and Contract Docs

### 5A. `alz/viewer/verify_payload_contract.py`

**What it checks:**
1. `meta.viewer_payload_schema_version == 2`
2. `meta.contexts` is a non-empty list; each entry has `{id, label, cohort, axis_kind, capabilities}` with unique ids
3. `meta.default_context` appears in `meta.contexts`
4. `kinases`, `celltypes`, `incytr_pathways` each have `by_context` present for all context ids; `by_donor` alias triggers error
5. `meta.transcript_trace.by_donor` alias triggers error
6. Per context: `incytr_pathways.by_context.<ctx>.slice_index.{filename_template, present}` present
7. Capability/data consistency: `capabilities.kinases` agrees with `kinases.by_context.<ctx>` row count > 0; same for celltypes and incytr `present` list

**What it does NOT check:** `attribution_index`, `decomposition_index`, `agreement_index`, `subclass_breakdown`, `human`, `supporting_5xfad`, `kinase_celltype_evidence`, `kinase_motifs`, `edge_slice_ref` URLs, shard presence on disk.

Expected output after 2026-06-01 refactor:
```
outputs/reports/unified_viewer/unified_viewer.payload.json: schema=2 default=song_ad contexts=song_ad pass=True
outputs/reports/tcell_viewer/tcell_viewer.payload.json: schema=2 default=donor1 contexts=donor1,donor2 pass=True
```

### 5B. `alz/viewer/verify_template.py`

**What it checks:**
1. Every `raw("...")` include in `viewer_shared/template/index.html.j2` resolves from local or shared template dir
2. Independent Jinja render is byte-equivalent to `build_unified_viewer._render_template()`
3. Required sentinels (`__APP_COLOR__`, `__TAU_COLOR__`, `__APTT_COLOR__`, `__PAYLOAD_SENTINEL__`) present in rendered HTML

**What it does NOT check:** Payload content, shard presence, JS correctness.

### 5C. `docs/foundation/viewer_payload_contract.md` — Reconciliation

The contract doc is current for the v2 schema skeleton but has several stale/incomplete points relative to the actual code as of 2026-06-17:

| Item | Contract doc says | Actual code |
|---|---|---|
| `supporting_5xfad.celltype_mea_index` | Doc mentions `celltype_mea_index` as replaced by `celltype_agreement_index` | Code emits `celltype_agreement_index` (correct); doc note is accurate |
| `celltype_attribution_summary_index` | Described | Emitted; accurate |
| `supporting_5xfad.contrast_qc`, `sample_counts` | Not mentioned | Emitted by `build_supporting_5xfad_slice()` at lines 2976–2979 |
| `supporting_5xfad.celltype_ols_shards` | Not mentioned | Emitted at line 2981; backed by `edge_slices/fivexfad_celltype_ols/` |
| `human.ctrl_donors`, `human.case_donors`, `human.winsor`, `human.recurrence_ctrl`, `human.stoich_by_site`, `human.raw_phospho_by_site`, `human.celltype_specificity` | Not mentioned | Emitted by `build_human_slice()` — significant undocumented fields |
| `incytr_pathways.global_index.score_columns` | Lists `TPDS, PPDS, PhPDS_ps, PhPDS_py, SiK_score` | Accurate; also `gene_node_index` emitted but not doc'd |
| `incytr_pathways.heatmap_counts_signed` | Not mentioned | Emitted at line 4551 |
| `incytr_pathways.trajectory_summary`, `gene_node_index`, `direction_flag_columns`, `path_metric_columns` | Not mentioned | All emitted; trajectory annotation is CR-04 addition post contract-doc |
| `edge_slice_ref` on T-cell | Doc shows `decomp_ols_url` in the ref | T-cell emits only `incytr_pathways_url` + `incytr_pathways_index` |
| T-cell `celltype_assignment` | Not mentioned | Key T-cell differentiator |
| T-cell `attribution_index` schema | Not mentioned (differs from unified) | T-cell attribution has `tcell_specificity, tcell_tier, tcell_lfc, tcell_concordance, tcell_concordant, tcell_consistency` instead of song/WMB/SEA-AD columns |
| Migration steps 1–5 | Doc says "implemented as of 2026-06-01" | Accurate; `by_donor` aliases removed |

**Summary:** The contract doc captures the v2 meta/context/by_context skeleton and the Incytr block structure well. It is most stale for: `human` block sub-fields, `supporting_5xfad` sub-fields, Incytr trajectory/direction-flag additions (CR-04), and the T-cell-specific keys (`celltype_assignment`, T-cell `attribution_index` schema).

---

## 6. Cohort-Namespace Tagging

| Payload section | Cohort owner | Notes |
|---|---|---|
| `meta` | shared/composer | Assembled by `build_payload()` drawing from all cohort data; capabilities flags gate per-cohort blocks |
| `kinases.by_context.song_ad` | **song** | Mouse bulk MEA kinase activity |
| `kinase_motifs` | shared | Union of all cohort kinase names; cross-cutting PSSM lookup |
| `celltypes.by_context.song_ad` | **song** | Levy-t5 31-cluster spine |
| `kinase_celltype_evidence` | **song** | From `celltype_evidence_table.csv` (mouse concordance data) |
| `attribution_index` | **song** | From `unified_attribution_full.csv` (mouse pipeline) |
| `decomposition_index` | **song** | From `levy_t5/mea_per_cluster.parquet` |
| `agreement_index` | **song** | Derived from song MEA + decomposition |
| `subclass_breakdown` | **song** | WMB subclass tooltips for mouse celltypes |
| `audit_tables` | **mixed** | Primarily song pipeline tables; 5xFAD manifest items in `_audit_specs()` (lines 178–184); measurement_trace is song |
| `edge_slice_ref` | shared | URL map cross-cutting all lazy shard families |
| `incytr_pathways.by_context.song_ad` | **song** | Pair-mode Incytr on the song/mouse 31-cluster spine |
| `human` | **mukesh** | Human NBB per-donor MEA entirely |
| `supporting_5xfad` | **fivexfad** | 5xFAD supporting cohort block |
| `edge_slices/decomp_ols/` | **song** | Mouse decomposition OLS |
| `edge_slices/song_concordance/` | **song** | Song scRNA concordance |
| `edge_slices/incytr_pathways/` | **song** | Pair-mode AD pathways |
| `edge_slices/human_perdonor/` | **mukesh** | Human perdonor substrate sidecars |
| `edge_slices/fivexfad_detail/` | **fivexfad** | 5xFAD kinase detail |
| `edge_slices/fivexfad_celltype_mea/` | **fivexfad** | 5xFAD cell-type MEA |
| `edge_slices/fivexfad_celltype_ols/` | **fivexfad** | 5xFAD cell-type OLS |
| `edge_slices/fivexfad_attribution/` | **fivexfad** | 5xFAD attribution sidecars |
| `audit_sources/measurement_trace/` | **song** | ST/pY phospho traces |
| `audit_sources/transcript_trace/` | **song** | Incytr pseudobulk substrate |
| `audit_sources/omics_trace/` | **song** | Protein + phospho per cluster |
| `audit_sources/omics_trace_normalized/` | **song** | Limma-normalized condition means |

---

## 7. Risky Extraction Points

### R1. `audit_tables` spans song and 5xFAD inputs

`build_audit_manifest()` (line 808) calls `_audit_specs()` which adds 5xFAD
manifest items at lines 178–184. If a per-cohort `build_audit_slice()` is naively
extracted, the 5xFAD rows in `audit_tables.tables["5xfad_sample_manifest"]` and
`audit_tables.tables["5xfad_dataset_index"]` will need to come from the fivexfad
adapter, not the song adapter. The composer must merge both sub-manifests.
**Risk level: medium.**

### R2. `build_supporting_5xfad_slice()` requires `data: UnifiedData`

`build_supporting_5xfad_slice(data)` (line 2812) takes a `UnifiedData` argument
and uses `data.edge_metadata` in `_build_fivexfad_attribution_rows()` (line 1693)
for the song bulk MEA rows — it builds 5xFAD attribution confidence by aligning
against the song MEA results (`_assign_fivexfad_song_aligned_confidence()` line
1823). This creates a **cross-cohort dependency** at build time: 5xFAD's
attribution confidence is derived from Song's MEA. A clean fivexfad adapter that
receives only 5xFAD inputs would need to receive the song MEA data as an
additional argument, or the alignment step must be lifted into the composer.
**Risk level: high.**

### R3. `kinase_motifs` is assembled from union of all cohort kinase names

`_build_kinase_motifs()` is called with `sorted(motif_names)` where `motif_names`
is the union of `kinases_slice.name`, `human_slice.kinases.name`, and
`supporting_5xfad.rows[*].kinase` (lines 4913–4918). This must happen after all
three cohort slices are available, so it belongs in the composer, not a
per-cohort adapter. **Risk level: low** (clean composer responsibility; no
per-cohort logic).

### R4. `meta.capabilities` is set in two passes

`meta.capabilities["human_reference"]` and `meta.capabilities["supporting_5xfad"]`
are initially `False` and conditionally set to `True` after building each optional
slice (lines 4903–4911). The composer must either: (a) collect availability from
each adapter's return value, or (b) let adapters return a capabilities dict to
merge. **Risk level: low.**

### R5. `_write_incytr_pair_pathways()` reads `INCYTR_PAIR_MODE_INPUT_DIR` as a whole

The Incytr shard writer reads all 9 contrast parquets and writes a single unified
`edge_slices/incytr_pathways/` tree (one per-pair parquet per the 31×31 pair grid,
plus one global `.bin.gz`). This is tightly tied to the song/AD context. No other
cohort uses this directory. Clean extraction: make the input dir and output dir
arguments of a shared `write_incytr_pair_shards()` function.
**Risk level: low** (self-contained, no cross-cohort data).

### R6. T-cell `attribution_index` has a different schema

The unified viewer's `attribution_index` has ~30 columns (song/WMB/SEA-AD
evidence). The T-cell `attribution_index` has 11 columns (`tcell_specificity`,
`tcell_tier`, etc.). Both are accessed as `PAYLOAD.attribution_index` by the
respective viewer's JS. They share the column names `kinase_id`, `contrast_id`,
`cell_type`, `nes`, `fdr`. If a future slice schema attempts to unify these, the
column set divergence must be handled. Phase 5B should define one schema per
cohort, not a merged schema. **Risk level: low** for isolated extraction; **high**
if unification is attempted.

### R7. Song concordance and decomp-OLS shards are data-volume sensitive

`_write_song_concordance_slices()` reads ~210 MB CSV and `_write_decomp_ols_slices()`
reads `site_level_ols.parquet` (size logged at build time). Both are song-only and
can be cleanly extracted. But the build-cache mechanism (`_load_build_cache`,
`_write_build_cache`) that makes these fast on repeated builds is currently
specific to the unified viewer. The fivexfad adapter does not use a cache.
**Risk level: low** (just need to pass cache mechanism along).

### R8. `edge_slice_ref` in the unified payload combines pointers from song, mukesh, and 5xFAD

`present_human_perdonor_kinase_ids` is set at line 4954, after `build_human_slice()`
runs. This key lives in `edge_slice_ref` alongside song-specific keys. The composer
must aggregate shard index data from each cohort adapter and assemble the final
`edge_slice_ref`. **Risk level: medium.**

---

## 8. Smallest-First Adapter Ordering

The planned order is 5C(mukesh) → 5D(tcell) → 5E(fivexfad), with Song deferred to 5F.

**Assessment: confirm with one revision.**

The planned order is correct with the following rationale:

1. **5C — mukesh** (`build_human_slice()`): Self-contained, no cross-cohort
   dependencies. Its output is gated by `HAS_HUMAN` — absent outputs produce no
   payload block. Reads only from `HUMAN_PERDONOR_DIR`. One lazy shard family
   (`human_perdonor`). Frontend access is isolated to `kinase_human.js` and
   `kinase_crosstable.js:220`. Smallest blast radius.

2. **5D — tcell** (`build_tcell_viewer.py`): Fully independent builder — does not
   share any module-level state with `build_unified_viewer.py`. Can be extracted as
   a standalone `build_tcell_viewer_slice()` that simply produces the current
   payload dict. The T-cell viewer already has its own builder, paths module, and
   template directory. Extraction here is mostly structural rename + composer
   wiring. One shard family (`incytr_pathways` with donor-scoped 3-part filenames).

3. **5E — fivexfad** (`build_supporting_5xfad_slice()`): Four shard families.
   The cross-cohort dependency (R2 above — Song MEA data needed for confidence
   alignment) must be resolved first. Recommend making song MEA a named parameter
   that the composer passes to the fivexfad adapter. Do 5E after 5C and 5D are
   stable so the composer pattern is established.

4. **5F — song** (deferred): Most complex. Owns the majority of payload keys and
   all measurement/transcript/omics trace shards. `load_all_data()` is large.
   Correctly deferred until the composer pattern is proven.

**One revision:** The `audit_tables` mixed ownership (R1) means 5E (fivexfad) must
also define how the 5xFAD audit entries are folded into the composer-level audit
manifest. Recommend that each adapter optionally returns an `audit_entries` dict
that the composer merges into the unified `audit_tables` — this pattern should be
established in 5C or 5D (whichever goes first), even if mukesh and tcell have zero
extra audit entries.
