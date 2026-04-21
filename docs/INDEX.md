# Document Map

Read by analytical role, not creation order.

## Start Here

| File | Role |
|:---|:---|
| [`pipeline_overview.md`](./pipeline_overview.md) | End-to-end narrative of the full pipeline (bulk + Incytr) with cross-references into every foundation spec |

## Foundation (start here for live work)

| File | Role |
|:---|:---|
| [`foundation/analysis_charter.md`](./foundation/analysis_charter.md) | Single source of truth for the live 72-sample analysis program |
| [`foundation/live_pipeline_contract.md`](./foundation/live_pipeline_contract.md) | Stage-by-stage runner contract: prerequisites, outputs, failure modes |
| [`foundation/concordance.md`](./foundation/concordance.md) | SEA-AD + Song concordance model: evidence sources, weights, confidence tiers |
| [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md) | Why the project pivoted from deconvolution to stoichiometry |
| [`foundation/statistical_constraints.md`](./foundation/statistical_constraints.md) | Governing identifiability and interpretation limits |
| [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) | Retention policy + file-level main / supporting / archived inventory; banned code paths |

## Supplementary Analysis

| File | Role |
|:---|:---|
| [`deconvolution_infeasibility.md`](./deconvolution_infeasibility.md) | Synthetic validation proving deconvolution is infeasible on this dataset (figures + summary) |
| [`report_writing_checklist.md`](./report_writing_checklist.md) | Reviewer-facing report writing guidance |

## Integrations

| File | Role |
|:---|:---|
| [`integrations/kinase_incytr_integration.md`](./integrations/kinase_incytr_integration.md) | Source of truth for the kinase ↔ Incytr integration (scoring model, runtime modes, config, outputs, limitations) |
| [`integrations/integrations-structure.md`](./integrations/integrations-structure.md) | Upstream data bundle structure (gdrive_shared) |

## Archive

`docs/archive/` is gitignored (2026-04-20). Local files persist on disk for provenance but are not tracked or linked. See [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) §Archived for what's there and why.

## Reading Rule

- **What should we do next?** → [`foundation/`](./foundation/analysis_charter.md)
- **How do external inputs map into runtime?** → [`integrations/`](./integrations/integrations-structure.md)
- **Why was a path closed?** → [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md) + [`deconvolution_infeasibility.md`](./deconvolution_infeasibility.md)
- **Historical context** → `archive/`

---

# Asset Map

Paths relative to repo root. Authoritative script docs live in [`CLAUDE.md`](../CLAUDE.md).

## Inputs

### Primary data (Song 72-animal cohort)
| Path | Contents |
|:---|:---|
| `data/incytr_collections/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx` | 72-animal TMT total proteome (6 plexes × 10 channels) |
| `data/incytr_collections/song/primary/proteomics/song_IMAC_sitequant_merged_labeled (2).xlsx` | Phospho site-level quant |
| `data/incytr_collections/song/primary/proteomics/song_IMAC_compositeSites_merged_labeled (2).xlsx` | Composite phospho sites |
| `data/incytr_collections/song/primary/proteomics/Sample_list_72mice (1).xlsx` | TMT channel ↔ animal mapping |
| `data/incytr_collections/song/method_records/aobs_desp_standardized/inputs/A_obs_fractions.tsv` | Cell-type composition fractions (24 groups × 10 cell types) |
| `data/incytr_collections/song/primary/snrnaseq/170_gex_celltypes_00.h5ad` | Paired snRNA-seq (63K nuclei, 28 animals) |

### External atlases
| Path | Contents |
|:---|:---|
| `data/external/allen_abc/` | Allen WMB + Aging Mouse cache (zstd-compressed) |
| `data/external/sea_ad/` | SEA-AD MTG effect sizes (`effect_sizes{,_early,_late}.h5ad`, 139 supertypes) |

### Analysis cache
| Path | Contents |
|:---|:---|
| `data/incytr_collections/song/analysis_cache/kinase_to_gene_mapping.csv` | Cached kinase → gene symbol mapping |
| `data/incytr_collections/song/analysis_cache/allen_expression_cache.csv` | Cached Allen queries |

## Pipeline

### Live bulk pipeline (`code/`)
| Stage | Script | Runner | Output dir |
|:---|:---|:---|:---|
| Config | `config.py` | — | — |
| 1. Ingest | `data_ingest.py` | `runners/main/run_data_ingest.sh` | `outputs/reports/data_ingest/` |
| 2. Normalize + MEA + attribute | `kinase_attribution.py` | `runners/main/run_kinase_attribution.sh` | `outputs/reports/kinase_attribution/` |
| 3. Recovery | `attribution_recovery.py` | `runners/main/run_attribution_recovery.sh` | `outputs/reports/attribution_recovery/` |
| Plots | `plot_attribution_bubbles.py` | — | `outputs/reports/attribution_recovery/bubble_plots/` |
| Bundled | — | `runners/main/run_live_pipeline.sh` | all of the above |
| Dual-track | — | `runners/main/run_dual_analysis.sh` | `*_males_only/`, `*_full_cohort/` |

### Supporting (`code/`)
| Script | Runner | Output dir |
|:---|:---|:---|
| `atlas_reference.py` | `runners/supporting/run_atlas_reference.sh` | `outputs/reports/atlas_reference/` |
| `wmb_expression.py` | `runners/supporting/run_wmb_expression.sh` | `outputs/reports/wmb_expression/` |
| `snrna_integration.py` | `runners/supporting/run_snrna_integration.sh` | `outputs/reports/snrna_integration/` |

### Supplementary diagnostics (`code/supplementary/`)
`fdr_stringent.py`, `threshold_sensitivity.py`, `aggregation_robustness.py`, `parent_protein_qc.py` — run via `runners/supplementary/run_reviewer_diagnostics.sh`. Output: `outputs/reports/supplementary/`.

### Integration pipeline (`code/integration/`)
Python adapters + R wrappers; config in `config_integration.py`.

| Component | Files |
|:---|:---|
| Python adapters | `adapters/export_expression{,_factorial}.py`, `export_phospho.py`, `export_kldata.py`, `export_kl_output{,_factorial}.py`, `export_kinase_imputed_genes{,_factorial}.py`, `compute_kinase_support{,_all_pairs,_factorial}.py`, `aggregate_cross_pair.py`, `aggregate_factorial.py`, `examine_factorial.py` |
| R wrappers | `wrappers/duckdb_enumeration.R`, `receiver_scoring.R`, `run_incytr{,_all_pairs,_factorial_all_pairs}.R`, `postprocess.R`, `verify_phase2.R`, `bootstrap_sensitivity.R` |
| Runners | `run_all_pairs.sh` (single-contrast), `run_factorial_all_pairs.sh`, `run_factorial_memory_gated.sh`, `run_factorial_permutations.sh`, `run_imputation_verification.sh`, `run_phase1.sh` |

## Outputs

### Bulk pipeline (`outputs/reports/`)
| Dir | Key files |
|:---|:---|
| `data_ingest/` | `sample_exclusions.csv`, `pca_plots/outlier_diagnostic.png` |
| `kinase_attribution/` | `stoichiometry_matrix.csv`, `mea_stoichiometry.csv`, `site_level_ols.csv`, `unified_attribution.csv`, `attribution_summary.json`, `mea_global_shift.csv`, `winsorized_sites.csv` |
| `attribution_recovery/` | **`kinase_hypothesis_table.csv` (primary deliverable)**, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, `bubble_plots/` |
| `atlas_reference/`, `wmb_expression/`, `snrna_integration/` | Supporting prerequisites |
| `supplementary/` | Reviewer-diagnostic results |

### Integration pipeline (`code/integration/intermediates/`)
| Dir | Key files |
|:---|:---|
| `factorial/` | Per-receiver kinase-imputed gene lists, expression matrices, `kldata.csv`, factorial kinase imputation summary |
| `factorial/all_pairs/` | `recv_{subclass}.parquet` × 22 (with `imputed_nodes` provenance), `pair_summary.csv` |
| `factorial/all_pairs/aggregation/` | `backbone_recurrence_by_contrast.csv`, `backbone_permutation_pvalues_by_contrast.csv` (superset with `significant_both` column), `kinase_backbone_edges.parquet`, `hub_matrix_by_contrast.csv`, `contrast_comparison.csv`, `temporal_dynamics.csv`, `target_convergence_by_contrast.csv`, `kinase_tpds_integration.csv`, `aggregation_metadata.json`, `hub_heatmap_grid.png`, `temporal_dynamics.png`, `kinase_coverage.png` |
| `factorial/all_pairs/aggregation/examination/` | Additivity, trajectory, celltype-centrality figures |
| `all_pairs/` | Single-contrast per-receiver parquets + aggregation |

## Viewers

| Viewer | Builder | Input | Output |
|:---|:---|:---|:---|
| Unified (kinase + pathway) | `code/build_unified_viewer.py` | `kinase_hypothesis_table.csv`, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, `mea_stoichiometry.csv`, `site_level_ols.csv`, `backbone_recurrence_by_contrast.csv`, `backbone_permutation_pvalues_by_contrast.csv`, `unified_attribution.csv`, `kinase_backbone_edges.parquet` | `outputs/reports/unified_viewer/index.html` + payload JSON + sharded edge slices |
