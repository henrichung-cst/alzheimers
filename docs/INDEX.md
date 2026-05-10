# Document Map

Read by analytical role, not creation order.

## Start Here

| File | Role |
|:---|:---|
| [`pipeline_overview.qmd`](./pipeline_overview.qmd) | End-to-end narrative of the full pipeline (bulk + Incytr) with cross-references into every foundation spec (renders to `pipeline_overview.html`) |

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
| [`report_writing_checklist.md`](./report_writing_checklist.md) | Reviewer-facing report writing guidance |
| [`result_analysis_plan.md`](./result_analysis_plan.md) | Interpretation framework: how to move from generated outputs to biological claims |
| [`allen_ctx_hpf_disagreement.md`](./allen_ctx_hpf_disagreement.md) | Reviewer FAQ: why our cell-type verdicts may differ from the Allen ctx+HPF Transcriptomics Explorer |
| [`viewer_style.md`](./viewer_style.md) | Writing-style guide for unified-viewer copy (panels, drawers, tooltips) |
| [`plex_covariate_plan.md`](./plex_covariate_plan.md) | **Deferred plan** — TMT plex covariate addition for males-only OLS (diagnostic complete, activation pending) |
| [`../archive/deconvolution/docs/deconvolution_infeasibility.md`](../archive/deconvolution/docs/deconvolution_infeasibility.md) | **Archived** — synthetic validation proving direct deconvolution is infeasible on this dataset. Source script + figures alongside under `archive/deconvolution/`. |

## Integrations

| File | Role |
|:---|:---|
| [`integrations/kinase_incytr_integration.md`](./integrations/kinase_incytr_integration.md) | In-tree Phase 1 stub inventory + one-way handoff diagram (thin pointer to the remediation plan) |
| [`incytr_remediation_plan.md`](./incytr_remediation_plan.md) | Authoritative architectural plan for the kinase ↔ Incytr integration rewrite |
| [`integrations/5xfad-lucie-manifest.json`](./integrations/5xfad-lucie-manifest.json) | Local inventory of Lucie 5xFAD upstream files |

## Archive

`docs/archive/` is gitignored (2026-04-20). Local files persist on disk for provenance but are not tracked or linked. See [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) §Archived for what's there and why.

## Reading Rule

- **What should we do next?** → [`foundation/`](./foundation/analysis_charter.md)
- **How do external inputs map into runtime?** → [`integrations/kinase_incytr_integration.md`](./integrations/kinase_incytr_integration.md) + [`incytr_remediation_plan.md`](./incytr_remediation_plan.md)
- **Why was a path closed?** → [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md) + [`../archive/deconvolution/docs/deconvolution_infeasibility.md`](../archive/deconvolution/docs/deconvolution_infeasibility.md)
- **Historical context** → `archive/`

---

# Asset Map

Paths relative to repo root. Authoritative script docs live in [`CLAUDE.md`](../CLAUDE.md).

## Inputs

### Primary data (Song 72-animal cohort)
| Path | Contents |
|:---|:---|
| `data/datasets/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx` | 72-animal TMT total proteome (6 plexes × 10 channels) |
| `data/datasets/song/primary/phospho/song_IMAC_{sitequant,compositeSites}_merged_labeled (2).xlsx` | Phospho IMAC (pS/pT) — site-level + composite |
| `data/datasets/song/primary/phospho/song_pY_{sitequant,compositeSites}_merged_labeled (2).xlsx` | Phospho pY — site-level + composite |
| `data/datasets/song/primary/metadata/Sample_list_72mice (1).xlsx` | TMT channel ↔ animal mapping |
| `data/datasets/song/transcriptomics/170_gex_celltypes_00.h5ad` | Paired snRNA-seq (63K nuclei, 28 animals) |

### External atlases
| Path | Contents |
|:---|:---|
| `data/external/allen_abc/` | Allen WMB cache (zstd-compressed) |
| `data/external/sea_ad/` | SEA-AD MTG effect sizes (`effect_sizes{,_early,_late}.h5ad`, 139 supertypes) |

### Analysis cache
| Path | Contents |
|:---|:---|
| `data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv` | Cached kinase → gene symbol mapping |

## Pipeline

### Live bulk pipeline (`alz/`)
| Stage | Script | Runner | Output dir |
|:---|:---|:---|:---|
| Config | `config.py` | — | — |
| 1. Ingest | `data_ingest.py` | `runners/main/run_data_ingest.sh` | `outputs/reports/data_ingest/` |
| 2. Normalize | `kinase_normalize.py` (CLI shim) / `kedro run --pipeline=normalize` | `runners/main/run_kinase_attribution.sh` (covers 2–4) | `outputs/reports/kinase_attribution/` |
| 3. Enrich | `kinase_enrich.py` (CLI shim) / `kedro run --pipeline=enrich` | (above) | `outputs/reports/kinase_attribution/` |
| 4. Attribute | `kinase_attribute.py` (CLI shim) / `kedro run --pipeline=attribute` | (above) | `outputs/reports/kinase_attribution/` |
| 5. Recovery | `attribution_recovery.py` (CLI shim) / `kedro run --pipeline=recovery` | `runners/main/run_attribution_recovery.sh` | `outputs/reports/attribution_recovery/` |
| Optional: mechanism | `kinase_mechanism.py` (CLI shim) / `kedro run --pipeline=mechanism` | — | `outputs/reports/kinase_attribution/` |
| Plots | `plot_attribution_bubbles.py` | — | `outputs/reports/attribution_recovery/bubble_plots/` |
| Bundled | — | `runners/main/run_live_pipeline.sh` | all of the above |
| Dual-track | — | `runners/main/run_dual_analysis.sh` | `*_males_only/`, `*_full_cohort/` |

### Standalone utilities (`alz/`)
| Script | Purpose |
|:---|:---|
| `map_kinases_to_genes.py` | Kinase → gene symbol mapping utility |
| `lucie_5xfad_manifest.py` | Proteomics manifest builder for Lucie 5xFAD integration |
| `build_unified_viewer.py` | See Viewers section below |

### Supporting (`alz/`)
| Script | Runner | Output dir |
|:---|:---|:---|
| `atlas_reference.py` | `runners/supporting/run_atlas_reference.sh` | `data/external/sea_ad/`, `data/external/allen_abc/` |
| `wmb_expression.py` | `runners/supporting/run_wmb_expression.sh` | `outputs/reports/wmb_expression/` |
| `snrna_integration.py` | `runners/supporting/run_snrna_integration.sh` | `outputs/reports/snrna_integration/` |

Additional supporting runners (ops utilities, no Python counterpart): `runners/supporting/compress_atlas_cache.sh`, `decompress_atlas_cache.sh`, `run_wmb_download.sh`, `run_extract_wmb_subset.sh`.

### Supplementary diagnostics (`alz/supplementary/`)
`fdr_stringent.py`, `threshold_sensitivity.py`, `aggregation_robustness.py`, `parent_protein_qc.py` — run via `runners/supplementary/run_reviewer_diagnostics.sh`. Output: `outputs/reports/supplementary/`. (The historical `deconvolution_infeasibility.py` proof has been frozen and moved to `archive/deconvolution/code/`.)

### Integration pipeline (`alz/integration/`)

Thin AD-specific client around the upstream `incytr` R package, gated on the production engine landing per [`incytr_remediation_plan.md`](./incytr_remediation_plan.md). The legacy `wrappers/`, `adapters/`, `sidecar/`, `tests/`, and orchestrator shell scripts are preserved under `archive/incytr_integration/`.

In-tree now:

| File | Role |
|:---|:---|
| `config_integration.py` | Filter values, design columns, contrast vectors, paths |
| `export_factorial_inputs.py` | Reads h5ad, writes the on-disk contract `load.R` consumes |
| `factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh` | Entry point + AD loader + parquet writer + DuckDB views (gated on production package API in `../incytr`) |
| `README.md` | File-by-file layout |

## Outputs

### Bulk pipeline (`outputs/reports/`)
| Dir | Key files |
|:---|:---|
| `data_ingest/` | `sample_exclusions.csv`, `pca_plots/outlier_diagnostic.png` |
| `kinase_attribution/` | `stoichiometry_matrix.csv`, `mea_stoichiometry.csv`, `site_level_ols.csv`, `unified_attribution.csv`, `attribution_summary.json`, `mea_global_shift.csv`, `winsorized_sites.csv` |
| `attribution_recovery/` | **`kinase_hypothesis_table.csv` (primary deliverable)**, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, `bubble_plots/` |
| `wmb_expression/`, `snrna_integration/` | Supporting prerequisites |
| `supplementary/` | Reviewer-diagnostic results |

### Integration pipeline outputs

Legacy outputs lived under `alz/integration/intermediates/` (gitignored). That tree is now orphaned by the integration rewrite (see [`incytr_remediation_plan.md`](./incytr_remediation_plan.md)); the new architecture writes to `outputs/reports/incytr_factorial/` instead. Nothing currently regenerates either.

## Viewers

| Viewer | Builder | Input | Output |
|:---|:---|:---|:---|
| Kinase + pathway | `alz/build_unified_viewer.py` | `kinase_hypothesis_table.csv`, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, `mea_stoichiometry.csv`, `site_level_ols.csv`, `backbone_recurrence_by_contrast.csv`, `backbone_permutation_pvalues_by_contrast.csv`, `unified_attribution.csv`, `kinase_backbone_edges.parquet` | `outputs/reports/unified_viewer/index.html` + payload JSON + sharded edge slices |
