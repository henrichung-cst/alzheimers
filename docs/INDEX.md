# Document Map

Read by analytical role, not creation order.

## Start Here

| File | Role |
|:---|:---|
| [`methods/pipeline_overview.qmd`](./methods/pipeline_overview.qmd) | End-to-end narrative of the full pipeline (bulk + Incytr) with cross-references into every foundation spec (renders to `pipeline_overview.html`) |

## Foundation (authoritative live specs)

| File | Role |
|:---|:---|
| [`foundation/analysis_charter.md`](./foundation/analysis_charter.md) | Single source of truth for the live 72-sample analysis program |
| [`foundation/live_pipeline_contract.md`](./foundation/live_pipeline_contract.md) | Stage-by-stage runner contract: prerequisites, outputs, failure modes |
| [`foundation/concordance.md`](./foundation/concordance.md) | SEA-AD + Song concordance model: evidence sources, weights, confidence tiers |
| [`foundation/specificity_confidence.md`](./foundation/specificity_confidence.md) | Recalculated confidence pill = within-cohort cell-type exclusivity over curated specificity units (collapse over-split Song clusters, keep distinct cell types split; references corroborate at WMB-class level; viewer shows collapsed units as expandable parents) |
| [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md) | Why the project pivoted from deconvolution to stoichiometry |
| [`foundation/statistical_constraints.md`](./foundation/statistical_constraints.md) | Governing identifiability and interpretation limits |
| [`foundation/multiple_testing.md`](./foundation/multiple_testing.md) | Multiple-testing policy across the pipeline |
| [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) | Retention policy + active vs archived inventory; banned code paths |
| [`foundation/cohort_contract.md`](./foundation/cohort_contract.md) | Canonical input/output schemas for the four shared analysis modes; per-cohort parameter knobs |
| [`foundation/viewer_payload_contract.md`](./foundation/viewer_payload_contract.md) | Shared frontend payload schema for current and future viewers; builders stay separate but emit a common context/capability contract |
| [`foundation/viewer_frontend_contract.md`](./foundation/viewer_frontend_contract.md) | Frontend sharing policy for AD/Song and T-cell viewers; documents shared modules, intentional forks, and consolidation targets |
| [`foundation/kinase_explorer_attribution.md`](./foundation/kinase_explorer_attribution.md) | Authoritative live contract for Kinase Explorer Attribution views across Song AD, human/Mukesh, 5xFAD, and T-cell viewer surfaces |
| [`foundation/mechanism_attribution_contract.md`](./foundation/mechanism_attribution_contract.md) | Mechanism-annotation contract: how mechanism labels merge into `unified_attribution.csv` |
| [`foundation/projected_state_mea_contract.md`](./foundation/projected_state_mea_contract.md) | Projected-state MEA contract: per-cluster decomposition → MEA substrate contract |
| [`foundation/mukesh_ingest_policies.yml`](./foundation/mukesh_ingest_policies.yml) | Mukesh / NBB human ingest edge-case policies (consumed by `alz/cohorts/mukesh/ingest.py`) |
| [`foundation/tcell_reference.md`](./foundation/tcell_reference.md) | T-cell NSCLC reference cohort constants (per-kinase cell-type detection basis) |
| [`foundation/pipeline_conventions.md`](./foundation/pipeline_conventions.md) | Cross-pipeline invariants: LFC/NES/sclog2FC/PDS sign convention, mechanism-after-attribute ordering, direct levy_t5 mapping |
| [`foundation/standard_attribution_metric.md`](./foundation/standard_attribution_metric.md) | The one cross-cohort attribution metric definition (`specificity.compute`); cited by path in 5 source files |
| [`foundation/backbone_incytr_track.md`](./foundation/backbone_incytr_track.md) | Authoritative backbone-grain spec — read before touching backbone / Incytr-viewer code |

## Reference (state of the world)

Consolidated status / decision / interpretation records — what exists, what's decided or closed, what results mean. Distilled from `project_*` memories; not authoritative specs, not plans.

| File | Role |
|:---|:---|
| [`reference/cohorts_and_data.md`](./reference/cohorts_and_data.md) | Per-cohort status + data conventions across Song, Mukesh, 5xFAD, T-cell |
| [`reference/tcell_exhaustion_analysis_summary.md`](./reference/tcell_exhaustion_analysis_summary.md) | Stable summary of the T-cell exhaustion cohort, attribution interpretation, and dedicated viewer |
| [`reference/incytr_sce4_reproduction.md`](./reference/incytr_sce4_reproduction.md) | sce4 reproduction status + closed dead ends (companion to CLAUDE.md parity constants) |
| [`reference/allen_ctx_hpf_disagreement.md`](./reference/allen_ctx_hpf_disagreement.md) | Reviewer FAQ: why our cell-type verdicts may differ from the Allen ctx+HPF Transcriptomics Explorer |

## Viewer Docs

Non-contract viewer docs. Binding viewer contracts live in `foundation/` (`viewer_frontend_contract.md`, `viewer_payload_contract.md`).

| File | Role |
|:---|:---|
| [`viewer/viewer_style.md`](./viewer/viewer_style.md) | Writing-style guide for unified-viewer copy (panels, drawers, tooltips) |
| [`viewer/viewer_crosstable_agreement.md`](./viewer/viewer_crosstable_agreement.md) | Unified-viewer Crosstable Agreement view: semantics + wiring |

## Methods

How the work is done and written up.

| File | Role |
|:---|:---|
| [`methods/pipeline_overview.qmd`](./methods/pipeline_overview.qmd) | End-to-end pipeline narrative (bulk + Incytr); renders to `pipeline_overview.html` |
| [`methods/report_writing_checklist.md`](./methods/report_writing_checklist.md) | Reviewer-facing report writing guidance |

## Integrations

| File | Role |
|:---|:---|
| [`integrations/kinase_incytr_integration.md`](./integrations/kinase_incytr_integration.md) | Current pair-mode integration architecture; in-tree file inventory; data-flow diagram |
| [`integrations/5xfad-kinase-mea-viewer.md`](./integrations/5xfad-kinase-mea-viewer.md) | 5xFAD kinase-MEA + viewer integration notes |
| [`integrations/5xfad-omics-status.md`](./integrations/5xfad-omics-status.md) | Current 5xFAD proteomics/transcriptomics availability, sample matching status, and unresolved label conflicts |

## Plans

Live plans live in `docs/plans/`, indexed by [`plans/README.md`](./plans/README.md). The
completed orchestration program's per-theme build record is archived under
`archive/archived_plans/orchestration/`; completed standalone plans under
`archive/archived_plans/standalone_done/`. See [`plans/README.md`](./plans/README.md) for the
active-work table and status legend.

## Archive

`archive/` (repo root) is gitignored and is not present in a fresh checkout; it holds
provenance material only where it has been locally regenerated. See
[`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) §Archived for what belongs there and why.

Archived material includes:

- Completed campaigns (e.g. CR-03 adoption, factorial-era Incytr docs)
- Superseded plans (WMB-34 spine, levy-19 spine, factorial deconvolution attempts)
- Historical audits + investigation notes

## Reading Rule

- **What should we do next?** → [`foundation/`](./foundation/analysis_charter.md) + [`plans/README.md`](./plans/README.md)
- **How do external inputs map into runtime?** → [`integrations/kinase_incytr_integration.md`](./integrations/kinase_incytr_integration.md)
- **Why was a path closed?** → [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md)
- **What plans are open?** → [`plans/README.md`](./plans/README.md)
- **Historical context** → `archive/` (gitignored; present only when locally regenerated)

---

# Asset Map

Paths relative to repo root. Authoritative script docs live in [`CLAUDE.md`](../CLAUDE.md); canonical data sources live in [`data/README.md`](../data/README.md).

## Inputs

### Primary data (Song 72-animal cohort)
| Path | Contents |
|:---|:---|
| `data/datasets/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx` | 72-animal TMT total proteome (6 plexes × 10 channels) |
| `data/datasets/song/primary/phospho/song_IMAC_{sitequant,compositeSites}_merged_labeled (2).xlsx` | Phospho IMAC (pS/pT) — site-level + composite |
| `data/datasets/song/primary/phospho/song_pY_{sitequant,compositeSites}_merged_labeled (2).xlsx` | Phospho pY — site-level + composite |
| `data/datasets/song/primary/metadata/Sample_list_72mice (1).xlsx` | TMT channel ↔ animal mapping |
| `data/datasets/song/transcriptomics/170_gex_celltypes_00.h5ad` | Paired snRNA-seq (63K nuclei, 28 animals) |
| `data/datasets/song/kinase/kldata_pspy.csv` | Song-built kinase-substrate library (canonical; symlinked into `data/derived/incytr_inputs/kldata.csv`) |

### Human cohort (Mukesh / NBB)
| Path | Contents |
|:---|:---|
| `data/datasets/mukesh/proteomics/` | DIA total proteome |
| `data/datasets/mukesh/phospho/IMAC/` | IMAC phospho |
| `data/datasets/mukesh/phospho/pY/` | pY phospho |

### External atlases
| Path | Contents |
|:---|:---|
| `data/external/allen_abc/` | Allen WMB cache (zstd-compressed) |
| `data/external/sea_ad/` | SEA-AD MTG effect sizes (`effect_sizes{,_early,_late}.h5ad`, 139 supertypes) |
| `data/external/allen_hbca/` | HBCA WHB-10Xv3 |

### Derived bridges + caches
| Path | Contents |
|:---|:---|
| `data/derived/bridges/cluster_to_{wmb_class,seaad_supertype,hbca_supercluster}.csv` | levy_t5 cluster → reference vocabulary (1-hop crosswalks) |
| `data/derived/bridges/wmb_subclass_to_class.csv` | WMB subclass → class |
| `data/derived/aggregates/seaad/expression_by_supertype.csv` | SEA-AD MTG per-supertype expression |
| `data/derived/aggregates/hbca/expression_by_class.csv` | HBCA per-class expression |
| `data/derived/caches/kinase_to_gene_mapping.csv` | Kinase abbreviation → gene symbol (MyGene cache) |
| `data/derived/caches/human_to_mouse_homologene.csv` | Homologene mapping |

## Pipeline

### Live bulk pipeline (`alz/`)
| Stage | Script | Runner | Output dir |
|:---|:---|:---|:---|
| Config | `alz/shared/config.py` | — | — |
| 1. Ingest | `alz/ingest/song.py` | `pixi run ingest` | `outputs/reports/data_ingest/` |
| 2. Normalize | `alz/bulk_mea/normalize.py` | `pixi run normalize` | `outputs/reports/kinase_attribution/` |
| 3. Enrich | `alz/bulk_mea/enrich.py` | `pixi run enrich` | `outputs/reports/kinase_attribution/` |
| 4. Attribute | `alz/bulk_mea/attribute.py` | `pixi run attribute` | `outputs/reports/kinase_attribution/` |
| 5. Mechanism | `alz/bulk_mea/mechanism.py` | `pixi run mechanism` (after attribute; merges into `unified_attribution.csv`) | `outputs/reports/kinase_attribution/` |
| 6. Recovery | `alz/bulk_mea/recover.py` | `pixi run recover` | `outputs/reports/attribution_recovery/` |
| Summary | `alz/bulk_mea/summary.py` | ad hoc | read-only report over cached outputs |
| Bundled | — | `pixi run live` | all of the above |
| Dual-track | — | `alz/runners/main/run_dual_analysis.sh` | `*_males_only/`, `*_full_cohort/` |
| End-to-end | — | `alz/runners/main/run_all.sh` (= `pixi run all`) | everything |

### Mouse decomposition + Incytr pair-mode (not yet under kedro)
| Stage | Script | Runner |
|:---|:---|:---|
| snRNA pseudobulk + concordance | `alz/reference/snrna_integration.py` | `alz/runners/supporting/run_snrna_integration.sh` |
| Per-(animal, cluster, gene) proportions | `alz/reference/snrna_proportions.py` | part of decomposition rerun |
| Per-cluster decomposition | `alz/decomposition_mea/build_celltype_decomposition.py` | `alz/runners/main/rerun_decomposition_chain.sh` |
| Per-cluster MEA | `alz/decomposition_mea/enrich_celltype.py` | above |
| Decomposition verification | `alz/decomposition_mea/verify_decomposition.py` | above |
| Incytr pair-mode | `alz/incytr_pair/*` + `alz/integration/*` | `alz/runners/main/run_pair_mode_pipeline.sh` |

### Human cohort pipeline (not yet under kedro)
| Stage | Script | Runner |
|:---|:---|:---|
| Ingest | `alz/cohorts/mukesh/ingest.py` | `alz/runners/main/run_mukesh_perdonor.sh` |
| Per-donor MEA | `alz/cohorts/mukesh/mea.py` | above |
| SEA-AD agreement | `alz/cross_reference/seaad_human_agreement.py` | above; = `pixi run human` |

### Supporting (`alz/`)
| Script | Runner | Output dir |
|:---|:---|:---|
| `alz/reference/atlas.py` | `alz/runners/supporting/run_atlas_reference.sh` | `data/external/sea_ad/`, `data/external/allen_abc/` |
| `alz/reference/wmb_expression.py` | `alz/runners/supporting/run_wmb_expression.sh` | `outputs/reports/wmb_expression/` |

### Supplementary diagnostics (`alz/supplementary/`)
`fdr_stringent.py`, `parent_protein_qc.py`, `deconvolution_feasibility.py` — run via `runners/supplementary/run_reviewer_diagnostics.sh`. Output: `outputs/reports/supplementary/`.

### Standalone utilities (`alz/`)
| Script | Purpose |
|:---|:---|
| `alz/shared/map_kinases_to_genes.py` | Kinase → gene symbol mapping; emits `data/derived/caches/kinase_to_gene_mapping.csv` |
| `alz/ingest/lucie.py` and `alz/ingest/build_5xfad_omics_join_manifest.py` | Proteomics and omics-join manifest builders for Lucie / 5xFAD integration |
| `alz/build_unified_viewer.py` | See Viewers section below |

## Outputs

### Bulk pipeline (`outputs/reports/`)
| Dir | Key files |
|:---|:---|
| `data_ingest/` | `sample_exclusions.csv`, `pca_plots/outlier_diagnostic.png` |
| `kinase_attribution/` | `stoichiometry_matrix.csv`, `mea_stoichiometry.csv`, `site_level_ols.csv`, `unified_attribution.csv`, `attribution_summary.json`, `mea_global_shift.csv`, `winsorized_sites.csv` |
| `attribution_recovery/` | **`kinase_hypothesis_table.csv` (primary deliverable)**, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, `bubble_plots/` |
| `wmb_expression/`, `snrna_integration/` | Supporting prerequisites |
| `supplementary/` | Reviewer-diagnostic results |
| `decomposition/levy_t5/` | per-(animal, cluster) projected bulk + per-cluster MEA |
| `incytr_pair_mode/` | 9 wide parquets + `receiver_cache/` (31² pair-mode results) |
| `kinase_attribution_human/` | Human per-donor MEA, recurrence, kinase_donor_nes |
| `unified_viewer/` | `index.html` + payload JSON + sharded edge slices |

## Viewers

| Viewer | Builder | Input | Output |
|:---|:---|:---|:---|
| Kinase + pathway | `alz/build_unified_viewer.py` | `kinase_hypothesis_table.csv`, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, `mea_stoichiometry.csv`, `site_level_ols.csv`, `backbone_recurrence_by_contrast.csv`, `backbone_permutation_pvalues_by_contrast.csv`, `unified_attribution.csv`, `kinase_backbone_edges.parquet` | `outputs/reports/unified_viewer/index.html` + payload JSON + sharded edge slices |
