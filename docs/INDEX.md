# Document Map

Read by analytical role, not creation order.

## Start Here

| File | Role |
|:---|:---|
| [`pipeline_overview.qmd`](./pipeline_overview.qmd) | End-to-end narrative of the full pipeline (bulk + Incytr) with cross-references into every foundation spec (renders to `pipeline_overview.html`) |

## Foundation (authoritative live specs)

| File | Role |
|:---|:---|
| [`foundation/analysis_charter.md`](./foundation/analysis_charter.md) | Single source of truth for the live 72-sample analysis program |
| [`foundation/live_pipeline_contract.md`](./foundation/live_pipeline_contract.md) | Stage-by-stage runner contract: prerequisites, outputs, failure modes |
| [`foundation/concordance.md`](./foundation/concordance.md) | SEA-AD + Song concordance model: evidence sources, weights, confidence tiers |
| [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md) | Why the project pivoted from deconvolution to stoichiometry |
| [`foundation/statistical_constraints.md`](./foundation/statistical_constraints.md) | Governing identifiability and interpretation limits |
| [`foundation/multiple_testing.md`](./foundation/multiple_testing.md) | Multiple-testing policy across the pipeline |
| [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) | Retention policy + active vs archived inventory; banned code paths |
| [`foundation/mukesh_ingest_policies.yml`](./foundation/mukesh_ingest_policies.yml) | Mukesh / NBB human ingest edge-case policies (consumed by `ingest_mukesh.py`) |

## Reference Guides

Stable docs that aren't authoritative specs but aren't plans either.

| File | Role |
|:---|:---|
| [`report_writing_checklist.md`](./report_writing_checklist.md) | Reviewer-facing report writing guidance |
| [`result_analysis_plan.md`](./result_analysis_plan.md) | Interpretation framework: how to move from generated outputs to biological claims |
| [`allen_ctx_hpf_disagreement.md`](./allen_ctx_hpf_disagreement.md) | Reviewer FAQ: why our cell-type verdicts may differ from the Allen ctx+HPF Transcriptomics Explorer |
| [`viewer_style.md`](./viewer_style.md) | Writing-style guide for unified-viewer copy (panels, drawers, tooltips) |

## Integrations

| File | Role |
|:---|:---|
| [`integrations/kinase_incytr_integration.md`](./integrations/kinase_incytr_integration.md) | Current pair-mode integration architecture; in-tree file inventory; data-flow diagram |
| [`integrations/5xfad-lucie-manifest.json`](./integrations/5xfad-lucie-manifest.json) | Local inventory of Lucie 5xFAD upstream files |

## Active Plans

Live plans in `docs/plans/`. Only the **currently-executing** master plan
lives here; everything else has been moved to `docs/archive/plans/` (see
Archive below). Plans graduate back to active status by explicit user
decision — the absence of an "active" plan does not imply the work is done.

| File | Status |
|:---|:---|
| [`plans/repo_organization_2026-05-21.md`](./plans/repo_organization_2026-05-21.md) | Master plan: two-layer architecture (bespoke ingest → canonical artifacts → shared analyses); methodical sequencing |

## Archive

`docs/archive/` is gitignored. Local files persist on disk for provenance but
are not tracked. See [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) §Archived for what's there and why.

Archived material includes:

- Completed campaigns (e.g. CR-03 adoption, factorial-era Incytr docs)
- Superseded plans (WMB-34 spine, levy-19 spine, factorial deconvolution attempts)
- Historical audits + investigation notes

## Reading Rule

- **What should we do next?** → [`foundation/`](./foundation/analysis_charter.md) + [`plans/repo_organization_2026-05-21.md`](./plans/repo_organization_2026-05-21.md)
- **How do external inputs map into runtime?** → [`integrations/kinase_incytr_integration.md`](./integrations/kinase_incytr_integration.md)
- **Why was a path closed?** → [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md)
- **What plans are open?** → [`plans/`](./plans/)
- **Historical context** → `archive/` (on disk; not tracked)

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
| Config | `config.py` | — | — |
| 1. Ingest | `data_ingest.py` / `kedro run --pipeline=ingest_mapping` | `runners/main/run_data_ingest.sh` | `outputs/reports/data_ingest/` |
| 2. Normalize | `kinase_normalize.py` / `kedro run --pipeline=normalize` | `runners/main/run_kinase_attribution.sh` (2–4) | `outputs/reports/kinase_attribution/` |
| 3. Enrich | `kinase_enrich.py` / `kedro run --pipeline=enrich` | (above) | `outputs/reports/kinase_attribution/` |
| 4. Attribute | `kinase_attribute.py` / `kedro run --pipeline=attribute` | (above) | `outputs/reports/kinase_attribution/` |
| 5. Recovery | `attribution_recovery.py` / `kedro run --pipeline=recovery` | `runners/main/run_attribution_recovery.sh` | `outputs/reports/attribution_recovery/` |
| Optional: mechanism | `kinase_mechanism.py` / `kedro run --pipeline=mechanism` | — | `outputs/reports/kinase_attribution/` |
| Plots | `plot_attribution_bubbles.py` | — | `outputs/reports/attribution_recovery/bubble_plots/` |
| Bundled | — | `runners/main/run_live_pipeline.sh` | all of the above |
| Dual-track | — | `runners/main/run_dual_analysis.sh` | `*_males_only/`, `*_full_cohort/` |
| End-to-end | — | `runners/main/run_all.sh` (= `pixi run all`) | everything |

### Mouse decomposition + Incytr pair-mode (not yet under kedro)
| Stage | Script | Runner |
|:---|:---|:---|
| snRNA pseudobulk + concordance | `snrna_integration.py` | `runners/supporting/run_snrna_integration.sh` |
| Per-(animal, cluster, gene) proportions | `snrna_proportions.py` | (part of decomposition rerun) |
| Per-cluster decomposition | `decomposition/build_celltype_decomposition.py` | `runners/main/rerun_decomposition_chain.sh` |
| Per-cluster MEA | `decomposition/enrich_celltype.py` | (above) |
| Decomposition verification | `decomposition/verify_decomposition.py` | (above) |
| Incytr pair-mode | `incytr/*` + `integration/pair_to_receiver_cache.py` | `runners/main/run_pair_mode_pipeline.sh` |

### Human cohort pipeline (not yet under kedro)
| Stage | Script | Runner |
|:---|:---|:---|
| Ingest | `ingest_mukesh.py` | `runners/main/run_mukesh_perdonor.sh` |
| Per-donor MEA | `ingest_mukesh_perdonor.py` | (above) |
| SEA-AD agreement | `seaad_human_agreement.py` | (above; = `pixi run human`) |

### Supporting (`alz/`)
| Script | Runner | Output dir |
|:---|:---|:---|
| `atlas_reference.py` | `runners/supporting/run_atlas_reference.sh` | `data/external/sea_ad/`, `data/external/allen_abc/` |
| `wmb_expression.py` | `runners/supporting/run_wmb_expression.sh` | `outputs/reports/wmb_expression/` |

### Supplementary diagnostics (`alz/supplementary/`)
`fdr_stringent.py`, `threshold_sensitivity.py`, `aggregation_robustness.py`, `parent_protein_qc.py`, `deconvolution_feasibility.py` — run via `runners/supplementary/run_reviewer_diagnostics.sh`. Output: `outputs/reports/supplementary/`.

### Standalone utilities (`alz/`)
| Script | Purpose |
|:---|:---|
| `map_kinases_to_genes.py` | Kinase → gene symbol mapping; emits `data/derived/caches/kinase_to_gene_mapping.csv` |
| `lucie_5xfad_manifest.py` | Proteomics manifest builder for Lucie 5xFAD integration |
| `build_unified_viewer.py` | See Viewers section below |

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
