# Repository Retention Policy

> **Note (2026-04-08):** The `archive/` directory was purged from git history to reduce repo size from 3.6 GB to ~5 MB. References to `archive/` paths below are retained as a decision record of what was archived and why. As of 2026-04-20, `docs/archive/` is gitignored — local files persist for provenance but are not tracked.

This document defines what remains mainline, what remains as supporting evidence, what is archived for provenance, which code paths should not be reopened, and the file-level inventory that backs those labels.

It is administrative and reproducibility-focused. The live scientific program is defined by [`analysis_charter.md`](./analysis_charter.md), justified by [`analysis_rationale.md`](./analysis_rationale.md), and bounded by [`statistical_constraints.md`](./statistical_constraints.md).

## Purpose

The repository contains material from multiple phases of the project:

- the live 72-sample stoichiometry-based analysis path,
- supporting integration and validation assets,
- archived records from the retired deconvolution-centered program.

These materials should not be treated as if they have the same status. This policy keeps the main path operational without deleting the provenance needed to justify why earlier branches closed.

## Classification Rules

| Class | Meaning |
|:---|:---|
| `main` | Required for current scientific claims and the live pipeline front door |
| `supporting` | Required inputs, references, provenance, or operational wrappers that support the main pipeline but are not co-equal analytic fronts |
| `supplementary` | Reviewer-response diagnostics that validate pipeline choices without modifying the main pipeline (read from canonical CSV outputs; no imports from main modules) |
| `archived` | Historical, compatibility, validation, or side-workflow code not needed for the current mainline conclusions |

Generated caches (`alz/__pycache__/`, etc.) are excluded from this index.

## Main

### Live documentation

- `docs/foundation/analysis_charter.md`
- `docs/foundation/analysis_rationale.md`
- `docs/foundation/statistical_constraints.md`
- `docs/foundation/live_pipeline_contract.md`
- `docs/foundation/concordance.md`
- `docs/foundation/repo_retention_policy.md` (this file)
- `docs/INDEX.md`

### Main code

| Path | Reason |
|:---|:---|
| `alz/data_ingest.py` | Live Stage 1 total-proteome ingestion and characterization |
| `alz/kinase_attribution.py` | Live stoichiometry, MEA enrichment, and unified cell-type attribution |
| `alz/attribution_recovery.py` | Live final attribution-table assembly |
| `alz/runners/main/run_data_ingest.sh` | Operational wrapper for data ingestion |
| `alz/runners/main/run_kinase_attribution.sh` | Operational wrapper for kinase attribution |
| `alz/runners/main/run_attribution_recovery.sh` | Operational wrapper for attribution recovery |
| `alz/runners/main/run_live_pipeline.sh` | Bundled end-to-end live runner |
| `alz/runners/main/run_dual_analysis.sh` | Dual-track (males-only primary + full-cohort sensitivity) |

### Outputs that must remain reproducible

- the 72-animal phospho-to-protein mapping and stoichiometry outputs
- the MEA kinase enrichment results on stoichiometry beta values
- the unified cell-type attribution (SEA-AD + WMB + Song concordance)
- the cross-contrast consistency analysis
- the final attribution table (`kinase_hypothesis_table.csv`)

## Supporting

### Integration and provenance references

- `docs/integrations/`
- `data/raw/external/gdrive_shared/` (on-demand via `pixi run ingest-gdrive-shared`)
- `data/raw/external/lucie_proteomics/` (on-demand via `pixi run ingest-lucie-proteomics`)

### Supporting code

| Path | Reason |
|:---|:---|
| `alz/config.py` | Shared configuration (still structurally mixed with legacy settings) |
| `alz/atlas_reference.py` | External-reference acquisition for SEA-AD and WMB |
| `alz/wmb_expression.py` | WMB expression export (required for unified attribution) |
| `alz/snrna_integration.py` | Song snRNA-seq pseudobulk, specificity, and concordance |
| `alz/plot_attribution_bubbles.py` | Attribution visualization |
| `alz/build_unified_viewer.py` | Kinase + pathway HTML viewer (cross-entity) |
| `alz/map_kinases_to_genes.py` | Shared kinase→gene mapping utility |
| `alz/lucie_5xfad_manifest.py` | Lucie 5xFAD integration/provenance utility |
| `alz/runners/supporting/run_atlas_reference.sh` | Atlas/reference setup |
| `alz/runners/supporting/run_wmb_expression.sh` | WMB expression export |
| `alz/runners/supporting/run_snrna_integration.sh` | snRNA-seq integration |
| `alz/integration/**` | Kinase ↔ Incytr integration (Python adapters + R wrappers; see `docs/integrations/kinase_incytr_integration.md`) |

## Supplementary

| Path | Reason |
|:---|:---|
| `alz/supplementary/fdr_stringent.py` | Q4: MEA at FDR < 0.10 vs < 0.25 |
| `alz/supplementary/threshold_sensitivity.py` | Q1: Confidence tier threshold sweep |
| `alz/supplementary/aggregation_robustness.py` | Q2: Supertype-to-subclass aggregation comparison |
| `alz/supplementary/parent_protein_qc.py` | Q5: Parent protein QC for activity-driven kinases |
| `alz/runners/supplementary/run_reviewer_diagnostics.sh` | Bundled reviewer diagnostics runner |

## Archived

Archived assets are preserved for design history, negative-result provenance, and auditability. They must not be presented as current guidance or invoked by default.

### Archived documentation (local only, gitignored)

`docs/archive/` holds prior-phase records kept on disk for provenance:

- SAP identifiability record, factor-model failure record, rescue-branch record
- Superseded transition summaries, atlas-series working notes
- Receiver-centric refactor narrative, Incytr input validation log, integration audit logs

### Archived code (purged from git 2026-04-08)

Removed from the working tree; decision record preserved here:

- `archive/code/sap_*.py` — retired SAP path (data, model, validate, perf_test, preflight, factor_model, model_2comp, module1_de, module2_triangulation, module5b_analysis, module5c_correlation, diagnostic_figures, tier_annotation)
- `archive/code/kl_analysis_*.py` — deconvoluted / bulk kinase-enrichment discovery paths
- `archive/code/{analysis_utils,downstream_utils,permutation_correction,plotting_utils}.py` — utility layer for retired paths
- `archive/code/analyze_{sensitivity,substrate_overlap,temporal_trajectories}.py` — retired side analyses
- `archive/code/{aptt_additivity_analysis,compare_corrections,compare_module6_sap,export_song_aobs_desp}.py`
- `archive/code/r/{run_bmind,reconstruct_5xad_seurat,run_incytr_ad_models_yuyu01,run_incytr_5xfad_lore00}.R`
- `archive/runners/{run_module5c,run_module5b,run_factor_model_validation,run_kinase_aggregate_validation,run_rerun_matrix}.sh`

## Reproducibility Requirements

1. Do not delete material needed to justify why a path was closed unless an equivalent archival record already exists.
2. Do not remove code or documentation required to regenerate the active 72-sample results.
3. If an asset is cited as provenance, keep either the original file or a stable archival replacement with an updated pointer.
4. If files move, update `docs/INDEX.md`, the README, and any supersession banners.
5. Keep archive documents clearly labeled as provenance records rather than live guidance.
6. Preserve enough context around negative-result branches that future contributors can see they were tested and closed for structural reasons.

## Banned Paths For New Work

1. Direct cell-type deconvolution from the 24-group composition design
2. Joint kinase-activity factor-model rescue
3. Two-compartment neuronal-versus-glial rescue
4. Transcript-only rescue as the main attribution route
5. Naive pre-stoichiometry atlas concordance as the primary cell-type attribution method

Operational bans:

- Do not present archived outputs as confirmed live findings.
- Do not use archived documents as the front door for contributor guidance.
- Do not add new one-off front-door guidance outside `docs/foundation/` without updating the document map.

## Bottom Line

Keep the live stoichiometry-based attribution workflow easy to find and reproduce. Keep supporting evidence available when it strengthens interpretation or provenance. Keep failed branches archived, visible, and clearly marked as closed.
