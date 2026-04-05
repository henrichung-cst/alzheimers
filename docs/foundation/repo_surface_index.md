# Repository Surface Index

This index makes the repository classification explicit without changing the
high-level layout. Executable code remains under `code/` unless it is moved
fully into `archive/`. The labels `main`, `supporting`, and `archived` are
semantic classifications, not top-level directory names.

Generated caches such as `code/__pycache__/` are excluded from this index.

## Classification Rules

| Class | Meaning |
|:---|:---|
| `main` | Required for the current scientific claims and the live pipeline front door |
| `supporting` | Required inputs, references, provenance, or operational wrappers that support the main pipeline but are not co-equal analytic fronts |
| `archived` | Historical, compatibility, validation, or side-workflow code not needed for the current mainline conclusions |

## Main Code

| Path | Class | Reason |
|:---|:---|:---|
| `code/data_ingest.py` | `main` | Live Stage 1 total-proteome ingestion and characterization |
| `code/kinase_attribution.py` | `main` | Live stoichiometry, MEA enrichment, and unified cell-type attribution |
| `code/attribution_recovery.py` | `main` | Live final attribution-table assembly |
| `code/runners/main/run_data_ingest.sh` | `main` | Operational wrapper for the data ingestion stage |
| `code/runners/main/run_kinase_attribution.sh` | `main` | Operational wrapper for the kinase attribution stage |
| `code/runners/main/run_attribution_recovery.sh` | `main` | Operational wrapper for the attribution recovery stage |
| `code/runners/main/run_live_pipeline.sh` | `main` | Bundled end-to-end live runner for the ordered pipeline sequence |

## Supporting Code

| Path | Class | Reason |
|:---|:---|:---|
| `code/config.py` | `supporting` | Shared configuration used by the live pipeline, but still structurally mixed with legacy settings |
| `code/atlas_reference.py` | `supporting` | External-reference acquisition and provenance setup for SEA-AD and WMB |
| `code/wmb_expression.py` | `supporting` | WMB expression export for Track B attribution (extracted from Module 5b) |
| `code/map_kinases_to_genes.py` | `supporting` | Shared mapping utility used by retained supporting surfaces |
| `code/lucie_5xfad_manifest.py` | `supporting` | Integration/provenance utility rather than a live analysis stage |
| `code/runners/supporting/run_atlas_reference.sh` | `supporting` | Supporting atlas/reference setup wrapper |
| `code/runners/supporting/run_wmb_expression.sh` | `supporting` | Targeted runner for WMB expression export (Track B dependency) |

## Supplementary Code

Reviewer-response diagnostics that validate pipeline choices without modifying
the main pipeline. Read from canonical pipeline CSV outputs; no imports from
main pipeline modules.

| Path | Class | Reason |
|:---|:---|:---|
| `code/supplementary/fdr_stringent.py` | `supporting` | Q4: Compare MEA results at FDR < 0.10 vs < 0.25 |
| `code/supplementary/threshold_sensitivity.py` | `supporting` | Q1: Sweep confidence tier thresholds, identify near-miss rows |
| `code/supplementary/aggregation_robustness.py` | `supporting` | Q2: Compare median/mean/weighted supertype-to-subclass aggregation |
| `code/supplementary/parent_protein_qc.py` | `supporting` | Q5: Parent protein quality diagnostics for activity-driven kinases |
| `code/runners/supplementary/run_reviewer_diagnostics.sh` | `supporting` | Bundled runner for all supplementary diagnostics |

## Archived Code

All archived code now lives under `archive/`:

| Path | Class | Reason |
|:---|:---|:---|
| `archive/code/sap_module4_rho.py` | `archived` | RNA-phospho coupling from rejected SAP Hurdle-Tweedie model; rho coefficients not trustworthy |
| `archive/code/sap_data.py` | `archived` | Direct deconvolution-era data layer |
| `archive/code/sap_model.py` | `archived` | Direct deconvolution-era model path |
| `archive/code/sap_validate.py` | `archived` | Validation matrix for the retired SAP path |
| `archive/code/sap_perf_test.py` | `archived` | Performance/self-test harness for the retired SAP model |
| `archive/code/sap_preflight.py` | `archived` | Factor-model/preflight branch |
| `archive/code/sap_factor_model.py` | `archived` | Retired factor-model rescue path |
| `archive/code/sap_model_2comp.py` | `archived` | Retired two-compartment rescue branch |
| `archive/code/sap_module1_de.py` | `archived` | Retired transcript-only rescue branch |
| `archive/code/sap_module2_triangulation.py` | `archived` | Retired transcript-only rescue branch |
| `archive/code/sap_module5b_analysis.py` | `archived` | Pre-stoichiometry concordance analysis; WMB expression export extracted to `wmb_expression.py` |
| `archive/code/sap_module5c_correlation.py` | `archived` | Retired pre-stoichiometry atlas-correlation branch |
| `archive/code/sap_diagnostic_figures.py` | `archived` | Diagnostic figure generation for the retired SAP path |
| `archive/code/kl_analysis_clusters.py` | `archived` | Deconvoluted kinase-enrichment discovery path |
| `archive/code/kl_analysis_bulk.py` | `archived` | Bulk kinase-enrichment side workflow |
| `archive/code/analysis_utils.py` | `archived` | Utility layer serving archived kinase-enrichment and factor-model code |
| `archive/code/downstream_utils.py` | `archived` | Helper module for archived downstream analyses |
| `archive/code/permutation_correction.py` | `archived` | Correction utilities tied to archived kinase-enrichment workflows |
| `archive/code/plotting_utils.py` | `archived` | Plotting utilities primarily used by archived kinase-enrichment surfaces |
| `archive/code/analyze_sensitivity.py` | `archived` | Archived side analysis for kinase-enrichment sensitivity |
| `archive/code/analyze_substrate_overlap.py` | `archived` | Archived kinase-enrichment side analysis |
| `archive/code/analyze_temporal_trajectories.py` | `archived` | Archived kinase-enrichment side analysis |
| `archive/code/aptt_additivity_analysis.py` | `archived` | Archived follow-up analysis outside the live Module 6 front door |
| `archive/code/compare_corrections.py` | `archived` | Archived correction-comparison analysis |
| `archive/code/sap_tier_annotation.py` | `archived` | Depends on archived sap_data; supporting annotation, not a live stage |
| `archive/code/compare_module6_sap.py` | `archived` | Depends on archived sap_data/sap_model; audit/provenance comparison |
| `archive/code/export_song_aobs_desp.py` | `archived` | Legacy Song/DESP compatibility builder, not a Module 6 prerequisite |
| `archive/code/r/run_bmind.R` | `archived` | Archived validation helper for the retired SAP path |
| `archive/code/r/reconstruct_5xad_seurat.R` | `archived` | Archived 5xFAD/InCytr preparation utility |
| `archive/code/r/run_incytr_ad_models_yuyu01.R` | `archived` | InCytr compatibility/example runner |
| `archive/code/r/run_incytr_5xfad_lore00.R` | `archived` | InCytr compatibility/example runner |
| `archive/runners/run_module5c.sh` | `archived` | Archived runner for the retired pre-stoichiometry atlas branch |
| `archive/runners/run_factor_model_validation.sh` | `archived` | Archived factor-model validation runner |
| `archive/runners/run_kinase_aggregate_validation.sh` | `archived` | Archived kinase-aggregate validation runner |
| `archive/runners/run_rerun_matrix.sh` | `archived` | Archived SAP rerun-matrix runner |
| `archive/runners/run_module5b.sh` | `archived` | Archived full Module 5b runner (replaced by targeted `run_wmb_expression.sh`) |

## Notes

- The repo front door should point users to the `main` pipeline path first, and
  only then to named `supporting` surfaces when those prerequisites are needed.
