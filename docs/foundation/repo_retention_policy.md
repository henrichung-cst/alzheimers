# Repository Retention Policy

This document defines what remains mainline, what remains as supporting evidence, what is archived for provenance, and which code paths should not be reopened for new work.

It is administrative and reproducibility-focused. The live scientific program is defined by [`analysis_charter.md`](./analysis_charter.md), justified by [`analysis_rationale.md`](./analysis_rationale.md), bounded by [`statistical_constraints.md`](./statistical_constraints.md), and made file-explicit in [`repo_surface_index.md`](./repo_surface_index.md).

## Purpose

The repository now contains material from multiple phases of the project:

- the live 72-sample stoichiometry-based analysis path,
- supporting integration and validation assets,
- archived records from the retired deconvolution-centered program.

These materials should not be treated as if they have the same status. This policy keeps the main path operational without deleting the provenance needed to justify why earlier branches closed.

## Retention Classes

### 1. Main

Main assets define and run the current analysis program. They are the only front door for new work and should stay easy to find and reproducible.

### 2. Supporting

Supporting assets are retained because they document inputs, validation, reviewer-facing context, or mechanistic bridge results that still matter to interpretation. They may be cited in methods, supplement, or provenance notes, but they are not co-equal with the main pipeline.

### 3. Archived

Archived assets are preserved for design history, negative-result provenance, and auditability. They should remain accessible, but they must not be presented as current guidance or invoked by default.

The repository keeps executable code under `code/` unless a surface is moved fully into `archive/`. The labels `main`, `supporting`, and `archived` are classifications, not top-level directory names.

## Main Assets

The following assets are mainline and should remain the only contributor-facing front door.

### Live documentation

- `docs/foundation/analysis_charter.md`
- `docs/foundation/analysis_rationale.md`
- `docs/foundation/statistical_constraints.md`
- `docs/foundation/repo_retention_policy.md`
- `docs/foundation/repo_surface_index.md`
- `docs/foundation/live_pipeline_contract.md`
- `docs/sap_document_map.md`

These are the authoritative package for understanding the current project.

### Operational data surfaces

- `data/incytr_collections/song/` is the authoritative Song workspace
- `data/incytr_collections/song/proteomics/` contains the active regenerated Song `pr` / `ps` / `py` bundle
- `data/incytr_collections/song/proteomics/legacy/` is retained inside the operational workspace as provenance, but the regenerated bundle is the default runtime source

### Main code paths

The mainline code surface should continue to support:

- total-proteome ingestion and characterization via `code/data_ingest.py`
- stoichiometry-corrected MEA enrichment and unified attribution via `code/kinase_attribution.py`
- final attribution-table assembly via `code/attribution_recovery.py`
- operational wrappers via `code/runners/main/run_data_ingest.sh`, `code/runners/main/run_kinase_attribution.sh`, and `code/runners/main/run_attribution_recovery.sh`
- bundled end-to-end runner via `code/runners/main/run_live_pipeline.sh`

### Outputs that must remain reproducible

At minimum, the repository should retain enough code, data references, and documentation to regenerate:

- the 72-animal phospho-to-protein mapping and stoichiometry outputs
- the MEA kinase enrichment results on stoichiometry beta values
- the unified cell-type attribution (SEA-AD concordance + WMB expression)
- the cross-contrast consistency analysis
- the final attribution table

## Supporting Assets

Supporting assets should remain available and documented, but they do not define the live program on their own.

### Integration and provenance references

- `docs/integrations/`
- `data/gdrive_shared/`
- `data/lucie_proteomics/`

These remain important for upstream provenance, input validation, and external-bundle mapping. They are not the default runtime surface for Song.

### Supporting scientific context

The following categories remain useful because they support interpretation or manuscript-facing justification:

- the RNA-phospho decoupling result and its summaries
- external atlas acquisition and taxonomy mapping used to support the attribution workflow
- reviewer-facing writing guidance in `docs/report_writing_checklist.md`
- key identifiability summaries, diagnostic summaries, and validation summaries that justify why the retired paths were closed

Supporting outputs may remain under `outputs/reports/` even when the generating code path is no longer part of the default live workflow.

### Supporting code that may still be useful

These code paths may remain accessible for reproducing supporting tables or bridge results, but they are not the front door for new work:

- `code/config.py`
- `code/atlas_reference.py`
- `code/wmb_expression.py`
- `code/map_kinases_to_genes.py`
- `code/lucie_5xfad_manifest.py`
- `code/runners/supporting/run_atlas_reference.sh`
- `code/runners/supporting/run_wmb_expression.sh`

## Archived Assets

Archived assets must remain available for provenance, but they should be clearly separated from the live package.

### Archived documentation

The archival SAP record lives under:

- `archive/sap_docs/legacy_design/`
- `archive/sap_docs/transitional_notes/`
- `archive/sap_docs/atlas_working_notes/`

These documents preserve:

- the full 24-group identifiability record,
- the factor-model failure record,
- the rescue-branch record,
- superseded transition summaries,
- atlas-series working notes.

### Archived benchmark workspace

- `archive/deconv/`

This is the retained benchmark and transition workspace. It should not be treated as the primary analysis surface.

### Archived code paths

The following code paths are retained only for provenance, auditability, or regeneration of archived/supporting negative results:

All archived code now lives under `archive/`:

- compatibility and side-workflow surfaces:
  - `archive/code/export_song_aobs_desp.py`
  - `archive/code/kl_analysis_clusters.py`
  - `archive/code/kl_analysis_bulk.py`
  - `archive/code/analysis_utils.py`
  - `archive/code/downstream_utils.py`
  - `archive/code/permutation_correction.py`
  - `archive/code/plotting_utils.py`
  - `archive/code/analyze_sensitivity.py`
  - `archive/code/analyze_substrate_overlap.py`
  - `archive/code/analyze_temporal_trajectories.py`
  - `archive/code/aptt_additivity_analysis.py`
  - `archive/code/compare_corrections.py`
- direct deconvolution stack:
  - `archive/code/sap_data.py`
  - `archive/code/sap_model.py`
  - `archive/code/sap_validate.py`
  - `archive/code/sap_perf_test.py`
- factor-model branch:
  - `archive/code/sap_preflight.py`
  - `archive/code/sap_factor_model.py`
- transcript-only rescue branch:
  - `archive/code/sap_module1_de.py`
  - `archive/code/sap_module2_triangulation.py`
- two-compartment rescue branch:
  - `archive/code/sap_model_2comp.py`
- pre-stoichiometry atlas concordance branches:
  - `archive/code/sap_module5b_analysis.py`
  - `archive/code/sap_module5c_correlation.py`
- legacy diagnostic and example runners:
  - `archive/code/sap_diagnostic_figures.py`
  - `archive/code/r/run_bmind.R`
  - `archive/code/r/reconstruct_5xad_seurat.R`
  - `archive/code/r/run_incytr_ad_models_yuyu01.R`
  - `archive/code/r/run_incytr_5xfad_lore00.R`
  - `archive/runners/run_module5c.sh`
  - `archive/runners/run_factor_model_validation.sh`
  - `archive/runners/run_kinase_aggregate_validation.sh`
  - `archive/runners/run_rerun_matrix.sh`

## Reproducibility Requirements

The following rules apply when cleaning up or reorganizing the repository.

1. Do not delete material that is needed to justify why a path was closed unless an equivalent archival record already exists.
2. Do not remove code or documentation required to regenerate the active 72-sample results.
3. If an asset is cited as provenance, keep either the original file or a stable archival replacement with an updated pointer.
4. If files move, update `docs/sap_document_map.md`, the README, and any supersession banners that point to them.
5. Keep archive documents clearly labeled as provenance records rather than live guidance.
6. Keep the Song runtime split explicit: `data/incytr_collections/song/` is active, `data/gdrive_shared/yuyu01/` is upstream archive.
7. Preserve enough context around negative-result branches that future contributors can see they were tested and closed for structural reasons, not forgotten accidentally.

## Banned Paths For New Work

The following paths should not be reopened as default analysis directions:

1. direct cell-type deconvolution from the 24-group composition design
2. joint kinase-activity factor-model rescue
3. two-compartment neuronal-versus-glial rescue
4. transcript-only rescue as the main attribution route
5. naive pre-stoichiometry atlas concordance as the primary cell-type attribution method

Additional operational bans:

- do not present archived outputs as confirmed live findings
- do not use archived documents as the front door for contributor guidance
- do not treat `data/gdrive_shared/yuyu01/` as the default Song runtime workspace
- do not add new one-off front-door guidance outside `docs/foundation/` without updating the document map

## Bottom Line

Keep the live stoichiometry-based attribution workflow easy to find and easy to reproduce. Keep supporting evidence available when it strengthens interpretation or provenance. Keep failed branches and old SAP material archived, visible, and clearly marked as closed.
