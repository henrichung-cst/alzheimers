# Repository Retention Policy

> The `archive/` directory (repo root) is gitignored and not tracked in git; local files persist for provenance only. `archive/` paths referenced below resolve only in a local checkout that retains them.

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
- `docs/foundation/specificity_confidence.md`
- `docs/foundation/cohort_contract.md`
- `docs/foundation/multiple_testing.md`
- `docs/foundation/mechanism_attribution_contract.md`
- `docs/foundation/projected_state_mea_contract.md`
- `docs/foundation/kinase_explorer_attribution.md`
- `docs/foundation/standard_attribution_metric.md`
- `docs/foundation/backbone_incytr_track.md`
- `docs/foundation/tcell_reference.md`
- `docs/foundation/viewer_payload_contract.md`
- `docs/foundation/viewer_frontend_contract.md`
- `docs/foundation/mukesh_ingest_policies.yml`
- `docs/foundation/repo_retention_policy.md` (this file)
- `docs/INDEX.md`

### Main code

| Path | Reason |
|:---|:---|
| `alz/ingest/song.py` | Live Stage 1 Song mouse ingestion and characterization; `pixi run ingest` |
| `alz/bulk_mea/normalize.py` | Live Stage 2 IRS normalization + stoichiometry; `pixi run normalize` |
| `alz/bulk_mea/enrich.py` | Live Stage 3 factorial OLS + MEA kinase enrichment; `pixi run enrich` |
| `alz/bulk_mea/attribute.py` | Live Stage 4 unified cell-type attribution; `pixi run attribute` |
| `alz/bulk_mea/mechanism.py` | Live Stage 5: raw-phospho MEA + mechanism classification; `pixi run mechanism` |
| `alz/bulk_mea/recover.py` | Live Stage 6 final hypothesis-table assembly; `pixi run recover` |
| `alz/cohorts/{song,fivexfad,mukesh,tcells}/` | Per-cohort ingest / MEA modules; each emits the `cohort_contract` artifacts (5xFAD, Mukesh human, T-cell) |
| `alz/pipelines/ingest/` | Current Kedro wrapper for ingest nodes; downstream bulk stages remain direct module entry points |
| `alz/pipeline_registry.py`, `alz/settings.py`, `pyproject.toml` | Kedro project bootstrap |
| `conf/base/{catalog,parameters}.yml`, `conf/{full_cohort,human_nbb}/parameters.yml` | Kedro Data Catalog + parameters (cohort selection lives here) |
| `alz/runners/main/run_all.sh` | Canonical full end-to-end build (kinase + decomposition + Incytr + human + viewer); resumable sentinels; `pixi run all`. Auto-resolves WMB/SEA-AD downloads unless `--skip-atlas` |
| `alz/runners/main/run_dual_analysis.sh` | Dual-track (males-only primary + full-cohort sensitivity); `pixi run dual` |
| `alz/runners/main/run_pair_mode_pipeline.sh` | Canonical pair-mode Incytr build (inputs → Incytr → viewer reshape) |
| `alz/runners/main/rerun_decomposition_chain.sh` | Standalone per-cluster decomposition rebuild + hard verification gate |
| `alz/runners/main/run_levy_t5_attribution_rebuild.sh` | Targeted rerun: attribute → recover → viewer |
| `alz/runners/main/run_pair_mode_viewer_build.sh` | Targeted rerun: pair receiver cache → viewer |
| `alz/runners/main/run_mukesh_perdonor.sh` | Targeted human per-donor rerun |

### Outputs that must remain reproducible

- the 72-animal phospho-to-protein mapping and stoichiometry outputs
- the MEA kinase enrichment results on stoichiometry beta values
- the unified cell-type attribution (SEA-AD + WMB + Song concordance)
- the cross-contrast consistency analysis
- the final attribution table (`kinase_hypothesis_table.csv`)

## Supporting

### Integration and provenance references

- `docs/integrations/`
- `conf/data_sources.yaml` — Drive ingest manifest (consumed by the `vendor/rclone-ingest` submodule engine)
- `data/raw/external/gdrive_shared/` (on-demand via `pixi run ingest-gdrive-shared`)
- `data/raw/external/lucie_proteomics/` (on-demand via `pixi run ingest-lucie-proteomics`)

### Supporting code

| Path | Reason |
|:---|:---|
| `alz/shared/config.py` | Shared configuration, paths, thresholds, vocabularies, and sample-filter parameter loader |
| `alz/reference/atlas.py` | External-reference acquisition for SEA-AD, WMB, and HBCA |
| `alz/reference/wmb_expression.py` | WMB expression export (required for unified attribution) |
| `alz/reference/snrna_integration.py` | Song snRNA-seq pseudobulk, specificity, and concordance |
| `alz/build_unified_viewer.py` + `alz/viewer/`, `alz/viewer_shared/` | Unified Song/5xFAD/Mukesh HTML viewer (builder + cohort payload/template packages) |
| `alz/build_tcell_viewer.py` + `alz/tcell_viewer/` | Dedicated T-cell HTML viewer (builder + package) |
| `alz/cross_reference/` | Cross-cohort specificity + T-cell within-cohort reference |
| `alz/decomposition_mea/` | Per-cluster decomposition MEA (levy_t5 forward projection) + verification |
| `alz/incytr_pair/` | Pair-mode Incytr driver, runners, and decomposition export |
| `alz/shared/map_kinases_to_genes.py` | Shared kinase-to-gene mapping utility |
| `alz/ingest/lucie.py` and `alz/ingest/build_5xfad_omics_join_manifest.py` | Lucie / 5xFAD integration and provenance utilities |
| `alz/runners/supporting/run_atlas_reference.sh` | Atlas/reference setup |
| `alz/runners/supporting/run_wmb_expression.sh` | WMB expression export |
| `alz/runners/supporting/run_snrna_integration.sh` | snRNA-seq integration |
| `alz/integration/**` | Consumer side of pair-mode Incytr outputs: viewer substrates, transcript/omics traces, cluster-spine config, and verification. Legacy factorial wrappers/adapters are preserved under `archive/incytr_integration/`; see `docs/integrations/kinase_incytr_integration.md` |

## Supplementary

| Path | Reason |
|:---|:---|
| `alz/supplementary/fdr_stringent.py` | Q4: MEA at FDR < 0.10 vs < 0.25 |
| `alz/supplementary/parent_protein_qc.py` | Q5: Parent protein QC for activity-driven kinases |
| `alz/supplementary/deconvolution_feasibility.py` | Q6: Marker→composition concordance via factorial OLS — closes the deconvolution-feasibility question with disease/age/sex correction and a specificity-threshold sweep |
| `alz/runners/supplementary/run_reviewer_diagnostics.sh` | Bundled reviewer diagnostics runner |

## Archived

Archived assets are preserved for design history, negative-result provenance, and auditability. They must not be presented as current guidance or invoked by default.

### Archived documentation (local only, gitignored)

`archive/` (repo root) holds prior-phase records kept on disk for provenance —
e.g. the deconvolution-infeasibility record (`archive/deconvolution/docs/`), the
archived orchestration and standalone plans (`archive/archived_plans/`), and the
factorial-Incytr archive (`archive/incytr_factorial_2026-05-18/`).

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
