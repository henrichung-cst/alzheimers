=== docs/report_writing_checklist.md ===
# Report Writing Checklist

Guidelines for writing analysis reports that anticipate reviewer concerns, derived from reviewer feedback on the male kinase enrichment report (March 2026).

---

## 1. Annotate rather than delete — show what was filtered out

When applying any filter (expression atlas, p-value, fold-change threshold), add a column flag or annotation to the filtered item instead of silently removing it. Include a supplemental table or "near-miss" section for entries that narrowly failed a threshold but show consistent biological patterns.

**Why:** Reviewers with a biology-first mindset distrust invisible filtering. If they can't tell what was removed and why, they assume something interesting may have been lost. Making the filtering traceable at the individual-entry level builds trust and lets the reader form their own judgment.

---

## 2. Lead with pattern consistency before effect size or p-value

Present evidence of reproducibility across conditions, timepoints, and cell types as the primary argument. Frame statistical metrics (p-values, effect sizes) as supporting evidence for a pattern the reader has already seen, not as the sole basis for claiming significance.

**Why:** Biological reviewers weight convergence across independent observations more heavily than the magnitude of any single measurement. A kinase that appears weakly but consistently across every condition and timepoint is more credible to this audience than one with a single dramatic p-value in one comparison.

---

## 3. Define every metric in a terminology section before first use

Include a short definitions box near the top of the report covering all abbreviations and derived metrics (e.g., LFF vs. LFC, enrichment score, adjusted p-value). Define at the point of introduction, not implicitly through usage.

**Why:** Even momentary confusion about a metric erodes trust in the precision of everything that follows. A reader who has to infer what LFF means from table context will question whether other details were equally loose. The cost of a 3-line terminology box is zero; the cost of ambiguity is cumulative.

---

## 4. Organize around the reader's primary analytical axis

Before structuring a report, identify which dimension the reader thinks along — cell type, timepoint, condition, pathway — and use that as the primary organizational axis. Nest other dimensions within it.

**Why:** A report organized comparison-first (by timepoint and condition) frustrates a reader who thinks cell-type-first. When the structure doesn't match the reader's mental model, they have to mentally re-sort every section, which increases cognitive load and produces requests to reorganize the data. Ask early: "What's your entry point into this data?"

---

## 5. Design figures for biological reading order: direction, then magnitude, then confidence

Encode the most biologically immediate information (up/down direction) most prominently. Use continuous scales for magnitude (percentile ranks over binary cutoffs where possible). Treat statistical confidence as accessible metadata, not a primary visual channel.

**Why:** Biologist reviewers read a figure by asking: "What went up? What went down? By how much?" If p-value occupies equal visual weight as direction, it competes for attention with the information the reader actually prioritizes. Confidence is important but should not dominate the visual hierarchy.

---

## 6. Include a "near-miss" section for threshold-adjacent results

For any hard threshold applied in the analysis, explicitly present entries that fell just below the cutoff but showed strong signals on other criteria (e.g., missed p-value but had high consistency, or missed fold-change but appeared in every condition).

**Why:** Hard cutoffs are necessary for primary analysis but create anxiety about lost signal. A near-miss section demonstrates the analyst considered what was excluded and provides the reviewer with the context to evaluate whether the thresholds were appropriate — preempting the "what did we lose?" question entirely.
=== docs/sap_document_map.md ===
# SAP Document Map

Read this folder by analytical role, not by creation order. The package now has a six-document foundation layer and a separated archive layer.

## Front Door

Start here for live work:

1. [`foundation/analysis_charter.md`](./foundation/analysis_charter.md)
2. [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md)
3. [`foundation/statistical_constraints.md`](./foundation/statistical_constraints.md)
4. [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md)
5. [`foundation/repo_surface_index.md`](./foundation/repo_surface_index.md)
6. [`foundation/live_pipeline_contract.md`](./foundation/live_pipeline_contract.md)

## What Each Foundation Document Does

| File | Role |
|:---|:---|
| [`foundation/analysis_charter.md`](./foundation/analysis_charter.md) | Single source of truth for the live 72-sample analysis program |
| [`foundation/analysis_rationale.md`](./foundation/analysis_rationale.md) | Concise explanation of why the project pivoted and why the current path is defensible |
| [`foundation/statistical_constraints.md`](./foundation/statistical_constraints.md) | Governing identifiability and interpretation limits carried forward from the old SAP |
| [`foundation/repo_retention_policy.md`](./foundation/repo_retention_policy.md) | Main / supporting / archived assets, reproducibility expectations, and banned code paths |
| [`foundation/repo_surface_index.md`](./foundation/repo_surface_index.md) | Explicit file-level `main` / `supporting` / `archived` inventory within the existing repo layout |
| [`foundation/live_pipeline_contract.md`](./foundation/live_pipeline_contract.md) | Stage-by-stage live runner contract: prerequisites, outputs, failure modes, and ordered run sequence |

## Active Package

The live analysis story is now:

- total proteome integration,
- stoichiometry correction,
- mechanism stratification,
- Track A abundance-coupled attribution,
- Track B activity-driven attribution,
- final attribution-table assembly.

That whole story should be readable from `foundation/` without opening archive material.

## Integrations Layer

The `integrations/` folder is a sidecar reference layer for external dataset bundles, InCytr input mapping, and provenance checks. It is not a replacement for the live SAP foundation, but it is the place to look when the question is "what external bundle do we actually have and how does it map into the runtime workflow?"

### Integration references

- [`integrations/integrations-structure.md`](./integrations/integrations-structure.md)
- [`integrations/alzheimers-incytr-input-validation.md`](./integrations/alzheimers-incytr-input-validation.md)

These documents explain:

- how the upstream collaborator-owned archive is organized,
- which locations are upstream provenance versus current operational workspaces,
- which Alzheimer bundles are currently plausible InCytr inputs,
- what has been validated directly versus what remains only cohort-level or post-collapse inference.

### Machine-readable manifest

- [`integrations/5xfad-lucie-manifest.json`](./integrations/5xfad-lucie-manifest.json)

This is a structured inventory of the local Lucie 5xFAD `.sne` files and should be treated as supporting integration evidence rather than a narrative guidance document.

## Archive Layer

### Legacy design records

- [`archive/legacy_design/sap_24group_identifiability_record.md`](../archive/sap_docs/legacy_design/sap_24group_identifiability_record.md)
- [`archive/legacy_design/sap_factor_model_failure_record.md`](../archive/sap_docs/legacy_design/sap_factor_model_failure_record.md)
- [`archive/legacy_design/sap_rescue_record.md`](../archive/sap_docs/legacy_design/sap_rescue_record.md)

These preserve the design constraints and the failed rescue branches that justify the live program.

### Transitional notes

- [`archive/transitional_notes/sap_atlas.md`](../archive/sap_docs/transitional_notes/sap_atlas.md)
- [`archive/transitional_notes/sap_primary_path_summary.md`](../archive/sap_docs/transitional_notes/sap_primary_path_summary.md)
- [`archive/transitional_notes/sap_cleanup.md`](../archive/sap_docs/transitional_notes/sap_cleanup.md)
- [`archive/transitional_notes/sap_rewrite.md`](../archive/sap_docs/transitional_notes/sap_rewrite.md)

These are superseded intermediary summaries from the cleanup phase.

### Atlas working notes

- [`archive/atlas_working_notes/sap_atlas_part2.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part2.md)
- [`archive/atlas_working_notes/sap_atlas_part3.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part3.md)
- [`archive/atlas_working_notes/sap_atlas_part4.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part4.md)
- [`archive/atlas_working_notes/sap_atlas_part5.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_part5.md)
- [`archive/atlas_working_notes/sap_atlas_series_distilled.md`](../archive/sap_docs/atlas_working_notes/sap_atlas_series_distilled.md)

These are provenance records for the atlas-series exploration and should not be treated as live guidance.

## Reading Rule

- If the goal is to understand what the team should do next, stay inside [`foundation/`](./foundation/analysis_charter.md).
- If the goal is to determine where an external Alzheimer/InCytr bundle lives or how upstream inputs map into the local runtime layout, open [`integrations/`](./integrations/integrations-structure.md).
- If the goal is to justify why a path was closed, open [`archive/legacy_design/`](../archive/sap_docs/legacy_design/sap_24group_identifiability_record.md).
- If the goal is historical context, open the transitional notes or atlas working notes.
=== docs/foundation/analysis_charter.md ===
# Analysis Charter

This is the single authoritative front door for the live analysis program. It defines what the project is doing now, what outputs matter, and what new code should support.

Use [`analysis_rationale.md`](./analysis_rationale.md) for the pivot logic, [`statistical_constraints.md`](./statistical_constraints.md) for hard design limits, and [`repo_retention_policy.md`](./repo_retention_policy.md) for active-versus-archived boundaries.

## Scope

The live program is the 72-sample pathway:

1. integrate the 72-animal total proteome (data ingestion),
2. compute phospho-to-protein stoichiometry and run MEA kinase enrichment,
3. run unified cell-type attribution (SEA-AD concordance + WMB expression specificity),
4. assemble the final attribution table with cross-contrast consistency.

## Closed Paths

These paths are closed and should not drive new code:

| Path | Why it is closed |
|:---|:---|
| Direct cell-type deconvolution from the 24-group design | Cell-type-specific condition effects are not identifiable |
| Joint kinase-activity factor model | Parameter reduction did not overcome the composition bottleneck |
| Two-compartment neuronal/glial simplification | Synthetic validation still failed |
| Transcript-only rescue | Matched RNA did not produce defensible cell-type attribution |
| Pre-stoichiometry atlas concordance | Mixed abundance and activity signal diluted concordance |

## Live Workflow

### 1. Total proteome integration

The total proteome is an enabling dataset, not a side analysis.

Retained facts:

- all 72 animals were mapped successfully,
- 14,772 of 16,114 phosphosites matched to parent proteins (`91.7%`),
- 2,697 of 3,117 unique parent proteins were matched (`86.5%`),
- kinase coverage was sufficient for mechanism classification,
- marker-protein behavior was too weak for composition recovery.

Operational consequence: stoichiometry is feasible; marker-based deconvolution is not.

### 2. Stoichiometry correction

Stoichiometry is the transition that turns the project onto its viable path:

`stoichiometry = log2(phospho) - log2(parent protein abundance)`

This removes parent-protein abundance confounding and exposes which kinase signals remain significant as activity-regulated events.

### 3. MEA kinase enrichment

All stoichiometry beta values are submitted to MEA (Motif Enrichment Analysis, GSEA-based) to produce a continuous NES (Normalized Enrichment Score) per kinase per contrast, without requiring arbitrary binarization cutoffs. Significance threshold: FDR < 0.25 (standard GSEA).

### 4. Unified cell-type attribution

All MEA-significant kinases are evaluated against both evidence sources at the subclass level (24 SEA-AD subclasses):

1. **SEA-AD concordance**: For each kinase gene, look up its differential expression in human AD (SEA-AD, 139 supertypes aggregated to 24 subclasses). Concordance score = `sign(NES) * median(sea_ad_lfc)` — positive when kinase activity direction matches human AD transcriptomic change in that subclass.

2. **WMB expression specificity**: How specifically each kinase gene is expressed in each of the 24 subclasses (Allen WMB 10Xv3 HPF dataset).

Combined confidence (thresholds expressed as multiples of uniform = 1/24):
- **High**: Concordant SEA-AD signal + WMB specificity >= 2x uniform + |LFC| > 0.1
- **Moderate**: Either SEA-AD or WMB evidence is strong
- **Low**: Weak evidence from both sources

### 5. Mechanism annotation (supplementary)

Optionally, the pipeline can classify kinases as abundance-driven, activity-driven, or both by comparing raw phospho MEA vs stoichiometry MEA significance. This is a descriptive annotation, not a routing variable for attribution.

### 6. Final attribution assembly

Attribution recovery adds cross-contrast consistency analysis and assembles the final attribution table. This is the main downstream deliverable.

## Outputs New Code Must Support

Code aligned to the live program should preserve:

- the 72-animal phospho-to-protein mapping,
- stoichiometry-corrected site-level OLS results,
- MEA kinase enrichment on stoichiometry beta values,
- unified cell-type attribution combining SEA-AD concordance and WMB expression,
- cross-contrast consistency analysis,
- the merged final attribution table.

## Rules For New Work

1. Treat stoichiometry correction as part of the primary workflow, not as an optional sensitivity analysis.
2. Attribution uses unified evidence (SEA-AD + WMB) for all significant kinases.
3. Do not reopen direct deconvolution or factor-model code paths for attribution.
4. Treat old deconvolution outputs only as provenance.

## Bottom Line

The live program is a stoichiometry-corrected, MEA-enriched, unified-attribution workflow. New documentation and new code should be judged against that charter, not against the retired 24-group deconvolution objective.
=== docs/foundation/analysis_rationale.md ===
# Analysis Rationale

This document explains why the project pivoted. Its purpose is not to narrate every failed branch; it is to preserve the minimum logic required to understand why the current path is defensible.

## 1. Why Direct Deconvolution Closed

The original question was whether cell-type-specific condition effects could be recovered directly from the 24-group bulk phosphoproteomics design. That path is closed because the composition matrix is rank-limited and the weak directions amplify noise too strongly for per-site cell-type condition recovery. The key conclusion is structural: the limitation comes from the design, not from a missing implementation detail.

Operational consequence: direct deconvolution outputs are negative-result provenance, not live biological findings.

## 2. Why Transcript-Only Rescue Closed

Matched pseudobulk RNA did not reopen that path. The snRNA-seq kinase/phosphatase differential-expression pipeline produced no FDR-significant rescue signal at the needed cell-type resolution, cross-modal triangulation did not exceed chance (`p = 0.556`), and even collapsing the problem to neuronal versus glial compartments still failed synthetic validation. Transcript-only rescue therefore closed for the same reason as direct deconvolution: it could not generate defensible cell-type attribution from the available design.

## 3. What Opened The Viable Path

The breakthrough came from changing the question rather than forcing the failed estimator class. The live path uses the 72-animal total proteome to compute phospho-to-protein stoichiometry, then separates kinase findings into abundance-driven, both, and activity-driven classes. That mechanism split produced the first defensible attribution signal because it stops treating all kinase signal as if it should map through the same transcriptomic logic.

The key retained numbers are:

- 14,772 of 16,114 phosphosites matched to parent proteins (`91.7%`),
- 379 abundance-driven kinase entries,
- 101 entries significant in both raw phospho and stoichiometry,
- 12 activity-driven kinases revealed only after stoichiometry correction,
- 214 of 492 final kinase-contrast entries attributed.

## 4. Why The Mechanism Split Is Biologically Justified

The mechanistic bridge is the RNA-phospho coupling analysis from the rescue effort. The double-transgenic `ApTt` condition shows strong RNA-phospho decoupling with Kruskal-Wallis `p = 1.4e-42`, concentrated in AD-relevant proline-directed kinase families rather than diffuse noise.

Retained family-level signals:

- `GSK3` with the strongest negative `ApTt` coupling,
- `MAPK/JNK` with strong negative `ApTt` coupling,
- `MAPK/ERK` with reversal from positive coupling in `Ttau` to negative in `ApTt`,
- `DYRK` with strong negative `ApTt` coupling.

This is why the live program separates abundance-coupled from activity-driven behavior:

- abundance-coupled kinase changes can be interrogated with transcriptomic concordance,
- activity-driven kinase changes cannot be expected to track transcript levels,
- `ApTt` is the condition where that distinction matters most.

## 5. Resulting Project Logic

The project now has a clean decision structure:

1. direct deconvolution is closed,
2. transcript-only rescue is closed,
3. stoichiometry plus mechanism stratification is the active route,
4. Track A and Track B should remain distinct by design,
5. the RNA-phospho decoupling result is the mechanistic explanation for that split.

## Bottom Line

The project pivoted because the old question was statistically non-identifiable, while stoichiometry plus mechanism stratification changed the question into one the data could answer.
=== docs/foundation/statistical_constraints.md ===
# Statistical Constraints

This document preserves the irreducible statistical facts that define the boundary conditions of the project. It is not a methods plan and it is not a recommendation to reopen the retired estimators.

## 1. Design Facts

The original deconvolution problem is anchored to a complete `2 x 3 x 4` factorial design:

- 2 sexes,
- 3 timepoints,
- 4 conditions,
- 24 bulk phosphoproteomics groups total,
- zero biological replication within group.

Each group is paired to a composition vector from matched snRNA-seq, but the composition matrix is the limiting object for cell-type-specific condition recovery.

## 2. Identifiability Boundary

The pooled composition matrix has effective rank near 2. The retained singular values are:

- `1.37`
- `0.14`
- `0.04`
- `0.02`
- `0.02`

Only the first two directions sit meaningfully above the noise floor. Inverting the weak directions amplifies noise by roughly:

- `1 / 1.37 ~= 0.7x` on the best direction,
- `1 / 0.14 ~= 7x`,
- `1 / 0.04 ~= 25x`,
- `1 / 0.02 = 50x` on the weakest directions.

That is the core identifiability boundary: the design does not contain enough independent composition variation to support unique cell-type-specific condition attribution at the per-site level.

## 3. Why Per-Site Recovery Fails

With 6 samples per condition and bulk residual MAD near 100 intensity units, the standard error of a bulk condition mean is about 41. After projection into weak composition directions, that error inflates into the hundreds or thousands, exceeding the effect sizes used in synthetic validation. The resulting per-site cell-type signal-to-noise ratio stays far below 1 in the poorly identified directions.

Practical implication: the data can support bulk condition effects, but not reliable decomposition of those effects across cell types.

## 4. Evidence That The Failure Is Structural

The non-identifiability conclusion is not based on one failed implementation.

Retained evidence:

- an 8-phase synthetic validation campaign returned essentially zero recovery,
- four additional audit-driven rescue strategies also returned `r ~= 0`,
- a joint kinase-activity factor model reduced the parameter count from roughly `146K` to `1,332` but still returned `r ~= 0`,
- two-compartment neuronal/glial collapse did not restore recovery,
- matched transcript-based rescue did not exceed chance.

The factor-model failure matters because it shows that parameter reduction and kinase-substrate structure did not solve the real problem. The bottleneck is the composition geometry, not the size of the parameterization.

## 5. Why Diagnostics Do Not Reopen The Old Path

The Hurdle-Tweedie deconvolution stack passed important internal diagnostics. That result should be retained, but interpreted correctly:

- it means the old model was built coherently,
- it argues against a trivial coding or fitting bug,
- it does not rehabilitate the model for biological attribution.

This distinction is important because it prevents future work from reopening the old branch under the assumption that better tuning alone would fix it.

## 6. Supported And Unsupported Statistical Claims

Supported:

- bulk phospho condition effects,
- total-proteome-enabled stoichiometry analysis,
- mechanism classification into abundance-driven, both, and activity-driven classes,
- Track A concordance results for abundance-coupled classes,
- Track B expression-constrained attributions for activity-driven kinases,
- the merged final attribution table within the current mechanism-stratified framework.

Not supported by the retired 24-group design:

- unique per-site cell-type condition estimates from direct deconvolution,
- claims that factor-model or compartment-collapse rescues solved the attribution problem,
- presenting old deconvolution outputs as confirmed biological localization.

## 7. Downstream Interpretation Rules

Use these constraints as guardrails:

1. treat old deconvolution outputs as provenance only,
2. treat bulk-level findings as the strongest directly supported signal from the 24-group branch,
3. keep mechanism-stratified attribution separate from retired deconvolution claims,
4. do not describe archived estimators in enough detail that they read like recommended live methods.

## Bottom Line

The project’s governing statistical fact is that cell-type-specific condition attribution is not uniquely determined by the original 24-group composition design. The live program works by changing the target of inference, not by rescuing that retired inverse problem.
=== docs/foundation/repo_retention_policy.md ===
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
=== docs/foundation/repo_surface_index.md ===
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
=== docs/foundation/live_pipeline_contract.md ===
# Live Pipeline Contract

This document defines the operational contract for the single live analysis
front door. It turns the charter into a stage-by-stage runtime specification:
what each stage requires, what it produces, and how the stages connect.

The live ordered sequence is:

1. `bash code/runners/main/run_data_ingest.sh`
2. `bash code/runners/main/run_kinase_attribution.sh`
3. `bash code/runners/main/run_attribution_recovery.sh`

For the bundled front door, use:

```bash
bash code/runners/main/run_live_pipeline.sh
```

Supporting setup is separate from the front door. In particular:

- `code/atlas_reference.py` remains supporting external-reference setup
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv` remains a required
  supporting input for unified attribution

## Stage 0 Supporting Prerequisites

These are not co-equal pipeline stages, but the live pipeline expects them.

| Surface | Status | Purpose |
|:---|:---|:---|
| `data/incytr_collections/song/` | required workspace | Local Song operational data surface |
| `outputs/reports/wmb_expression/wmb_kinase_expression.csv` | required for unified attribution | WMB expression specificity for cell-type attribution |
| SEA-AD effect sizes under `config.SEA_AD_DIR` | required for unified attribution | External transcriptomic concordance reference |

## Data Ingestion (`data_ingest.py`)

Data ingestion is the total-proteome integration stage. It establishes sample
mapping, phosphosite-to-protein linkage, marker-protein diagnostics, and
total-proteome quality control.

Inputs:

- `data/incytr_collections/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx`
- `data/incytr_collections/song/primary/proteomics/Sample_list_72mice (1).xlsx`
- `data/incytr_collections/song/primary/proteomics/song_IMAC_compositeSites_merged_labeled (2).xlsx`
- `data/incytr_collections/song/primary/proteomics/song_IMAC_sitequant_merged_labeled (2).xlsx`
- `config.A_OBS_FILE`
- `config.MAPPING_CACHE_FILE`

Canonical outputs:

- `outputs/reports/data_ingest/sample_mapping.csv`
- `outputs/reports/data_ingest/phospho_protein_matching.csv`
- `outputs/reports/data_ingest/matching_summary.json`
- `outputs/reports/data_ingest/datadriven_marker_assessment.csv`
- `outputs/reports/data_ingest/data_quality.json`
- `outputs/reports/data_ingest/pca_plots/`

Failure modes:

- missing mounted upstream Song proteomics files
- inconsistent TMT/sample-list naming
- missing mapping cache or A_obs inputs for marker summaries
- sparse or malformed total-proteome matrices causing PCA or matching failure

## Kinase Attribution (`kinase_attribution.py`)

Kinase attribution performs IRS cross-plex normalization, stoichiometry
computation, OLS site-level modelling, MEA (GSEA-based) kinase enrichment on
stoichiometry beta values, and unified cell-type attribution combining SEA-AD
concordance and WMB expression specificity for all significant kinases.

Inputs:

- `outputs/reports/data_ingest/sample_mapping.csv`
- `data/incytr_collections/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx`
- `data/incytr_collections/song/primary/proteomics/song_IMAC_sitequant_merged_labeled (2).xlsx`
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv`
- SEA-AD effect sizes under `config.SEA_AD_DIR`
- `config.MAPPING_CACHE_FILE` (kinase-to-gene mapping)

Canonical outputs:

- `outputs/reports/kinase_attribution/stoichiometry_matrix.csv`
- `outputs/reports/kinase_attribution/raw_phospho_normalized.csv`
- `outputs/reports/kinase_attribution/mea_stoichiometry.csv`
- `outputs/reports/kinase_attribution/site_level_ols.csv`
- `outputs/reports/kinase_attribution/unified_attribution.csv`
- `outputs/reports/kinase_attribution/attribution_summary.json`

Optional supplementary output (via `--mechanism-annotation`):

- `outputs/reports/kinase_attribution/mechanism_annotation.csv`

Failure modes:

- data ingestion outputs missing or stale
- missing `outputs/reports/wmb_expression/wmb_kinase_expression.csv`
- missing SEA-AD reference files
- mismatch between phospho site IDs and protein mapping

## Attribution Recovery (`attribution_recovery.py`)

Attribution recovery is the final attribution-table assembly stage. It adds
cross-contrast consistency analysis and produces the canonical deliverable.

Inputs:

- `outputs/reports/kinase_attribution/unified_attribution.csv`
- `outputs/reports/kinase_attribution/mea_stoichiometry.csv`

Canonical outputs:

- `outputs/reports/attribution_recovery/cross_contrast_matrix.csv`
- `outputs/reports/attribution_recovery/cross_contrast_heatmap.png`
- `outputs/reports/attribution_recovery/final_attribution_table.csv`

Failure modes:

- missing unified attribution outputs from kinase attribution
- empty or malformed MEA results

## Canonical Deliverables

The current mainline deliverables under `outputs/reports/` are:

- Data ingestion outputs under `outputs/reports/data_ingest/`
- Kinase attribution outputs under `outputs/reports/kinase_attribution/`
- Attribution recovery outputs under `outputs/reports/attribution_recovery/`

The canonical downstream table for the live program is:

- `outputs/reports/attribution_recovery/final_attribution_table.csv`
