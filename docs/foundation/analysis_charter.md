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

These paths are closed and should not drive new code. See [`analysis_rationale.md`](./analysis_rationale.md) for the full pivot logic.

| Path | Status |
|:---|:---|
| Direct cell-type deconvolution from the 24-group design | Closed — not identifiable |
| Proportional decomposition with snRNA-seq prior (`alz/deconvolution/`) | Branch-only, not in live path; CTM-native on WMB-class spine; uses the snRNA pseudobulk as a per-(group, WMB class, gene) prior rather than inferring cell-type effects from `A_obs + bulk` alone |
| Joint kinase-activity factor model | Closed — composition bottleneck |
| Two-compartment neuronal/glial simplification | Closed — failed validation |
| Transcript-only rescue | Closed — no defensible attribution |
| Pre-stoichiometry atlas concordance | Closed — diluted signal |

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

All MEA-significant kinases are evaluated against three evidence sources at the **WMB class level (34 classes)** — the published Allen Whole Mouse Brain class taxonomy, used directly to avoid silent dropping of cells outside cortical/glial coverage (e.g., hippocampal CA, dentate granule, striatal MSN, olfactory bulb, cerebellar):

1. **WMB expression specificity** (Allen WMB 10Xv3, ~4M cells brain-wide): cell-type mean log2 expression divided by the sum across all 34 WMB classes — a share-of-total measure of cell-type concentration. Spine evidence; always present.

2. **Song within-cohort concordance** (paired snRNA-seq, 28 animals): per-class disease LFC from factorial OLS. Present for ~21 of 34 classes (those Song's dissection captures with ≥10 male animals); `n/a` otherwise.

3. **SEA-AD cross-species concordance** (human AD MTG, 139 supertypes): supertypes aggregated to WMB class via `seaad_subclass_to_wmb_class.csv`. Present for ~9 of 34 classes (cortical neurons + glia + vascular + immune); `n/a` for non-MTG classes (hippocampal pyramidals, subcortical, brainstem, cerebellar).

Combined confidence (thresholds expressed as multiples of uniform = 1/34):
- **High**: Within-cohort Song supports the direction + WMB specificity ≥ 2× uniform + at least one |LFC| > 0.1
- **Moderate**: Either Song or SEA-AD provides directional evidence with WMB plausibility
- **Low**: Weak evidence from all sources

SEA-AD `n/a` (out-of-MTG classes) does not preclude high confidence — Song + WMB alone suffice when Song concordance is significant.

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
