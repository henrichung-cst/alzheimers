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
