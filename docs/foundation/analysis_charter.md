# Analysis Charter

This is the single authoritative front door for the live analysis program. It defines what the project is doing now, what outputs matter, and what new code should support.

Use [`analysis_rationale.md`](./analysis_rationale.md) for the pivot logic, [`statistical_constraints.md`](./statistical_constraints.md) for hard design limits, and [`repo_retention_policy.md`](./repo_retention_policy.md) for active-versus-archived boundaries.

## Scope

The primary deliverable is the 72-sample Song mouse bulk pathway:

1. integrate the 72-animal total proteome (data ingestion),
2. compute phospho-to-protein stoichiometry and run MEA kinase enrichment,
3. run unified cell-type attribution on the levy_t5 spine (Song within-cohort primary; WMB + human SEA-AD/HBCA corroborate),
4. assemble the final hypothesis table with cross-contrast consistency.

Three further layers extend the program on the same shared contracts: per-cluster proportional decomposition + per-cluster MEA (`alz/decomposition_mea/`), pair-mode Incytr intercellular signaling on the levy_t5 spine (`alz/incytr_pair/`), and parallel cohorts — human NBB/Mukesh (cross-species support), the T-cell exhaustion donors, and 5xFAD. Their canonical I/O schemas live in [`cohort_contract.md`](./cohort_contract.md).

## Closed Paths

These paths are closed and should not drive new code. See [`analysis_rationale.md`](./analysis_rationale.md) for the full pivot logic.

| Path | Status |
|:---|:---|
| Direct (statistical) cell-type deconvolution from the 24-group design | Closed — not identifiable |
| Per-cluster stoichiometry | Closed — algebraically collapses to bulk under same-proportion-for-both numerator and denominator |
| Proportional decomposition with snRNA-seq prior (`alz/decomposition_mea/`) | **Active branch (levy_t5 31-cluster spine)** — forward projection only (`P_c = f_c × bulk`), not statistical recovery; consumed by per-cluster MEA and the pair-mode Incytr inputs. See [`cohort_contract.md`](./cohort_contract.md) §4–5 |
| Factorial Incytr engine | Closed — archived 2026-05-18; superseded by pair-mode Incytr (`Incytr::Cal_pairwise_grid`) |
| Levy-19 / WMB-34 cluster spine | Closed — superseded by the levy_t5 31-cluster spine |
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

Every MEA-significant kinase is attributed on the **levy_t5 31-cluster spine** (`config.CLUSTER_SPINE`, `CLUSTER_SPINE_NAME = "levy_t5"`) — the active Song snRNA-seq cluster taxonomy. Three evidence sources feed the attribution; each reference rolls onto the spine through a 1-hop bridge (`data/derived/bridges/cluster_to_*`), never a chained mapping:

1. **Song within-cohort** (paired snRNA-seq, same animals as the bulk MEA): the **primary** signal — per-cluster expression specificity + disease LFC from factorial OLS. Sets both the confidence pill and the direction tier.
2. **WMB expression specificity** (Allen WMB 10Xv3, ~4M cells brain-wide): mouse-atlas corroborator, rolled up at the WMB **class** level (`cluster_to_wmb_class`).
3. **SEA-AD / HBCA cross-species** (human AD MTG + HBCA whole-brain): human corroborator; SEA-AD supertypes and HBCA classes roll directly onto the spine.

Two orthogonal per-kinase outputs result:

- **`confidence_tier`** — the headline **cell-type exclusivity pill** (`none…very_high`): how exclusively the kinase is expressed in one curated specificity unit, with references corroborating the home cell class. Full spec: [`specificity_confidence.md`](./specificity_confidence.md).
- **`direction_tier`** — the disease-**direction** concordance tier (info-only): whether the kinase's activity moves with disease across bulk MEA, within-cohort expression, and the decomposition layer. Full spec: [`concordance.md`](./concordance.md).

Within-cohort Song alone can reach a high pill; references only ever raise it by one step. No source has veto power, and a cell type a source cannot witness carries `n/a` rather than being silently dropped.

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
2. Attribution uses unified evidence on the levy_t5 spine (Song within-cohort primary; WMB + human SEA-AD/HBCA corroborate) for all significant kinases.
3. Do not reopen direct deconvolution or factor-model code paths for attribution.
4. Treat old deconvolution outputs only as provenance.

## Bottom Line

The live program is a stoichiometry-corrected, MEA-enriched, unified-attribution workflow. New documentation and new code should be judged against that charter, not against the retired 24-group deconvolution objective.
