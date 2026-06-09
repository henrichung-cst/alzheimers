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

## 5. Cross-Species Inference Scope

The unified attribution arm uses human SEA-AD transcriptomic data (139
supertypes, 24 subclasses from the Allen Institute MTG dataset) to interpret
mouse 5xFAD phosphoproteomics. This is a deliberate design choice, not an
accidental conflation. The implicit assumption is that cell-type-specific
transcriptomic changes in human AD are directionally conserved in the 5xFAD
mouse model — that is, if a kinase gene is upregulated in human AD microglia,
the corresponding kinase activity signal in mouse microglia should point the
same way.

This assumption is well-supported for certain pathways and cell types. The
GSK3, MAPK/JNK, MAPK/ERK, and DYRK families that drive the strongest signals
in this pipeline have established cross-species conservation in AD models.
Microglial and astrocytic transcriptomic responses to amyloid pathology are
among the most conserved features between human AD and 5xFAD mice. For other
cell types and pathways, cross-species conservation is an open question.

To make this transparent, the final attribution table includes a
`confidence_basis` column that classifies each attribution as:

- `cross_species` — supported by both human SEA-AD concordance and mouse WMB
  expression specificity
- `mouse_expression_only` — supported only by WMB expression specificity (no
  meaningful human SEA-AD signal)
- `human_concordance_only` — supported only by SEA-AD concordance (WMB
  specificity below threshold)
- `weak` — positive concordance but both evidence sources below their strong
  thresholds

This lets downstream consumers identify which attributions depend on the
cross-species assumption and which are supported by mouse-internal evidence
alone.

## 6. Resulting Project Logic

The project now has a clean decision structure:

1. direct deconvolution is closed,
2. transcript-only rescue is closed,
3. stoichiometry plus mechanism stratification is the active route,
4. Track A and Track B should remain distinct by design,
5. the RNA-phospho decoupling result is the mechanistic explanation for that split.

## Bottom Line

The project pivoted because the old question was statistically non-identifiable, while stoichiometry plus mechanism stratification changed the question into one the data could answer.
