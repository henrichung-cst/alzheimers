
## Questions and Concerns

**1. The unified attribution confidence tiers need more justification.** The high/moderate/low scheme combining SEA-AD concordance and WMB specificity makes intuitive sense, but the specific thresholds (≥2x uniform for WMB, |LFC| > 0.1 for SEA-AD) feel somewhat arbitrary. How sensitive is the final attribution table to these cutoffs? A threshold sensitivity analysis showing that the main conclusions are stable across reasonable perturbations would strengthen this considerably. The report writing checklist actually anticipates this concern with its "near-miss" guidance — I'd apply that logic here too.

**2. The 24 SEA-AD subclass aggregation deserves scrutiny.** You're collapsing 139 supertypes down to 24 subclasses. The aggregation method matters: are you taking the median LFC across supertypes within a subclass? Mean? Are you weighting by cell count or treating each supertype equally? Different choices here could shift concordance scores meaningfully, particularly for heterogeneous subclasses. I'd want to see how robust the concordance calls are to the aggregation strategy.

**3. Cross-species inference carries real risk.** The entire attribution arm leans on human SEA-AD transcriptomic data to interpret mouse 5xFAD phosphoproteomics. That's a reasonable strategy, but the implicit assumption is that cell-type-specific transcriptomic changes in human AD are directionally conserved in this mouse model. For some cell types and pathways that's well-supported; for others it's an open question. I'd want the final deliverable to flag which attributions depend heavily on cross-species concordance versus which are supported by mouse-internal evidence (like WMB expression specificity alone).

**4. The FDR < 0.25 threshold for MEA is standard GSEA, but worth discussing.** In a manuscript context, reviewers may push back on 0.25 as permissive, especially if the number of kinase-contrast tests is large. What's the effective multiple-testing burden across all contrasts? Is there a secondary analysis at FDR < 0.10 to show the core findings are robust?

**5. The activity-driven class is small and high-leverage.** Twelve activity-driven kinases is a meaningful finding, but the small count means each one carries disproportionate weight in the narrative. Are there diagnostics to confirm these aren't edge cases of the stoichiometry correction (e.g., noisy parent protein estimates creating spurious stoichiometry signals)? I'd want to see parent protein detection quality metrics specifically for this subset.

**6. I'd like to understand the OLS modeling choices better.** The charter mentions site-level OLS on stoichiometry betas, but the documents don't specify the model formula. With a 2×3×4 factorial and 72 animals, the contrast structure matters. Are you fitting the full interaction model? Main effects only? How are you handling the sex × timepoint × condition contrasts that feed into MEA?

