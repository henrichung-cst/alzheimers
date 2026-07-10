# Bioinformatician feedback: cycle-independent per-cell T-cell annotation

## Executive Summary

Scientific soundness is **medium**, novelty is **medium**, antibody-design relevance
is **low**, and signaling-analysis applicability is **medium**. Restoring per-cell
marker classification fixes a major composition artifact caused by day-confounded
Seurat clusters. The output is research-grade for descriptive stratification, not
functional proof of exhaustion.

## Scientific and Biological Assessment

Raw CITE-seq CD4/CD8 measurements establish lineage. Every named state requires
direct positive detection in each defining non-cycle RNA module. Standardized
positive-module strength selects among competing eligible states; expected-low
evidence only breaks exact ties. Cycling is ignored because proliferation was
artificially induced.

This revision materially improves the time-course interpretation. Cluster-only
labels were dominated by clusters nearly unique to individual days. Per-cell labels
instead measure marker heterogeneity within those clusters. Separating naive-like
from resting/memory explains most late composition shifts and reduces generic bins
to low single digits. A hierarchical HAVCR2/LAG3/ENTPD1/TOX/NR4A1 aggregate that
must exceed both acute activation and effector function restores the expected
trajectory without a sparse Boolean AND gate: donor 1 CD8 TEX is 10.7% at day 2
and 30.8% at day 20; donor 2 is 6.7% and 40.2%.
The donor 2 day-11 value is a validation flag: it is a relative TEX-like RNA
program, not evidence that terminal dysfunction has been established by day 11.

## Antibody Design and Signaling Relevance

CITE-seq CD4/CD8 antibodies materially strengthen lineage assignment. Most
exhaustion checkpoints were not measured as proteins, so the RNA categories should
guide follow-up antibody-panel design rather than be presented as protein-confirmed
phenotypes.

## Novelty and Impact

Positive module detection with signed tie-breaking is standard, but explicitly
excluding the induced cycle program and separating naive-like from resting/memory
is a useful, transparent adaptation. Its main value is preventing experimental-day
structure from masquerading as biological state composition.

## Key Risks and Limitations

- RNA dropout makes individual labels uncertain and increases use of `CD4`/`CD8`.
- AUROCs describe separation created by the same panels and are not independent
  validation.
- ProjecTILs uses tumor-infiltrating reference atlases; confidence 1 is internal
  reference certainty, not guaranteed correctness in this experiment.
- TEX is a marker-program call, not measured dysfunction. The provisional TPEX
  category is conservatively collapsed to `CD8`.
- One sample per donor/day makes trajectories descriptive; per-cell tests would be
  pseudoreplication.
- Composition values are within-day proportions of recovered T cells, not absolute
  cell abundances; relative enrichment must not be described as absolute expansion.

## Actionable Recommendations

Must retain for validity:

1. Keep cycle evidence outside lineage and state assignment.
2. Preserve `CD4`/`CD8` as legitimate fallback labels.
3. Keep ProjecTILs as cluster/cell-level corroboration without automatic override.
4. Report donor trajectories separately and do not impose monotonic exhaustion.

Useful follow-up work includes checkpoint proteins (PD-1, TIM-3, LAG-3, TIGIT,
CD39), TCF1, cytokine production, cytotoxicity, persistence, and restimulation.
