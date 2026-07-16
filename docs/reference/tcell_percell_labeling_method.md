# T-cell lineage and state annotation

## Lineage assignment

CD4 and CD8 lineages were assigned to individual cells from CITE-seq surface
protein counts. A cell was assigned to the CD4 lineage when its CD4 antibody
count exceeded both its CD8 antibody count and the isotype-control background,
and to the CD8 lineage under the converse condition. Cells whose antibody counts
were inconclusive were assigned the lineage of their transcriptomic cluster. Two
clusters per donor were identified as contaminants and were reported separately
from the T-cell lineages.

## State assignment

Each cell was annotated with one functional state independently, considering
only the five states defined for its lineage. Within each donor and lineage,
marker expression was standardized to zero mean and unit variance across cells,
and a panel score was computed as the mean standardized expression of the genes
in that panel.

Terminal exhaustion was evaluated first. A cell was assigned the exhaustion
state when its late exhaustion signature score was positive and exceeded both
its acute activation score and its effector function score.

For the remaining states, a cell was eligible for a given state only when each
of that state's positive panels contained at least one detected gene. Among the
eligible states, each state was scored by its lowest-scoring positive panel, and
the cell was assigned the state with the highest such score. Ties were resolved
in favor of the state with the lowest expression across its expected-low panels.
Cells that were eligible for no named state were annotated by lineage alone.

The positive and expected-low panels defining each state, and their constituent
genes, are given in Table 1. For the exhaustion states, the acute activation and
effector function panels served as the comparators in the hierarchical
exhaustion criterion.

**Table 1. State definitions and marker genes.**

| State | Positive panels | Expected-low panels |
|---|---|---|
| CD8 exhausted | Late exhaustion signature: HAVCR2, LAG3, ENTPD1, TOX, NR4A1 | Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG. Effector function: GZMB, PRF1, IFNG, TNF |
| CD8 cytotoxic effector | Granzyme program: GZMB, GZMH, GNLY. Perforin: PRF1 | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1 |
| CD8 activated/effector | Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG. Effector function: GZMB, PRF1, IFNG, TNF | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1 |
| CD8 naive-like | Naive stemness: TCF7, LEF1. Naive homing: CCR7, SELL | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1. Cytotoxic machinery: GZMB, GZMH, GNLY, NKG7, PRF1, EOMES. Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG |
| CD8 resting/memory | Resting/memory identity: IL7R, CD27 | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1. Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG |
| CD4 exhaustion-associated | Late exhaustion signature: HAVCR2, LAG3, ENTPD1, TOX, NR4A1 | Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG. Effector function: GZMB, PRF1, IFNG, TNF |
| CD4 cytotoxic | Granzyme program: GZMB, GZMH, GNLY. Perforin: PRF1 | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1 |
| CD4 activated/effector | Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG. Effector function: GZMB, PRF1, IFNG, TNF | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1 |
| CD4 naive-like | Naive stemness: TCF7, LEF1. Naive homing: CCR7, SELL | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1. Cytotoxic machinery: GZMB, GZMH, GNLY, NKG7, PRF1, EOMES. Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG |
| CD4 resting/memory | Resting/memory identity: IL7R, CD27 | Inhibitory receptors: PDCD1, HAVCR2, LAG3, CTLA4, TIGIT, ENTPD1. Exhaustion transcription factors: TOX, NR4A1. Acute activation: CD69, IL2RA, TNFRSF4, ICOS, CD40LG |
