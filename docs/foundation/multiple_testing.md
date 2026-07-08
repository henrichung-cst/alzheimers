# Multiple-testing correction policy

Single source of truth for how the pipeline handles multiple-testing correction (MTC).

## Rule

**Use Benjamini–Hochberg (BH) via `statsmodels.stats.multitest.multipletests(method="fdr_bh")`
everywhere.** All test universes in the live pipeline are kinase-scale (tens to low thousands of
tests), where BH is the appropriate correction; there is no Storey / q-value path.

## Test sites

| Site | File | Universe | Threshold |
|---|---|---|---|
| Bulk MEA (stoichiometry) | `alz/bulk_mea/enrich.py` | ~78 kinases × 9 contrasts | FDR < 0.25 |
| Bulk MEA (raw phospho, mechanism) | `alz/bulk_mea/mechanism.py` | ~78 kinases × 9 contrasts | FDR < 0.25 |
| Site-level OLS | `alz/bulk_mea/enrich.py` | ~7,000 sites × 9 contrasts | FDR < 0.25, BH per-contrast |
| snRNA concordance | `alz/reference/snrna_integration.py` | ~385 kinase genes × (cell_type × contrast) | FDR < 0.25, BH per (cell_type, contrast), restricted to kinase universe (informational; cohort underpowered, see below) |
| FDR-stringent supplementary | `alz/supplementary/fdr_stringent.py` | same as bulk MEA | FDR < 0.10 (sensitivity check) |

## Why the snRNA universe is restricted to kinase genes

Pseudobulk OLS is computed for every detected gene (~13,000 per cell type), but the only rows
downstream attribution joins on are the ~385 kinase symbols. Correcting BH over the full
13,000-gene universe inflates FDR for the kinase rows we actually consume. Restricting BH to the
kinase universe makes q-values comparable to the bulk MEA pipeline (which is also kinase-scale).

Even after restriction, the snRNA cohort (~15 male animals, df_resid = 5) is underpowered: only
~0.4% of kinase rows pass FDR < 0.25. The unified viewer surfaces the **uncorrected p-value**
(`song_pval`, bold at p < 0.05) as the directional flag, with `song_fdr` available for users who
want a stringent gate.

## Why FDR < 0.25 default (0.10 stringent)

`q < 0.25` is GSEA convention for kinase-scale tests. With ~78 kinases, BH at 0.25 corresponds to a
per-test FDR similar to permutation methods' defaults and is what kinase-library / gseapy recommend
for hypothesis generation at this scale. `alz/supplementary/fdr_stringent.py` re-runs the same
pipeline at FDR < 0.10 as a sensitivity check.

## Implementation conventions

- Always import from `statsmodels.stats.multitest`, not hand-rolled BH.
- The `_bh_fdr` helper in `alz/bulk_mea/enrich.py` is a NaN-safe wrapper around
  `multipletests(method="fdr_bh")`; numerics are identical.
- When BH is applied per-stratum (per contrast, per cell type, etc.), document the stratum in the
  column name or accompanying comment so readers know what universe a given FDR is over.

## Track-specific KL_THRESH (Ser/Thr vs Tyr MEA)

Adjacent to MTC: substrate-set membership for MEA is governed by `KL_THRESH`, the percentile-rank
cutoff for declaring a phosphosite a kinase's substrate. This is set **per track**, not globally:

- Ser/Thr (`st`): `kl_thresh = 15` (kinase-library default)
- Tyrosine  (`py`): `kl_thresh = 7` (tightened)

The thresholds are different because the Ser/Thr and tyrosine kinomes have different intrinsic
specificity. At the kinase-library default of 15, the median pairwise within-family Jaccard overlap
of substrate sets is 0.034 on ST (kinase-specific signal) but 0.244 on Tyr (family-redundant signal
— when one Tyr kinase moves, its family-mates move with it because they share motif features).
Lowering the Tyr threshold to 7 brings Tyr's within-family overlap to 0.122, into the same
interpretability regime as ST without making substrate sets too small for stable GSEA enrichment
(median 84 substrates per kinase at thr=7).

We did **not** raise the Ser/Thr threshold to equalize Jaccard exactly, because that would loosen ST
substrate sets and introduce family co-firing on the track that does not currently have it. The goal
is per-kinase NES interpretability on each track, not equalized Jaccard between tracks; MEA runs
separately per track and never compares ST and Tyr NES directly.

The asymmetry reflects published kinome biology: the tyrosine kinome (~90 kinases, recently
expanded, structurally tighter) is more conserved within families than the Ser/Thr kinome (~390
kinases, ancient, more diverse), and Tyr-kinase substrate recognition relies on features (SH2
docking, +3 hydrophobic) more correlated across families than the flanking-residue signatures
Ser/Thr kinases use.

Implementation: `alz/shared/config.py:PHOSPHO_TRACKS[track]["kl_thresh"]`. Read by
`alz/bulk_mea/enrich.py`, `alz/bulk_mea/mechanism.py`, and `alz/decomposition_mea/enrich_celltype.py`.

## Things this policy does **not** do

- It does not change MEA NES values or pvals (BH was already in use).
- It does not change site-level OLS LFCs or pvals.
- It does not introduce a new significance gate in attribution; the existing `mea_significant` flag
  (based on bulk MEA FDR < 0.25) remains the gate.
