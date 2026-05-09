# Multiple-testing correction policy

This document is the single source of truth for how the pipeline handles
multiple-testing correction (MTC). It exists because the codebase previously
mixed BH and Storey's q-value across sites with very different test universes,
making "FDR" labels ambiguous.

## Rule

**Use Benjamini–Hochberg (BH) via `statsmodels.stats.multitest.multipletests(method="fdr_bh")` everywhere, except in two specific sites that use Storey's q-value.**

Storey is reserved for sites where:
1. The test universe is large (≥ ~1,000 tests), and
2. The expected null fraction `π₀` is close to 1 (most tests are nulls).

Below ~1,000 tests, `π₀` estimation is unstable and Storey collapses to BH
anyway; using BH unconditionally avoids spurious differences in q-value
between similar-sized universes.

## Test sites

| Site | File | Universe | Method | Threshold |
|---|---|---|---|---|
| Bulk MEA (stoichiometry) | `kinase_attribution.py` | ~78 kinases × 9 contrasts | BH (gseapy internal) | FDR < 0.25 |
| Bulk MEA (raw phospho, mechanism) | `kinase_attribution.py` | ~78 kinases × 9 contrasts | BH (gseapy internal) | FDR < 0.25 |
| Site-level OLS | `kinase_attribution.py` | ~7,000 sites × 9 contrasts | BH per-contrast | FDR < 0.25 |
| snRNA concordance | `snrna_integration.py` | ~385 kinase genes × (cell_type × contrast) | BH per (cell_type, contrast), restricted to kinase universe | FDR < 0.25 (informational; cohort underpowered, see below) |
| Cohort concordance (deconvolution) | `deconvolution/cohort_concordance.py` | ~43 strata | BH | FDR < 0.25 |
| Per-cell-type kinase MEA (deconvolution) | `deconvolution/mea_per_celltype.py` | ~78 kinases × strata | BH (gseapy internal) | FDR < 0.25 |
| Backbone permutation recurrence | `integration/adapters/aggregate_factorial.py` | thousands of backbones | **Storey's q** | q < 0.05 |
| Marker-protein assessment | `data_ingest.py` | ~6,300 proteins × 10 cell types | **Storey's q** | q < 0.10 |
| Cross-pair pathway integration | `integration/adapters/aggregate_cross_pair.py` | varies (often 100s) | BH | FDR < 0.25 |
| Kinase-support null tests | `integration/adapters/compute_kinase_support.py` | varies | BH | FDR < 0.25 |
| FDR-stringent supplementary | `supplementary/fdr_stringent.py` | same as bulk MEA | BH (gseapy internal) | FDR < 0.10 (sensitivity check) |

## Why the snRNA universe is restricted to kinase genes

Pseudobulk OLS is computed for every detected gene (~13,000 per cell type),
but the only rows downstream attribution joins on are the ~385 kinase
symbols. Correcting BH over the full 13,000-gene universe inflates FDR for
the kinase rows we actually consume. Restricting BH to the kinase universe
makes q-values comparable to the bulk MEA pipeline (which is also kinase-scale).

Even after restriction, the snRNA cohort (~15 male animals, df_resid = 5)
is underpowered: only ~0.4% of kinase rows pass FDR < 0.25. The unified
viewer surfaces the **uncorrected p-value** (`song_pval`, bold at p < 0.05)
as the directional flag, with `song_fdr` available for users who want a
stringent gate.

## Why two thresholds (0.05 vs 0.25)

- **q < 0.05**: applied to large universes where MTC is the dominant cost
  of inference (markers, backbone recurrence). Storey gives back power
  Storey power against a tight gate; the tight gate is appropriate.
- **q < 0.25**: GSEA convention for kinase-scale tests. With ~78 kinases,
  BH at 0.25 corresponds to a per-test FDR similar to permutation methods'
  default and is what kinase-library / gseapy recommend for hypothesis
  generation at this scale.

## Implementation conventions

- Always import from `statsmodels.stats.multitest`, not hand-rolled BH.
- The `_bh_fdr` helper in `kinase_attribution.py` is a NaN-safe wrapper
  around `multipletests(method="fdr_bh")`; numerics are identical.
- When BH is applied per-stratum (per contrast, per cell type, etc.),
  document the stratum in the column name or accompanying comment so
  readers know what universe a given FDR is over.
- Storey implementation lives in `integration/adapters/aggregate_factorial.py`
  (`_storey_qvalue`); `data_ingest.py` has its own inlined Storey for
  marker assessment. Both use the same standard π₀ estimator at λ = 0.5–0.95.

## Track-specific KL_THRESH (Ser/Thr vs Tyr MEA)

Adjacent to MTC: substrate-set membership for MEA is governed by `KL_THRESH`,
the percentile-rank cutoff for declaring a phosphosite a kinase's substrate.
This is set **per track**, not globally:

- Ser/Thr (`st`): `kl_thresh = 15` (kinase-library default)
- Tyrosine  (`py`): `kl_thresh = 7` (tightened)

The thresholds are different because the Ser/Thr and tyrosine kinomes have
different intrinsic specificity. At the kinase-library default of 15, the
median pairwise within-family Jaccard overlap of substrate sets is 0.034 on
ST (kinase-specific signal) but 0.244 on Tyr (family-redundant signal —
when one Tyr kinase moves, its family-mates move with it because they
share motif features). Lowering the Tyr threshold to 7 brings Tyr's
within-family overlap to 0.122, into the same interpretability regime as
ST without making substrate sets too small for stable GSEA enrichment
(median 84 substrates per kinase at thr=7).

We did **not** raise the Ser/Thr threshold to equalize Jaccard exactly,
because that would loosen ST substrate sets and introduce family co-firing
on the track that does not currently have it. The goal is per-kinase NES
interpretability on each track, not equalized Jaccard between tracks; MEA
runs separately per track and never compares ST and Tyr NES directly.

The asymmetry reflects published kinome biology: the tyrosine kinome
(~90 kinases, recently expanded, structurally tighter) is more conserved
within families than the Ser/Thr kinome (~390 kinases, ancient, more
diverse), and Tyr-kinase substrate recognition relies on features
(SH2 docking, +3 hydrophobic) more correlated across families than the
flanking-residue signatures Ser/Thr kinases use.

Implementation: `alz/config.py:PHOSPHO_TRACKS[track]["kl_thresh"]`.
Read by `alz/kinase_attribution.py` (`step_enrich`) and
`alz/deconvolution/mea_per_celltype.py`. The legacy `config.KL_THRESH`
constant is retained for backwards-compat but should not be added to
new code paths.

## Things this policy does **not** do

- It does not change MEA NES values or pvals (BH was already in use).
- It does not change site-level OLS LFCs or pvals.
- It does not introduce a new significance gate in attribution; the existing
  `mea_significant` flag (based on bulk MEA FDR < 0.25) remains the gate.
