# Incytr factorial kldata correction note

_Generated 2026-05-14T19:36:02.304694+00:00 from `alz/integration/diff_kldata_factorial.py`._

## Summary

The live factorial integration was previously sourcing its kinase library from `data/datasets/5xFAD/kinase/kldata_pspy.csv` (`alz/integration/export_factorial_inputs.py:58`, pre-correction). The kinase library is study-specific — its substrate row set must come from the sites actually phosphoprofiled in the cohort. Using 5xFAD's kldata silently scored Song factorial paths against a different study's substrate set.

Phase 0a regenerated kldata from this cohort's IMAC + pY sitequant tables; Phase 0c re-ran the factorial integration. This note quantifies the per-(sender, receiver, contrast) shift in PDS.

## Substrate-set overlap

- Yuyu kldata: 17,407 unique substrate sites; 376 mouse kinases; 101,987 (site × kinase) rows.
- 5xFAD kldata: 19,034 unique substrate sites; 101,558 rows.
- Site-level intersection: **5,828** (≈33% of Yuyu's sites). **11,579** Yuyu-measured sites were absent from the 5xFAD kldata; **13,206** 5xFAD-only sites were being scored despite not appearing in this study's data.

## Pre/post PDS concordance

Streamed all 342 (sender, receiver) pair shards × 9 contrasts (=3078 cells). Each cell inner-joined pre vs post on `(ID_1)`; metrics computed only when ≥3 paths overlapped.

**Overall PDS Spearman ρ (across all cells):**

- Median: **1.000**
- 25th–75th percentile: 1.000 — 1.000
- Min: 0.975; max: 1.000
- Cells with ρ < 0.5: **0** (0.0%)
- Cells with ρ < 0.0: **0** (0.0%)

**Sign agreement** (fraction of paths with same PDS sign in pre and post):

- Median: 1.000
- Cells with sign agreement < 0.8: **0** (0.0%)

## By contrast

| contrast | n_cells | median_n_paths | median_rho | min_rho | median_frac_pds_changed | median_n_sik_pre_nz | median_n_sik_post_nz | median_n_sik_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ApTt_2mo | 342 | 916.5 | 1.0 | 0.985 | 0.0021 | 0.0 | 2.0 | 2.0 |
| ApTt_4mo | 342 | 916.5 | 1.0 | 0.9863 | 0.0011 | 0.0 | 2.0 | 2.0 |
| ApTt_6mo | 342 | 916.5 | 1.0 | 0.9839 | 0.0007 | 0.0 | 2.0 | 2.0 |
| App_2mo | 342 | 916.5 | 1.0 | 0.9899 | 0.0021 | 0.0 | 2.0 | 2.0 |
| App_4mo | 342 | 916.5 | 1.0 | 0.9873 | 0.0006 | 0.0 | 0.0 | 0.0 |
| App_6mo | 342 | 916.5 | 1.0 | 0.9886 | 0.0012 | 0.0 | 2.0 | 2.0 |
| Tau_2mo | 342 | 916.5 | 1.0 | 0.9896 | 0.0021 | 0.0 | 2.0 | 2.0 |
| Tau_4mo | 342 | 916.5 | 1.0 | 0.9858 | 0.0006 | 0.0 | 1.0 | 0.0 |
| Tau_6mo | 342 | 916.5 | 1.0 | 0.9748 | 0.0007 | 0.0 | 1.0 | 1.0 |

## Why PDS barely moves despite the substrate-set swap

PDS is a weighted combination of PPDS (protein), PhPDS_ps and PhPDS_py (phospho stoichiometry per modality), and the kinase-arm score (driven by SiK_score columns). Across the 342 pair shards, the kinase arm contributes a small fraction of PDS magnitude — the protein and phospho arms dominate. Swapping the kldata changes **which** paths get a nonzero SiK_score (see the per-contrast table above), but the resulting PDS shift is below 0.01 for the vast majority of paths.

## Implication for prior interpretations

Path-level PDS rankings, sign calls, and top-N tables built against the pre-fix outputs are essentially unchanged. **However**, any analysis that interprets the *kinase arm specifically* (which kinases were predicted to act on a given path's substrates, kinase-driven hypotheses, attribution back to specific kinases) must be re-derived against the corrected outputs — the substrate set was the wrong study's. The pre-fix snapshot at `outputs/reports/incytr_factorial_5xfad_kldata/` is preserved for audit only and should not seed new analysis.

## Files

- Per-cell summary: `outputs/reports/incytr_factorial/kldata_correction/pre_post_kldata_concordance.parquet`
- Pre-fix snapshot: `outputs/reports/incytr_factorial_5xfad_kldata/`
- Post-fix outputs: `outputs/reports/incytr_factorial/`
- kldata generator: `alz/integration/build_yuyu_kldata.py`
- kldata + provenance: `data/datasets/song/kinase/`
