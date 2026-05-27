# Human CTRL outlier audit + single AD-vs-CTRL kinase metric

Date: 2026-05-25
Status: **DRAFT — awaiting approval. No analysis run, no pipeline edits, until approved.**

## Motivation

Two linked concerns about human (Mukesh/NBB) kinase MEA.

1. **No single AD-vs-CTRL fold-change metric.** MEA needs a baseline→condition LFC. Mouse
   (Song) has one: factorial OLS on site stoichiometry `log2(phospho) − log2(protein)`,
   disease-vs-WTyp, → `NES` per contrast (`mea_stoichiometry.csv` → `kinase_activity_matrix.csv`).
   Human uses a **per-donor** design: `stoich(donor_i) − nanmean(stoich(all 7 CTRL))`
   (`alz/ingest/mukesh_perdonor.py:_build_donor_deltas`, L80–101), one MEA per donor →
   `kinase_donor_nes.csv`, recurrence stats in `recurrence.csv`. There is **no group-level
   AD-vs-CTRL phospho MEA NES**. Closest existing single number, `sea_ad_lfc_median` in
   `human_seaad_agreement.csv`, is transcriptomic RNA, not phospho.

2. **The last 3 sequential controls look AD-like.** `CTRL-07, CTRL-08, CTRL-10` (column
   positions 15–17 in every raw file). At the derived-NES level their NES vectors correlate
   **+0.74 / +0.89 / +0.84** with the AD-mean NES, while `CTRL-01–04` are negatively correlated.
   We must confirm or refute this **below the MEA black box** — at the phospho-omic data
   itself — and separate genuine biology from technical artifact, so a statement like
   "they look unusual but after triple-checking data and method it is genuine" is defensible.

**Linkage:** the per-donor contrast (and any group metric for Concern 1) uses all 7 controls,
including 07/08/10, in the reference mean. If those three are genuinely AD-like they pull the
CTRL baseline toward AD and attenuate every AD-vs-CTRL effect. So Concern 2's verdict decides
the correct control set for Concern 1. **Concern 2 is run first and gates Concern 1.**

## Candidate artifacts to rule out (named up front)

- `CTRL-07`: low protein coverage, `median_log2_protein = 8.534`, robust_z −2.08
  (`sample_exclusions.csv`). Low coverage can manufacture spurious AD-similarity.
- `AD-15`: `outlier_flag = True`, robust_z −3.49, kept because `sample_outlier_exclusion: false`
  (`docs/foundation/mukesh_ingest_policies.yml:17`).
- Run-order / batch: sample IDs are sequential column positions; 07/08/10 being "the last
  three" could be a run-order/batch position effect rather than biology.

## Inputs (all small; safe to read in full)

- `outputs/reports/kinase_attribution_human/stoichiometry_matrix.csv` (3.4 MB, 9,554 IMAC sites × 17)
- `outputs/reports/kinase_attribution_human/stoichiometry_matrix_pY.csv` (311 KB, 1,133 pY sites × 17)
- `outputs/reports/kinase_attribution_human/raw_phospho_normalized{,_pY}.csv` (3.3 MB / 295 KB)
- `outputs/reports/data_ingest_human/sample_mapping.csv` (group labels)
- `outputs/reports/data_ingest_human/sample_exclusions.csv` (coverage/z-score)
- `outputs/reports/kinase_attribution_human/perdonor/kinase_donor_nes{,_pY}.csv`, `recurrence{,_ctrl}.csv`
- Raw peptide reports under `data/datasets/mukesh/` only for run-order/coverage metadata —
  **89.7 MB IMAC / 39.2 MB pY / 281 MB peptide report: do NOT load into memory.** Use DuckDB
  with column selection / `LIMIT`, or read headers only, per memory-safety rules.

## Concern 2 — phospho-omic audit (read-only; gating)

### Phase A — sample-level structure on the underlying data (not NES)
- Sample×sample correlation matrix (Pearson + Spearman) on `stoichiometry_matrix.csv`, complete-case
  and pairwise, IMAC and pY separately.
- Hierarchical clustering (correlation distance) + PCA on the site matrix. Question: do
  CTRL-07/08/10 cluster with AD at the **data** level, mirroring the NES-level signal?
- Per-sample summary stats: n quantified sites, missingness, median/IQR stoichiometry,
  % sites above CTRL-mean. Compare 07/08/10 vs CTRL-01–04 vs AD.
- Deliverable: clustered heatmap, PCA scatter (colored by group, labeled by sample), summary table.

### Phase B — artifact controls (the "triple-check")
- **Coverage confound:** recompute Phase-A clustering/correlation on the subset of sites
  quantified in **all 17 samples**. If 07/08/10 still group with AD, AD-similarity is not a
  coverage artifact. Report the AD-correlation before/after the coverage filter, per sample.
- **CTRL-07 specifically:** repeat with CTRL-07 down-weighted/excluded to confirm its similarity
  is not purely its low coverage.
- **Run-order / batch:** extract acquisition/run order from raw-report column order or any
  timestamp in `data/datasets/mukesh/` headers (DuckDB header read only). Test whether
  AD-similarity tracks run position rather than the AD/CTRL label.
- **Normalization sensitivity:** re-derive the comparison on `raw_phospho_normalized.csv`
  (no protein denominator) to confirm the AD-like signal is not an artifact of the
  stoichiometry denominator (protein-report) coverage.

### Phase C — site-level attribution
- Rank sites by AD-vs-CTRL discrimination (e.g. |mean(AD) − mean(CTRL-01..04)|). Confirm
  CTRL-07/08/10 sit on the AD side at exactly those discriminating sites, not at random sites.
- Identify the kinases (and their leading substrate sites) most responsible for the
  +0.74/+0.89/+0.84 NES correlation; check those sites are well-quantified (not low-coverage noise).

### Phase D — targeted per-kinase leading-edge proof (the granular element of proof)

Phases A–C show broad trends across all kinases. This phase is the highly granular,
mechanistic proof that the **omic data is similar, therefore the MEA is similar** — done
kinase-by-kinase, at the level of the motif signal MEA actually consumes.

Rationale: an MEA `NES` for a kinase is GSEA-prerank of **that kinase's substrate motif set**
against the site-level LFC ranking. If the same kinases are enriched in CTRL-07/08/10 as in AD,
then the leading-edge substrate sites and their LFCs must be similarly ranked in both. Showing
that directly closes the loop from raw signal → NES.

- **Selection:** pick **5–10 kinases** that are FDR-significant *and* same-direction in both the
  peculiar controls (07/08/10) and the AD group — from `kinase_donor_nes{,_pY}.csv` /
  `recurrence.csv` (high `n_donors_sig`, concordant `median_nes` sign), prioritizing kinases that
  also drive the Phase-C NES correlation. Span up and down kinases and both tracks (IMAC + pY).
- **Substrate / leading-edge sites:** for each selected kinase, recover its substrate motif set
  and leading-edge sites via the same `kinase_library.RankedPhosData` motif scoring used by the
  pipeline (reuse `mea_substrate_sets.csv` / `Leading substrates` where available; otherwise
  re-score the human site table once and cache). These are the exact sites whose LFC drives the NES.
- **Proof per kinase:**
  - Per-site LFC scatter at the kinase's substrate sites: AD-mean delta (x) vs CTRL-07/08/10-mean
    delta (y), with CTRL-01–04 overlaid as the contrast group. AD-like controls should land on
    the AD diagonal; clean controls should not.
  - GSEA running-enrichment / rank-position plot of the substrate set for AD vs CTRL-07/08/10 vs
    CTRL-01–04 — visually identical enrichment curves for AD and the peculiar controls is the proof.
  - Leading-edge site overlap (Jaccard) between AD and 07/08/10 vs between AD and 01–04.
- **Deliverable:** a small-multiple figure (one panel per selected kinase: substrate-site LFC
  scatter + enrichment curve) plus a table of per-kinase NES (AD vs 07/08/10 vs 01–04) and
  leading-edge overlap. This is the artifact that backs "I triple-checked the underlying data —
  it is genuinely the same motif signal."

### Phase E — verdict
Per-sample (07, 08, 10) classification: **genuine AD-like** vs **technical artifact**, with the
supporting numbers (AD-correlation before/after coverage filter, run-order test, site-level
attribution, and the Phase-D per-kinase leading-edge proof). Written conclusion that either
supports or refutes "it's genuine."

## Concern 1 — single AD-vs-CTRL kinase metric (design deferred)

Decide the metric's shape **after** Phase E, because the verdict sets the control set:
- Group contrast `mean(AD) − mean(CTRL*)` stoichiometry per site → one MEA → one `NES` per kinase
  (mirrors mouse). The CTRL set (`CTRL*`) = all 7, or the "clean" 4 (CTRL-01–04), depending on Phase D.
- Open question to resolve post-audit (per user): add group-level NES **alongside** per-donor,
  **replace** per-donor (anti-shim), or hold — decided once heterogeneity is shown real or artifactual.
- If 07/08/10 are genuine AD-like CTRLs, that is itself a finding about the control cohort and
  must be stated, not silently dropped.

## Deliverables

- `docs/plans/` audit writeup with figures (heatmap, PCA, before/after-coverage correlation table,
  site-level attribution) and the Phase-D verdict.
- Concern 1 metric proposal (shape + control set) for approval — **no pipeline edits** in this audit.

## Guardrails

- Read-only audit. No edits to `mukesh_perdonor.py` / `enrich.py` / pipeline until Concern 1 is approved.
- Memory safety: do not load the 89.7 MB / 39.2 MB / 281 MB raw reports into memory; DuckDB
  column-pushdown or header reads only. Derived matrices (≤3.4 MB) are safe to load fully.
