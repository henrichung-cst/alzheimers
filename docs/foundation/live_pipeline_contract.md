# Live Pipeline Contract

This document defines the operational contract for the live analysis arc.
It turns the charter into a stage-by-stage runtime specification: what each
stage requires, what it produces, and how the stages connect.

The live arc is invoked through Pixi task aliases. Kedro has been reintroduced
incrementally for orchestration, but the only currently registered Kedro
pipeline is `ingest` in `alz/pipelines/ingest/`; the downstream analysis stages
still call their package modules directly.

| Stage | Pixi task | Current entry point |
|:---|:---|:---|
| 1. Ingest | `pixi run ingest` | `python alz/ingest/song.py --run` |
| 2. Normalize | `pixi run normalize` | `python alz/bulk_mea/normalize.py` |
| 3. Enrich | `pixi run enrich` | `python alz/bulk_mea/enrich.py` |
| 4. Attribute | `pixi run attribute` | `python alz/bulk_mea/attribute.py` |
| 5. Recover | `pixi run recover` | `python alz/bulk_mea/recover.py` |
| Optional: mechanism | `pixi run mechanism` | `python alz/bulk_mea/mechanism.py` |

Bundled front door (sequences ingest → normalize → enrich → attribute →
recover):

```bash
pixi run live
```

Dual-track runner (males-only primary + full-cohort sensitivity, archives
outputs by mode):

```bash
pixi run dual
```

Cohort selection lives in Kedro parameters
(`conf/base/parameters.yml:analysis_mode`, default `males_only`); set
`KEDRO_ENV=full_cohort` to overlay `conf/full_cohort/parameters.yml` for
the sensitivity track.

Supporting setup is separate from the front door:

- `alz/reference/atlas.py` — external-reference setup (SEA-AD + WMB)
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv` — required
  supporting input for unified attribution
- `outputs/reports/snrna_integration/song_*.csv` — Song within-cohort
  evidence (optional; falls through to SEA-AD + WMB if absent)

## Stage 0 — Supporting Prerequisites

These are not co-equal pipeline stages, but the live pipeline expects them.

| Surface | Status | Purpose |
|:---|:---|:---|
| `data/datasets/song/` | required workspace | Local Song operational data surface |
| `outputs/reports/wmb_expression/wmb_kinase_expression.csv` | required for unified attribution | WMB expression specificity for cell-type attribution |
| SEA-AD effect sizes under `config.SEA_AD_DIR` | required for unified attribution | External transcriptomic concordance reference |

## Stage 1 — Data Ingestion (`alz/ingest/song.py`, `alz/pipelines/ingest/`)

Establishes TMT channel-to-animal sample mapping. Phosphosite-to-protein
matching, marker-protein diagnostics, PCA quality control, and outlier
detection run from `alz/ingest/song.py`. The current Kedro ingest pipeline
wraps the sample-mapping and phospho-match steps.

Inputs:

- `data/datasets/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx`
- `data/datasets/song/primary/metadata/Sample_list_72mice (1).xlsx`
- `data/datasets/song/primary/phospho/song_IMAC_compositeSites_merged_labeled (2).xlsx`
- `data/datasets/song/primary/phospho/song_IMAC_sitequant_merged_labeled (2).xlsx`
- `data/datasets/song/primary/phospho/song_pY_*_merged_labeled (2).xlsx`
- `config.MAPPING_CACHE_FILE`

Canonical outputs:

- `outputs/reports/data_ingest/sample_mapping.csv`
- `outputs/reports/data_ingest/phospho_protein_matching.csv`
- `outputs/reports/data_ingest/matching_summary.json`
- `outputs/reports/data_ingest/data_quality.json`
- `outputs/reports/data_ingest/sample_exclusions.csv` (from `--outliers`)
- `outputs/reports/data_ingest/pca_plots/`

Failure modes:

- missing mounted upstream Song proteomics files
- inconsistent TMT/sample-list naming
- missing mapping cache for marker summaries
- sparse or malformed total-proteome matrices causing PCA or matching failure

## Stage 2 — Normalize (`alz/bulk_mea/normalize.py`)

IRS cross-plex normalization (all 72 samples, mode-independent) +
stoichiometry computation (`log2 phospho − log2 protein`). Track-namespaced
(`st` for IMAC pS/pT, `py` for Tyr).

Inputs:

- `outputs/reports/data_ingest/sample_mapping.csv`
- Song total-proteome + per-track phospho sitequant Excel workbooks

Canonical outputs (per track; `st` is the suffix-less default, `py` adds
`_pY`):

- `outputs/reports/kinase_attribution/stoichiometry_matrix{,_pY}.csv`
- `outputs/reports/kinase_attribution/raw_phospho_normalized{,_pY}.csv`
- `outputs/reports/kinase_attribution/stoichiometry_qc{,_pY}.csv`
- `outputs/reports/kinase_attribution/normalization_summary{,_pY}.json`

Implementation: `alz/bulk_mea/normalize.py`.

## Stage 3 — Enrich (`alz/bulk_mea/enrich.py`)

Sample filtering (outlier exclusion + sex per `analysis_mode`), factorial
OLS with disease × timepoint interactions, and MEA (GSEA-based) kinase
enrichment on stoichiometry β values. Track-namespaced.

Inputs:

- `outputs/reports/kinase_attribution/{,_pY}stoichiometry_matrix.csv`
- `outputs/reports/kinase_attribution/{,_pY}raw_phospho_normalized.csv`
- `outputs/reports/data_ingest/sample_mapping.csv`
- `outputs/reports/data_ingest/sample_exclusions.csv` (optional)
- Kedro params: `analysis_mode`, `track`, `sample_exclusions_path`

### OLS Model Specification

The site-level OLS fits a factorial model with disease × timepoint
interactions. Design matrix shape depends on `analysis_mode`:

- `males_only` (default): N × 10 — `const`, `App`, `Tau`, `Int`,
  `time_4mo`, `time_6mo`, `App_x_time4`, `App_x_time6`, `Tau_x_time4`,
  `Tau_x_time6`.
- `full_cohort` (sensitivity): N × 11 — adds a `female` main effect.

Genotype coding: `WT = (App=0, Tau=0, Int=0)`, `APP = (1,0,0)`,
`T22 = (0,1,0)`, `T22/APP = (1,1,1)`. The `Int` column captures the
synergistic interaction between the two transgenes, not an interaction
with sex or timepoint.

Nine time-resolved contrasts are derived from the coefficient vector
(`alz/shared/config.py:CONTRAST_COEFS`):

| Contrast | Coefficient vector |
|:---|:---|
| `App_2mo` | `App` |
| `App_4mo` | `App + App_x_time4` |
| `App_6mo` | `App + App_x_time6` |
| `Tau_2mo` | `Tau` |
| `Tau_4mo` | `Tau + Tau_x_time4` |
| `Tau_6mo` | `Tau + Tau_x_time6` |
| `ApTt_2mo` | `App + Tau + Int` |
| `ApTt_4mo` | `App + Tau + Int + App_x_time4 + Tau_x_time4` |
| `ApTt_6mo` | `App + Tau + Int + App_x_time6 + Tau_x_time6` |

P-values use the contrast variance formula `Var(c'β) = c'(X'X)⁻¹c · σ²`
with two-tailed t-tests, corrected to FDR via Benjamini-Hochberg per
contrast.

Sex enters as a main effect only (nuisance covariate) in `full_cohort`
mode; not interacted with genotype or timepoint. Disease × timepoint
interactions are first-class because the temporal trajectory is the
biology of interest.

Implementation: `alz/bulk_mea/enrich.py` (`_build_design_matrix`,
`GENOTYPE_CODING`, `CONTRAST_COEFS`, `_run_ols_all_sites`, `_run_mea`).

Canonical outputs (per track):

- `outputs/reports/kinase_attribution/site_level_ols{,_pY}.csv`
- `outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv`
- `outputs/reports/kinase_attribution/mea_global_shift{,_pY}.csv`
- `outputs/reports/kinase_attribution/winsorized_sites{,_pY}.csv`
- `outputs/reports/kinase_attribution/mea_substrate_sets{,_pY}.csv`

## Stage 4 — Attribute (`alz/bulk_mea/attribute.py`)

Unified cell-type attribution combining SEA-AD transcriptomic concordance,
WMB expression specificity, and Song within-cohort concordance. Single
namespace (consumes both `st.*` and `py.*` MEA outputs).

Inputs:

- `outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv`
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv`
- SEA-AD effect sizes under `config.SEA_AD_DIR`
- `outputs/reports/snrna_integration/song_*.csv` (optional)
- `data/derived/caches/kinase_to_gene_mapping.csv` (`config.MAPPING_CACHE_FILE`)

Canonical outputs:

- `outputs/reports/kinase_attribution/unified_attribution.csv`
- `outputs/reports/kinase_attribution/unified_attribution_full.csv`
- `outputs/reports/kinase_attribution/sea_ad_supertype_lfc.csv`
- `outputs/reports/kinase_attribution/attribution_summary.json`

Implementation: `alz/bulk_mea/attribute.py`. Concordance model documented in
[`concordance.md`](concordance.md).

Failure modes:

- normalize/enrich outputs missing or stale
- missing `wmb_kinase_expression.csv`
- missing SEA-AD reference files
- mismatch between MEA kinase names and WMB / kinase-to-gene mapping

## Stage 5 — Recover (`alz/bulk_mea/recover.py`)

Cross-contrast consistency analysis and final hypothesis-table assembly.

Inputs:

- `outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv`
- `outputs/reports/kinase_attribution/unified_attribution_full.csv`

Canonical outputs:

- `outputs/reports/attribution_recovery/kinase_activity_matrix.csv`
- `outputs/reports/attribution_recovery/celltype_evidence_table.csv`
- `outputs/reports/attribution_recovery/kinase_hypothesis_table.csv`

Implementation: `alz/bulk_mea/recover.py`.

Failure modes:

- missing unified attribution outputs from Stage 4
- empty or malformed MEA results

## Optional — Mechanism (`alz/bulk_mea/mechanism.py`)

Off the live arc. Reviewer-response stage that re-runs MEA on raw
(uncorrected) phospho LFCs and classifies each (kinase, contrast) as
`activity_driven`, `abundance_driven`, or `both` against the
stoichiometry MEA from Stage 3. Track-namespaced for the per-track raw MEA;
single-namespace combiner produces `mechanism_annotation.csv` and merges
the result back into `unified_attribution.csv`.

Canonical outputs:

- `outputs/reports/kinase_attribution/mea_raw_phospho{,_pY}.csv`
- `outputs/reports/kinase_attribution/mechanism_annotation.csv`
- `outputs/reports/kinase_attribution/unified_attribution.csv` (in-place
  merge — gets a `mechanism_annotation` column added)

## Canonical Deliverables

The mainline deliverables under `outputs/reports/` are:

- Data ingestion outputs under `outputs/reports/data_ingest/`
- Stoichiometry / MEA / unified attribution under
  `outputs/reports/kinase_attribution/`
- Hypothesis tables under `outputs/reports/attribution_recovery/`

The canonical downstream table for the live program is:

- `outputs/reports/attribution_recovery/kinase_hypothesis_table.csv`
