# Live Pipeline Contract

This document defines the operational contract for the single live analysis
front door. It turns the charter into a stage-by-stage runtime specification:
what each stage requires, what it produces, and how the stages connect.

The live ordered sequence is:

1. `bash alz/runners/main/run_data_ingest.sh`
2. `bash alz/runners/main/run_kinase_attribution.sh`
3. `bash alz/runners/main/run_attribution_recovery.sh`

For the bundled front door, use:

```bash
bash alz/runners/main/run_live_pipeline.sh
```

Supporting setup is separate from the front door. In particular:

- `alz/atlas_reference.py` remains supporting external-reference setup
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv` remains a required
  supporting input for unified attribution

## Stage 0 Supporting Prerequisites

These are not co-equal pipeline stages, but the live pipeline expects them.

| Surface | Status | Purpose |
|:---|:---|:---|
| `data/datasets/song/` | required workspace | Local Song operational data surface |
| `outputs/reports/wmb_expression/wmb_kinase_expression.csv` | required for unified attribution | WMB expression specificity for cell-type attribution |
| SEA-AD effect sizes under `config.SEA_AD_DIR` | required for unified attribution | External transcriptomic concordance reference |

## Data Ingestion (`data_ingest.py`)

Data ingestion is the total-proteome integration stage. It establishes sample
mapping, phosphosite-to-protein linkage, marker-protein diagnostics, and
total-proteome quality control.

Inputs:

- `data/datasets/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx`
- `data/datasets/song/primary/proteomics/Sample_list_72mice (1).xlsx`
- `data/datasets/song/primary/proteomics/song_IMAC_compositeSites_merged_labeled (2).xlsx`
- `data/datasets/song/primary/proteomics/song_IMAC_sitequant_merged_labeled (2).xlsx`
- `config.A_OBS_FILE`
- `config.MAPPING_CACHE_FILE`

Canonical outputs:

- `outputs/reports/data_ingest/sample_mapping.csv`
- `outputs/reports/data_ingest/phospho_protein_matching.csv`
- `outputs/reports/data_ingest/matching_summary.json`
- `outputs/reports/data_ingest/datadriven_marker_assessment.csv`
- `outputs/reports/data_ingest/data_quality.json`
- `outputs/reports/data_ingest/pca_plots/`

Failure modes:

- missing mounted upstream Song proteomics files
- inconsistent TMT/sample-list naming
- missing mapping cache or A_obs inputs for marker summaries
- sparse or malformed total-proteome matrices causing PCA or matching failure

## Kinase Attribution (`kinase_attribution.py`)

Kinase attribution performs IRS cross-plex normalization, stoichiometry
computation, OLS site-level modelling, MEA (GSEA-based) kinase enrichment on
stoichiometry beta values, and unified cell-type attribution combining SEA-AD
concordance and WMB expression specificity for all significant kinases.

Inputs:

- `outputs/reports/data_ingest/sample_mapping.csv`
- `data/datasets/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx`
- `data/datasets/song/primary/proteomics/song_IMAC_sitequant_merged_labeled (2).xlsx`
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv`
- SEA-AD effect sizes under `config.SEA_AD_DIR`
- `config.MAPPING_CACHE_FILE` (kinase-to-gene mapping)

### OLS Model Specification

The site-level OLS fits a main-effects factorial model with 7 parameters across
72 animals (65 residual degrees of freedom):

| Column | Meaning |
|:---|:---|
| `const` | Intercept |
| `App` | APP transgene main effect |
| `Tau` | Tau (T22) transgene main effect |
| `Int` | App × Tau interaction (1 only for T22/APP) |
| `female` | Sex main effect (1 = female) |
| `time_4mo` | Timepoint indicator (1 = 4 months) |
| `time_6mo` | Timepoint indicator (1 = 6 months) |

Genotype coding: `WT = (0,0,0)`, `APP = (1,0,0)`, `T22 = (0,1,0)`,
`T22/APP = (1,1,1)`. The `Int` column captures the synergistic interaction
between the two transgenes, not an interaction with sex or timepoint.

Three contrasts are derived from the coefficient vector:

- **App**: APP vs WT main effect (`β_App`)
- **Tau**: T22 vs WT main effect (`β_Tau`)
- **ApTt**: T22/APP vs WT full effect (`β_App + β_Tau + β_Int`)

P-values use the contrast variance formula `Var(c'β) = c'(X'X)⁻¹c · σ²` with
two-tailed t-tests, corrected to FDR via Benjamini-Hochberg per contrast.

Sex and timepoint enter as main effects only (nuisance covariates). They are not
interacted with genotype. This is a deliberate design choice: with 72 animals
across a 2×3×4 factorial, a full interaction model would consume degrees of
freedom that provide little benefit for the downstream GSEA-based enrichment,
which aggregates across all sites for a given kinase. The site-level genotype
contrast betas are what feed into MEA, so sex and timepoint are controlled for
but not interrogated as moderators.

See `alz/kinase_attribution.py` (`_build_design_matrix`, `GENOTYPE_CODING`,
`CONTRAST_COEFS`) and `config.py` for the implementation.

Canonical outputs:

- `outputs/reports/kinase_attribution/stoichiometry_matrix.csv`
- `outputs/reports/kinase_attribution/raw_phospho_normalized.csv`
- `outputs/reports/kinase_attribution/mea_stoichiometry.csv`
- `outputs/reports/kinase_attribution/site_level_ols.csv`
- `outputs/reports/kinase_attribution/unified_attribution.csv`
- `outputs/reports/kinase_attribution/attribution_summary.json`

Optional supplementary output (via `--mechanism-annotation`):

- `outputs/reports/kinase_attribution/mechanism_annotation.csv`

Failure modes:

- data ingestion outputs missing or stale
- missing `outputs/reports/wmb_expression/wmb_kinase_expression.csv`
- missing SEA-AD reference files
- mismatch between phospho site IDs and protein mapping

## Attribution Recovery (`attribution_recovery.py`)

Attribution recovery is the final attribution-table assembly stage. It adds
cross-contrast consistency analysis and produces the canonical deliverable.

Inputs:

- `outputs/reports/kinase_attribution/unified_attribution.csv`
- `outputs/reports/kinase_attribution/mea_stoichiometry.csv`

Canonical outputs:

- `outputs/reports/attribution_recovery/cross_contrast_matrix.csv`
- `outputs/reports/attribution_recovery/cross_contrast_heatmap.png`
- `outputs/reports/attribution_recovery/final_attribution_table.csv`

Failure modes:

- missing unified attribution outputs from kinase attribution
- empty or malformed MEA results

## Canonical Deliverables

The current mainline deliverables under `outputs/reports/` are:

- Data ingestion outputs under `outputs/reports/data_ingest/`
- Kinase attribution outputs under `outputs/reports/kinase_attribution/`
- Attribution recovery outputs under `outputs/reports/attribution_recovery/`

The canonical downstream table for the live program is:

- `outputs/reports/attribution_recovery/final_attribution_table.csv`
