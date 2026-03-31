# Alzheimer's Kinase Analysis

This repository is the study-specific workspace for Alzheimer's multi-omic kinase and signaling analysis.

It now has three clearly separated surfaces:

- main analysis code in `code/`
- authoritative Song dataset workspace in `data/incytr_collections/song/`
- archived deconvolution benchmark and transition work in `deconv/`

The reusable signaling method implementation lives in the separate `incytr` repository and is loaded from the local environment when needed.

## Current Status

For the Song (`yuyu01`) dataset:

- `data/incytr_collections/song/` is the authoritative local workspace
- `data/gdrive_shared/yuyu01/` should be treated as upstream archive and provenance source
- the active Song `pr` / `ps` / `py` files are regenerated with the standardized `A_obs + DESP` workflow
- the exact historical collaborator outputs are preserved under `data/incytr_collections/song/proteomics/legacy/`

For benchmarking and method-transition history:

- `deconv/` is retained as an archive
- it should not be treated as the main live analysis surface

## Top-Level Layout

```text
alzheimers/
├── code/                           # Main analysis scripts and helpers
│   ├── export_song_aobs_desp.py    # Regenerate Song pr/ps/py with A_obs + DESP
│   ├── kl_analysis_clusters.py     # Deconvoluted kinase-enrichment pipeline
│   ├── kl_analysis_bulk.py         # Bulk kinase-enrichment pipeline
│   ├── sap_data.py                 # SAP Phase 0–1: data ingestion + RNA preprocessing
│   ├── sap_model.py                # SAP Phase 2: Hurdle-Tweedie GLM fitting + LOCO-CV
│   ├── sap_validate.py             # SAP Phase 3: validation suite (synthetic, permutation, etc.)
│   ├── sap_perf_test.py            # SAP numerical regression tests
│   └── r/                          # Alzheimer's-specific InCytr runners
├── data/
│   ├── incytr_collections/song/    # Authoritative localized Song workspace
│   ├── gdrive_shared/              # Upstream archive mounts
│   └── lucie_proteomics/           # 5xFAD upstream proteomics sources
├── docs/                           # Live repo docs
├── outputs/                        # Generated outputs and reports
├── sap.md                          # Statistical Analysis Plan for SAP deconvolution model
├── deconv/                         # Archived benchmark and transition workspace
├── scripts/setup_gdrive_mounts.sh  # Mount helper
├── regenerate_outputs_and_reports.sh
├── environment.yml
└── README.md
```

## Documentation Entry Points

Start here if you need orientation:

- [data/incytr_collections/song/INDEX.md](/home/hchung/Projects/work/alzheimers/data/incytr_collections/song/INDEX.md)
- [data/incytr_collections/song/source/INDEX.md](/home/hchung/Projects/work/alzheimers/data/incytr_collections/song/source/INDEX.md)
- [data/incytr_collections/song/CODE_DATA_RELATIONSHIP.md](/home/hchung/Projects/work/alzheimers/data/incytr_collections/song/CODE_DATA_RELATIONSHIP.md)
- [docs/downstream_output_index.md](/home/hchung/Projects/work/alzheimers/docs/downstream_output_index.md)
- [deconv/docs/deconvolution-transition-aobs-desp.md](/home/hchung/Projects/work/alzheimers/deconv/docs/deconvolution-transition-aobs-desp.md)

## Environment Setup

Create and activate the mixed Python/R environment:

```bash
mamba env create -f environment.yml
mamba activate alzheimers
```

Install the local `incytr` package into the environment:

```bash
mamba run -n alzheimers Rscript -e 'devtools::install("/home/hchung/Projects/work/incytr", dependencies=FALSE)'
```

If you need the study data mounts:

```bash
bash scripts/setup_gdrive_mounts.sh
```

## Authoritative Data Layout

### Song workspace

The active Song dataset lives under:

- `data/incytr_collections/song/source/`
- `data/incytr_collections/song/transcriptomics/`
- `data/incytr_collections/song/proteomics/`
- `data/incytr_collections/song/markers/`
- `data/incytr_collections/song/kinase/`
- `data/incytr_collections/song/analysis_support/`
- `data/incytr_collections/song/analysis_cache/`
- `data/incytr_collections/song/method_records/`

Important files:

- foundational source records:
  - `data/incytr_collections/song/source/single_cell/`
  - `data/incytr_collections/song/source/bulk_omics/`
  - `data/incytr_collections/song/source/metadata/`
- active proteomics:
  - `data/incytr_collections/song/proteomics/pr_yuyu_deconvoluted.csv`
  - `data/incytr_collections/song/proteomics/ps_yuyu_deconvoluted.csv`
  - `data/incytr_collections/song/proteomics/py_yuyu_deconvoluted.csv`
- preserved historical bundle:
  - `data/incytr_collections/song/proteomics/legacy/*.csv`
- localized bulk median inputs:
  - `data/incytr_collections/song/proteomics/source/pr_median.csv`
  - `data/incytr_collections/song/proteomics/source/imac_median.csv`
  - `data/incytr_collections/song/proteomics/source/py_median.csv`
- current support and cache files:
  - `data/incytr_collections/song/analysis_support/median_cluster_sizes.csv`
  - `data/incytr_collections/song/analysis_cache/kinase_to_gene_mapping.csv`
  - `data/incytr_collections/song/analysis_cache/allen_expression_cache.csv`

### Upstream archive

The collaborator-owned mounted tree under `data/gdrive_shared/` is still useful for provenance and source recovery, but it is no longer the default runtime dependency for Song.

## Main Workflows

### 1. Regenerate Song `pr` / `ps` / `py`

This rebuilds the active Song proteomics bundle with the current `A_obs + DESP` workflow:

```bash
python code/export_song_aobs_desp.py --modalities pr,ps,py
```

Inputs come from the localized Song workspace and the localized `A_obs` records under `data/incytr_collections/song/method_records/aobs_desp_standardized/`.

### 2. Run deconvoluted kinase enrichment

Serine/threonine:

```bash
python code/kl_analysis_clusters.py --step all --kin-type ser_thr
```

Tyrosine:

```bash
python code/kl_analysis_clusters.py --step all --kin-type tyrosine
```

These write to:

- `outputs/deconv/`
- `outputs/deconv_tyrosine/`

### 3. Run bulk kinase enrichment

Serine/threonine:

```bash
python code/kl_analysis_bulk.py --step all --kin-type ser_thr
```

Tyrosine:

```bash
python code/kl_analysis_bulk.py --step all --kin-type tyrosine
```

These write to:

- `outputs/bulk/`
- `outputs/bulk_tyrosine/`

### 4. Refresh plots, summaries, and reports

Fast mode recomputes summaries and plots from existing enrichment CSVs:

```bash
bash regenerate_outputs_and_reports.sh
```

Full mode reruns enrichment first:

```bash
bash regenerate_outputs_and_reports.sh --full
```

### 5. SAP model (condition-specific deconvolution)

Cell-type-resolved phosphoproteomic deconvolution via Hurdle-Tweedie GLM. See [`sap.md`](sap.md) for the full statistical analysis plan and `CLAUDE.md` for all CLI flags.

```bash
python code/sap_model.py --fit       # full LOCO-CV + model fit
python code/sap_validate.py --all    # validation suite
```

### 6. Run the local Song InCytr example

```bash
Rscript code/r/run_incytr_ad_models_yuyu01.R
```

This consumes the localized Song bundle under `data/incytr_collections/song/`.

## Analysis Logic

The main kinase-enrichment pipeline currently uses:

- percentile-based site classification
- `kinase-library` enrichment
- ser/thr and tyrosine modes via `--kin-type`
- Song-owned support and cache files under `data/incytr_collections/song/`

For the Song dataset, the important provenance split is:

1. historical collaborator deconvolution outputs preserved in `proteomics/legacy/`
2. active standardized `A_obs + DESP` outputs in `proteomics/`

## Outputs

The main generated output trees are:

- `outputs/deconv/`
- `outputs/deconv_tyrosine/`
- `outputs/bulk/`
- `outputs/bulk_tyrosine/`
- `outputs/reports/`

See [docs/downstream_output_index.md](/home/hchung/Projects/work/alzheimers/docs/downstream_output_index.md) for the current output map.

Important note:

- some existing files under `outputs/` may still reflect older mixed-history runs from before the Song workspace migration
- if you need a clean current-state result set, regenerate the relevant output tree

## Repository Conventions

- prefer `data/incytr_collections/song/` over ad hoc files elsewhere in `data/`
- treat `data/gdrive_shared/yuyu01/` as upstream archive, not active workspace
- treat `deconv/` as archived benchmark context, not the primary analysis surface
- use the docs indexes before adding new one-off notes

## 5xFAD Note

`5xFAD` work remains in this repository, but it is separate from the Song-localization effort. The current 5xFAD provenance and input validation notes live under:

- `docs/integrations/`

and the upstream proteomics source area is:

- `data/lucie_proteomics/`
