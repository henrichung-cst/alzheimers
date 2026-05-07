# OUT OF DATE — IGNORE

References the retired live `data/gdrive_shared/` mount. The current pattern is `pixi run ingest-gdrive-shared` (rclone copy into `data/raw/external/gdrive_shared/`, on demand); the local mirror was deleted on 2026-05-07. See `CLAUDE.md` for the live layout.

---

# Integrations Directory Structure (Upstream Archive View)

This document describes the structure of the collaborator-owned upstream archive under `data/gdrive_shared/`.

It is not the authoritative runtime layout for the Song dataset anymore. For Song (`yuyu01`), the authoritative localized workspace is now:

- `data/incytr_collections/song/`

Use this document to understand the upstream archive and original source locations, not to choose default runtime paths for Song analyses.

## Directory Overview

The `data/gdrive_shared` directory contains subdirectories organized by data batch or study.

Interpretation:

- for `5xFAD`, this tree is still an important upstream source area
- for `Song` / `yuyu01`, this tree should now be treated as upstream archive and provenance source, while `data/incytr_collections/song/` is the operational workspace

### `lore00` (5xFAD Source - Primary)

The `lore00` directory contains the multi-modal data for the 5xFAD mouse model, processed by Ivan Gregoretti, PhD. This is the **most recent** 5xFAD dataset (July 2024).

#### Documentation: `data/gdrive_shared/lore00/transcriptomics/log00.md`
- **Primary Source of Truth**: Describes the complete processing pipeline.
- **Genome**: Custom **mm10** (Ensembl 112) with five human transgenes (**TgAPP** and **TgPSEN1**).
- **Pipeline**: Parse Biosciences `split-pipe` v1.3.1.

#### Transcriptomics: `data/gdrive_shared/lore00/transcriptomics/`
- **`obs_df.csv`**: The primary metadata and annotation file (91,905 cells).
- **`analysis/240712_ex1_comb1/all-sample/DGE_filtered/`**: Merged count matrix for all samples.
- **`reclustering/named_lore00.RDS`**: Refined R object ready for Seurat/Incytr (790MB).

#### Proteomics & Kinase Data (Staged): `examples/5xad_data/`
While the raw FASTA is in `lore00/proteomics`, the processed inputs for Incytr are staged in the examples folder:
- `processed_pr_5X_v2.csv` / `processed_pr_WT_v2.csv`: Proteomics (PR).
- `processed_ps_5X_v2.csv` / `processed_ps_WT_v2.csv`: Phospho-Ser (PS).
- `processed_py_5X_v2.csv` / `processed_py_WT_v2.csv`: Phospho-Tyr (PY).
- `kldata_pspy.csv`: Validated kinase-substrate mapping data.

---

### `yuyu01` (Tau/AppP Consolidated Data)

Contains consolidated data for Alzheimer's models (`Ttau`, `AppP`, `ApTt`). This dataset **supersedes `yuyu00`**.

#### Documentation: `data/gdrive_shared/yuyu01/transcriptomics/log01.md`
- **Status**: Multi-batch consolidated data (January 2024).

#### Transcriptomics: `data/gdrive_shared/yuyu01/transcriptomics/scanpy/`
- **`170_gex_celltypes_00.h5ad`**: Final annotated object with cell type assignments (606MB).
- **`165_gex_clusters_00_named.h5ad`**: Alternate annotated object (2.2GB).

#### Proteomics (Raw Sources): `data/gdrive_shared/yuyu01/proteomics/`
- Contains TMT 6-plex Excel files (`song_IMAC_sitequant_merged_labeled (2).xlsx`, `song2024_tmttotal_protein_quant_merged_labeled (2).xlsx`, etc.). These are the raw site-quantification sources.

#### Incytr-Specific Inputs (Historical upstream bundle): `data/gdrive_shared/yuyu01/documentation/incytr/incytr input/`
Historical pre-processed CSVs derived from the raw proteomics and transcriptomics:
- `pr_yuyu_deconvoluted.csv`, `ps_yuyu_deconvoluted.csv`, `py_yuyu_deconvoluted.csv`.
- `kldata.csv`: Kinase library mapping.
- `incytr_obj.rds`: Pre-built Incytr object for this model (664MB).

Current operational note:

- these historical upstream inputs have been localized and reorganized under `data/incytr_collections/song/`
- the active Song `pr` / `ps` / `py` files in the localized workspace are regenerated `A_obs + DESP` outputs
- the exact old collaborator outputs are preserved separately under `data/incytr_collections/song/proteomics/legacy/`

---

### `MC-38` (Colorectal Cancer Model)

Used as a reference/benchmark dataset for pipeline validation.

#### Transcriptomics
- **`data/gdrive_shared/MC-38/130_mc38_expression_counts_00.tsv`**: Raw count matrix.

#### Proteomics
- **`data/gdrive_shared/MC-38/results_proteomics/KR MC38 siteQuant_19_export_pY.xls`**: Phospho-Tyr (pY) site quantification.

---

### `yuyu00` (Older AD Data)
- **Status**: Deprecated. Single batch data, superseded by the multi-batch consolidation in `yuyu01`.
- **Recommendation**: Do not use for new analyses.

## Summary of Matching Sets for Incytr Pipeline

| Model | Transcriptomics (Object/Matrix) | Proteomics (PR/PS/PY) | Kinase Data |
| :--- | :--- | :--- | :--- |
| **5xFAD** | `lore00/.../named_lore00.RDS` | `examples/5xad_data/processed_*.csv` | `examples/5xad_data/kldata_pspy.csv` |
| **AD Models (upstream archive view)** | `yuyu01/.../170_gex_celltypes_00.h5ad` | `yuyu01/.../incytr input/*_deconvoluted.csv` | `yuyu01/.../incytr input/kldata.csv` |
| **MC-38** | `MC-38/130_mc38_expression_counts_00.tsv` | `MC-38/results_proteomics/...pY.xls` | N/A |
