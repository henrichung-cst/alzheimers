# 5xFAD Omics Status

Updated: 2026-06-15

This note records the current status of 5xFAD omics data in this repository and
the candidate matching transcriptomics data in Google Drive. It is a provenance
snapshot, not a completed integration plan.

Implementation status as of this update:

- `conf/data_sources.yaml` now declares a `5xfad-snrna` Drive source.
- `pixi.toml` now exposes `ingest-5xfad-snrna` for small metadata files and
  `ingest-5xfad-snrna-rds` for the large reclustering RDS group.
- `alz/ingest/build_5xfad_omics_join_manifest.py` builds the explicit
  transcriptomics/proteomics join manifest at
  `data/datasets/5xFAD/metadata/omics_join_manifest.csv`.
- `alz/ingest/audit_5xfad_proteomics_sample_lists.py` audits the delivered
  proteomics DOCX sample lists and writes
  `outputs/reports/5xfad_proteomics_sample_list_audit.csv`.
- `alz/ingest/inspect_5xfad_snrna_rds.R` inspects downloaded reclustering RDS
  objects and writes `outputs/reports/5xfad_snrna_rds_inspection.json`.
- `alz/ingest/audit_5xfad_snrna_transgenes.R` audits `TgAPP` and `TgPSEN1`
  expression per snRNA sample and writes
  `outputs/reports/5xfad_snrna_transgene_audit.csv`.
- The newest reclustering object,
  `data/datasets/5xFAD/primary/scrna/reclustering/fivex_renamed_from_merged.RDS`,
  has been retrieved locally for inspection.
- A separate local target root, `data/derived/5xfad_incytr_inputs/`, has been
  created for future 5xFAD Incytr input generation.
- The proteomics genotype audit found delivered DOCX sample lists in the Lucie
  proteomics report bundle. Those lists correct the previous inferred genotype
  map for 6-month sample 15 and 12-month sample 6.
- The current cross-omics foundation is 31 direct per-animal joins plus 1
  pooled-only audit-hold row excluded from per-animal integration.

## Scope

This status is specific to the Lucie/Lorenzo 5xFAD mouse cohort. The available
design variables are:

- Genotype: WT / WildT and 5xFAD / TG
- Age: 3, 6, 9, and 12 months
- Tissue: cortex and hippocampus
- Sex: male, based on the delivered proteomics filenames and local dataset
  index

## Sample Count Levels

The dataset has several different count levels that should not be conflated:

| Level | Count | Meaning |
|:---|---:|:---|
| Local proteomics-related primary files | 18 | Proteomics report/correction files under `data/datasets/5xFAD/primary/` |
| Local transcriptomics/snRNA primary files | 5 | snRNA metadata/provenance/object files under `data/datasets/5xFAD/primary/scrna/` |
| snRNA samples | 32 | One row per transcriptomics biological sample label in `obs_df.csv`; balanced at 2 samples per tissue x age x genotype condition |
| snRNA cells/nuclei | 91,904 | Cell/nucleus metadata rows in `obs_df.csv` and the inspected Seurat object |
| Proteomics individual biological sample IDs | 57 | Unique non-pool tissue-level biological sample IDs represented across proteomics assays |
| Proteomics pool runs | 3 | Explicit pooled cortex runs retained as provenance and excluded from individual-sample contrasts |
| Proteomics manifest rows | 279 | Raw-run rows across tissues and assay families, including repeated measurements of the same biological sample across assays |
| Cross-omics join rows | 32 | One row per snRNA sample in `data/datasets/5xFAD/metadata/omics_join_manifest.csv` |
| Cross-omics direct individual joins | 31 | snRNA samples with direct individual proteomics biological-sample counterparts |
| Cross-omics excluded rows | 1 | `WildT_06mo_C_11`, excluded because its cortex proteomics counterpart is present only in a pool |

The short version is: snRNA is a 32-sample balanced subset, while proteomics is
a larger tissue-level cohort with more animals per condition and multiple assay
measurements per animal. The matched per-animal cross-omics subset is therefore
defined by the 32-row join manifest, not by the full proteomics manifest row
count.

## Local Proteomics

The local repository has analysis-ready 5xFAD proteomics reports under:

`data/datasets/5xFAD/primary/`

Available local assay families:

| Tissue | Available assays | Current kinase-analysis use |
|:---|:---|:---|
| cortex | total, IMAC, pY, KGG, AcK | IMAC/ST and pY are in `kinase_mea_v1`; total/KGG/AcK are retained as available provenance or support inputs |
| hippocampus | total, IMAC, pY, KGG, AcK | IMAC/ST and pY are in `kinase_mea_v1`; total/KGG/AcK are retained as available provenance or support inputs |

Current local manifest outputs:

| Path | Role |
|:---|:---|
| `outputs/reports/kinase_attribution_5xfad/sample_manifest.csv` | Raw-run to biological-sample manifest, including parsed tissue, assay, age, genotype, pool flag, duplicate group, and action |
| `outputs/reports/kinase_attribution_5xfad/dataset_index.csv` | Per-tissue/per-assay source-file index |
| `outputs/reports/unified_viewer/audit_sources/5xfad_sample_manifest.csv` | Viewer copy of the sample manifest |

Current manifest summary:

- 279 total manifest rows
- 276 primary raw-run rows
- 3 explicit pool rows excluded from primary contrasts
- 57 unique biological sample IDs represented across assays
- Genotype calls in the local proteomics manifest now come from delivered Lucie
  proteomics DOCX sample lists:
  - `data/raw/external/lucie_proteomics/reports/Male Cortex/5xFAD Male Cortex Sample List.docx`
  - `data/raw/external/lucie_proteomics/reports/Male Hippocampus/Sample IDs 5xFAD Male Hippocampus.docx`

The three pool runs are:

| Tissue | Assay | Raw run | Delivered pool meaning | Current action |
|:---|:---|:---|:---|:---|
| cortex | total | `260203_LD_CTX_M6_Pool_TP_DIA.raw` | M6 Pool (11 + 14) WT | Exclude from individual-sample contrasts and matched per-animal joins |
| cortex | total | `260203_LD_CTX_M12_Pool_TP_DIA.raw` | M12 Pool (5 + 9) TG | Exclude from individual-sample contrasts and matched per-animal joins |
| cortex | pY | `011626_LD_Cort_M6_pool_pY.raw` | M6 WT pool, interpreted with the delivered M6 Pool (11 + 14) WT provenance | Exclude from individual-sample contrasts and matched per-animal joins |

Proteomics biological-sample availability by condition:

| Tissue | Age | Genotype | Unique biological samples in local proteomics |
|:---|:---|:---|---:|
| cortex | 3mo | WT | 3 |
| cortex | 3mo | TG | 4 |
| cortex | 6mo | WT | 2 |
| cortex | 6mo | TG | 4 |
| cortex | 9mo | WT | 2 |
| cortex | 9mo | TG | 3 |
| cortex | 12mo | WT | 5 |
| cortex | 12mo | TG | 3 |
| hippocampus | 3mo | WT | 3 |
| hippocampus | 3mo | TG | 4 |
| hippocampus | 6mo | WT | 5 |
| hippocampus | 6mo | TG | 4 |
| hippocampus | 9mo | WT | 2 |
| hippocampus | 9mo | TG | 3 |
| hippocampus | 12mo | WT | 5 |
| hippocampus | 12mo | TG | 5 |

## Local Transcriptomics

The local folder:

`data/datasets/5xFAD/primary/scrna/`

now contains the original starter R Markdown file plus retrieved Drive
transcriptomics artifacts:

| Local path | Role |
|:---|:---|
| `5xFAD_scRNAseq_Clustering.rmd` | Starter/default R Markdown file; not a usable analysis object |
| `obs_df.csv` | Cell/sample metadata table used for sample matching |
| `log00.md` | Parse Biosciences processing log with repeated `--sample` assignments |
| `analysis/240712_ex1_comb1/agg_samp_ana_summary.csv` | Combined Parse sample summary |
| `reclustering/fivex_renamed_from_merged.RDS` | Retrieved 2025 Seurat reclustering object; current canonical snRNA object candidate |
| `outputs/reports/5xfad_snrna_transgene_audit.csv` | Per-sample `TgAPP`/`TgPSEN1` audit validating the snRNA genotype labels |
| `outputs/reports/5xfad_proteomics_sample_list_audit.csv` | Delivered proteomics DOCX sample-list audit validating the proteomics genotype map |

The older R Markdown file should not be used as an analysis input. The current
usable transcriptomics foundation is the retrieved Seurat RDS plus explicit
metadata/join manifests.

## Candidate Drive Transcriptomics

Candidate matching transcriptomics data was found in the provided Google Drive
folder:

`https://drive.google.com/drive/folders/1Ad6Xc-hrBEz02PrrRw5DP6N165dQQyBG`

The folder was visible through the project `rclone` Google Drive remotes. The
Google Drive connector did not list it, but `rclone` did.

Important Drive artifacts observed:

| Drive path | Status |
|:---|:---|
| `obs_df.csv` | Cell/sample metadata table inspected for sample matching |
| `scanpy/165_gex_clusters_00.h5ad` | Processed Scanpy object, about 782 MB |
| `scanpy/165_gex_clusters_00.html` | Scanpy report |
| `reclustering/fivex_renamed_from_merged.RDS` | Processed R/Seurat-style object, about 829 MB |
| `reclustering/named_lore00.RDS` | Processed R/Seurat-style object, about 829 MB |
| `reclustering/with_cluster_names_merged_object.RDS` | Processed R/Seurat-style object, about 1.9 GB |
| `log00.md` | Processing log with Parse Biosciences sample assignments |

The inspected `obs_df.csv` metadata contained:

- 32 transcriptomics samples
- 91,904 cell/nucleus metadata rows after filtering
- 2 transcriptomics samples per genotype x age x tissue condition
- sample labels following the pattern:
  - `5XFAD_03mo_C_10`
  - `WildT_12mo_H_08`

Transcriptomics sample availability by condition:

| Tissue | Age | Genotype | snRNA samples |
|:---|:---|:---|:---|
| cortex | 3mo | WT | `WildT_03mo_C_01`, `WildT_03mo_C_02` |
| cortex | 3mo | TG | `5XFAD_03mo_C_10`, `5XFAD_03mo_C_11` |
| cortex | 6mo | WT | `WildT_06mo_C_11`, `WildT_06mo_C_15` |
| cortex | 6mo | TG | `5XFAD_06mo_C_18`, `5XFAD_06mo_C_19` |
| cortex | 9mo | WT | `WildT_09mo_C_10`, `WildT_09mo_C_11` |
| cortex | 9mo | TG | `5XFAD_09mo_C_13`, `5XFAD_09mo_C_14` |
| cortex | 12mo | WT | `WildT_12mo_C_04`, `WildT_12mo_C_08` |
| cortex | 12mo | TG | `5XFAD_12mo_C_06`, `5XFAD_12mo_C_11` |
| hippocampus | 3mo | WT | `WildT_03mo_H_01`, `WildT_03mo_H_02` |
| hippocampus | 3mo | TG | `5XFAD_03mo_H_10`, `5XFAD_03mo_H_11` |
| hippocampus | 6mo | WT | `WildT_06mo_H_11`, `WildT_06mo_H_15` |
| hippocampus | 6mo | TG | `5XFAD_06mo_H_18`, `5XFAD_06mo_H_19` |
| hippocampus | 9mo | WT | `WildT_09mo_H_10`, `WildT_09mo_H_11` |
| hippocampus | 9mo | TG | `5XFAD_09mo_H_13`, `5XFAD_09mo_H_14` |
| hippocampus | 12mo | WT | `WildT_12mo_H_04`, `WildT_12mo_H_08` |
| hippocampus | 12mo | TG | `5XFAD_12mo_H_06`, `5XFAD_12mo_H_11` |

Observed transcriptomics metadata fields include sample, batch, sublibrary,
QC counts, Leiden cluster, fine cluster label, and coarse cluster label.

Coarse cell-type labels observed in the transcriptomics metadata include:

- Excitatory neurons
- Interneurons
- Oligodendrocytes
- Astrocytes
- Medium spiny neurons
- Microglia
- OPCs
- Endothelial cells
- Choroid plexus
- Unknown / high-mitochondrial categories

## Drive Reclustering Folder

The Drive `reclustering/` folder currently contains three RDS objects:

| Drive path | Size | Modified | Initial interpretation |
|:---|---:|:---|:---|
| `reclustering/fivex_renamed_from_merged.RDS` | 828,769,899 bytes | 2025-11-19 | Newest object; likely the first candidate for the canonical 5xFAD snRNA source because the name suggests renamed/curated identities from a merged object |
| `reclustering/with_cluster_names_merged_object.RDS` | 1,892,232,087 bytes | 2025-11-05 | Larger merged object with cluster names; likely useful as a fallback or provenance source if the newest object is stripped down |
| `reclustering/named_lore00.RDS` | 828,639,889 bytes | 2025-09-24 | Older named object; likely lower priority unless it carries metadata missing from the newer files |

These 2025 reclustering files are newer than the observed 2024 Scanpy object
(`scanpy/165_gex_clusters_00.h5ad`). They should be treated as the likely
canonical transcriptomics inputs for any new 5xFAD Incytr work, pending direct
inspection of their Seurat/R object schema, assays, identities, and metadata.

Inspection of `fivex_renamed_from_merged.RDS` under the project pixi R
environment found:

- Class: Seurat
- Dimensions: 30,701 genes x 91,904 cells
- Assay: `originalexp`, with `counts` and `data`
- Reduction: `umap`
- Sample metadata: 32 samples, all parseable from the `sample` column
- Derived design balance: 2 transcriptomics samples for every tissue x age x
  genotype condition across cortex/hippocampus, 3/6/9/12 months, and WT/TG
- Cluster metadata columns: `leiden`, `coarse_cluster`, `fine_cluster`,
  `new_clusters`
- `new_clusters` contains 46 labels and is the likely first cluster-column
  candidate for 5xFAD Incytr, while `fine_cluster` contains 33 labels and
  mirrors the earlier observed fine cluster labels.

The object does not carry explicit `condition`, `Group`, `age`, `genotype`, or
`tissue` metadata columns. Those fields must be derived from `sample` and the
join manifest before the object is used by the Incytr driver.

## Transcriptomics-to-Proteomics Matching

The Drive transcriptomics design matches the local proteomics design at the
condition level:

- WT/WildT versus 5xFAD/TG
- cortex and hippocampus
- 3, 6, 9, and 12 months

For per-sample matching, transcriptomics sample IDs were normalized as:

- `5XFAD` -> `TG`
- `WildT` -> `WT`
- `C` -> `cortex`
- `H` -> `hippocampus`
- `03mo`, `06mo`, `09mo` -> `3mo`, `6mo`, `9mo`

After the completed proteomics DOCX sample-list audit and manifest rebuild, 31
of 32 transcriptomics samples directly match biological sample IDs in the local
proteomics manifest.

Directly matched transcriptomics sample IDs:

| Transcriptomics sample | Proteomics biological sample ID |
|:---|:---|
| `5XFAD_03mo_C_10` | `cortex_3mo_TG_10` |
| `5XFAD_03mo_C_11` | `cortex_3mo_TG_11` |
| `5XFAD_03mo_H_10` | `hippocampus_3mo_TG_10` |
| `5XFAD_03mo_H_11` | `hippocampus_3mo_TG_11` |
| `5XFAD_06mo_C_18` | `cortex_6mo_TG_18` |
| `5XFAD_06mo_C_19` | `cortex_6mo_TG_19` |
| `5XFAD_06mo_H_18` | `hippocampus_6mo_TG_18` |
| `5XFAD_06mo_H_19` | `hippocampus_6mo_TG_19` |
| `5XFAD_09mo_C_13` | `cortex_9mo_TG_13` |
| `5XFAD_09mo_C_14` | `cortex_9mo_TG_14` |
| `5XFAD_09mo_H_13` | `hippocampus_9mo_TG_13` |
| `5XFAD_09mo_H_14` | `hippocampus_9mo_TG_14` |
| `5XFAD_12mo_C_06` | `cortex_12mo_TG_6` |
| `5XFAD_12mo_C_11` | `cortex_12mo_TG_11` |
| `5XFAD_12mo_H_06` | `hippocampus_12mo_TG_6` |
| `5XFAD_12mo_H_11` | `hippocampus_12mo_TG_11` |
| `WildT_03mo_C_01` | `cortex_3mo_WT_1` |
| `WildT_03mo_C_02` | `cortex_3mo_WT_2` |
| `WildT_03mo_H_01` | `hippocampus_3mo_WT_1` |
| `WildT_03mo_H_02` | `hippocampus_3mo_WT_2` |
| `WildT_06mo_C_15` | `cortex_6mo_WT_15` |
| `WildT_06mo_H_11` | `hippocampus_6mo_WT_11` |
| `WildT_06mo_H_15` | `hippocampus_6mo_WT_15` |
| `WildT_09mo_C_10` | `cortex_9mo_WT_10` |
| `WildT_09mo_C_11` | `cortex_9mo_WT_11` |
| `WildT_09mo_H_10` | `hippocampus_9mo_WT_10` |
| `WildT_09mo_H_11` | `hippocampus_9mo_WT_11` |
| `WildT_12mo_C_04` | `cortex_12mo_WT_4` |
| `WildT_12mo_C_08` | `cortex_12mo_WT_8` |
| `WildT_12mo_H_04` | `hippocampus_12mo_WT_4` |
| `WildT_12mo_H_08` | `hippocampus_12mo_WT_8` |

## Sample-Label Reconciliation

One transcriptomics sample label does not make a direct label-identical
per-animal join to the proteomics manifest:

| Transcriptomics sample | Normalized target ID | Current issue |
|:---|:---|:---|
| `WildT_06mo_C_11` | `cortex_6mo_WT_11` | The delivered cortex sample list identifies M6 sample 11 as WT, but only inside the M6 Pool (11 + 14) WT entry. No individual cortex proteomics raw run for sample 11 is present in the local reports. |

Current reconciliation evidence:

- Delivered proteomics sample-list DOCX files were found in the Lucie proteomics
  report bundle for both cortex and hippocampus.
- The reproducible audit
  `outputs/reports/5xfad_proteomics_sample_list_audit.csv` parsed 59 delivered
  records and found 0 individual-sample mismatches against the corrected
  proteomics genotype map.
- The DOCX lists resolve the prior genotype conflict: 6-month sample 15 is WT
  in both cortex and hippocampus, and 12-month sample 6 is TG in both cortex and
  hippocampus.
- The Parse `log00.md` sample-assignment blocks and the downloaded Parse
  `run_proc_def.json` files repeatedly encode the same 32 biological sample
  labels and well assignments. Those labels explicitly contain genotype, age,
  tissue, and sample number.
- The Illumina `SampleSheet.csv` only maps sublibraries to sequencing indexes;
  it does not contain biological animal genotype or tissue assignments.
- The 2025 Seurat object contains `TgAPP` and `TgPSEN1` rows. The reproducible
  audit in `outputs/reports/5xfad_snrna_transgene_audit.csv` supports the snRNA
  genotype labels: all 16 `5XFAD_*` samples show high transgene expression,
  while all 16 `WildT_*` samples are near background.
- For the disputed `5XFAD_12mo_C_06` and `5XFAD_12mo_H_06` samples, `TgAPP`
  count sums are 1496.69 and 1985.61, and `TgPSEN1` count sums are 205.32 and
  287.09. These are consistent with the `5XFAD` label.
- For the disputed `WildT_06mo_C_11`, `WildT_06mo_C_15`, and
  `WildT_06mo_H_15` samples, `TgAPP` count sums are 16.49, 21.26, and 62.83,
  and `TgPSEN1` count sums are 2.06, 0.97, and 3.58. These are consistent with
  wild-type/background expression relative to the transgenic samples.
- Therefore, both the snRNA genotype labels and the corrected proteomics
  genotype labels are now supported by independent provenance.
- `WildT_06mo_C_11` remains excluded from matched per-animal integration because
  the corresponding cortex proteomics material exists only in a pooled run, not
  as an individual biological-sample column.

The current generated join manifest has:

- 31 `direct` joins supported by normalized sample-label equality.
- 1 `audit_hold_pooled_only` row for `WildT_06mo_C_11`.
- 31 rows with `per_animal_integration_action=use`.
- 1 row with `per_animal_integration_action=exclude_until_resolved`.

## Current Interpretation

The current safe interpretation is:

- Local 5xFAD proteomics is available and analysis-ready for the existing
  kinase attribution workflow.
- The Drive transcriptomics folder is very likely the intended matching 5xFAD
  scRNA/snRNA dataset at the cohort-design level.
- The snRNA sample labels are internally consistent and biologically supported
  by `TgAPP`/`TgPSEN1` expression.
- The local proteomics genotype labels are now grounded in delivered DOCX sample
  lists rather than inferred sample-number layout.
- Per-animal transcriptomics/proteomics integration is currently locked for 31
  direct joins.
- `WildT_06mo_C_11` should remain excluded from matched per-animal integration
  because the cortex proteomics counterpart is pooled-only.

## Recommended Next Steps

1. Create a small 5xFAD omics join manifest with one row per transcriptomics
   sample and columns for transcriptomics sample ID, proposed proteomics
   biological sample ID, tissue, age, transcriptomics genotype, local proteomics
   genotype, join status, integration action, and provenance note. Implemented by
   `alz/ingest/build_5xfad_omics_join_manifest.py`.
2. Audit the delivered proteomics DOCX sample lists and use them as the
   proteomics genotype foundation. Implemented by
   `alz/ingest/audit_5xfad_proteomics_sample_lists.py`; the audit found 0
   individual-sample mismatches after correction.
3. Encode `WildT_06mo_C_11` as `audit_hold_pooled_only` rather than mapping it
   to `cortex_6mo_WT_12`. Implemented in the current manifest.
4. Use `alz/ingest/audit_5xfad_snrna_transgenes.R` as the repeatable snRNA
   genotype audit and retain `outputs/reports/5xfad_snrna_transgene_audit.csv`
   as raw evidence for `TgAPP`/`TgPSEN1` expression.
5. Add a `5xfad-snrna` source to `conf/data_sources.yaml` with the small
   metadata files (`obs_df.csv`, `log00.md`) plus an on-demand `scrna` transfer
   group for the `reclustering/*.RDS` objects. Implemented.
6. Retrieve and inspect `reclustering/fivex_renamed_from_merged.RDS` first. If
   it lacks required assays, metadata, or cluster identities, inspect
   `reclustering/with_cluster_names_merged_object.RDS` next; keep
   `named_lore00.RDS` as older provenance unless it contains unique metadata.
   The newest RDS has been retrieved and inspected successfully.
7. Write an R inspection script that reports each RDS object's class,
   dimensions, assays, reductions, active identities, metadata columns, sample
   IDs, condition counts, and cluster-label columns before selecting a canonical
   object. Implemented by `alz/ingest/inspect_5xfad_snrna_rds.R`.
8. Build a separate 5xFAD Incytr input root, for example
   `data/derived/5xfad_incytr_inputs/`, rather than mixing these inputs into the
   existing Song/sce4 `data/derived/incytr_inputs/` bundle. **Implemented** as a
   per-tissue root (`<tissue>/`), see below.
9. Normalize the selected Seurat object to the Incytr driver contract:
   `Type` for sender/receiver cluster identity and `condition`/`Group` columns
   that match the 5xFAD proteomics condition columns. **Implemented** by
   `alz/incytr_pair/build_5xfad_seurat.R`.
10. Build 5xFAD-specific proteomics/deconvolution inputs from local total,
    IMAC/ST, and pY reports after applying the 31 direct joins and excluding the
    pooled-only row from matched per-animal analysis. KGG and AcK remain
    provenance/support channels. **Implemented** by `fivexfad.py --export-bulk`
    (linear per-group bulk) + `alz/ingest/fivexfad_decompose.py`.
11. Run a smoke Incytr contrast on one tissue/age pair, then document the exact
    canonical RDS, manifest version, excluded row, and input-root path used for
    the run.

## Incytr Pair-Mode Inputs

The 5xFAD Incytr path mirrors the **t-cell cohort** (native-aggexp deconvolution +
live `DEG ∪ prG` gene.use + per-unit input bundles + a dedicated runner), NOT the
AD frozen-provenance path. The AD "aggexp is unrecoverable" caveat does not apply:
the 5xFAD Seurat is on-box, so the transcript-share matrix is regenerated natively
via `AggregateExpression(slot="data")`.

Design (locked):

- **Bundles:** per tissue, `data/derived/5xfad_incytr_inputs/<cortex|hippocampus>/`.
- **Contrasts:** within each tissue, TG vs WT at 3/6/9/12 mo = 4 contrasts/tissue,
  8 total. `condition = <geno>_<age>` (e.g. `TG_3mo`).
- **Cell types:** `new_clusters` with unnamed `^cluster-[0-9]+$` dropped → 31
  (set-equal to the levy_t5 31; the 46-name spine is a name cross-check only, never
  an `in_spine` whitelist).
- **Channels:** `pr` (total) + `ps` (IMAC/ST) + `py` (pY) + `Ack` (acetylation) +
  `KGG` (ubiquitination). The PTM channels are 5xFAD-only and always included.
- **gene.use:** derived live (`SCE4_GENEUSE_DIR` unset). No transgene force-include
  (the driver's hardcoded `App/Psen1/Mapt` list no-ops — 5xFAD rows are
  `TgAPP`/`TgPSEN1`).
- **Species:** mouse; `kldata.csv` symlinks to `data/datasets/song/kinase/kldata_pspy.csv`.

Pipeline (pixi tasks, in order):

| Task | Script | Output |
|:---|:---|:---|
| `5xfad-export-bulk` | `fivexfad.py --export-bulk` | `<tissue>/{pr,ps,py}_bulk_linear.csv` |
| `5xfad-incytr-scrna-extract` | `fivexfad_scrna_extract.R` | `<tissue>/scrna/{aggexp_data,cell_counts}.csv` |
| `5xfad-incytr-decompose` | `fivexfad_decompose.py` | `<tissue>/{pr,ps,py,ack,kgg}_deconvoluted.csv` |
| `5xfad-build-incytr-seurat` | `build_5xfad_seurat.R` | `<tissue>/incytr_obj.rds` |
| `5xfad-build-incytr-gene-list` | `build_5xfad_input_gene_list.R` | `<tissue>/allmarkers.csv` |
| `5xfad-incytr` | `run_pair_mode_5xfad.sh` | `outputs/reports/incytr_pair_mode_5xfad/<tissue>/wide/` |

Provenance for the run: canonical RDS
`data/datasets/5xFAD/primary/scrna/reclustering/fivex_renamed_from_merged.RDS`;
join manifest `data/datasets/5xFAD/metadata/omics_join_manifest.csv` (31 `use` +
1 excluded `WildT_06mo_C_11`, pooled-only); deconvolution mass-identity max
`|rel err|` ≈ 1e-15 (machine precision) on every condition. snRNA caveat: with the
pooled-only exclusion, **cortex 6mo WT** has a single snRNA animal
(`WildT_06mo_C_15`) — its per-cell-type share rests on one sample (the proteomics
bulk for that group still uses both individual WT samples).
