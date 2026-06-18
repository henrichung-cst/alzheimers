# alz/cohorts/fivexfad/

Cohort namespace for the 5xFAD mouse AD cohort.

## Modules

| Module | Moved from | CLI entry | Notes |
|--------|------------|-----------|-------|
| `ingest.py` | `alz/ingest/fivexfad.py` | `python alz/cohorts/fivexfad/ingest.py --ingest` | Manifest build, reshape, bulk MEA (TG vs WT contrasts across cortex and hippocampus); bulk linear export for Incytr decomposition inputs. |
| `celltype_mea.py` | `alz/ingest/fivexfad_celltype_mea.py` | `pixi run 5xfad-celltype-mea` | Per-(tissue, track, cell-type) MEA using deconvoluted pseudobulk inputs from the snRNA decomposition step. |

## Modules remaining in alz/ingest/

The following 5xFAD-related scripts remain in `alz/ingest/` pending a later sub-phase
that covers the R/scRNA decomposition layer:

- `fivexfad_decompose.py` — provenance deconvolution for Incytr inputs
- `fivexfad_scrna_extract.R` — memory-safe scRNA aggexp + cell-count extraction
- `audit_5xfad_proteomics_sample_lists.py`, `build_5xfad_omics_join_manifest.py`,
  `inspect_5xfad_snrna_rds.R`, `test_fivexfad.py` — audit, manifest, and test scripts
  with heavier non-Python call graphs or explicit dependency on the ingest layer location
