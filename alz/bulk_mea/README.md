# `alz/bulk_mea/` — Mode 1: Bulk MEA

Stoichiometry-corrected kinase attribution for bulk phosphoproteomics on the
Song mouse cohort (and, via the human chain in `alz/ingest/mukesh_perdonor.py`,
the NBB human cohort).

## Stage order

| Stage | Script | Role |
|------:|--------|------|
| 1 | `normalize.py` | IRS cross-plex normalization (all 72 samples) + stoichiometry `log2(phospho) − log2(protein)` |
| 2 | `enrich.py` | Sample filter + factorial OLS (9 contrasts) + MEA on stoichiometry β |
| 3 | `attribute.py` | Unified cell-type attribution (SEA-AD + WMB + Song concordance) |
| 4 | `recover.py` | Cross-contrast consistency + final hypothesis tables (primary deliverable: `kinase_hypothesis_table.csv`) |
|  ⸻ | `mechanism.py` | Optional: raw-phospho MEA + abundance/activity/both classification (reuses Stage 2 helpers) |
|  ⸻ | `summary.py` | Read-only: prints cached results across all stages |

Run via pixi tasks (`pixi run normalize / enrich / attribute / mechanism /
recover`), or the bundled `pixi run live` to chain ingest → recover.

## Inputs

- `outputs/reports/data_ingest/sample_mapping.csv` (from
  `alz/ingest/song.py --mapping`)
- `outputs/reports/data_ingest/sample_exclusions.csv` (from
  `alz/ingest/song.py --outliers`)
- Raw TMT total proteome + phospho IMAC/pY workbooks under
  `data/datasets/song/primary/` (read by `normalize.py`)
- WMB per-class expression matrix at
  `outputs/reports/wmb_expression/wmb_kinase_expression.csv` (built by
  `alz/reference/wmb_expression.py`; required for `attribute.py`)
- SEA-AD effect-size h5ads under `data/external/sea_ad/`
  (required for `attribute.py`)

## Outputs

All under `outputs/reports/kinase_attribution/` (stages 1–3) and
`outputs/reports/attribution_recovery/` (stage 4). See `docs/foundation/
live_pipeline_contract.md` for the per-file schema contract.

## Sample filtering

`analysis_mode` lives in `conf/base/parameters.yml` (default `males_only`).
Set `KEDRO_ENV=full_cohort` to overlay `conf/full_cohort/parameters.yml` and
run the both-sexes sensitivity analysis. `normalize.py` always uses all 72
samples; filtering applies starting at `enrich.py`.
