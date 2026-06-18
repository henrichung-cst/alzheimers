# alz/ingest/

Layer 1 of the two-layer architecture: bespoke, per-collaborator data
ingest. Each module knows one dataset's quirks (channel layouts, sample
keys, file naming) and emits canonical artifacts the shared analysis
pipelines consume without further translation.

The Song cohort ingest lives here. Mukesh, T-cell, and 5xFAD cohort modules
have moved to `alz/cohorts/<cohort>/` (Phase 4 refactor, 2026-06-17).

| File | Cohort | CLI entry | Notes |
|------|--------|-----------|-------|
| `song.py` | Song mouse (72 animals, 6 plexes) | `python alz/ingest/song.py --run` | TMT mapping, phospho-protein matching, PCA QC, outlier detection. Was `alz/data_ingest.py`. |
| `lucie.py` | Lucie 5xFAD | `python alz/ingest/lucie.py` | Proteomics manifest builder. |

Moved cohort modules (now under `alz/cohorts/`):

| New path | Cohort | Old path |
|----------|--------|----------|
| `alz/cohorts/mukesh/ingest.py` | Mukesh / NBB human | `alz/ingest/mukesh.py` |
| `alz/cohorts/mukesh/mea.py` | Mukesh / NBB human | `alz/ingest/mukesh_perdonor.py` |
| `alz/cohorts/tcells/ingest.py` | T-cell exhaustion | `alz/ingest/tcells.py` |
| `alz/cohorts/tcells/mea.py` | T-cell exhaustion | `alz/ingest/tcells_perdonor.py` |
| `alz/cohorts/fivexfad/ingest.py` | 5xFAD mouse | `alz/ingest/fivexfad.py` |
| `alz/cohorts/fivexfad/celltype_mea.py` | 5xFAD mouse | `alz/ingest/fivexfad_celltype_mea.py` |

Downstream consumers (shared analysis):
- Mode 1 — `alz/bulk_mea/normalize.py` reads Song ingest outputs.
- Mode 4 — `alz/cross_reference/seaad_human_agreement.py` reads Mukesh
  per-donor outputs (via `alz/cohorts/mukesh/ingest.py`).
