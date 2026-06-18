# alz/cohorts/tcells/

Cohort namespace for the T-cell exhaustion cohort.

## Modules

| Module | Moved from | CLI entry | Notes |
|--------|------------|-----------|-------|
| `ingest.py` | `alz/ingest/tcells.py` | `python alz/cohorts/tcells/ingest.py --reshape` | Reshape T-cell DIA exports into stoichiometry and raw-phospho matrices; bulk export for Incytr inputs. |
| `mea.py` | `alz/ingest/tcells_perdonor.py` | `python alz/cohorts/tcells/mea.py --donor both` | Per-donor time-course MEA (donor1: st + py; donor2: all tracks skip by design — no IMAC, no flanking motif). |
