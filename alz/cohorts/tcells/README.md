# alz/cohorts/tcells/

Cohort namespace for the T-cell exhaustion cohort.

## Modules

| Module | Moved from | CLI entry | Notes |
|--------|------------|-----------|-------|
| `ingest.py` | `alz/ingest/tcells.py` | `python -m alz.cohorts.tcells.ingest --reshape` | Reshape T-cell DIA exports into stoichiometry and raw-phospho matrices; bulk export for Incytr inputs. |
| `mea.py` | `alz/ingest/tcells_perdonor.py` | `python -m alz.cohorts.tcells.mea --donor both` | Per-donor time-course MEA (donor1: st + py; donor2: all tracks skip by design — no IMAC, no flanking motif). Writes additive `mechanism_attribution{_pY}.csv` files with stoich/raw classification. |
