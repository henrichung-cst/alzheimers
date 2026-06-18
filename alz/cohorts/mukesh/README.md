# alz/cohorts/mukesh/

Cohort namespace for the Mukesh / NBB human AD cohort.

## Modules

| Module | Moved from | CLI entry | Notes |
|--------|------------|-----------|-------|
| `ingest.py` | `alz/ingest/mukesh.py` | `python alz/cohorts/mukesh/ingest.py --reshape` | UniProt canonical-isoform cache, diagnostic pass, reshape of NBB donor × site tables into Song-shaped artifacts. |
| `mea.py` | `alz/ingest/mukesh_perdonor.py` | `python alz/cohorts/mukesh/mea.py --track both` | Per-donor MEA on stoichiometry and raw-phospho tracks; writes kinase × donor NES/FDR matrices and recurrence summary. |
