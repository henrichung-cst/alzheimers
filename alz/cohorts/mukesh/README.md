# alz/cohorts/mukesh/

Cohort namespace for the Mukesh / NBB human AD cohort.

## Modules

| Module | Moved from | CLI entry | Notes |
|--------|------------|-----------|-------|
| `ingest.py` | `alz/ingest/mukesh.py` | `python -m alz.cohorts.mukesh.ingest --reshape` | UniProt canonical-isoform cache, diagnostic pass, reshape of NBB donor x site tables into Song-shaped artifacts. |
| `mea.py` | `alz/ingest/mukesh_perdonor.py` | `python -m alz.cohorts.mukesh.mea --track both` | Per-donor MEA on stoichiometry and raw-phospho tracks; writes kinase x donor NES/FDR matrices and recurrence summary. |

`mea.py` also emits additive mechanism-attribution outputs:
- `mechanism_attribution{suffix}.csv` (additive; canonical outputs unchanged), generated after both stoichiometry and raw per-donor runs for a track.
- Suffix is `""` for ST and `"_pY"` for pY.

Canonical add-on command:
- `python -m alz.cohorts.mukesh.mea --track both`
