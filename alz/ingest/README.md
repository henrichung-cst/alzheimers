# alz/ingest/

Layer 1 of the two-layer architecture: bespoke, per-collaborator data
ingest. Each module knows one dataset's quirks (channel layouts, sample
keys, file naming) and emits canonical artifacts the shared analysis
pipelines consume without further translation.

| File | Cohort | CLI entry | Notes |
|------|--------|-----------|-------|
| `song.py` | Song mouse (72 animals, 6 plexes) | `python alz/ingest/song.py --run` | TMT mapping, phospho-protein matching, PCA QC, outlier detection. Was `alz/data_ingest.py`. |
| `mukesh.py` | Mukesh / NBB human | `python alz/ingest/mukesh.py --reshape` | Reshape Mukesh CSVs into normalized phospho + stoichiometry. |
| `mukesh_perdonor.py` | Mukesh / NBB human | `python alz/ingest/mukesh_perdonor.py --track both` | Per-donor MEA on stoichiometry + raw-phospho tracks. |
| `lucie.py` | Lucie 5xFAD | `python alz/ingest/lucie.py` | Proteomics manifest builder. |

Downstream consumers (shared analysis):
- Mode 1 — `alz/bulk_mea/normalize.py` reads Song ingest outputs.
- Mode 4 — `alz/cross_reference/seaad_human_agreement.py` reads Mukesh
  per-donor outputs.
