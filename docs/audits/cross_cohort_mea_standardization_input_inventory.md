# Cross-Cohort MEA Input Inventory — Packet 0C

Date: 2026-06-18

| Cohort | Path pattern | Present locally | Expected producer | Safe to run now | Notes |
| --- | --- | --- | --- | --- | --- |
| Song bulk stoich/raw MEA | `outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv`, `outputs/reports/kinase_attribution/mea_raw_phospho{,_pY}.csv` | yes | `alz/bulk_mea/mechanism.py` | yes | Both ST and pY stoichiometry/raw MEA long tables are present locally under the Song output root. |
| Mukesh stoich/raw MEA | `outputs/reports/kinase_attribution_human/perdonor/mea_perdonor{,_raw}{,_pY}.csv` | yes | `alz/cohorts/mukesh/mea.py` | yes | Per-donor long MEA outputs exist for both stoich and raw tracks (with `_pY` variants). |
| T-cell bulk stoich/raw MEA | `outputs/reports/kinase_attribution_tcells/donor{1,2}/mea/mea_timecourse{,_raw}{,_pY}.csv` | no (donor2 missing) | `alz/cohorts/tcells/mea.py` | no | T-cell uses `mea_timecourse*` rather than `mea_stoichiometry*`; donor1 has all four base raw/stoich tables, donor2 has only `mea_manifest.json` currently. |
| 5xFAD bulk stoich/raw MEA | `outputs/reports/kinase_attribution_5xfad/{cortex,hippocampus}_{st,py}_{mea_stoichiometry,mea_raw_phospho}.csv` | yes | `alz/cohorts/fivexfad/ingest.py` | yes | All eight expected tissue×track long-table pairs are present with `_st`/`_py` naming. |
| T-cell projected state inputs | `data/derived/tcells_incytr_inputs/{donor1,donor2}/*deconvoluted.csv` | yes (donor1 yes, donor2 partial) | `alz/incytr_pair/export_decomposition_for_pair.py` / `alz/incytr_pair/run_pair_mode_tcells.sh` | no (donor2 cannot support ST) | First T-cell projected-state target is donor1 ST. Donor1 has `pr/ps/py_deconvoluted.csv`; donor2 has `pr/py_deconvoluted.csv` only (no ps IMAC layer). |
