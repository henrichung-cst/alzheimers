# alz/cohorts/song/

Cohort namespace for the Song mouse cohort (72 animals, 6 plexes).

## Modules

Song cohort code remains in its legacy location pending a later sub-phase:

- `alz/ingest/song.py` — TMT mapping, phospho-protein matching, PCA QC, outlier
  detection. Primary ingest entry point for the 72-animal mouse cohort.
- `alz/decomposition_mea/` — per-cluster decomposition and cell-type MEA chain
  (frozen Incytr invariants; heavier R + scRNA call graph).

The decision to defer Song was made at the Phase 3 / Phase 4 boundary gate: the
Song ingest and decomposition chain have tighter coupling to frozen Incytr
invariants recorded in `CLAUDE.md` and `docs/audits/cohort_abstraction_refactor/`,
and their move constitutes a separate sub-phase with its own parity verification.
