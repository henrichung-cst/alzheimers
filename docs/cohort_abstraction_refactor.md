# Cohort Abstraction Refactor — Phase History and Carryover

Source: `project_cohort_abstraction_refactor.md`

Control pack: `docs/plans/cohort_abstraction_refactor/` (README + agent_protocol + phase_0..5). Branch: `refactor/cohort-namespaces`. **MERGED TO MAIN 2026-06-18** (tip `ad23c86`; 15 commits, linear ff history).

## Architecture after the refactor

- `alz/cohorts/{mukesh,tcells,fivexfad}/{ingest,mea,celltype_mea}.py` (moved from flat `alz/ingest/`, `alz/bulk_mea/` — import-line-only changes).
- `alz/viewer/cohorts/{song,mukesh,fivexfad}.py` — per-cohort `CohortViewerSlice` emitters.
- `alz/viewer/shared/{cohort_slice,compose}.py` — slice contract + reducer.
- `alz/core/mea_outputs.py` — shared `build_nes_fdr_matrices`, `build_recurrence_summary`, `KIND_SPEC`, `mea_output_path`.
- `alz/core/mea_runner.py` — shared MEA runner wrapping the frozen `_run_mea`.
- `build_payload` builds 3 slices + song meta and composes; `meta`-building stays in `build_payload` (its `ensure_*_trace_sources`/`build_audit_manifest` deps are shared → moving would cycle).

## Phase summary (all CLOSED + committed)

| Phase | Description | Status |
|---|---|---|
| 0 | Read-only inventory (`alz/core/baseline_inventory.py`); tightened protected surface 215→165 | CLOSED 2026-06-17 |
| 1 | Read-only validators (`alz/core/{phospho_schema,cohort_manifest,validation,validate_cohort}.py`), 0 FAIL × 4 cohorts | CLOSED 2026-06-17 |
| 2 | Deduplication of char-identical recurrence/pivot/write blocks → `alz/core/mea_outputs.py`; inline blocks deleted (90 deletions) | CLOSED 2026-06-17 |
| 3 | Shared MEA runner (`alz/core/mea_runner.py`); two fold patterns: table-writing shell + `mea_caller` injection; waves 3A–3D + 3F feasibility doc | CLOSED 2026-06-17 |
| 4 | Cohort namespace migration (6 scaffold Python modules `git mv` + import-line updates); no wrappers (anti-shim) | CLOSED + committed |
| 5 | Viewer slice extraction: 5A–5F, build_unified_viewer monolith decomposed into per-cohort adapters + composer | CLOSED |

**Parity standard used throughout:** structural exact + numerics `isclose(rtol=1e-6, atol=1e-9)`, NaN-positions exact. Verifier ≠ implementer for all waves.

## Design corrections discovered during the refactor

**5E R2 correction:** the 5B design stated "fivexfad confidence derives from SONG's bulk MEA; adapter takes song MEA as an arg; composer orders song-before-fivexfad." This was a documentation error. `_assign_fivexfad_song_aligned_confidence` consumes 5xFAD's OWN bulk rows (`build_unified_viewer.py:2410`); "song-aligned" = shared semantics, not shared data. The slice's only cross-cohort input is `data.celltype_evidence`. No cross-cohort data seam, no composer ordering constraint.

**3D side-finding:** the on-disk 5xFAD celltype canonical at the time of Phase 3 (`2026-06-15 18:08`) was stale and didn't reproduce from the current pseudobulk. Human chose regenerate; done 2026-06-17 via `outputs/reports/refactor_audit/phase_3/regen_5xfad_celltype.sh`. New dims reflect the current smaller pseudobulk (substrate 21.3M rows, was 24.9M).

## Uncommitted perf change (intentionally left in working tree)

`alz/bulk_mea/enrich.py` + `alz/shared/config.py` carry an LRU percentile cache in `_run_mea` plus `threads=config.MEA_THREADS` (default 8) on the GSEApy prerank. `seed`/`permutation_num` untouched, but `threads=` is a determinism risk on seeded prerank. Do NOT commit without a frozen-output fingerprint (cache+threads on vs off → identical).

## Carryover (not part of the refactor phases, awaiting explicit go)

1. **Producer-side never-read sidecar cleanup:** stop producing `_raw`/`_all`/`_per_cluster` audit sidecars at writers (`mukesh_perdonor.py`, `tcells_perdonor.py`, `enrich_celltype.py`, `mukesh.py _concat`) and delete on-disk copies. Deferred from Phase 2/3.
2. **Deferred cohort-file sub-phase:** R extractors, `*_decompose.py`, `build_5xfad_*`, Song `song.py` + `alz/decomposition_mea/`, `lucie.py` namespace moves. Deferred from Phase 4 to keep the frozen Incytr/decomposition layer out of a path-only move.

## Key audit artifacts

- `docs/audits/cohort_abstraction_refactor/phase_0_protected_surface_audit.md`
- `docs/audits/cohort_abstraction_refactor/phase_3_decisions.md`
- `docs/plans/cohort_abstraction_refactor/phase_3F_song_feasibility.md`
- `docs/plans/cohort_abstraction_refactor/phase_4_decisions.md`
- `outputs/reports/refactor_audit/phase_{2,3,4}/` (monitoring + parity reports)
