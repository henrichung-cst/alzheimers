# Phase 4 Decisions — Cohort Namespace Creation and Module Moves

**Date:** 2026-06-17
**Branch:** `refactor/cohort-namespaces`
**Base commit:** `53531cc`
**Implementer:** Claude (Sonnet 4.6)

---

## Packets 4A–4D: What was done

### Packet 4A — Namespace skeleton

Created:
- `alz/cohorts/__init__.py` (empty)
- `alz/cohorts/mukesh/__init__.py`, `__init__.py`s for tcells, fivexfad, song (all empty)
- `README.md` in each of `mukesh/`, `tcells/`, `fivexfad/`, `song/` documenting modules present and (for song/) the deferred-move rationale

### Packets 4B–4D — Module moves (via `git mv`)

| Old path | New path | New import |
|----------|----------|------------|
| `alz/ingest/mukesh.py` | `alz/cohorts/mukesh/ingest.py` | `alz.cohorts.mukesh.ingest` |
| `alz/ingest/mukesh_perdonor.py` | `alz/cohorts/mukesh/mea.py` | `alz.cohorts.mukesh.mea` |
| `alz/ingest/tcells.py` | `alz/cohorts/tcells/ingest.py` | `alz.cohorts.tcells.ingest` |
| `alz/ingest/tcells_perdonor.py` | `alz/cohorts/tcells/mea.py` | `alz.cohorts.tcells.mea` |
| `alz/ingest/fivexfad.py` | `alz/cohorts/fivexfad/ingest.py` | `alz.cohorts.fivexfad.ingest` |
| `alz/ingest/fivexfad_celltype_mea.py` | `alz/cohorts/fivexfad/celltype_mea.py` | `alz.cohorts.fivexfad.celltype_mea` |

---

## No-wrapper decision

**Anti-shim rule applies.** This is a solo-dev repository with all consumers in-repo. No compatibility wrappers, stubs, or re-export shims were created at the old paths. The global anti-shim rule (CLAUDE.md) overrides any "migration window" convention from the control pack: the old import paths are wrong now, not "one of two valid options." Every consumer was updated in the same pass.

---

## Modules intentionally left in `alz/ingest/` (deferred, NOT forgotten)

The following stay at their current locations, with rationale:

| Path | Reason deferred |
|------|-----------------|
| `alz/ingest/song.py` | Tight coupling to frozen Incytr invariants (`CLAUDE.md` §Pair-mode Incytr correctness invariants); move constitutes a separate sub-phase with its own parity verification. |
| `alz/ingest/lucie.py` | Proteomics manifest builder for Lucie / 5xFAD integration; no active consumers; low priority. |
| `alz/ingest/tcells_decompose.py` | Part of the R/scRNA decomposition layer (calls `tcells_scrna_extract.R`); non-Python call graph, deferred with fivexfad_decompose. |
| `alz/ingest/tcells_projectils_map.R` | R script; out of scope for Python namespace refactor. |
| `alz/ingest/tcells_scrna_extract.R` | R script; out of scope. |
| `alz/ingest/fivexfad_decompose.py` | Part of R/scRNA decomposition layer; coupled to frozen Incytr inputs. |
| `alz/ingest/fivexfad_scrna_extract.R` | R script; out of scope. |
| `alz/ingest/audit_5xfad_proteomics_sample_lists.py` | Audit/diagnostic; stays in ingest alongside its input data; import updated to new path. |
| `alz/ingest/build_5xfad_omics_join_manifest.py` | Manifest builder; no cohort namespace yet for this utility. |
| `alz/ingest/inspect_5xfad_snrna_rds.R` | R script; out of scope. |
| `alz/ingest/test_fivexfad.py` | Test file; stays in ingest alongside the audit chain; import updated to new path. |
| `alz/decomposition_mea/` | Frozen Song decomposition chain with hard parity invariants; separate sub-phase. |

---

## Consumers updated

### Python imports
- `alz/cohorts/mukesh/mea.py` — self-import: `alz.ingest.mukesh` → `alz.cohorts.mukesh.ingest`
- `alz/cohorts/tcells/mea.py` — self-import: `alz.ingest.tcells` → `alz.cohorts.tcells.ingest`
- `alz/cohorts/fivexfad/celltype_mea.py` — self-import: `from alz.ingest import fivexfad` → `from alz.cohorts.fivexfad import ingest as fivexfad`
- `alz/core/fivexfad_bulk_mea_adapter.py`
- `alz/core/fivexfad_celltype_mea_adapter.py`
- `alz/core/mukesh_mea_adapter.py`
- `alz/core/phase3_parity_harness.py`
- `alz/core/tcells_mea_adapter.py`
- `alz/cross_reference/seaad_human_agreement.py`
- `alz/ctrl_outlier_audit/concordance_overlap_AD_excl_01_03.py`
- `alz/ingest/audit_5xfad_proteomics_sample_lists.py`
- `alz/ingest/test_fivexfad.py` (rebind via `from alz.cohorts.fivexfad import ingest as fivexfad` etc.)
- `alz/ingest/fivexfad_decompose.py`
- `alz/ingest/tcells_decompose.py`

### pixi.toml tasks
- `5xfad-ingest`, `5xfad-mea`, `5xfad-celltype-mea`, `5xfad-celltype-mea-smoke`, `5xfad-export-bulk`
- `tcells-reshape`, `tcells-perdonor`, `tcells-export-bulk`
- `human-ingest`, `human-perdonor`

### Shell scripts
- `alz/runners/main/run_all.sh` (H-ingest, H-perdonor steps)
- `alz/runners/main/run_mukesh_perdonor.sh` (python invocation + echo)
- `alz/runners/main/run_pair_mode_pipeline.sh` (F1, F2 steps)

### Docs / comments
- `alz/ingest/README.md` — full rewrite to record the moves; old paths listed as provenance in "Old path" column
- `alz/bulk_mea/README.md` — `mukesh_perdonor.py` reference updated
- `docs/INDEX.md` — mukesh_ingest_policies.yml note + Human cohort pipeline table
- `docs/foundation/cohort_contract.md` — Mode 1 entry updated
- `docs/foundation/mukesh_ingest_policies.yml` — header comment
- `docs/integrations/5xfad-kinase-mea-viewer.md` — opening reference
- `alz/viewer/template/js/tabs/kinase_human.js` — three user-facing "Re-run python …" strings
- `conf/human_nbb/parameters.yml` — comment
- `README.md` — ingest descriptions (§3 + code block + dir tree comment)
- `alz/core/mukesh_mea_adapter.py` — docstring path
- `alz/core/tcells_mea_adapter.py` — docstring path
- `alz/build_tcell_viewer.py` — two inline comments naming moved modules
- `alz/cross_reference/seaad_human_agreement.py` — error message string

---

## End-check results

- `py_compile` on all 17 moved/edited Python files: PASSED
- `import alz.cohorts.mukesh.ingest, alz.cohorts.mukesh.mea, alz.cohorts.tcells.ingest, alz.cohorts.tcells.mea, alz.cohorts.fivexfad.ingest, alz.cohorts.fivexfad.celltype_mea`: ALL OK
- Old files absent: `ls alz/ingest/{mukesh,mukesh_perdonor,tcells,tcells_perdonor,fivexfad,fivexfad_celltype_mea}.py` → all "No such file"
- Stale-reference scan (excluding provenance-record README "Old path" / "Moved from" columns, excluding `docs/{plans,audits}/cohort_abstraction_refactor/`): **EXIT 1 (no matches)**

---

## Packet 4E — Song assessment

**Decision: Song does NOT move in Phase 4.** Recorded explicitly per the control
pack's 4E requirement.

Song is not symmetric with the other three cohorts, so the scaffold's
"`<cohort>/ingest.py` + `<cohort>/mea.py`" shape does not apply cleanly:

| Song module | Disposition | Reason |
|---|---|---|
| `alz/bulk_mea/enrich.py` (`_run_mea`, Song bulk MEA at `:455`) | **STAYS — not a Song file.** | This is the *shared* MEA engine every cohort calls (and the FROZEN `_run_mea`). It is infrastructure, not cohort-specific; it belongs in `alz/bulk_mea/`, never under `alz/cohorts/song/`. Moving it is out of scope for the entire refactor. |
| `alz/bulk_mea/normalize.py` | STAYS | Shared 72-sample normalization (all cohorts / always-on); not Song-specific. |
| `alz/ingest/song.py` | DEFERRED | Song ingest. Coupled to the frozen pair-mode Incytr input chain (`CLAUDE.md` §Pair-mode Incytr correctness invariants). A move is a separate sub-phase with its own parity gate. |
| `alz/decomposition_mea/` (`enrich_celltype.py`, `build_celltype_decomposition.py`, `build_per_animal_site_ols.py`, `verify_decomposition.py`) | DEFERRED | The Song decomposition + verification chain carries hard mass-identity / spine-coverage invariants and the levy_t5 forward-projection contract. Path-only relocation is low-risk in principle but must be gated separately, not ridden along on the proteomics-ingest move. |

**Net:** the only genuinely Song-specific *ingest/decomposition* code (`song.py`,
`alz/decomposition_mea/`) is deferred to a later sub-phase; the shared MEA engine
and normalization are permanent residents of `alz/bulk_mea/` and were never move
candidates. An empty `alz/cohorts/song/` package exists with a README pointing at
the legacy locations so the namespace is uniform across the four cohorts. The 3F
feasibility report (`docs/plans/cohort_abstraction_refactor/phase_3F_song_feasibility.md`)
already concluded Song's *MEA callers* fold cleanly onto the runner in a future
phase; this 4E decision concerns only *file relocation* and reaches the same
"defer Song, keep shared infra put" conclusion.

---

## Post-verification touch-up

`alz/ingest/fivexfad_decompose.py:11` carried a stale bare-name docstring
reference (`` `fivexfad.py --export-bulk` ``) flagged by the verifier as a
non-blocking nit. Corrected by the orchestrator to
`` `alz/cohorts/fivexfad/ingest.py --export-bulk` `` (docstring only; no logic
change).

---

## Verification (independent verifier ≠ implementer)

`audit-pipeline` agent ran the full 11-check adversarial checklist. **Verdict:
PASS** (11/11). Highlights: old paths absent; no stub/shim re-exports; `py_compile`
+ import of all 6 new module paths and all `alz.core` adapters clean; zero stale
references outside provenance README columns; all pixi tasks resolve to existing
files; **Phase-1 validators 0 FAIL across all four cohorts** (song 94/0/4, mukesh
106/0/1, tcells 106/0/1, fivexfad 166/0/5); no `outputs/` drift; frozen
`enrich.py` / `mea_runner.py` untouched in `git diff`.
