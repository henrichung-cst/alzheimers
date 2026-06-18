# Phase 4 — Cohort Directory Migration

## Goal

Move cohort-specific modules into explicit cohort namespaces after shared
runners are stable.

Target layout:

```text
alz/cohorts/
  song/
  mukesh/
  tcells/
  fivexfad/
```

This phase is about source organization and ownership boundaries. It should not
change behavior.

## Non-Goals

- Do not change analysis logic.
- Do not change output schemas.
- Do not change canonical output roots.
- Do not remove compatibility wrappers until downstream references are updated
  and verified.
- Do not combine this move with shared-runner behavior changes.

## Prerequisites

- Phase 3 shared runner is stable for non-Song cohorts.
- Song migration feasibility report accepted.
- Import/path scan patterns agreed.

## Implementation Scaffold

Suggested moves:

```text
alz/ingest/mukesh.py              -> alz/cohorts/mukesh/ingest.py
alz/ingest/mukesh_perdonor.py     -> alz/cohorts/mukesh/mea.py
alz/ingest/tcells.py              -> alz/cohorts/tcells/ingest.py
alz/ingest/tcells_perdonor.py     -> alz/cohorts/tcells/mea.py
alz/ingest/fivexfad.py            -> alz/cohorts/fivexfad/ingest.py
alz/ingest/fivexfad_celltype_mea.py -> alz/cohorts/fivexfad/celltype_mea.py
```

Exact names should be decided during the phase. Keep wrappers at old paths
initially.

## Agent Work Packets

### Packet 4A — Namespace Skeleton

Create `alz/cohorts/` package and empty cohort packages with README files.

### Packet 4B — Mukesh Move

Move Mukesh-specific modules and add compatibility wrappers.

### Packet 4C — T-cell Move

Move T-cell-specific modules and add compatibility wrappers.

### Packet 4D — 5xFAD Move

Move 5xFAD-specific modules and add compatibility wrappers.

### Packet 4E — Song Assessment

Decide whether Song should move in this phase or remain in legacy location until
later. Log the decision.

### Packet 4F — Runner and Docs Update

Update runner scripts, pixi tasks, and docs. Existing commands should still work
or fail with clear migration messages.

## Required Checks

- `python -m py_compile` for moved Python modules and wrappers.
- Old import paths still import where compatibility is promised.
- New import paths import.
- Stale direct-import scan passes.
- Phase 1 validators still pass.
- No protected output files change.

## Exit Criteria

- Cohort namespaces exist.
- Compatibility wrappers are present and documented.
- Import scans are clean.
- No behavior drift.

## Rollback Criteria

Revert the move commit. Because this phase should not change behavior or
outputs, rollback should be a normal source revert.

## Decision Log Instructions

Log decisions for:

- final module names,
- compatibility-wrapper duration,
- any old command retired,
- any module intentionally left in legacy location,
- ownership boundaries between `alz/cohorts`, `alz/core`, and `alz/viewers`.

Decision log path:

```text
docs/audits/cohort_abstraction_refactor/phase_4_decisions.md
```
