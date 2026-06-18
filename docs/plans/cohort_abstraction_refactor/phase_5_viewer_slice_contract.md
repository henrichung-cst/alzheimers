# Phase 5 — Viewer Slice Contract

## Goal

Reduce viewer coupling to cohort-specific file layouts by defining a shared
viewer slice contract. Cohort adapters should emit common slice structures, and
viewer builders should compose slices.

## Non-Goals

- Do not redesign the frontend.
- Do not change viewer payload semantics without a versioned contract change.
- Do not remove existing viewer fields until consumers are migrated.
- Do not combine payload schema changes with visual redesign.

## Prerequisites

- Phase 3 shared runner stable.
- Phase 4 source layout either complete or explicitly deferred.
- Existing payload validators and template checks pass.

## Implementation Scaffold

Suggested module (corrected to the implemented singular viewer namespace):

```text
alz/viewer/shared/cohort_slice.py
```

Shared slice concepts:

```text
cohort_id
context_id
kinase_index
contrast_index
mea_rows
attribution_rows
audit_tables
edge_slice_indexes
capabilities
provenance
```

Each cohort adapter should expose:

```python
build_viewer_slice(cohort_id, output_root) -> CohortViewerSlice
```

The unified viewer should become a composer:

```python
payload = compose_viewer_slices([
    load_song_slice(),
    load_mukesh_slice(),
    load_fivexfad_slice(),
])
```

T-cell can remain a separate HTML deliverable, but its backend adapter should
share the slice machinery when possible.

## Agent Work Packets

### Packet 5A — Payload Field Inventory

Inventory current unified and T-cell payload fields, including lazy shard
indexes and frontend consumers.

### Packet 5B — Slice Schema Draft

Define `CohortViewerSlice` and serialize it to JSON-compatible dictionaries.

### Packet 5C — Mukesh Slice Adapter

Implement one small adapter first. Mukesh is a good candidate because it is
mostly human per-donor MEA and SEA-AD agreement.

### Packet 5D — T-cell Slice Adapter

Implement T-cell slice adapter while preserving current dedicated viewer output.

### Packet 5E — 5xFAD Slice Adapter

Extract current 5xFAD payload construction from the unified viewer into a
cohort adapter.

### Packet 5F — Song Slice Feasibility

Document which Song payload pieces are core and which are presentation-specific
before moving Song payload logic.

## Required Checks

- Existing payload validators pass.
- Template verifier passes.
- Payload JSON key sets match unless a versioned change is approved.
- Lazy shard indexes match.
- Frontend source references remain resolvable.
- File size changes are explained.

## Exit Criteria

- At least one cohort slice adapter is used by the viewer builder without
  payload drift.
- Payload contract documentation updated.
- Any non-migrated cohort has a logged reason and migration plan.

## Rollback Criteria

Viewer builder can return to old direct file-loading path. Keep old path until
slice-composed payload parity is accepted.

## Decision Log Instructions

Log decisions for:

- slice field names,
- payload versioning,
- backward compatibility strategy,
- lazy shard ownership,
- cohort capability flags,
- any frontend-facing behavior change.

Decision log path:

```text
docs/audits/cohort_abstraction_refactor/phase_5_decisions.md
```
