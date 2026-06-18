# Phase 2 — Shared Output and Recurrence Helpers

## Goal

Extract repeated MEA output-writing and recurrence-summary code into shared
helpers while preserving each cohort's current contrast logic.

Start with Mukesh/NBB and T-cell because they already follow a similar pattern:

- load stoichiometry/raw matrices,
- build cohort-specific LFC vectors,
- call shared MEA,
- write MEA tables,
- write NES/FDR matrices,
- write recurrence summaries,
- write manifests.

## Non-Goals

- Do not change contrast construction.
- Do not change MEA parameters.
- Do not change output file names in canonical paths.
- Do not switch canonical outputs to the new helper until scratch-output parity
  passes.
- Do not touch Song in this phase.

## Prerequisites

- Phase 0 baseline locked.
- Phase 1 validators available.
- Mukesh and T-cell protected outputs identified.

## Implementation Scaffold

Suggested module:

```text
alz/core/mea_outputs.py
```

Suggested functions:

```python
write_mea_audit_tables(...)
build_nes_fdr_matrices(...)
build_recurrence_summary(...)
write_mea_result_bundle(...)
```

The functions should be pure where possible and accept DataFrames rather than
reading cohort-specific paths internally.

Scratch output roots:

```text
outputs/reports/refactor_audit/phase_2/mukesh_new/
outputs/reports/refactor_audit/phase_2/tcells_new/
```

## Agent Work Packets

### Packet 2A — Recurrence Helper

Extract recurrence logic into a shared function. Preserve current definitions:

- significant = `FDR < config.MEA_FDR_THRESH`,
- up/down determined by NES sign,
- tested count from non-null FDR/NES availability,
- median NES behavior unchanged.

### Packet 2B — MEA Bundle Writer

Extract writing of:

- MEA long table,
- global shifts,
- winsorized sites,
- substrate sets,
- NES/FDR matrices.

### Packet 2C — Mukesh Scratch Adapter

Add an opt-in scratch path for Mukesh using the shared helpers. Do not replace
canonical output writing yet.

### Packet 2D — T-cell Scratch Adapter

Add an opt-in scratch path for T-cell using the shared helpers. Do not replace
canonical output writing yet.

### Packet 2E — Parity Comparator

Implement old-vs-new comparison for protected Mukesh/T-cell outputs.

## Required Checks

- Existing canonical Mukesh and T-cell commands still run.
- New scratch commands run side by side.
- Old vs scratch row counts match.
- Old vs scratch key sets match.
- Protected numeric fields match within exact or declared tolerance.
- Protected categorical fields match exactly.
- Validators still pass on canonical outputs.

## Exit Criteria

- Mukesh scratch outputs match canonical outputs.
- T-cell scratch outputs match canonical outputs.
- Any discrepancy has a drift exception or blocks the phase.
- Human reviewer approves replacing duplicated writer code in a later phase.

## Rollback Criteria

Remove `alz/core/mea_outputs.py` and scratch adapter paths. Canonical outputs and
old code paths remain available.

## Decision Log Instructions

Log decisions for:

- recurrence field definitions,
- ordering rules for wide matrices,
- numeric tolerance policy,
- whether to preserve legacy column order exactly,
- scratch output root naming,
- any skipped parity file.

Decision log path:

```text
docs/audits/cohort_abstraction_refactor/phase_2_decisions.md
```
