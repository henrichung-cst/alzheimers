# Phase 3 — Shared MEA Runner

## Goal

Introduce a shared MEA runner that owns common orchestration:

- track iteration,
- stoichiometry/raw-phospho analysis-kind iteration,
- absent-track skip recording,
- motif availability checks,
- MEA invocation,
- audit table writing,
- recurrence writing,
- run manifest/provenance writing.

Cohorts should supply contrast adapters, not copy the full MEA pipeline.

## Non-Goals

- Do not redesign statistical models.
- Do not change cohort contrast semantics.
- Do not change MEA thresholds, winsorization, centering, or kinase-library
  behavior.
- Do not retire old cohort scripts until parity passes.
- Do not move cohort modules yet.

## Prerequisites

- Phase 2 shared output helpers pass parity for Mukesh and T-cell.
- Validators exist.
- Baseline outputs are protected.

## Implementation Scaffold

Suggested modules:

```text
alz/core/contrast.py
alz/core/mea_runner.py
alz/core/provenance.py
```

Conceptual API:

```python
run_cohort_mea(
    cohort_id,
    matrices,
    sample_manifest,
    contrast_adapter,
    output_root,
    tracks,
    analysis_tracks,
    run_label,
)
```

Contrast adapter concept:

```python
class ContrastAdapter:
    def build_contrasts(matrix, sample_manifest, analysis_track) -> dict:
        ...

    def summarize_strata() -> DataFrame:
        ...
```

## Migration Order

1. Mukesh/NBB
2. T-cell
3. 5xFAD bulk MEA
4. 5xFAD cell-type MEA, only after bulk MEA is stable
5. Song, last

Song goes last because it has the richest downstream attribution and recovery
dependencies.

## Agent Work Packets

### Packet 3A — Runner Skeleton

Implement the shared runner using scratch output only.

### Packet 3B — Mukesh Contrast Adapter

Implement donor-vs-CTRL-mean adapter. Preserve current CTRL mean behavior and
AD/CTRL ordering.

### Packet 3C — T-cell Contrast Adapter

Implement later-day-vs-Day-2 adapter. Preserve donor-level baseline behavior and
skip handling for donor2.

### Packet 3D — 5xFAD Bulk Adapter

Implement TG-vs-WT within tissue/age adapter for bulk MEA only.

### Packet 3E — 5xFAD Cell-Type Adapter

Implement cell-type MEA adapter only after Packet 3D passes.

### Packet 3F — Song Adapter Feasibility Report

Before coding Song migration, write a feasibility report listing every Song
downstream dependency on current file shapes.

## Required Checks

For each migrated cohort:

- old command still runs,
- new shared-runner command writes scratch output,
- validators pass,
- old vs scratch protected outputs match,
- run manifest records old/new command lines,
- skip manifests match,
- sign convention is stamped.

## Exit Criteria

- Mukesh, T-cell, and 5xFAD run through shared runner in scratch mode with
  parity.
- Song feasibility report accepted.
- No canonical output switch happens without reviewer approval.

## Rollback Criteria

Disable shared-runner entrypoints. Existing cohort scripts remain the fallback.

## Decision Log Instructions

Log decisions for:

- contrast adapter boundaries,
- runner API fields,
- skip-manifest schema,
- provenance-manifest schema,
- cohort migration order changes,
- any accepted parity tolerance.

Decision log path:

```text
docs/audits/cohort_abstraction_refactor/phase_3_decisions.md
```
