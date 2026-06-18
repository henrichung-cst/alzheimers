# Phase 0 — Inventory and Baseline Lock

## Goal

Freeze the current behavior before refactoring. This phase defines exactly what
must remain non-regressive by inventorying current outputs and creating parity
contracts for Song, Mukesh/NBB, T-cell, and 5xFAD.

## Non-Goals

- Do not change analysis code.
- Do not regenerate canonical outputs.
- Do not move files or imports.
- Do not introduce new schemas beyond read-only inventory metadata.
- Do not decide that current behavior is correct; only record what it is.

## Prerequisites

- Clean git working tree.
- Current canonical outputs available locally, or missing outputs explicitly
  recorded.
- Human reviewer confirms which output roots are canonical for each cohort.

## Implementation Scaffold

Create a read-only inventory command, for example:

```text
alz/core/baseline_inventory.py
```

The command should accept:

```bash
python -m alz.core.baseline_inventory \
  --cohort song \
  --output outputs/reports/refactor_audit/phase_0/song_inventory.json
```

The first version may be a simple script under `tools/` or `alz/core/` as long
as it is read-only.

Inventory fields:

```text
path
exists
file_size_bytes
sha256
mtime
row_count, for table files
column_names, for table files
key_columns, when known
key_unique, when known
numeric_columns
categorical_columns
producer_command, when known
notes
```

## Agent Work Packets

### Packet 0A — Output Root Discovery

Identify current output roots for each cohort:

- Song mouse AD
- Mukesh/NBB human AD
- T-cell exhaustion
- 5xFAD mouse
- Incytr pair-mode outputs
- viewer outputs

Deliverable:

```text
outputs/reports/refactor_audit/phase_0/output_roots.json
```

### Packet 0B — Protected File List

Define protected outputs by cohort:

- MEA long tables
- NES/FDR matrices
- recurrence tables
- site-level OLS/effect tables
- audit tables
- attribution tables
- viewer payloads and shard indexes

Deliverable:

```text
outputs/reports/refactor_audit/phase_0/protected_files.json
```

### Packet 0C — Inventory Generator

Implement the read-only inventory generator and run it on the protected files.

Deliverables:

```text
outputs/reports/refactor_audit/phase_0/<cohort>_inventory.json
outputs/reports/refactor_audit/phase_0/<cohort>_inventory.csv
```

### Packet 0D — Baseline Summary

Write a human-readable summary:

```text
docs/audits/cohort_abstraction_refactor/phase_0_baseline_summary.md
```

## Required Checks

- `git status --short` shows only intentional Phase 0 files.
- Inventory command is read-only.
- Missing files are reported, not created.
- Hashes and row counts are stable across two consecutive inventory runs.

## Exit Criteria

- Protected file list exists.
- Inventory reports exist for all available cohorts.
- Missing outputs are explicitly listed.
- Human reviewer accepts the protected-file list and parity policy.

## Rollback Criteria

Rollback is simple: remove Phase 0 inventory scripts and reports. No production
outputs should have changed.

## Decision Log Instructions

Log decisions for:

- choosing canonical output roots,
- excluding any output from protection,
- accepting missing outputs,
- choosing key columns,
- choosing hash/row-count tolerance policy.

Decision log path:

```text
docs/audits/cohort_abstraction_refactor/phase_0_decisions.md
```
