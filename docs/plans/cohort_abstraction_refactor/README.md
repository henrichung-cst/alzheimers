# Cohort Abstraction Refactor Control Pack

Date: 2026-06-17

Implementation status update (2026-06-18): phases 0-5 have been implemented.
This pack is now historical control documentation, with the final source state
recorded in `docs/audits/cohort_abstraction_refactor/`. Current policy:
`alz/core/` exists, viewer shared code lives under `alz/viewer/`, the moved
Mukesh/T-cell/5xFAD modules have no old-path wrappers, and cohort CLIs are
invoked with `python -m alz.cohorts...`.

## Purpose

This control pack prepares the cohort abstraction refactor for implementation by
a multi-agent orchestrator system. The refactor is important for maintainability
and for adding future datasets, but it touches analysis paths whose outputs are
scientifically consequential. The work must therefore proceed as a regulated,
non-destructive migration with explicit parity evidence at every phase.

The target architecture is:

```text
cohort-specific ingest + cohort-specific contrast design
    -> canonical cohort artifacts
    -> shared analysis kernels
    -> shared result contracts
    -> viewer-specific presentation
```

## Cohorts In Scope

- Song mouse AD
- Mukesh/NBB human AD
- T-cell exhaustion
- 5xFAD mouse

## Prime Directive

Default expectation: **no output drift**.

Any row, key, numeric, categorical, sample-inclusion, sign-convention, threshold,
or viewer-payload drift must be declared, justified, reviewed, and logged before
it can be accepted.

## Control Files

| File | Purpose |
| --- | --- |
| `agent_protocol.md` | Shared operating rules for all agents. |
| `phase_0_inventory_baseline_lock.md` | Freeze the current output baseline and define parity contracts. |
| `phase_1_read_only_validators.md` | Add schema/provenance validators without changing producers. |
| `phase_2_shared_output_recurrence_helpers.md` | Extract shared output-writing and recurrence helpers. |
| `phase_3_shared_mea_runner.md` | Introduce the shared MEA runner with cohort contrast adapters. |
| `phase_4_cohort_directory_migration.md` | Move cohort-specific modules into `alz/cohorts/` after parity is stable. |
| `phase_5_viewer_slice_contract.md` | Extract a shared viewer slice contract. |

## Required Phase File Sections

Each phase file contains:

- goals,
- non-goals,
- prerequisites,
- implementation scaffold,
- agent work packets,
- required checks,
- exit criteria,
- rollback criteria,
- decision log instructions.

## Decision Log Rule

Every phase must maintain a decision log before and during implementation. If a
decision affects analysis behavior, output schemas, naming, provenance, or
backward compatibility, it must be logged.

Recommended log location:

```text
docs/audits/cohort_abstraction_refactor/<phase>_decisions.md
```

If an automated orchestrator cannot write under `docs/audits/`, it should write
the log under the phase output directory and copy a summary into docs during the
review step.

## Protected Outputs

At minimum, protect:

- MEA long tables,
- kinase x contrast NES/FDR matrices,
- recurrence tables,
- site-level OLS/effect tables,
- global-shift and winsorized-site audit tables,
- substrate-set audit tables,
- attribution tables,
- viewer payload JSON and lazy shard indexes,
- Incytr wide outputs and receiver-cache outputs, when touched.

## Required Monitoring Report

Each phase must produce a monitoring report containing:

```text
phase id
agent ids / work packet ids
git base commit
git final commit or working-tree identifier
commands run
input roots
output roots
protected files checked
pass/fail status per protected file
known skipped checks and why
drift exceptions
rollback status
```

Recommended report location:

```text
outputs/reports/refactor_audit/<phase>/
```

## Phase Order

Phases must run in order:

1. Phase 0: inventory and baseline lock
2. Phase 1: read-only validators
3. Phase 2: shared output/recurrence helpers
4. Phase 3: shared MEA runner
5. Phase 4: cohort directory migration
6. Phase 5: viewer slice contract

Do not start a later phase if a previous phase has unresolved parity failures on
protected outputs.

## Immediate Human Review Gate

Before implementation begins, a human reviewer should confirm:

- which current dirty/WIP changes are part of the refactor,
- which changes are unrelated and should remain isolated,
- where Phase 0 baseline reports will be stored,
- which outputs are available locally for baseline locking,
- whether viewer directory reorganization should be committed before or after
  Phase 0.
