# Phase 1 — Read-Only Validators

## Goal

Add validators for current canonical artifacts without changing any producer.
Validators should make implicit contracts explicit and prepare the codebase for
safe shared runners.

## Non-Goals

- Do not change ingest modules.
- Do not change MEA, attribution, Incytr, or viewer behavior.
- Do not rewrite output files.
- Do not enforce new schemas that current outputs cannot satisfy unless the
  legacy deviation is documented and accepted.

## Prerequisites

- Phase 0 baseline inventory accepted.
- Protected files and canonical output roots identified.
- Clean working tree or isolated Phase 1 branch.

## Implementation Scaffold

Suggested modules:

```text
alz/core/__init__.py
alz/core/phospho_schema.py
alz/core/cohort_manifest.py
alz/core/validation.py
alz/core/validate_cohort.py
```

CLI concept:

```bash
python -m alz.core.validate_cohort --cohort mukesh
python -m alz.core.validate_cohort --cohort tcells
python -m alz.core.validate_cohort --cohort fivexfad
python -m alz.core.validate_cohort --cohort song
```

The validators should emit reports only:

```text
outputs/reports/refactor_audit/phase_1/<cohort>_validation.json
outputs/reports/refactor_audit/phase_1/<cohort>_validation.md
```

## Validation Scope

Validate:

- required columns,
- site metadata presence,
- sample metadata presence,
- sample columns in matrices match sample manifests,
- track suffix conventions,
- motif availability for MEA-capable tracks,
- duplicate key behavior,
- numeric value coercibility,
- sign-convention metadata when available,
- absent-by-design fields.

## Agent Work Packets

### Packet 1A — Schema Definitions

Implement schema objects or simple dataclasses for:

- sample manifest concepts,
- site matrix concepts,
- contrast manifest concepts,
- MEA output concepts.

### Packet 1B — Cohort Validators

Implement validators for current files without requiring producers to move.
Each validator may have cohort-specific path adapters.

### Packet 1C — Report Writer

Implement common JSON/Markdown validation reports.

### Packet 1D — Legacy Deviation Register

Create:

```text
docs/audits/cohort_abstraction_refactor/phase_1_legacy_deviations.md
```

Use it to record accepted current deviations.

## Required Checks

- Validators run without modifying canonical outputs.
- Validator reports are deterministic across repeated runs.
- All failures are either fixed in validator assumptions or recorded as accepted
  legacy deviations.
- `python -m py_compile` passes for new Python files.

## Exit Criteria

- Validators pass or produce accepted deviations for all four cohorts.
- Reports are written under `outputs/reports/refactor_audit/phase_1/`.
- Phase 1 decision log is complete.

## Rollback Criteria

Remove validator modules and reports. No production output or producer code
should need rollback.

## Decision Log Instructions

Log decisions for:

- required vs optional columns,
- accepted missing metadata,
- how absent tracks are represented,
- how duplicate site/sample keys are treated,
- whether wide matrices remain canonical for now,
- report format choices.

Decision log path:

```text
docs/audits/cohort_abstraction_refactor/phase_1_decisions.md
```
