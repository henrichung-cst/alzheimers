# Cohort Refactor Cleanup Decisions

Date: 2026-06-18

## Module-only command policy

Canonical cohort commands are module invocations:

```text
python -m alz.cohorts.mukesh.ingest
python -m alz.cohorts.mukesh.mea
python -m alz.cohorts.tcells.ingest
python -m alz.cohorts.tcells.mea
python -m alz.cohorts.fivexfad.ingest
python -m alz.cohorts.fivexfad.celltype_mea
```

The old moved paths under `alz/ingest/{mukesh,tcells,fivexfad}*.py` remain
retired. They should appear only as historical provenance in refactor audit logs
or "moved from" tables, not as live commands.

## Direct new-path execution

The moved cohort modules keep a repo-root bootstrap so direct execution from the
new `alz/cohorts/...` paths is not accidentally broken during local debugging.
This is not the public command contract; documentation and task runners should
prefer `python -m`.

## Shared runner contract

`MeaAdapter.load_inputs()` may return `None` to record a structured skip before
contrast construction. `MeaRunner.run_unit(..., skip_check_fn=...)` honors the
caller-supplied skip callback before adapter-level skip checks.
