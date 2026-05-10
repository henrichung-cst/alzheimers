# Kinase ↔ Incytr Integration

**Status:** Phase 1 stub. Architecturally mid-rewrite per
[`docs/incytr_remediation_plan.md`](../incytr_remediation_plan.md).

This document describes what currently lives in-tree under
`alz/integration/`. The legacy shadow-fork architecture (`wrappers/`,
`adapters/`, `sidecar/`, `tests/` + orchestrator shells) is preserved
under `archive/incytr_integration/`; for the historical design, see
[`docs/archive/kinase_incytr_integration_pre_remediation.md`](../archive/kinase_incytr_integration_pre_remediation.md).

## Architecture (target)

The remediation plan moves the integration math into the upstream
`incytr` R package proper (`../incytr`). This repository becomes a thin
AD-specific shell that:

1. Materializes per-contrast inputs from the live kinase pipeline
   outputs (`outputs/reports/kinase_attribution/*`) into the shapes
   `Incytr` consumes.
2. Calls `Incytr::*` functions directly from R — no shadow wrappers.
3. Persists results into the per-cohort DuckDB at
   `outputs/reports/incytr_factorial/`.

Until upstream `Incytr` finalizes the consuming API, the in-tree
surface is a set of incomplete Phase 1 stubs.

## In-tree inventory

Under `alz/integration/`:

| File | Role |
|---|---|
| `config_integration.py` | Paths, thresholds, contrast definitions, scoring/universe/config-id digests. Kept by the remediation plan; possibly trimmed. |
| `factorial.R` | Per-contrast `Incytr` invocation (Phase 1 stub — incomplete; awaits the upstream package API). |
| `load.R` | Loads materialized inputs (kinase data, expression, phospho) into the R session. |
| `persist.R` | Writes per-pair Parquet outputs into the per-cohort DuckDB. |
| `views.sql` | DuckDB view definitions over the persisted outputs. |
| `run_factorial.sh` | Two-line wrapper: `Rscript alz/integration/factorial.R "$@"`. |
| `MOVED.txt` | One-page note documenting the 2026-05-08 archive relocation. |
| `README.md` | Pointer file — points readers at `MOVED.txt` and the remediation plan. |
| `intermediates/` | Gitignored output dir from the legacy pipeline (orphaned; harmless). |

## Running the stub

```bash
pixi run install-incytr     # installs the upstream package from ../incytr
pixi run incytr-factorial   # currently exits early — stubs are incomplete
```

The `incytr-factorial` task is exposed in `pixi.toml` so the wiring
exists when the upstream API is ready, but it is not part of the live
arc.

## R dependencies

Required by the Phase 1 stubs (and by the upstream `Incytr` package):
`Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`. All conda-forge
packages, pinned in `pixi.toml`.

## Configuration

`config_integration.py` is the integration-only config (kept separate
from `alz/config.py` to avoid dragging integration deps into the live
arc).

Notable knobs:

- `CONTRAST` — single primary contrast for the Phase 1 stub.
- `CONDITION_WT` / mutant column values — match h5ad metadata.
- `resolve_universe_id` / `resolve_scoring_id` / `resolve_config_id` —
  stable digest helpers used to key persisted outputs.

## Pipeline relationship

The integration is a one-way handoff from the live kinase arc to
upstream `Incytr`:

```
alz/pipelines/{enrich,attribute,recovery}/  →  outputs/reports/kinase_attribution/
                                            →  outputs/reports/attribution_recovery/
                                                       │
                                                       ▼
                                            alz/integration/load.R
                                                       │
                                                       ▼
                                            ../incytr (Incytr::*)
                                                       │
                                                       ▼
                                            alz/integration/persist.R
                                                       │
                                                       ▼
                                            outputs/reports/incytr_factorial/
```

No backflow. The kinase pipeline does not consume any output from
`Incytr`; the Phase 1 stubs read kinase outputs and produce
`incytr_factorial/` artifacts.

## See also

- [`docs/incytr_remediation_plan.md`](../incytr_remediation_plan.md) —
  authoritative architectural plan for the rewrite.
- [`docs/archive/kinase_incytr_integration_pre_remediation.md`](../archive/kinase_incytr_integration_pre_remediation.md) —
  legacy shadow-fork architecture (historical reference only).
- [`alz/integration/MOVED.txt`](../../alz/integration/MOVED.txt) —
  in-tree note documenting the 2026-05-08 archive relocation.
