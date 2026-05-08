# Incytr Integration

**Status (2026-05-08):** the legacy R wrappers and Python adapters that previously lived here have been relocated to `~/Projects/work/incytr_integration_archive/`. See [`MOVED.txt`](./MOVED.txt) for the full list and rationale.

This directory now holds only:

- `config_integration.py` — paths, thresholds, contrast definitions (kept per the remediation plan)
- `factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh` — Phase 1 stubs for the new architecture (entry point + AD-specific loaders + parquet persistence + SQL views + one-line shell entry)
- `intermediates/` — gitignored output dir from the legacy pipeline (orphaned; safe to delete)

The structural rewrite that produced the stubs is documented in [`../../docs/incytr_remediation_plan.md`](../../docs/incytr_remediation_plan.md). The end-state target is a thin AD-specific shell that calls into the upstream `incytr` R package directly, with no shadow-forked math and no materialized-derivation CSVs.

The integration spec ([`../../docs/integrations/kinase_incytr_integration.md`](../../docs/integrations/kinase_incytr_integration.md)) describes the legacy architecture and is also being superseded.
