# Cross-Cohort MEA Standardization Packets 6-7

Date: 2026-06-18

## Decision

Viewer integration is implemented as optional payload support only. The T-cell
viewer builder can expose `payload.projected_state_mea` when projected-state
MEA files are present under the promoted donor state-MEA output directory. If
those files are absent, the block is omitted and existing viewer behavior is
unchanged.

Unified viewer cohort adapters can expose categorical mechanism attribution
rows when standardized mechanism files are present:

- Mukesh: `payload.human.mechanism_attribution`
- 5xFAD: `payload.supporting_5xfad.mechanism_attribution`
- Song: top-level `payload.mechanism_attribution`

These payload rows carry categorical `mechanism_call`/`sign_relation` plus raw
NES/FDR evidence columns. No numeric mechanism score is exported.

## Verification Status

Code-level verification passed for the optional payload loaders and mechanism
contracts. Full viewer regeneration could not be completed in the current shell:

- `python alz/build_tcell_viewer.py --payload --html --validate` stopped at
  missing `duckdb`.
- `python alz/build_unified_viewer.py` stopped at missing `kinase_library`.
- `pixi run ...` could not be used because `pixi` is not installed in this
  shell.

These are environment blockers, not analysis-design blockers. Re-run the viewer
build commands in the project environment that provides `duckdb`,
`kinase_library`, and Pixi.
