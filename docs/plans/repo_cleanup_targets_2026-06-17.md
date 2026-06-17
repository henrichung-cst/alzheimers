# Repo cleanup targets — 2026-06-17

Read-only audit performed after checkpoint commit `d756bc7`. The tracked source tree is small
(262 files, ~4.4 MB); the checkout size is dominated by ignored/local data, outputs, environments,
and provenance archives.

## Pass 1 — live documentation path drift

Status: completed.

Contributor-facing docs still point at pre-reorg module names such as
`alz/data_ingest.py`, `alz/kinase_enrich.py`, `alz/atlas_reference.py`, and missing decision docs.
Update the live docs to match the current package layout:

- `alz/ingest/song.py`
- `alz/bulk_mea/{normalize,enrich,attribute,mechanism,recover}.py`
- `alz/reference/{atlas,wmb_expression,snrna_integration,snrna_proportions,human_expression}.py`
- `alz/shared/config.py`

Scope: live docs and viewer-facing command hints only. Historical plans can keep old paths when they
are clearly old decision records.

Completed in this pass:

- Updated `docs/INDEX.md`, `docs/foundation/live_pipeline_contract.md`,
  `docs/foundation/repo_retention_policy.md`, `docs/foundation/multiple_testing.md`,
  `docs/foundation/cohort_contract.md`, and `docs/foundation/concordance.md`.
- Updated viewer/runtime help strings that pointed users at removed root-level scripts.
- Replaced missing deconvolution-pivot and spine-rethreshold pointers with current foundation docs
  and `alz/decomposition_mea/README.md`.

## Pass 2 — plan triage

Status: completed.

`docs/INDEX.md` described one active plan that no longer exists, while `docs/plans/` contains 33
plan files. Decide which plans are active, superseded, or provenance-only, then either update the
index or move superseded records to an archived location with pointers.

Completed in this pass:

- Removed superseded plan files from `docs/plans/`.
- Kept this cleanup ledger as the sole active plan in `docs/plans/`.
- Updated `docs/INDEX.md` to describe the simplified active plan directory.

## Pass 3 — generated cache cleanup

Status: completed.

Ignored/generated local files are present under `alz/**/__pycache__`, `tests/__pycache__`,
`vendor/rclone-ingest/**/__pycache__`, `bench/__pycache__`, `.ruff_cache`, and `graphify-out`.
These are low-risk local cleanup candidates.

Completed in this pass:

- Removed Python `__pycache__` directories outside `.pixi/`.
- Removed `.ruff_cache/` and `graphify-out/`.
- Removed ignored rendered docs (`docs/**/*.pdf` and generated HTML, preserving the tracked probe
  page at `docs/probes/s3_gzip_probe/index.html`).

## Pass 4 — viewer template duplication

Status: partially completed.

`alz/viewer/template` and `alz/tcell_viewer/template` still share 9 same-path template files after
the identical `head.html` chunk and near-identical `index.html.j2` shell were moved to
`alz/viewer_shared`. The remaining overlaps all diverge. Consolidation should be done against
`alz/viewer_shared` with payload/context checks, not by blind deduplication.

Completed in this pass:

- Moved identical `head.html` into `alz/viewer_shared/template/head.html`; both viewers now resolve
  it through the existing shared-template fallback.
- Removed the inactive legacy T-cell `TemporalV2` implementation and its orphaned `.tv2-*` styles;
  the live T-cell `Temporal` tab remains intact.
- Restored the existing T-cell cell-type assignment workbench by adding its missing template include,
  tab panel, `TAB_GUIDE`, and `TAB_MANIFEST` entry.
- Moved the common `index.html.j2` shell into `alz/viewer_shared/template/index.html.j2`; builders
  now pass only viewer-specific tab include paths into the shared shell.

Still pending:

- Function-level consolidation of shared helper code in `js/01_state.js`, `js/02_ui_chrome.js`, and
  the kinase tab scripts.

## Pass 5 — storage/provenance policy cleanup

Status: partially completed.

Large ignored/local surfaces:

- `data/` ~62 GB
- `outputs/` ~8.3 GB
- `.pixi/` ~4.4 GB
- `archive/` ~2.7 GB
- `bench/` ~297 MB

Treat these as storage-policy cleanup. Do not delete provenance or upstream data without a
manifest of what is regenerable, externally recoverable, or uniquely local.

Completed in this pass:

- Removed large ignored benchmarking payloads under `bench/` (`find bench -type f -size +1M`).
- Removed empty benchmark directories left behind by that deletion.

Still pending:

- `archive/`, `data/`, and `outputs/` remain intentionally untouched pending a manifest-based
  review.

## Pass 6 — test layout

Status: pending.

Only two Python tests are discoverable: `tests/test_confidence.py` and
`alz/ingest/test_fivexfad.py`. Consider moving package-local tests into `tests/` and adding a
single documented test command once the layout is settled.
