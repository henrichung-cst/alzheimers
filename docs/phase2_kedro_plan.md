---
status: completed 2026-05-09
phase: 2 of 5 (per CURRENT_SPRINT.md §Sequencing)
date_drafted: 2026-05-08
---

# Phase 2 — Kedro Skeleton Plan

## Goal

Stand up Kedro as the workflow framework with one end-to-end proof node. No live-pipeline migration. Phase 3 (cohort param) and Phase 4 (per-phase migration) build on this skeleton.

End state: `kedro run --pipeline=ingest_mapping` runs the TMT channel-mapping node end-to-end via the Data Catalog and `parameters.yml`, producing `outputs/reports/data_ingest/sample_mapping.csv` byte-identical to the current `data_ingest.py --mapping` output.

## Decisions (locked in via discussion)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Rename `code/` → `alz/`** as part of Phase 2 | `code` shadows the stdlib `code` module; Phase 4 shouldn't have to do this rename mid-migration. One-shot churn now, paid back across every later phase. |
| 2 | **Flat layout under `alz/`** (no `pipelines/<phase>/` subdirs yet) | Kedro registers pipelines from any importable location. Phase 4 lifts modules into pipeline subpackages as each phase migrates; doing it before then is premature. |
| 3 | **Proof-of-concept node = TMT channel mapping** (`data_ingest.py --mapping`) | Smallest real I/O contract on the live arc. No atlas/external deps. Exercises Excel reader + CSV writer + a parameter injection. Survives Phase 4 unchanged. |
| 4 | **Kedro from PyPI, pinned `kedro>=1.3,<2`** (revised at execution) | Kedro 1.3.1 is the current stable on conda-forge; the originally planned `~=0.19` was outdated. Conda-forge install conflicted with kinase-library's `myst-parser` pre-release marker, so kedro went into `[pypi-dependencies]` alongside kinase-library and py-spy rather than `[dependencies]`. |
| 5 | **Catalog grown incrementally** | Declare only the two datasets the proof node touches. Pre-declaring all 30+ live-arc datasets before any node consumes them is bookkeeping that rots. |
| 6 | **`parameters.yml` exists with one stub param** read by the proof node | Proves the parameter-injection wiring so Phase 3's cohort migration is "thread the param through more nodes," not "build the plumbing." |

## Node design philosophy

Nodes are **general (shape-bound), not dataset-specific**. Dataset specifics live in the catalog and parameters. This matches the sprint's "light reusability across similarly-shaped TMT / phospho / snRNA cohorts" goal without crossing into the "generic dataset abstraction layer" non-goal.

Three layers:

1. **Pure math nodes** — fully generic. `compute_stoichiometry(phospho, protein)`, `factorial_ols(df, design)`, `mea_enrich(ranks, sets)`. No dataset coupling.
2. **TMT-shape-bound nodes** — assumes plex/channel structure. IRS normalization, phospho-protein matching. Reusable across Song + Lucie 5xFAD; won't fit DIA (non-goal).
3. **Ingest nodes** — closest to dataset-specific, but still general functions: `parse_tmt_channel_map(metadata_df, params)` rather than `parse_song_metadata()`. Song-ness lives in the catalog entry (which file, which sheet) and parameter overrides.

The cohort param case forces this: `males_only` vs `full_cohort` works only because nodes are general w.r.t. cohort and run twice with different params. The framework choice has already committed us to general nodes.

**Not built:** dataset adapter classes, registry/plugin patterns, abstract base classes. Pure functions + catalog + parameters is the whole abstraction.

Phase 2 proof node demonstrating this:

```python
# alz/pipelines/ingest_mapping/nodes.py
def build_tmt_channel_mapping(metadata: pd.DataFrame, params: dict) -> pd.DataFrame:
    ...
```

Catalog wires `song_metadata_xlsx` → this node. No "song" in the function name or body.

## Out of scope for Phase 2

- Migrating any live-pipeline stage other than the proof node.
- Moving `ANALYSIS_MODE` from env var to `parameters.yml` (that's Phase 3).
- Retiring `pixi run live`, `dual`, or any shell runner.
- Rewriting `kinase_normalize.py`, `kinase_enrich.py`, etc. into nodes.
- Touching R / Incytr code.

## Concrete steps

### Step 1 — Rename `code/` → `alz/`

`git mv code alz` plus a sweep of every importer/path reference:
- All Python files: `from code.X` → `from alz.X`, `import code.X` → `import alz.X`
- `pixi.toml` `[tasks]`: `python code/foo.py` → `python alz/foo.py` (and re-add `incytr-factorial` line untouched, it points at `alz/integration/run_factorial.sh` after rename)
- Shell runners under `alz/runners/`: paths inside the scripts
- `CLAUDE.md`, `README.md`, `docs/INDEX.md`, `docs/foundation/repo_retention_policy.md`, `docs/integrations/*.md`, every `code/foo.py` reference
- `pyproject.toml` (created in Step 2) declares `alz` as the package

Validation: `pixi run live` (or its individual stages) produces byte-identical outputs to a pre-rename run.

### Step 2 — Add Kedro + project scaffolding

- `pixi add kedro~=0.19` (conda-forge)
- Create `pyproject.toml` at repo root with Kedro project metadata + `alz` package declaration
- Create `conf/base/catalog.yml` with the two datasets the proof node touches
- Create `conf/base/parameters.yml` with one stub param (e.g. `proof_marker: phase2_smoke_test`)
- Create `conf/base/logging.yml` (Kedro default)
- Create `conf/local/.gitkeep`, add `conf/local/` to `.gitignore`
- Create `alz/settings.py`, `alz/pipeline_registry.py` (Kedro hooks)

### Step 3 — Wire the proof node

- Extract the TMT channel-mapping logic from `alz/data_ingest.py` into a pure function in `alz/pipelines/ingest_mapping/nodes.py` (input: dataframe; output: dataframe). The original `--mapping` CLI path keeps working.
- `alz/pipelines/ingest_mapping/pipeline.py` builds the Kedro pipeline from the node.
- Catalog declares the input Excel and the output CSV as Kedro datasets.
- Pipeline registered in `alz/pipeline_registry.py`.

### Step 4 — Validate

- `kedro run --pipeline=ingest_mapping` produces `outputs/reports/data_ingest/sample_mapping.csv`.
- Diff against a snapshot of the pre-Phase-2 output — must be byte-identical.
- `pixi run ingest` (which still calls `data_ingest.py --run`) continues to work end-to-end.
- Smoke-check `pixi run live` to confirm the rename didn't break the live arc.

### Step 5 — Commit

Single commit per logical unit, conventional commit messages:
1. `refactor: rename code/ → alz/ (avoid stdlib shadow, prep for Kedro)`
2. `feat: add Kedro project scaffolding (pyproject, conf/, settings)`
3. `feat(ingest): wire TMT channel mapping as first Kedro pipeline`
4. `docs: record Phase 2 plan + completion notes`

## Risks

- **Rename touches a lot of files.** Mitigation: `rg "code/" --type-add 'all:*' -tall` + `rg "from code" -tall` to find every reference; do the rename in one commit so reviewers can see the diff together.
- **Kedro 0.19 layout assumptions.** Kedro expects the package root via `pyproject.toml`; with a flat layout under `alz/` (no `pipelines/` subdir), pipeline discovery is explicit in `pipeline_registry.py` rather than auto-discovered. We sidestep auto-discovery by registering manually — fine for one pipeline, may revisit in Phase 4.
- **`pixi run live` regression** from the rename. Mitigation: Step 4 explicit smoke check before commit.

## Definition of done

- `kedro run --pipeline=ingest_mapping` succeeds and produces byte-identical output to `python alz/data_ingest.py --mapping`.
- `pixi run live` smoke-passes (gets through all 5 stages without import errors).
- 4 commits landed on `incytr-cleanup` (or successor branch).
- `CURRENT_SPRINT.md` §Sequencing updated to mark Phase 2 done.

## Open: branching

Phase 1 work is on `incytr-cleanup`. Phase 2 can either continue on the same branch or fork `kedro-skeleton`. Lean toward continuing on `incytr-cleanup` — same sprint, no value in splitting reviews. Defer to your call.

## Completion notes (2026-05-09)

Landed on `incytr-cleanup` as four commits:

1. `ffaefac refactor: rename code/ → alz/` — 117 files (text rewrite + git mv); archive paths preserved
2. `e332116 feat: add Kedro project scaffolding` — `pyproject.toml`, `alz/{__init__,settings,pipeline_registry}.py`, `conf/base/{catalog,parameters}.yml`, kedro 1.3.1 PyPI dep
3. `fbb7794 feat(ingest): wire TMT channel mapping as first Kedro pipeline` — proof node + catalog entries; output byte-identical to `python alz/data_ingest.py --mapping`; `kedro-datasets[pandas-csv,pandas-excel]` PyPI dep
4. `<this commit> docs: record Phase 2 plan + completion`

Validation: `kedro run --pipeline=ingest_mapping` produces byte-identical CSV; `python alz/data_ingest.py --mapping` continues to work in CLI mode; all 6 live-arc entry scripts (`data_ingest`, `kinase_normalize`, `kinase_enrich`, `kinase_attribute`, `kinase_mechanism`, `attribution_recovery`) load without import errors.

**Surprises during execution, recorded so Phase 4 doesn't re-stumble:**

- The flat-import pattern (`import config` resolved via `sys.path[0]` when scripts run as `python alz/foo.py`) breaks when Kedro imports modules as `alz.foo`. Fixed in Phase 2 with a 4-line `sys.path.insert(_ALZ_DIR)` bridge in `alz/__init__.py`. **Phase 4 must rewrite these as package-relative imports** (`from alz import config`) and switch live-arc invocation to `python -m alz.foo` so the bridge can be deleted.
- `kedro-datasets` is a separate PyPI package in Kedro 1.x — `pandas.CSVDataset` etc. aren't shipped in the core. Phase 4 will need additional extras for parquet, h5ad, plotly, etc., as more datasets land in the catalog.
- Conda-forge resolution of `kedro` collides with kinase-library's transitive `myst-parser` pre-release marker. PyPI install dodges this; revisit if kinase-library's metadata is fixed upstream.
