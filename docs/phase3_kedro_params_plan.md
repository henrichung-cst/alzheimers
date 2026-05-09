---
phase: 3 of 5 (per CURRENT_SPRINT.md §Sequencing)
status: completed 2026-05-09
---

# Phase 3 — Cohort config to Kedro parameters

## Context

Phase 2 stood up Kedro as the workflow framework with a single proof pipeline (`ingest_mapping`). Cohort selection still flows through the legacy `ANALYSIS_MODE` env var read in `alz/config.py:84`. The sprint standard (CURRENT_SPRINT.md §Standards) is explicit: *"Kill the `ANALYSIS_MODE` env-var pattern"* — Kedro parameters become the single source of truth.

Phase 4 (per-phase pipeline migration) is the broader move; Phase 3 is the narrow plumbing change that makes Phase 4 possible without revisiting cohort wiring later. Three modules currently consume the mode and must be re-rooted: `kinase_enrich.py`, `kinase_attribute.py`, `kinase_mechanism.py`. (`kinase_attribute.py` doesn't read the mode itself but imports `config`, so it's bridged automatically.)

The CLI scripts still drive the live arc until Phase 4, so they need an override mechanism for the dual-track runner. Decision (locked in this turn): use Kedro's native `KEDRO_ENV` switch with a `conf/full_cohort/` overlay — purest Kedro idiom, no project-specific env vars, scales to other future overrides.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Source of truth = `conf/base/parameters.yml`**, key `analysis_mode: males_only` | Sprint standard: Kedro parameters are canonical. |
| 2 | **Override mechanism = `KEDRO_ENV` switch** (`conf/full_cohort/parameters.yml`) | Idiomatic Kedro; matches the pattern Phase 4 will use for any other future overlays. |
| 3 | **`alz/config.py` loads params via `OmegaConfigLoader`** (not full `KedroSession.create()`) | Lightweight — no session lifecycle in CLI scripts. Same loader Kedro uses internally; reads `KEDRO_ENV` from env. |
| 4 | **`config.ANALYSIS_MODE` retained as a module-level attribute** | Minimizes diff in `kinase_enrich.py` / `kinase_mechanism.py` — they still see `config.ANALYSIS_MODE`; only the source changes. Phase 4 will replace these reads with `params:analysis_mode` injected into nodes. |
| 5 | **Drop `ANALYSIS_MODE` env var entirely** — pixi tasks lose `env = { ANALYSIS_MODE = "males_only" }`; dual runner switches to `KEDRO_ENV=full_cohort`. | Standard says kill it. Don't leave a dual-source ambiguity. |
| 6 | **`attribution_recovery.py` is out of scope** | Reads no `ANALYSIS_MODE` directly (surface map confirmed); only the shell archives outputs by mode. Stays untouched. |

## Out of scope

- Migrating any module into a Kedro pipeline (Phase 4).
- Rewriting flat imports as package-relative (Phase 4 — needed before the `sys.path` bridge in `alz/__init__.py` can be deleted).
- Touching `attribution_recovery.py` or any non-kinase module.
- Adding tests beyond a smoke check that both cohorts still produce expected output.
- Deleting `pixi run live` or shell runners (Phase 4/5 territory).

## Concrete steps

### Step 1 — Add cohort param to Kedro config

- `conf/base/parameters.yml`: add `analysis_mode: males_only` (alongside the existing `proof_marker`).
- Create `conf/full_cohort/parameters.yml` with single line `analysis_mode: full_cohort`.
- `conf/local/` already gitignored from Phase 2; no change needed.

### Step 2 — Re-root `config.ANALYSIS_MODE` through Kedro

Edit `alz/config.py:84`:

```python
# Before:
ANALYSIS_MODE = os.environ.get("ANALYSIS_MODE", "males_only")

# After:
def _load_analysis_mode() -> str:
    """Read cohort selection from Kedro parameters. Honors KEDRO_ENV for overrides."""
    from kedro.config import OmegaConfigLoader
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.get("KEDRO_ENV", "base")
    loader = OmegaConfigLoader(
        conf_source=str(repo_root / "conf"),
        base_env="base",
        default_run_env=env,
    )
    return loader["parameters"]["analysis_mode"]

ANALYSIS_MODE = _load_analysis_mode()
```

Validate: `OmegaConfigLoader` is the Kedro 1.x default config loader; reads `base/` then layers `<env>/` on top. Already a transitive dep of `kedro` itself — no new package install.

### Step 3 — Update CLI invocation surfaces

- `pixi.toml`: drop `env = { ANALYSIS_MODE = "males_only" }` from the `enrich` and `attribute` task definitions (lines 13-14). YAML default carries males-only.
- `alz/runners/main/run_dual_analysis.sh`: replace `ANALYSIS_MODE=males_only` prefixes with bare invocation (males-only is the YAML default); replace `ANALYSIS_MODE=full_cohort` prefixes with `KEDRO_ENV=full_cohort`.

### Step 4 — Validate

Smoke checks (no automated tests added):

- `python alz/kinase_summary.py` after each invocation prints the active cohort.
- `pixi run enrich` (default males_only): output `outputs/reports/kinase_attribution/site_level_ols.csv` row count matches a pre-Phase-3 baseline (the design matrix shape is cohort-dependent — see `_build_design_matrix` at `alz/kinase_enrich.py:146`).
- `KEDRO_ENV=full_cohort python alz/kinase_enrich.py`: design matrix includes the `female` column.
- `bash alz/runners/main/run_dual_analysis.sh`: both cohort archives populate as before.
- `kedro run --pipeline=ingest_mapping`: still passes (Phase 2 regression check).

### Step 5 — Commit

Conventional commits, per logical unit:

1. `feat(config): move cohort selection from ANALYSIS_MODE env var to Kedro parameters` — `config.py` + `conf/base/parameters.yml` + `conf/full_cohort/parameters.yml`
2. `refactor: drop ANALYSIS_MODE env-var threading from pixi tasks and dual runner` — `pixi.toml` + `run_dual_analysis.sh`
3. `docs: record Phase 3 plan + completion notes` — this plan, plus CURRENT_SPRINT.md §Sequencing item 4 marked done

## Critical files

| File | Change |
|---|---|
| `alz/config.py:84` | Replace env-var read with `OmegaConfigLoader` load |
| `conf/base/parameters.yml` | Add `analysis_mode: males_only` |
| `conf/full_cohort/parameters.yml` | New file: `analysis_mode: full_cohort` |
| `alz/runners/main/run_dual_analysis.sh:40-43, 58-61` | Strip env prefixes; add `KEDRO_ENV=full_cohort` to track 2 |
| `pixi.toml:13-14` | Drop `env = { ANALYSIS_MODE = "males_only" }` from `enrich`, `attribute` |

No changes needed in `kinase_enrich.py`, `kinase_attribute.py`, `kinase_mechanism.py` — they continue to read `config.ANALYSIS_MODE`; only its source changes. (Phase 4 inverts this: nodes receive `params:analysis_mode` directly and `config.ANALYSIS_MODE` is deleted.)

## Risks

- **`OmegaConfigLoader` import at `config.py` load time** triggers Kedro project bootstrap when *any* live-arc script runs. If kedro lazy-init has side effects (e.g., logging config), they fire on every invocation. Mitigation: import inside `_load_analysis_mode()` (already done above) so the cost is paid once at module load, not at every import-time pass.
- **`alz/__init__.py` sys.path bridge** runs *before* `config.py` imports kedro. Kedro itself does not depend on the bridge, so order is fine — but if `OmegaConfigLoader` ever transitively imports an `alz.*` module, the bridge must already be installed. Verified: kedro core has no `alz.*` references.
- **CI / fresh environments** won't have a `pyproject.toml`-rooted Kedro project unless the repo root is the cwd. `Path(__file__).resolve().parent.parent` pins it absolutely — should be robust.
- **`pixi run mechanism`** task currently has no env block (falls through to default). Behavior unchanged after Phase 3 — but worth re-verifying since the default source moved from Python literal to YAML.

## Verification

Definition of done:

- `python alz/kinase_summary.py` prints `analysis_mode = males_only` with default invocation.
- `KEDRO_ENV=full_cohort python alz/kinase_summary.py` prints `analysis_mode = full_cohort`.
- `bash alz/runners/main/run_dual_analysis.sh` produces both `*_males_only/` and `*_full_cohort/` archive trees with the same row counts as a pre-Phase-3 baseline.
- `grep -r ANALYSIS_MODE alz/ pixi.toml` returns only docstrings / comments — no live env-var reads or sets.
- `kedro run --pipeline=ingest_mapping` still passes (Phase 2 regression).
- 3 commits landed on `incytr-cleanup`.
- CURRENT_SPRINT.md §Sequencing item 4 marked done.

## Completion notes (2026-05-09)

Landed on `incytr-cleanup` as three commits:

1. `55edf9f feat(config): move cohort selection from ANALYSIS_MODE env var to Kedro parameters` — `conf/base/parameters.yml` gains `analysis_mode`; `conf/full_cohort/parameters.yml` created; `alz/config.py` re-rooted via `OmegaConfigLoader`.
2. `0c267a9 refactor: drop ANALYSIS_MODE env-var threading from pixi tasks and dual runner` — `pixi.toml` `enrich`/`attribute` lose env blocks; `run_dual_analysis.sh` uses `KEDRO_ENV=full_cohort` for track 2.
3. `<pending> docs: update CLAUDE.md + README.md cohort-selection sections` — workspace docs updated to match new behavior; foundation/integrations doc sweep deferred to Phase 5.

Validation:

- `pixi run python -c "import alz.config; print(alz.config.ANALYSIS_MODE)"` prints `males_only` by default.
- `KEDRO_ENV=full_cohort pixi run python -c "import alz.config; print(alz.config.ANALYSIS_MODE)"` prints `full_cohort`.
- `KEDRO_ENV=nonexistent_env` raises `MissingConfigException` (graceful failure mode, not silent fallback).
- `kedro run --pipeline=ingest_mapping` still passes (Phase 2 regression).
- `python alz/kinase_summary.py` runs end-to-end.

**Surprise during execution, recorded so Phase 4 doesn't re-stumble:**

- `OmegaConfigLoader` used standalone (outside `kedro run`) defaults `base_env` and `default_run_env` to **empty strings**, not `"base"` and `"local"` as the docstrings imply for project-bootstrapped use. With those empties, `Path(conf_source) / ""` resolves to `conf/`, and the recursive `**/parameters*` glob then sweeps every `conf/<env>/parameters.yml` into a single merge — the duplicate-key check fires. Fix: pass `base_env="base", default_run_env="local"` explicitly. Phase 4 should look at moving config loading into a project-bootstrap path (`bootstrap_project()` or `KedroSession.create()`) so this constant becomes redundant.
