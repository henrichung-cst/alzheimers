# CLAUDE.md

Project context, commands, architecture, and outputs: see [`README.md`](README.md) and [`docs/foundation/`](docs/foundation/). This file holds agent-specific operating overrides, correctness invariants, and traps.

## Project-specific behavior overrides

- **DuckDB spill directory** is `~/.cache/duckdb` via `.envrc` to avoid OOM on tmpfs `/tmp`. If DuckDB hits disk-full, verify with `echo $DUCKDB_TEMP_DIR`.

## Mass identity (decomposition verification)

`Σ_c [P_c × (N_c / N_total)] ≈ bulk`, **not** `Σ_c P_c = bulk`. The `f_c` weights are per-cell rates × N_total/N_c; literal summation overshoots.

## Project rules

- **Gated plans: stop at each phase boundary.** When the user defines a checkpoint ("run one, then show me, then run the rest"), stop and report before launching the next phase even when the outcome looks settled. Never fold sequential-but-approved steps into one irreversible action.
- **Intermediates are derivable.** Missing `*.rds` / derived CSVs / cached matrices are not blockers — reconstructing them from raw inputs + method IS the assignment. "We can't reproduce because X.rds isn't on disk" is a restatement of the problem, not a finding. Do not propose accepting stale artifacts as authoritative.
- **No collaborator contact — ever.** Do not propose emailing, asking, or requesting anything from external collaborators (e.g. Yuyu Song, CST team). Not as an option, not as a dismissed alternative. All investigation proceeds from artifacts already on disk + what is derivable from them. If those routes are exhausted, report that honestly — do not escalate externally.
- **Viewer ports = lift, not rewrite.** When porting a viewer to a new cohort, reuse `alz/viewer/template/js/tabs/*.js` / `js/widgets/*.js` / `js/01_state.js` verbatim — reshape the new cohort's payload to fit the unified viewer's `PAYLOAD.*` contract. Greenfielding a new app.js ships a permanently inferior feature surface. Genuinely-N/A features get dropped at `TAB_MANIFEST` level, not silently reimplemented.
- **Multi-session plans go in `docs/plans/`.** Write final plans to `docs/plans/<descriptive-noun-phrase>.md` in the repo, not `~/.claude/plans/`. The staging path is ephemeral; the repo is the durable review location.
- **Use `pixi run` for all verification.** Bare `python` is the system interpreter (no pyarrow/duckdb/project-deps). Always invoke `pixi run python …` or the relevant `pixi run <task>`. If a dep import fails unexpectedly, confirm the interpreter with `pixi run python -c "import sys; print(sys.executable)"` before concluding a real failure.

## Pipeline-specific gotchas

- **`analysis_mode` flow** — Kedro parameter in `conf/base/parameters.yml`, default `males_only`. `KEDRO_ENV=full_cohort` overlays the sensitivity config. Affects `enrich.py`, `attribute.py`, `mechanism.py` — **not** `normalize.py` (always uses all 72 samples). Legacy `ANALYSIS_MODE` env var was retired and is silently ignored.
- **Outlier detection requires stoichiometry** — `alz/ingest/song.py --outliers` reads `stoichiometry_matrix.csv`, so `normalize.py` must run first. Falls back to total proteome if unavailable.
- **Unified-viewer hard refresh** — `build_unified_viewer.py` inlines PAYLOAD as `<script type="application/json" id="payload-data">` into `index.html`. After `pixi run viewer`, hard-refresh (Ctrl+Shift+R / Cmd+Shift+R) or the cached HTML serves stale data. Quick check: `PAYLOAD.meta.generated_at` in DevTools.
- **API caching** — delete `data/derived/caches/` to force re-fetch from MyGene / Allen Brain Atlas.

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in `henrichung-cst/alzheimers` via the `gh` CLI. External PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, label strings equal to their role names (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
