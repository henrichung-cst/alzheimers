---
phase: 5 of 5 (per CURRENT_SPRINT.md §Sequencing)
status: audit (awaiting approval)
created: 2026-05-09
---

# Phase 5 — Doc Sweep

## Context

CURRENT_SPRINT.md §Sequencing item 6 scopes Phase 5 to a doc sweep:
> Reconcile `docs/foundation/*` and `docs/integrations/*` with the contracted
> shape. Archive what's no longer load-bearing. Foundation charter
> incorporates the Kedro decision.

Phase 4 (4a–4d) is complete: all five live-arc kinase modules
(`normalize`, `enrich`, `attribute`, `mechanism`, `recovery`) are now Kedro
pipelines under `alz/pipelines/`, registered in `alz/pipeline_registry.py`,
with cohort selection driven by `params:analysis_mode`. The legacy
integration tree (`alz/integration/{wrappers,adapters,sidecar,tests}/` plus
all orchestrator shell scripts) was relocated to
`~/Projects/work/incytr_integration_archive/` on 2026-05-08, leaving only
Phase 1 stubs in-tree.

Many docs in `foundation/` and `integrations/` were authored before either
of those moves and now describe code paths that no longer exist.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Doc-by-doc verdict matrix first**, edits second | CLAUDE.md *audit-first* rule; the integrations/ folder is dominated by 100 KB+ of audit forensics about archived code — bulk archival is the right tool, but only after explicit approval. |
| 2 | **Verdicts: KEEP-AS-IS / REFRESH / ARCHIVE / DELETE.** ARCHIVE means move to `docs/archive/<filename>` with a one-line frozen-snapshot header; DELETE is reserved for self-declared obsolete files with no historical value. | Three-state isn't enough: the `incytr_audit_*` series is closed forensic gold (archive); `integrations-structure.md` self-labels as "OUT OF DATE — IGNORE" (delete). |
| 3 | **Refresh = surgical text edits to live docs**, not rewrites. Update file:line references that point to dead code; do not re-design any live spec. | The methodology (concordance model, statistical constraints, charter) is correct; only the implementation pointers rotted. |
| 4 | **`kinase_incytr_integration.md` gets the heaviest refresh** | CLAUDE.md names it the source of truth, but it documents archived code. Remediation plan owns the new architecture. The doc must shrink to a thin pointer + Phase 1 stub description. |
| 5 | **Archive before refresh for high-value docs.** Snapshot `kinase_incytr_integration.md` to `docs/archive/kinase_incytr_integration_pre_remediation.md` before the rewrite. | The shadow-fork architecture was a real piece of work; the historical record stays accessible. |
| 6 | **No code changes in Phase 5.** | Doc sweep only. The `sys.path` bridge in `alz/__init__.py` (still needed by ~7 supporting modules) is out of scope — that's a code phase. |

## Out of scope

- Retiring the `sys.path` bridge in `alz/__init__.py` (requires migrating ~7 supporting modules to package-relative imports — separate phase).
- Rewriting `alz/runners/main/run_kinase_attribution.sh` to call `kedro run` directly (works as-is via the CLI shims; cosmetic).
- Touching `docs/foundation/working/`, `docs/archive/` (existing archive content), or anything outside `docs/foundation/` / `docs/integrations/`.
- Updating CLAUDE.md or README.md beyond the minimum needed to remove dead doc pointers (Phase 3 + 4 already updated the live arc).

## Verdict matrix

### Foundation docs (`docs/foundation/`)

| File | Verdict | Stale references | Action |
|---|---|---|---|
| `analysis_charter.md` | **KEEP-AS-IS** | None — charter is implementation-agnostic. | No edits. |
| `live_pipeline_contract.md` | **REFRESH** | Lines 8–11 (shell-runner front door); §66–128 references nonexistent `kinase_attribution.py` monolith; §84–115 describes 7-param OLS / 3 contrasts (live: 11-param, 9 contrasts); line 132 (`--mechanism-annotation` flag); lines 154–155 (`final_attribution_table.csv` — dead, replaced by `kinase_hypothesis_table.csv`). | Rewrite the stage breakdown to reflect the 5 split pipelines + Kedro entry points; correct OLS shape; rename canonical deliverable. |
| `concordance.md` | **REFRESH** | Lines 234–238 ("Implementation locations" table) — all four rows point to `kinase_attribution.py`. | Update table to point to `alz/pipelines/attribute/{nodes,pipeline}.py` and `alz/kinase_attribute.py` (CLI shim). Methodology body untouched. |
| `analysis_rationale.md` | **KEEP-AS-IS** | None — methodology only. | No edits. |
| `statistical_constraints.md` | **KEEP-AS-IS** | None — math facts about the design. | No edits. |
| `multiple_testing.md` | **REFRESH** | Lines 24–26, 63, 105 reference dead `kinase_attribution.py` / `step_enrich`. Lines 30, 32–33, 68 reference `integration/adapters/aggregate_factorial.py`, `aggregate_cross_pair.py`, `compute_kinase_support.py` — all moved to `incytr_integration_archive/` on 2026-05-08. | Update live-pipeline rows to point to `alz/pipelines/{enrich,attribute}/`; mark the three `integration/adapters/` rows "archived 2026-05-08 — see incytr_remediation_plan.md". |
| `repo_retention_policy.md` | **REFRESH** | Line 47 (`alz/kinase_attribution.py` listed as main code — file does not exist); line 86 (`alz/integration/**` listed as supporting integration — surface is now Phase 1 stubs only). | Replace the `kinase_attribution.py` entry with the split modules + `alz/pipelines/` subtree; update `integration/**` entry to reflect stub-only state. |

### Integration docs (`docs/integrations/`)

| File | Verdict | Notes | Action |
|---|---|---|---|
| `kinase_incytr_integration.md` | **REFRESH** (preceded by archive snapshot) | Documents the archived shadow-fork architecture as if live. §§3–7 (component inventory, runtime modes, invocation) all reference relocated files. CLAUDE.md still names it source of truth — direct contradiction. | (1) Snapshot current content to `docs/archive/kinase_incytr_integration_pre_remediation.md`. (2) Rewrite as a thin pointer to `incytr_remediation_plan.md` + description of the in-tree Phase 1 stubs (`factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh`) + `config_integration.py`. |
| `incytr_audit_ledger.md` (56 KB) | **ARCHIVE** | Closed 2026-05-05; audits 40+ shadow-fork commits all now in archive tree. Forensic gold but no forward utility. | Move to `docs/archive/`; prepend one-line frozen-snapshot header. |
| `incytr_audit_pre_diff.md` (22 KB) | **ARCHIVE** | Sprint 0 diff between upstream `93b9881` and shadow-fork; all 5 sprint addenda closed. | Move to `docs/archive/`. |
| `incytr_audit_plan.md` (20 KB) | **ARCHIVE** | Working plan for the 5-sprint shadow-fork audit; all sprints checked off. | Move to `docs/archive/`. |
| `incytr_invocation_audit.md` (17 KB) | **ARCHIVE** | Column-family comparison + remediation recommendations; recommendations subsumed by `incytr_remediation_plan.md`. | Move to `docs/archive/`. |
| `incytr_layer_inventory.md` (16 KB) | **ARCHIVE** | Cleanup triage of optional/parked layers; Phases 1–3 complete on now-archived code. | Move to `docs/archive/`. |
| `incytr_audit_commit_list.md` (6 KB) | **ARCHIVE** | Raw commit enumeration feeding the ledger; no forward content. | Move to `docs/archive/`. |
| `integrations-structure.md` (5 KB) | **DELETE** | Self-labeled "OUT OF DATE — IGNORE"; describes retired `data/gdrive_shared/` mounts; live ingest pattern already documented in CLAUDE.md §"Layer-2 drive access". | Delete outright. No archive value — header already says ignore. |

### Pointer hygiene

After the moves, search-and-update inbound references:

- `docs/INDEX.md` — likely indexes some of the moved/archived files.
- `CLAUDE.md` — line ~`docs/integrations/kinase_incytr_integration.md` ("Source of truth for the kinase ↔ Incytr integration") needs to either change to point at the rewritten thin doc or shift the source-of-truth label to `incytr_remediation_plan.md`.
- `README.md` — verify no inbound references to archived integration docs.

## Concrete steps

### Step 1 — Archive the closed `incytr_audit_*` series (5 files)

```bash
git mv docs/integrations/incytr_audit_ledger.md docs/archive/
git mv docs/integrations/incytr_audit_pre_diff.md docs/archive/
git mv docs/integrations/incytr_audit_plan.md docs/archive/
git mv docs/integrations/incytr_invocation_audit.md docs/archive/
git mv docs/integrations/incytr_audit_commit_list.md docs/archive/
git mv docs/integrations/incytr_layer_inventory.md docs/archive/
```

Then prepend a one-line frozen-snapshot header to each (sed in-place):

```
> **Archived 2026-05-09.** This document covers the legacy shadow-fork
> integration code at `alz/integration/{wrappers,adapters,sidecar,tests}/`
> + orchestrator shells, all relocated to
> `~/Projects/work/incytr_integration_archive/` on 2026-05-08. Forward-
> looking guidance lives in `docs/incytr_remediation_plan.md`.
```

### Step 2 — Delete `integrations-structure.md`

```bash
git rm docs/integrations/integrations-structure.md
```

### Step 3 — Snapshot then rewrite `kinase_incytr_integration.md`

```bash
git mv docs/integrations/kinase_incytr_integration.md docs/archive/kinase_incytr_integration_pre_remediation.md
```

Prepend frozen header (same template). Then create a new fresh
`docs/integrations/kinase_incytr_integration.md` (~100 lines) covering:

- Architecture pointer (one paragraph): math now lives upstream in
  `../incytr`; this repo holds an AD-specific shell that calls it.
- In-tree inventory: `config_integration.py` (paths/thresholds/contrasts),
  Phase 1 stubs (`factorial.R`, `load.R`, `persist.R`, `views.sql`,
  `run_factorial.sh`).
- Pixi tasks (`install-incytr`, `incytr-factorial`).
- R deps still required (`Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`).
- Pointer to `docs/incytr_remediation_plan.md` for the architectural plan.
- Pointer to `docs/archive/kinase_incytr_integration_pre_remediation.md`
  for the legacy architecture (historical reference only).

### Step 4 — Refresh `live_pipeline_contract.md`

Surgical edits, not rewrite:

- §"Front door" — replace shell-runner table with `kedro run --pipeline=<name>` table covering `normalize`, `enrich`, `attribute`, `mechanism`, `recovery`. Keep pixi task aliases as a row.
- §"Stage 2: Kinase Attribution" → split into three sub-sections (Normalize / Enrich / Attribute) reflecting the actual module split. Update implementation pointers to `alz/pipelines/{normalize,enrich,attribute}/` and `alz/kinase_{normalize,enrich,attribute}.py` shims.
- §"OLS model" — update to 11-param factorial / 9 time-resolved contrasts (matches `alz/kinase_enrich.py:CONTRAST_COEFS`). The `kinase_enrich.py:_build_design_matrix` is the source of truth; mirror its parameter names.
- §"Mechanism annotation" — replace `--mechanism-annotation` flag invocation with `kedro run --pipeline=mechanism`.
- §"Outputs" — replace `final_attribution_table.csv` with `kinase_hypothesis_table.csv`.

### Step 5 — Refresh `concordance.md` (table only)

Lines 234–238: update implementation locations table.

| Function | New location |
|---|---|
| `_compute_effective_concordance` | `alz/kinase_attribute.py` |
| `_assign_confidence_and_basis` | `alz/kinase_attribute.py` |
| Pipeline orchestration | `alz/pipelines/attribute/{nodes,pipeline}.py` |
| CLI shim | `alz/kinase_attribute.py:main` |

### Step 6 — Refresh `multiple_testing.md`

- Live-pipeline rows (24–26, 63, 105): redirect to `alz/pipelines/enrich/` and `alz/pipelines/attribute/`. Add a one-line note that `_bh_fdr` lives in `alz/kinase_enrich.py`.
- Three `integration/adapters/` rows (30, 32–33, 68): replace path with annotation `archived 2026-05-08 — see docs/archive/kinase_incytr_integration_pre_remediation.md`.

### Step 7 — Refresh `repo_retention_policy.md`

- Line 47: replace `alz/kinase_attribution.py` row with the four split modules (`kinase_normalize.py`, `kinase_enrich.py`, `kinase_attribute.py`, `kinase_mechanism.py`) + `alz/pipelines/` subtree.
- Line 86: update `alz/integration/**` row — surface is now Phase 1 stubs (`factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh`) + `config_integration.py`. Cross-reference `incytr_remediation_plan.md`.

### Step 8 — Pointer hygiene

```bash
# After all moves, sweep:
grep -rn "incytr_audit_" docs/ CLAUDE.md README.md
grep -rn "integrations-structure" docs/ CLAUDE.md README.md
grep -rn "kinase_incytr_integration" docs/ CLAUDE.md README.md
```

Update any inbound references to archived files; in CLAUDE.md update the
"Source of truth" line for kinase ↔ Incytr to point at the rewritten thin
doc + remediation plan.

Update `docs/INDEX.md` to remove archived entries and add a one-line note
for the new thin `kinase_incytr_integration.md`.

### Step 9 — Commit

Conventional commits per logical unit:

1. `docs: archive closed incytr_audit_* series and obsolete integrations-structure.md` — Steps 1–2.
2. `docs: snapshot legacy kinase_incytr_integration to archive; rewrite as thin pointer + Phase 1 stub inventory` — Step 3.
3. `docs(foundation): refresh live_pipeline_contract for Phase-4 Kedro pipelines + 9-contrast factorial` — Step 4.
4. `docs(foundation): refresh implementation pointers in concordance, multiple_testing, repo_retention_policy` — Steps 5–7.
5. `docs: pointer hygiene — update CLAUDE.md, README.md, INDEX.md after Phase 5 moves` — Step 8.

## Critical files

| File | Change |
|---|---|
| 6 × `docs/integrations/incytr_*` | Move to `docs/archive/` + frozen header |
| `docs/integrations/integrations-structure.md` | Delete |
| `docs/integrations/kinase_incytr_integration.md` | Snapshot to archive, rewrite as thin pointer (~100 lines) |
| `docs/foundation/live_pipeline_contract.md` | Stage breakdown rewrite + OLS correction + invocation refresh |
| `docs/foundation/concordance.md` | Implementation-locations table only (lines ~234–238) |
| `docs/foundation/multiple_testing.md` | Implementation pointers (lines 24–26, 30, 32–33, 63, 68, 105) |
| `docs/foundation/repo_retention_policy.md` | Lines 47 + 86 |
| `CLAUDE.md`, `README.md`, `docs/INDEX.md` | Inbound-reference hygiene |

## Risks

- **`kinase_incytr_integration.md` is named the source of truth in CLAUDE.md.** Replacing it with a thin pointer means CLAUDE.md must update in the same commit, or there's a window where docs disagree.
- **Archive moves change git history blame** for the moved files (now git mv'd to archive). Acceptable — CURRENT_SPRINT.md sanctions this.
- **Inbound references** outside `docs/`, `CLAUDE.md`, `README.md` (e.g., comments in shell scripts, R headers) may reference the doc paths. The Step 8 grep sweep should be widened to the full repo if any are found.
- **Phase 1 stub state may drift** — the in-tree `factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh` are described as "incomplete; awaiting the production package API in `../incytr`" (CLAUDE.md). The new thin `kinase_incytr_integration.md` should mirror that disclaimer.

## Verification

Definition of done:

- `ls docs/integrations/` returns only `kinase_incytr_integration.md` (the new thin version).
- `ls docs/archive/` includes the 6 archived `incytr_*` files plus `kinase_incytr_integration_pre_remediation.md`.
- `grep -rn "alz/kinase_attribution\.py" docs/` returns 0 hits (excluding archive/).
- `grep -rn "integration/adapters/" docs/` returns hits only inside `docs/archive/` or annotated archive lines.
- `grep -rn "incytr_audit_" CLAUDE.md README.md docs/INDEX.md` returns 0 hits.
- CLAUDE.md "Source of truth" line for kinase ↔ Incytr points at the rewritten doc and `incytr_remediation_plan.md`.
- Spot-check by reading: `live_pipeline_contract.md` describes the 5 Kedro pipelines, the 11-param/9-contrast OLS, and `kinase_hypothesis_table.csv` as the canonical deliverable.
- 5 commits land on `incytr-cleanup`.
- CURRENT_SPRINT.md §Sequencing item 6 marked done.

## Approval

Awaiting user approval of:

1. The 8 verdicts (foundation table + integrations table).
2. The Step 3 strategy (snapshot legacy `kinase_incytr_integration.md` to archive, then write a fresh thin replacement) — the alternative would be a heavy in-place refactor of the existing doc.
3. The 5-commit decomposition (alternative: collapse into 2–3 commits if preferred).
