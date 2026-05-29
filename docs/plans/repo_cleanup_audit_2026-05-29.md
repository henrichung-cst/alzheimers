# Repo Cleanup Audit — 2026-05-29

**Status:** AWAITING APPROVAL. No files have been edited, moved, or deleted. This document
is the approval gate; execution happens only after sign-off.

**Scope:** Full repo audit across four buckets — (1) untracked code, (2) cruft/sprawl,
(3) `docs/plans/` triage, (4) README layout sync. Driven by `docs/foundation/repo_retention_policy.md`
(`main` / `supporting` / `archived` taxonomy) and the global anti-shim / closed-paths rules.

**Headline verdict:** The README + `docs/foundation/` remain a genuine canonical contract —
the 4-component analysis program (bulk mouse, Incytr pair-mode, human NBB/Mukesh, T-cell cohort)
is correctly described. The drift is **organizational, not analytical**: two whole live
workstreams (Kedro reintroduction, T-cell cohort + viewer) are untracked in git and absent
from the README "Repository Layout" block, and there is accumulated local cruft. Nothing
analytical is wrong; the map is just stale.

---

## Bucket 1 — Untracked code (assess → recommend)

Every untracked code file is either live production wired to a pixi task, or a completed
one-off investigation whose outputs are already on disk. **There are no abandoned/DELETE
candidates.** One ARCHIVE candidate.

### 1a. Kedro/Argo reintroduction — ADD (all)

Fully wired: `pyproject.toml` has `[tool.kedro]`; `pixi.toml` pins `kedro ~=0.19.11` +
`kedro-datasets` + `s3fs`; `pipeline_registry.py` registers the `ingest` pipeline; P1 runs
2/2 nodes. P2–P6 are intentionally deferred (registering nonexistent pipelines would break
`kedro run`) — incremental scaffolding per `kedro_argo_reintroduction_2026-05-26.md`, not abandonment.

| File | Verdict |
|---|---|
| `alz/pipeline_registry.py` | ADD |
| `alz/settings.py` | ADD |
| `alz/pipelines/__init__.py`, `pipelines/ingest/{__init__,nodes,pipeline}.py` | ADD |
| `conf/base/catalog.yml` | ADD |

> Note: `conf/base/parameters.yml` still carries a stale comment "Not a Kedro project — see
> docs/plans/repo_organization_2026-05-21.md". Should be corrected during execution (Phase 7
> of the Kedro plan).

### 1b. T-cell viewer — ADD (all)

Wired via `pixi.toml` `tcell-viewer = "python alz/build_tcell_viewer.py"`. Builder is complete
(1402 lines). The `template/` JS is an **adapted lift** of `alz/viewer/` (shared structure,
T-cell-specific tabs/coloring; mouse-only tabs dropped) — consistent with the "port = lift
verbatim, adapt where the data differs" rule, not a greenfield reimplementation.

| File | Verdict |
|---|---|
| `alz/build_tcell_viewer.py` | ADD |
| `alz/tcell_viewer/{__init__,paths}.py` | ADD |
| `alz/tcell_viewer/template/*` | ADD |

### 1c. T-cell cohort code — ADD (all but one)

Every file below maps directly to an existing pixi task:

| File | pixi task | Verdict |
|---|---|---|
| `alz/ingest/tcells_projectils_map.R` | `tcells-projectils-map` | ADD |
| `alz/ingest/tcells_scrna_extract.R` | `tcells-scrna-extract` | ADD |
| `alz/ingest/tcells_decompose.py` | `tcells-decompose` | ADD |
| `alz/incytr_pair/build_tcells_seurat.R` | `tcells-build-incytr-seurat` | ADD |
| `alz/incytr_pair/build_tcells_input_gene_list.R` | `tcells-build-input-gene-list` | ADD |
| `alz/incytr_pair/run_pair_mode_tcells.sh` | `tcells-incytr` | ADD |
| `alz/runners/supporting/install_projectils.sh` | `install-projectils` | ADD |
| `alz/runners/supporting/tcells_projectils_map.sh` | `tcells-projectils-map` | ADD |
| `alz/runners/supporting/tcells_scrna_extract.sh` | `tcells-scrna-extract` | ADD |
| `alz/ingest/probe_tcells_scrna.R` | *(none)* | **ARCHIVE** |

`probe_tcells_scrna.R` is a one-shot metadata reconnaissance script (D4 in the meeting-notes
triage) with no pixi task and no importers; its job is done. → `archive/tcells_probe_2026-05-27/`.

### 1d. Other untracked cross_reference code — ADD (both)

| File | pixi task | Verdict |
|---|---|---|
| `alz/cross_reference/human_group_mea_reanalysis.py` | `mea-suspect-reanalysis` | ADD |
| `alz/cross_reference/ctrl_outlier_suspect_lfc_table.py` | *(none, but outputs tracked)* | ADD |

`ctrl_outlier_suspect_lfc_table.py` has no task but is the canonical generator of
`suspect_vs_ad_lfc_*.csv` referenced in `outputs/.../ctrl_audit/INDEX.md`; leaving it untracked
while its outputs are referenced is inconsistent.

---

## Bucket 2 — Cruft / sprawl

All items below are **untracked and already gitignored** (root `*` rule) unless noted, so git
history is unaffected — these are local-disk hygiene actions. Tiered by confidence.

### Tier A — safe deletes (empty / generated / stray, gitignored)

| Path | Why |
|---|---|
| `logs/` | empty |
| `.tmp/` | empty |
| `.agents/` | empty |
| `.codex` | 0-byte empty file |
| `pipeline_notes/` (1 file `tcell_payload_report.md`) | machine-generated build-size report, not a doc |
| root `__pycache__/` (3 stale SAP-profiling `.pyc`) | dead artifacts from a closed path |

### Tier B — likely deletes, confirm first (tool scaffolding, gitignored)

These are tool configs that leaked into the project root. **Flagging rather than asserting** —
delete only if you're not using these tools locally. `latexmkrc` + `.texmf-*` relate to LaTeX/Quarto
rendering of `docs/pipeline_overview.qmd`, so they may be live.

| Path | Caveat |
|---|---|
| `robots.txt` | no web server here; almost certainly stray |
| `latexmkrc`, `.texmf-config/`, `.texmf-home/`, `.texmf-var/` | LaTeX build config — keep if you render `.qmd`/PDFs locally |
| `.rtk/`, `.antigravitycli/`, `.telemetry` | other-tool metadata — keep if those tools are in use |

### Tier C — move tracked files out of the source tree

| Path | Action |
|---|---|
| `alz/decomposition_mea/_archive/` (6 tracked files: WMB-34 crosswalks — **closed path**) | → `archive/decomposition_mea/` (git mv) |
| `alz/decomposition_mea/_results/{cohort_concordance_calibration,variance_audit}.md` | → `docs/archive/` (audit result docs don't belong under `alz/`) |

### Tier D — archive consolidation (no git change; bulk untracked)

- `archive/deconv/` (636 KB, 0 tracked) appears to duplicate `archive/deconvolution/` (3 tracked).
  Consolidate to eliminate the duplicate dir.
- `archive/` is 5.5 GB, almost all untracked historical data (`pre_levy19` 4 GB, `incytr_factorial_inputs`
  1.5 GB). Correct per CLAUDE.md ("prior output artifacts may stay on disk"). **No action** — noted for completeness.

### Bucket 2 also surfaced a *tracking gap* (the opposite of cruft)

- **`bench/bench.md`** (the parity-audit history that CLAUDE.md cites as authoritative) is silently
  untracked because `bench/` is swallowed by the `*` rule. **Recommend: whitelist + track `bench/bench.md`.**
  (The 55.6 MB `bench/sce4_*.csv` reference table and `bench/perf/` 226 MB should stay untracked.)

---

## Bucket 3 — `docs/plans/` triage (24 files)

**6 ACTIVE (keep) · 18 archivable → `docs/archive/`.** Evidence (commit / on-disk feature) was
verified per file by the triage pass.

### KEEP (active)
- `kedro_argo_reintroduction_2026-05-26.md` — P2–P7 outstanding
- `levers_AB_implementation_2026-05-28.md` — Lever A/B not yet shipped
- `meeting_notes_triage_2026-05-27.md` — master stream tracker (A/B/C/D)
- `tcells_incytr_pair_2026-05-28.md`, `tcells_incytr_run_2026-05-28.md`, `tcells_percell_aggregation_2026-05-28.md` — T-cell in-flight
- `tcell_viewer_lift_2026-05-29.md`, `tcell_viewer_payload_2026-05-29.md` — viewer in-flight (counts toward active)

### MOVE → `docs/archive/` (completed / superseded)
`critical_audit_fixes_2026-05-22`, `human_ctrl_outlier_audit_2026-05-25` (+`_findings`),
`incytr_significance_filter_2026-05-26`, `memory_investigation_2026-05-28`,
`optimization_levers_2026-05-28`, `pairmode_memory_audit_2026-05-27` (+`_retrospective_2026-05-28`),
`pairmode_perf_oom_2026-05-25`, `pathway_storage_redesign_2026-05-26` (superseded — no redesign adopted),
`primary_method_divergence_2026-05-24`, `repo_organization_2026-05-21`,
`roundtrip_substrate_staleness_2026-05-26`, `sce4_decomposition_reconciliation_2026-05-24`,
`sce4_fix_propagation_2026-05-23`, `suspect_ctrl_mea_reanalysis_2026-05-27`.

**Consolidation note:** the four perf/memory plans are one investigation arc closed by the
retrospective (`bench/bench.md` is the living summary) — archive them as a unit.

---

## Bucket 4 — README "Repository Layout" sync (last step, after 1–3 land)

The layout block in `README.md` (lines 171–201) omits live dirs. Proposed edits:

- Add `alz/pipelines/` + `pipeline_registry.py` + `settings.py` (Kedro), with `conf/base/catalog.yml` under `conf/`.
- Add `alz/tcell_viewer/` alongside `alz/viewer/`; note both `build_unified_viewer.py` and `build_tcell_viewer.py`.
- `alz/supplementary/` exists as a sibling of `runners/supplementary/` — clarify or de-dupe in the layout.
- Reflect Tier-C moves (`decomposition_mea/_archive`, `_results` leaving the source tree).

No analytical sections of the README need changing — the 4-component program and key-outputs
tables are current.

---

## Proposed execution order (on approval)

1. **Bucket 1 ADDs** — `git add` the live Kedro / T-cell / viewer / cross_reference files; fix the
   stale `parameters.yml` comment. (One `feat:`/`chore:` commit, logically grouped.)
2. **Bucket 1 ARCHIVE** — `git mv probe_tcells_scrna.R → archive/tcells_probe_2026-05-27/`.
3. **Bucket 2 Tier C** — `git mv` the two `decomposition_mea/` subdirs out of the source tree.
4. **Bucket 2 tracking gap** — whitelist + track `bench/bench.md`.
5. **Bucket 3** — `git mv` the 18 archivable plans into `docs/archive/`.
6. **Bucket 2 Tier A** — delete empty/generated cruft. **Tier B only after you confirm** the tools are unused.
7. **Bucket 4** — update README layout; refresh `repo_retention_policy.md` inventory if needed.

### Open decisions for you
- **Tier B deletes** (LaTeX / tool-config dirs): delete or keep? (Default: keep `latexmkrc`/`.texmf-*`, delete `robots.txt`/`.codex`.)
- **Plans archive location**: `docs/archive/` (existing) vs a dated subdir `docs/archive/plans_2026-05/`?
- Commit granularity: one commit per bucket, or one cleanup commit?
