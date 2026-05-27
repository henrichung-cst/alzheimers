# Kedro + Argo Reintroduction — 2026-05-26

## Why this reverses the 2026-05-21 strip

Kedro was stripped at commit `0c2a997` because it was overkill for a solo-dev
local repo with a Mode-1-only, half-finished migration. That cost/benefit call
is now void: the **cloud compute infrastructure orchestrates Kedro pipelines
through Argo (Argo Workflows)**. This is the "genuine external compatibility"
exception to the anti-shim rule — a hard platform constraint, not a "switch
back later" hedge. We reintroduce Kedro **in place** (no new repo / folder).

Decision record on why *not* a new folder: the prior skeleton is recoverable
from git history *here*; per-node Argo + a full DataCatalog **wrap existing
node-shaped helpers** rather than rewrite them; and forking would duplicate
code guarded by byte-exact parity gates (sce4 reproduction, mass identity,
ε=0.01) — the exact "two valid states coexist" trap. See chat 2026-05-26.

## Locked constraints (already decided)

1. **In place** — reintroduce into this repo, not a new folder.
2. **Argo via plugins** — `kedro-argo` / `kedro-docker` translate the Kedro DAG
   into an Argo `WorkflowTemplate`. We do **not** hand-author Argo templates.
   ⇒ Node granularity + tagging is the lever; the catalog must be
   deployment-portable (containers must resolve every dataset).
3. **Full DataCatalog** — every cross-node artifact modeled in
   `conf/base/catalog.yml`; nodes take/return objects, Kedro owns I/O.
4. **All four modes** in scope — but formalized methodically, one at a time,
   after the Phase-0 inventory is agreed.
5. **Cohort selection stays `KEDRO_ENV` overlay** (`conf/{full_cohort,human_nbb}/`)
   — already Kedro-native, no change needed.

## Working principle

Slowly and methodically. Phase 0 (this inventory) must be **agreed before any
code**. Each pipeline thereafter gets its own architecture pass — we decide the
node boundaries and catalog entries for that pipeline, implement, verify against
its existing gate, *then* move to the next. Open architectural questions
(below) are resolved one at a time, not pre-committed here.

---

## Phase 0 — Inventory & triage (THE AGREEMENT GATE)

Itemizes what becomes a Kedro pipeline/node vs what stays a flat one-off.
Derived from a full data-flow contract map (agent recon, 2026-05-26).

### CORE — formalize as Kedro pipelines

| # | Pipeline | Nodes (from existing helpers) | Terminal artifact |
|---|----------|-------------------------------|-------------------|
| P1 | **ingest** | `alz/ingest/song.py`, `alz/ingest/mukesh.py` → sample mapping / exclusions / matching | `data_ingest/sample_mapping.csv` |
| P2 | **bulk_mea** | normalize (`step_normalize`) → enrich (`_fit_and_contrast`, `_run_mea`) → attribute (`_assemble_unified`, `_combine_mea_tracks`, `_map_kinases_to_genes`) → recover (`step_attribution_recovery`); mechanism optional | `attribution_recovery/kinase_hypothesis_table.csv` |
| P3 | **decomposition_mea** | proportions (`step_proportions`) → decompose (`build_celltype_decomposition` — *needs factoring*) → enrich_celltype (`_ols_for_cluster`) → per_animal_ols | `decomposition/levy_t5/mea_per_cluster*.parquet` |
| P4 | **incytr_pair** | export (`export_protein`/`export_phospho`) → **[R: `incytr_commandline.R`]** → filter (`filter_one`) → reshape (`reshape`) | `incytr_pair_mode/receiver_cache/` |
| P5 | **reference** (supporting) | snrna pseudobulk, `wmb_expression.py`, atlas reference | `wmb_expression/*.csv`, `snrna_integration/*.csv` |
| P6 | **viewer** (terminal) | `build_unified_viewer.py` | `index.html` payload |

Library modules with no `main()` become node-internal calls, not nodes:
`cross_reference/evidence.py`, `decomposition_mea/{cohort_concordance,confidence,factorial_ols,load_deconvoluted,snrna_concordance}.py`.

### ONE-OFF — stay flat scripts, NOT Kedro

These are investigations / reporters / gates with no place in the production DAG:

- `cross_reference/ctrl_outlier_audit.py`, `..._kinases.py`, `..._report_figs.py`
  — human CTRL outlier investigation (untracked; tied to
  `docs/plans/human_ctrl_outlier_audit_*`). Figures + audit JSON. **One-off.**
- `cross_reference/human_group_mea.py` — clean-CTRL sensitivity. **One-off.**
- `cross_reference/seaad_human_agreement.py` — human SEA-AD agreement. **One-off.**
  (decided 2026-05-26)
- `cross_reference/human_celltype_attribution.py` — human celltype specificity.
  **One-off.** (decided 2026-05-26)
- `bulk_mea/summary.py` — read-only reporter. **One-off.**
- `decomposition_mea/verify_decomposition.py`, `incytr_pair/verify_sce4_parity.py`,
  `integration/verify_pathway_round_trip.py` — parity/verification gates. Run as
  **gates** (CI-style checks or Kedro `after_pipeline_run` hooks), not DAG nodes.

### REFERENCE INPUTS — build-once, modeled as catalog inputs (not DAG nodes)

`alz/integration/*` (cluster spine, bridges, transcript trace, normalized
substrate) are **stable build-once artifacts** (decided 2026-05-26). Generate
outside the production DAG; their outputs (`data/derived/bridges/*`, the
levy_t5 spine, normalized substrate) enter the catalog as **inputs** to P2/P3/P4.
They are not regenerated per pipeline run.

**Phase 0 exit criterion:** the tables above are confirmed by the user. No code
is written until then. (Triage-pending items resolved 2026-05-26.)

---

## Architectural decisions (resolved 2026-05-26)

- **Q1 — R engine strategy (Mode 3). DECIDED: decoupled R container, Argo
  composes.** Mode 3 is a sandwich: Python prep (`export_decomposition_for_pair.py`)
  → **R engine** (`incytr_commandline.R` + the `Incytr` package, entirely R) →
  Python post (`filter_significant_paths.py`, `pair_to_receiver_cache.py`).
  - We do **not** force R into a Kedro Python node. Build a dedicated,
    version-pinned **R container image** (own Dockerfile) with `Incytr` + R
    deps (`DBI`, `duckdb`, `data.table`, `arrow`). The R engine runs as its
    **own Argo step** in that image.
  - Kedro (via the plugin) owns all Python work — modes 1/2/4, viewer, and
    Mode-3 prep/post. A top-level Argo workflow wires
    `[python prep] → [R engine] → [python post]`, edges carried by **S3 files**
    (the Q2 catalog). R reads/writes files only; it never imports Kedro.
  - Each tool stays in its lane; nothing bends the plugin to emit R steps.
  - **The six sce4 parity overrides stay untouched** — we containerize and run
    the R script as-is, never rewrite it. `verify_sce4_parity.py` is the gate.
  - **Dependency wrinkle:** the `Incytr` package lives **outside this repo**
    (`~/Projects/work/incytr`); the R image must install it from a **pinned git
    ref / tarball**, and that source must be reachable from the image build.
  - **Optional finer split (see Q4):** `incytr_commandline.R` emits 9 parquets
    (one per contrast). Parameterizing it for a single contrast ⇒ **9 parallel
    R steps** (wall-clock + per-contrast retry). Needs a small R-driver refactor
    (Q5) — pursue if scheduling benefits justify it.
  - Rejected alternative: single Kedro DAG with a per-node R image via
    `kedro-docker` — more plugin friction, less isolation.
- **Q2 — Catalog portability. DECIDED: S3.** Artifacts live in **S3 cloud
  storage**. Catalog written so the same recipe resolves locally
  (`outputs/reports/...`) and on the cluster (S3) via a swappable base-path /
  env-specific `conf/{local,argo}/catalog.yml`.
- **Q3 — Heavy nodes. DECIDED: precompute once, read in.** `wmb_expression.py`
  (~30 GB RAM), the zstd atlas h5ads, and the `alz/integration/*` reference
  artifacts are **built once, outside the production DAG**, and enter the
  catalog as **inputs**. Not regenerated per run. Shared-box memory rule applies
  to those one-time local builds.
- **Q4 — Node granularity. DECIDED: finer where reasonable.** Prefer fine nodes
  (e.g. split OLS vs MEA) for retry/cache granularity and parallelism, stopping
  short of unreasonable container sprawl on cheap steps. The R engine is the
  coarse exception unless the per-contrast split (Q1) is adopted.
- **Q5 — Refactoring. DECIDED: refactor freely, discuss as we go.** Any node
  that isn't a clean `inputs → outputs` function gets refactored when we reach
  it (e.g. `build_celltype_decomposition.py`'s flat `main()`; the Mode-3
  per-contrast R split). Refactoring is welcomed, not avoided; each one is
  raised for discussion at its pipeline's pass.

---

## Phase 1 — Restore skeleton as template (after Phase 0)

The strip deleted a complete layout; recover as a **structural reference**
(not drop-in — the reorg renamed `kinase_*.py` → `bulk_mea/*.py`, so old node
files reference dead module paths):

```
git show 0c2a997^:alz/pipeline_registry.py
git show 0c2a997^:alz/settings.py
git show 0c2a997^:conf/base/catalog.yml
git show 0c2a997^:alz/pipelines/<name>/{nodes,pipeline}.py   # 6 subpipelines
```

Re-add deps with the CLAUDE.md pins respected: `kedro` (<0.20 ⇒ needs
`rich<15`, `click<8.2`), `kedro-datasets`, plus `kedro-argo` / `kedro-docker`.
Mirror any `~=` PyPI constraints on the conda side before `pixi install`.
Re-add `[tool.kedro]` to `pyproject.toml`.

## Phases 2–5 — Per-pipeline formalization (one pass each)

For each CORE pipeline, in dependency order (P1 → P5/P6): decide node
boundaries (Q4) + catalog entries (Q2) → wire `nodes.py`/`pipeline.py` over
existing helpers → register in `pipeline_registry.py` → run end-to-end →
verify against that pipeline's existing gate (sce4 for P4, `verify_decomposition`
for P3, `summary.py` numbers for P2). Mode 3 (P4) blocks on Q1.

## Phase 6 — Argo generation + deployment carve-out

Generate the `WorkflowTemplate` via the plugin; validate the per-node DAG.
Deployment scaffolding (Dockerfile, manifests, ArgoCD pointer, bioplat
overlays) is a **separate artifact** with its own lifecycle — handled by the
bioplat-deployer conventions, not mixed into the pipeline code.

## Phase 7 — Reconcile stale docs (closes the "confusing files" concern)

- Rewrite `docs/foundation/live_pipeline_contract.md` to the reintroduced
  layout (it currently documents the pre-strip `alz/pipelines/...` paths and
  pre-reorg `alz/data_ingest.py` shims — stale on both axes).
- Update `CLAUDE.md` integration-config note, README run surface, and the
  `alz/runners/*.sh` front doors (or replace with `kedro run` / pixi-task
  aliases). Per anti-shim: the flat-script CLIs either become thin
  `KedroSession` shims or are removed — they do not coexist as a second way to
  run the same pipeline.

## Success criteria

- All CORE pipelines run via `kedro run --pipeline=<name>` and reproduce
  current outputs (parity gates green: sce4, mass identity, ε=0.01).
- `kedro-argo`/`kedro-docker` emit a valid `WorkflowTemplate` with correct
  per-node DAG + resource hints.
- One-off scripts remain flat and are not in the DAG.
- No stale Kedro/CLI references left in `docs/foundation/` or `CLAUDE.md`.
```
