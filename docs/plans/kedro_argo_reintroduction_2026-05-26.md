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
2. **Argo via the bioplat convention** (revised 2026-05-27 — supersedes the
   original "via plugins" wording). `kedro-argo` is abandoned (PyPI `0.0.2`,
   kedro-0.16 era) and `kedro-docker` is unnecessary. bioplat already runs
   Kedro projects on Argo with a **thin, hand-authored `WorkflowTemplate`**
   (in `manifests/base/`) whose steps invoke `kedro run --pipeline=<name>
   --env={{workflow.namespace}}` (or `--to-nodes=` for finer steps). Kedro
   owns the inner DAG; Argo owns outer step orchestration. The template +
   Dockerfile + manifests are scaffolded by the **`bioplat-deployer`** agent,
   not a kedro plugin. ⇒ Granularity is the lever via how steps are authored
   (`--pipeline` coarse, `--to-nodes`/`--nodes` finer); the catalog must be
   deployment-portable (containers must resolve every dataset).
   Evidence: bioplat `services/conjugation-portfolio` (parameterized
   `kedro run --to-nodes=$NODE_NAME`) and `services/protein-structure`
   (`steps:` each running `kedro run --pipeline ... --env={{workflow.namespace}}`).
3. **Full DataCatalog** — every cross-node artifact modeled in
   `conf/base/catalog.yml`; nodes take/return objects, Kedro owns I/O.
4. **All four modes** in scope — but formalized methodically, one at a time,
   after the Phase-0 inventory is agreed.
5. **Cohort selection stays `KEDRO_ENV` overlay** (`conf/{full_cohort,human_nbb}/`)
   — already Kedro-native, no change needed. On-cluster this maps to Argo via
   `kedro run --env={{workflow.namespace}}` (confirmed bioplat convention).

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
`cross_reference/evidence.py`. (The `decomposition_mea` WMB-class statistical-deconvolution
cluster — `cohort_concordance`, `confidence`, `factorial_ols`, `load_deconvoluted`,
`snrna_concordance`, `paths`, `per_animal_extension`, `cohort_concordance_audit` — was
deleted 2026-05-29 as a closed path; nothing to factor.)

### ONE-OFF — stay flat scripts, NOT Kedro

These are investigations / reporters / gates with no place in the production DAG:

- `ctrl_outlier_audit/ctrl_outlier_audit.py`, `..._kinases.py`, `..._report_figs.py`,
  `ctrl_outlier_suspect_lfc_table.py`, `human_group_mea_reanalysis.py`
  — human CTRL outlier investigation (moved out of `cross_reference/` 2026-05-29; tied to
  `docs/plans/human_ctrl_outlier_audit_*`). Figures + audit JSON + clean-baseline group MEA.
  **One-off.** (`human_group_mea.py` archived 2026-05-29 — superseded by the reanalysis twin.)
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
  - Kedro owns all Python work — modes 1/2/4, viewer, and Mode-3 prep/post. A
    hand-authored Argo `WorkflowTemplate` (bioplat convention) wires
    `[kedro run prep] → [R engine step] → [kedro run post]`, edges carried by
    **S3 files** (the Q2 catalog). R reads/writes files only; never imports Kedro.
  - Each tool stays in its lane; the R engine is simply another Argo step
    running its own image — no plugin needs to emit R steps.
  - **The six sce4 parity overrides stay untouched** — we containerize and run
    the R script as-is, never rewrite it. `verify_sce4_parity.py` is the gate.
  - **Dependency wrinkle:** the `Incytr` package lives **outside this repo**
    (`~/Projects/work/incytr`); the R image must install it from a **pinned git
    ref / tarball**, and that source must be reachable from the image build.
  - **Optional finer split (see Q4):** `incytr_commandline.R` emits 9 parquets
    (one per contrast). Parameterizing it for a single contrast ⇒ **9 parallel
    R steps** (wall-clock + per-contrast retry). Needs a small R-driver refactor
    (Q5) — pursue if scheduling benefits justify it.
  - Rejected alternative: single Kedro DAG with a per-node R image — irrelevant
    now that orchestration is hand-authored Argo steps, not a plugin.
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
  coarse exception unless the per-contrast split (Q1) is adopted. **Note
  (2026-05-27):** Argo step granularity is independent of Kedro node
  granularity — fine Kedro nodes always exist; the WorkflowTemplate chooses how
  many to bundle per step (`--pipeline` = whole sub-DAG in one step, `--nodes`/
  `--to-nodes` = finer parallel steps). Start coarse (pipeline-per-step); split
  hot/expensive nodes into their own steps only where scheduling pays off.
- **Q5 — Refactoring. DECIDED: refactor freely, discuss as we go.** Any node
  that isn't a clean `inputs → outputs` function gets refactored when we reach
  it (e.g. `build_celltype_decomposition.py`'s flat `main()`; the Mode-3
  per-contrast R split). Refactoring is welcomed, not avoided; each one is
  raised for discussion at its pipeline's pass.

---

## Phase 1 — Restore skeleton as template ✅ DONE 2026-05-27

The strip deleted a complete layout; recover as a **structural reference**
(not drop-in — the reorg renamed `kinase_*.py` → `bulk_mea/*.py`, so old node
files reference dead module paths):

```
git show 0c2a997^:alz/pipeline_registry.py
git show 0c2a997^:alz/settings.py
git show 0c2a997^:conf/base/catalog.yml
git show 0c2a997^:alz/pipelines/<name>/{nodes,pipeline}.py   # 6 subpipelines
```

Re-add deps (revised 2026-05-27 — no plugins). Just `kedro` + `kedro-datasets`
(with S3 support for the Q2 catalog). conda-forge has `kedro` 1.1.1 and the
0.19.x line; pick a version and confirm it matches what bioplat's Kedro services
run. **No `kedro-argo`** (abandoned) and **no `kedro-docker`** (bioplat builds
the image its own way) — so the `click<8.2` cap (which came from `kedro-docker`)
is no longer needed. `rich<14` is still required (kedro 0.19's own bound) and a
`markdown-it-py<4` cap is needed to coexist with `kinase-library` — see the
Phase 1 outcome below. Mirror any `~=` PyPI constraints on the conda side before
`pixi install`. Re-add
`[tool.kedro]` to `pyproject.toml`. The minimal scaffolding (empty registry +
catalog that grow per-pipeline in Phases 2–5) goes in here — `alz/settings.py`,
`alz/pipeline_registry.py` (empty `register_pipelines`), `conf/base/catalog.yml`
(grows incrementally — pre-emptive entries rot).

### Phase 1 outcome (actuals)

- **Version:** `kedro 0.19.14` (`pixi.toml` pin `kedro ~=0.19.11`, matches the
  canonical bioplat Kedro-service line; conda-forge resolved 0.19.14).
  `kedro-datasets >=3,<5`, `s3fs` added. Old skeleton API (`from kedro.pipeline
  import Pipeline, node`) stays valid on 0.19.x.
- **Pin reconciliation (CLAUDE.md trap, hit):** adding kedro pulled conda
  `markdown-it-py` 4.x, but `kinase-library` → `myst-parser>=4.0,<4.1` requires
  `markdown-it-py<4` ⇒ env unsatisfiable. Fixed by mirroring on the conda side:
  `rich >=12,<14` (kedro 0.19 requires rich<14) + `markdown-it-py >=3,<4`.
  `click<8.2` was **not** needed (it came from the now-dropped `kedro-docker`).
- **Flat layout:** repo is `alz/` at root, not kedro's default `src/`. Set
  `source_dir = "."` in `[tool.kedro]`, else `bootstrap_project` raises
  `NotADirectoryError: src cannot be found`.
- **Telemetry:** `.telemetry` → `consent: false` (silences the prompt; kedro
  also auto-added `[tool.kedro_telemetry] project_id` to pyproject — left as-is,
  inert with consent denied).
- **Files written:** `alz/settings.py`, `alz/pipeline_registry.py` (empty
  `register_pipelines` → `{"__default__": Pipeline([])}`), `conf/base/catalog.yml`
  (empty), `[tool.kedro]` block, `.telemetry`, pixi deps + lock.
- **Verification:** `kedro registry list` → `- __default__` (project bootstraps
  cleanly). `kedro run` errors `Pipeline contains no nodes` — **expected**; kedro
  refuses a node-less pipeline. First real node arrives in Phase 2.
- **Not committed** pending user review.

## Phases 2–5 — Per-pipeline formalization (one pass each)

For each CORE pipeline, in dependency order (P1 → P5/P6): decide node
boundaries (Q4) + catalog entries (Q2) → wire `nodes.py`/`pipeline.py` over
existing helpers → register in `pipeline_registry.py` → run end-to-end →
verify against that pipeline's existing gate (sce4 for P4, `verify_decomposition`
for P3, `summary.py` numbers for P2). Mode 3 (P4) blocks on Q1.

### P1 — ingest (song) ✅ DONE 2026-05-27

Scope decisions (user, 2026-05-27): **two pure song nodes only** —
`sample_mapping` + `phospho_match`. Outlier exclusions (`step_outliers` →
`sample_exclusions.csv`) move to **P2/bulk_mea** (reads `stoichiometry_matrix.csv`,
a normalize output — belongs downstream of normalize, not in ingest). The §3
quality diagnostic stays a flat one-off. The human (mukesh) ingest becomes its
**own pipeline, deferred** (its downstream is all one-offs).

**Wiring pattern (sets the precedent for P2–P5):** each song step was split
into a pure `build_*` core (no file I/O — takes pre-loaded inputs, returns
objects) plus a thin CLI wrapper (`step_*`) that loads/saves. Kedro nodes
(`alz/pipelines/ingest/nodes.py`) call the same `build_*` cores; the catalog
(`conf/base/catalog.yml`) owns all I/O. Single implementation, two callers
(Kedro + `song.py` CLI) — not a shim.

- **Cores:** `song.build_sample_mapping(layout, proteome_columns, snrna_samples)`,
  `song.build_phospho_matching(total_proteome, phospho_sitequant, imac_composite)`.
- **Catalog inputs:** `song_sample_list`, `song_total_proteome`,
  `song_imac_sitequant`, `song_imac_composite` (all `pandas.ExcelDataset`, ≤12 MB).
  The snRNA manifest is read inside the node (optional → `{}` when absent), not
  cataloged.
- **Catalog outputs:** `song_sample_mapping`, `song_phospho_protein_matching`
  (`pandas.CSVDataset`), `song_proteome_gene_list` (`text.TextDataset`,
  feeds WMB reference build), `song_matching_summary` (`json.JSONDataset`).
- **Files:** `alz/pipelines/{__init__,ingest/__init__,ingest/nodes,ingest/pipeline}.py`;
  registry now `{"ingest", "__default__": ingest}`; catalog P1 block;
  `alz/ingest/song.py` refactored (cores + wrappers).
- **Verification:** `kedro run --pipeline ingest` → 2/2 nodes OK. All four
  outputs **byte-identical** to the CLI baseline (`sample_mapping.csv`,
  `phospho_protein_matching.csv`, `total_proteome_genes.txt`; `matching_summary.json`
  semantically equal). Refactored CLI (`song.py --phospho-match`) re-verified
  identical — both callers agree. **Not committed** pending user review.

## Phase 6 — Argo authoring + deployment carve-out

No plugin. Hand-author a thin `WorkflowTemplate` (bioplat convention) whose
steps invoke `kedro run --pipeline=<name> --env={{workflow.namespace}}`, plus a
dedicated R-engine step (R container image) for Mode 3, wired
`[prep] → [R] → [post]` over S3 edges. Deployment scaffolding (Dockerfile,
VERSION.txt, `manifests/base` + overlays, build-caller workflow, ArgoCD
pointer, SealedSecrets) is a **separate artifact** with its own lifecycle —
delegated to the **`bioplat-deployer`** agent, not mixed into pipeline code.
Match the live examples: `conjugation-portfolio` (param node/pipeline) and
`protein-structure` (`steps:` per pipeline).

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
- A hand-authored bioplat `WorkflowTemplate` runs the pipelines via
  `kedro run --pipeline=… --env={{workflow.namespace}}` (+ R-engine step),
  with correct step ordering + resource hints.
- One-off scripts remain flat and are not in the DAG.
- No stale Kedro/CLI references left in `docs/foundation/` or `CLAUDE.md`.
```
