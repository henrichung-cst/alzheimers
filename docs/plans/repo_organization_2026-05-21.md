# Repo Organization Plan — 2026-05-21

## Broader Goal

Reach a state where the repo has a clean **two-layer architecture**:

- **Layer 1 — bespoke ingest (per-dataset, one-time):** dataset-specific
  modules (`ingest_song.py`, `ingest_mukesh.py`, `ingest_lucie.py`, …) that
  know each collaborator's quirks and emit a small set of **canonical
  artifacts** with a stable, documented schema.
- **Layer 2 — shared analysis pipelines (data-agnostic, kedro):** the four
  primary analysis modes consume only canonical artifacts. Cohort identity is
  carried in parameters, not hardcoded in nodes.

The four shared analysis modes are:

1. **Bulk MEA** (mouse + human) — IRS normalize → factorial OLS → MEA on
   stoichiometry β. Today: kedro pipelines for mouse only; human runs via
   shell scripts.
2. **Decomposition MEA** (mouse only — requires matched snRNA) — pseudobulk →
   per-(animal, cluster, gene) proportions → forward-projected per-cluster
   bulk → per-cluster factorial OLS + MEA.
3. **Incytr pair-mode** (mouse only — requires snRNA) — 31² sender × receiver
   pair-mode on the Levy-t5 spine, scored by `|PDS|` (pvalue untrustworthy).
4. **Cross-reference correlation** — map outside references (SEA-AD, WMB,
   HBCA, Song within-cohort) to the Levy-t5 spine via 1-hop bridges and check
   directional agreement with MEA outputs.

## Working Principle

Tackle items **slowly and methodically**. When a blocker surfaces while doing
work toward the broader goal, **stop and capture it as the next item to
address** — but don't ignore it, and don't pretend it doesn't exist. Always
hold the broader thread; never let blockers reset the plan.

## Phase Order

### Phase 1 — Write down the plan (this file)

Done. This document is the durable artifact so the work survives session
breaks and compaction.

### Phase 2 — Clean up `docs/` folder

The `docs/` tree itself has accumulated cruft and needs structural pruning
before more docs land in it. Concrete tasks (to be expanded once we audit):

- Inventory `docs/`: current contents vs. `docs/INDEX.md` claims.
- Identify stale plans (`docs/plans/`), orphaned audits, archived material
  still mixed with live docs.
- Reconcile `docs/INDEX.md` with reality.
- Move stale plans to `docs/plans/archive/` (or delete if superseded).
- Confirm `docs/foundation/` only holds authoritative live specs.

Do NOT add new authoritative docs (canonical contract spec, etc.) until this
cleanup is done — otherwise we're piling on top of the same mess.

### Phase 3 — Address factorial fragments

From the analysis-mode inventory (see prior session notes / chat), the
following code paths still carry the Song-specific 4-genotype × 3-timepoint
factorial vocabulary as hardcoded constants:

- `alz/kinase_enrich.py` — `GENOTYPE_CODING`, `CONTRAST_COEFS`
- `alz/decomposition/enrich_celltype.py` — reuses the same coding
- `alz/snrna_integration.py` — `SAP_FACTORIAL`
- `alz/incytr_pair/export_decomposition_for_pair.py` — `GENO_DECODE`, `ANIMAL_RE`

Plus the SAP vocabulary (`WTyp/AppP/Ttau/ApTt`, `ma/fe`, `2mo/4mo/6mo`)
appears across ≥7 files spanning modes 1–3.

The goal here is to **finish the factorial cleanup we already started** —
not to design a new abstraction. Concretely:

- Find the single canonical source of the genotype × timepoint × sex coding
  (probably `alz/config.py` or a new `alz/factorial.py`).
- Remove duplicate definitions in the four files above.
- Make sure the dual-track runner (`males_only` + `full_cohort`) still
  resolves correctly through the Kedro parameter system, not through
  hardcoded env vars.
- Cross-check that nothing references the deleted factorial Incytr code
  (already deleted at upstream commit `424119f` — see CLAUDE.md gotcha).

This phase is **scoped to Song mouse cohort vocabulary**. A separate, more
ambitious effort to make the factorial design parametric for arbitrary
cohorts is part of Phase 4, not here.

### Phase 4 — One organizational pattern: subdirectory per mode

**Pivot 2026-05-21.** Initial direction was a kedro migration (4a contract +
4b wiring). After surveying the actual repo state we concluded kedro is
overkill for a solo-dev research repo with four pipelines, and the parallel
kedro entry points (added for Mode 1 only) duplicate the flat-script CLI
without adding enough value to justify finishing. Decision: **stop the kedro
migration and adopt one consistent organizational style for all four modes**
— the subdirectory pattern that already works for `alz/decomposition_mea/`.

Target layout:

```
alz/
  bulk_mea/                 # Mode 1 — flat scripts (was alz/kinase_*.py + alz/pipelines/)
  decomposition_mea/        # Mode 2 — renamed from alz/decomposition/
  incytr_pair/              # Mode 3 — renamed from alz/incytr_pair/ + reshape moved in
  cross_reference/          # Mode 4 — carved out (currently scattered)
  ingest/                   # Layer 1 — bespoke per-dataset modules
  reference/                # shared atlas/expression (feeds 1+4)
  shared/                   # config.py + cross-mode utilities
  integration/              # cross-mode glue (cluster spine, bridges, omics trace)
  viewer/                   # unchanged
  runners/                  # unchanged shell entry points
```

Order is documented in per-subdir READMEs, not filename prefixes. Per-cohort
parameters live in plain YAML loaded by ingest modules (no kedro
env-overlay; cohort identity propagates via CLI args).

**4a (done 2026-05-21, superseded but retained).** Wrote
`docs/foundation/cohort_contract.md`. Still useful as the canonical I/O
schema spec independent of which framework reads it.

**Migration order (tracked as TaskCreate tasks #12–18):**

1. **Strip kedro (done 2026-05-21).** Three-pass strip:
   - Pass 1: replaced `OmegaConfigLoader` with plain `yaml.safe_load` in
     `alz/config._load_analysis_mode`, decoupling module-load from kedro.
   - Pass 2: rewrote `main()` in `kinase_normalize.py`, `kinase_enrich.py`,
     `kinase_attribute.py`, `kinase_mechanism.py`, `attribution_recovery.py`
     to do direct orchestration (load inputs from disk → call existing
     `step_*` helpers → save outputs). `_fit_and_contrast` inlined into
     `kinase_enrich.py` after the deleted `pipelines/enrich/nodes.py`
     was found to be its only home.
   - Pass 3: deleted `alz/pipelines/`, `alz/pipeline_registry.py`,
     `alz/settings.py`, `conf/base/catalog.yml`. Removed `[tool.kedro]` +
     `[tool.kedro_telemetry]` from `pyproject.toml`. Removed `kedro` +
     `kedro-datasets` from `pixi.toml`. Stripped kedro-only keys
     (`proof_marker`, `track_st`, `track_py`, `contrast_coefs`) from
     `conf/base/parameters.yml`.
   - Verified: `python alz/attribution_recovery.py` and
     `python alz/kinase_enrich.py` both run end-to-end with no regressions
     (after regenerating stale pre-`abd394a` `sample_mapping.csv`).
2. **Move Mode 1 → `alz/bulk_mea/` (next, task #13).** Renames:
   - `kinase_normalize.py` → `alz/bulk_mea/normalize.py`
   - `kinase_enrich.py` → `alz/bulk_mea/enrich.py`
   - `kinase_attribute.py` → `alz/bulk_mea/attribute.py`
   - `kinase_mechanism.py` → `alz/bulk_mea/mechanism.py`
   - `kinase_summary.py` → `alz/bulk_mea/summary.py`
   - `attribution_recovery.py` → `alz/bulk_mea/recover.py`

   Then: update every import site (decomposition/enrich_celltype, viewer,
   supplementary scripts, runners), update pixi-task command paths
   (`pixi.toml` lines for `normalize`/`enrich`/`attribute`/`mechanism`/
   `recover`), update `alz/runners/main/*.sh` paths, write
   `alz/bulk_mea/README.md` documenting the order
   `normalize → enrich → attribute → recover` (+ optional `mechanism`,
   read-only `summary`). Smoke-test `pixi run live` end-to-end.

   Known import surfaces to update (from `grep -l kinase_normalize\|
   kinase_enrich\|kinase_attribute\|kinase_mechanism\|attribution_recovery\|
   kinase_summary alz/ docs/ -r`):
   - `alz/decomposition/enrich_celltype.py` imports
     `from alz.kinase_enrich import ...`
   - `alz/build_unified_viewer.py` reads attribution-recovery outputs by path
     (no import; just verify the output dir hasn't moved)
   - `pixi.toml [tasks]` block
   - Every script under `alz/runners/main/` and `alz/runners/supplementary/`
     that invokes `python alz/kinase_*.py` or `python alz/attribution_recovery.py`

3. **Rename `alz/decomposition_mea/` → `alz/decomposition_mea/` (task #14)** for
   parallelism with `bulk_mea/`. Update `alz/decomposition/paths.py` import
   sites, `alz/runners/main/rerun_decomposition_chain.sh`,
   `alz/runners/main/run_pivot_smoke.sh`, and any `from alz.decomposition`
   imports across the repo. Internal file layout stays the same.

4. **Consolidate Mode 3 under `alz/incytr_pair/` (task #15).**
   - Rename `alz/incytr_pair/` → `alz/incytr_pair/`
   - Move `alz/incytr_pair/pair_to_receiver_cache.py` →
     `alz/incytr_pair/pair_to_receiver_cache.py` (it's purely a Mode-3
     output reshaper; doesn't belong in cross-mode integration glue)
   - `alz/integration/` retains: cluster spine builder, omics/transcript
     trace, normalized substrate, bridge builders, config_integration —
     these are genuinely cross-mode (consumed by Mode 2, Mode 3, AND viewer)
   - Update `alz/runners/main/run_pair_mode_pipeline.sh` +
     `run_pair_mode_viewer_build.sh` paths

5. **Carve out Mode 4 as `alz/cross_reference/` (task #16).** Mode 4 is
   currently the most scattered:
   - Move `alz/cross_reference/seaad_human_agreement.py` → `alz/cross_reference/`
   - Move `alz/cross_reference/human_celltype_attribution.py` → `alz/cross_reference/`
   - Extract SEA-AD/WMB/Song evidence loading currently inlined in
     `kinase_attribute.py` (functions `_compute_sea_ad_concordance`,
     `_prepare_wmb_specificity`, `_prepare_song_specificity`,
     `_prepare_song_concordance`) into a shared module under
     `alz/cross_reference/`. `bulk_mea/attribute.py` (post-task-#13) and
     viewer consumers import from there.
   - This is the most-invasive of #13–#17; defer until #13 lands and the
     bulk_mea/attribute import surface is stable

6. **Group `alz/ingest/` and `alz/reference/` (task #17).** Ingest:
   - `data_ingest.py` → `alz/ingest/song.py`
   - `ingest_mukesh.py` + `ingest_mukesh_perdonor.py` →
     `alz/ingest/mukesh.py` + `alz/ingest/mukesh_perdonor.py`
   - `lucie_5xfad_manifest.py` → `alz/ingest/lucie.py`

   Reference (atlas + cross-cohort expression):
   - `atlas_reference.py` → `alz/reference/atlas.py`
   - `wmb_expression.py` → `alz/reference/wmb_expression.py`
   - `human_reference_expression.py` → `alz/reference/human_expression.py`
   - `snrna_integration.py` → `alz/reference/snrna_integration.py`
   - `snrna_proportions.py` → `alz/reference/snrna_proportions.py`

   Update import sites + runners.

7. **`alz/shared/` for `config.py` + small utilities (task #18).**
   - `alz/config.py` → `alz/shared/config.py` (single biggest import-site
     fan-out — every script imports `from alz import config`. Plan to update
     to `from alz.shared import config` in one pass)
   - `alz/map_kinases_to_genes.py` → `alz/shared/`
   - Consider extracting the repeated `_load_params()` helper that appears
     in 5 CLI shims into `alz/shared/params.py`

Each step is independently shippable; the repo stays runnable between
commits. **Recommended commit cadence: one commit per task.**

**Why not numbered files inside each subdir.** Considered briefly. Numbered
prefixes break when steps get inserted/reordered, and the existing
`alz/decomposition_mea/` (six descriptive filenames + README documenting order)
reads fine without them. Stick with descriptive names + README.

## Open Blockers (the stack)

These are individual items that may surface during the phase work above. Add
to this list as new blockers appear; don't ignore them. Address one at a
time; don't lose the broader thread.

- **B1.** Lucie 5xFAD path: `data/lucie_proteomics` appears to still be a
  live FUSE mount (`mv` → `Device or resource busy`). Defer — need explicit
  user ack to `umount` + move, or to run `pixi run ingest-lucie-proteomics`.
  (Surfaced 2026-05-21 during data-layout consolidation; see
  `docs/archive/plans/data_layout_consolidation_2026-05-21.md`.)
- **B2.** Factorial fragments (see Phase 3 above) — promoted to Phase 3,
  not a side blocker.

## Reference Documents

- Prior data-layout consolidation: `docs/archive/plans/data_layout_consolidation_2026-05-21.md`
- Canonical data sources: `data/README.md`
- Live analysis charter: `docs/foundation/analysis_charter.md`
- Repository-level instructions: `CLAUDE.md`

## Success Criteria

The plan is "done" when:

- `docs/` reflects only live, authoritative material. (Phase 2 — done.)
- The Song factorial coding lives in exactly one place. (Phase 3 — done.)
- The four analysis modes each live in their own `alz/<mode>/` subdirectory
  with a README documenting stage order, and a single shared `config.py`
  + `alz/shared/` for cross-mode utilities. (Phase 4 — task #12 done;
  tasks #13–18 pending.)
- A new cohort can be onboarded by writing one ingest module + adding one
  `conf/<cohort>/parameters.yml`, with no changes to shared analysis code.
  (Validation step, post-#18.)

These are aspirational endpoints, not deadlines. Methodical beats fast.

## Progress Snapshot — 2026-05-21

- Phase 1 (write the plan): done
- Phase 2 (docs cleanup): done
- Phase 3 (factorial fragments): done (B3/B4/B5 cleanups landed in
  `alz/integration/config_integration.py` + `alz/attribution_recovery.py` +
  `alz/build_unified_viewer.py`)
- Phase 4:
  - Task #12 — Strip Kedro: **done 2026-05-21** (commit `0c2a997`)
  - Task #13 — Move Mode 1 → `alz/bulk_mea/`: **done 2026-05-21**
    (commit `7e73ecb`)
  - Task #14 — Rename `alz/decomposition/` → `alz/decomposition_mea/`:
    **done 2026-05-21** (commit `629446e`)
  - Task #15 — Consolidate Mode 3 under `alz/incytr_pair/`: **done 2026-05-21**
    (renamed `alz/incytr/` → `alz/incytr_pair/` and moved
    `pair_to_receiver_cache.py` into it)
  - Task #16 — Carve out `alz/cross_reference/` for Mode 4: **done 2026-05-21**
    in two commits — (a) `8e8bd5e` moved `seaad_human_agreement.py` +
    `human_celltype_attribution.py` into `alz/cross_reference/`; (b) the
    follow-up extracted the SEA-AD / WMB / Song evidence loaders out of
    `bulk_mea/attribute.py` into `alz/cross_reference/evidence.py` and
    promoted them to public (no leading underscore).
  - Task #17 — Group ingest + reference: **done 2026-05-21** —
    `git mv`'d 9 files into `alz/ingest/` (`song.py`, `mukesh.py`,
    `mukesh_perdonor.py`, `lucie.py`) and `alz/reference/`
    (`atlas.py`, `wmb_expression.py`, `human_expression.py`,
    `snrna_integration.py`, `snrna_proportions.py`); fixed sys.path
    bootstraps, repointed every import + CLI string + runner script
    + pixi.toml task + CLAUDE.md reference.
  - Task #18 — Move config.py to `alz/shared/`: pending
