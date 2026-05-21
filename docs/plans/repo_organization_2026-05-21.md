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
- `alz/incytr/export_decomposition_for_pair.py` — `GENO_DECODE`, `ANIMAL_RE`

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

### Phase 4 — Canonical contract + kedro pipeline structure (deferred)

Only after Phase 3 closes. Two options surfaced in conversation:

- **(a) Contract-first:** write `docs/foundation/cohort_contract.md` locking
  down columns/dtypes/sample-ID convention/group vocabulary for each
  canonical artifact. Then refactor the four non-kedro modes onto that
  contract.
- **(b) Structure-first:** add `pipeline_registry.py` entries for all four
  modes against today's de-facto contract, then iterate on cleaning up
  couplings.

Working preference (subject to change): **(a) contract-first**. Without the
contract, new kedro pipelines just lock in today's hardcoded vocabulary.

## Open Blockers (the stack)

These are individual items that may surface during the phase work above. Add
to this list as new blockers appear; don't ignore them. Address one at a
time; don't lose the broader thread.

- **B1.** Lucie 5xFAD path: `data/lucie_proteomics` appears to still be a
  live FUSE mount (`mv` → `Device or resource busy`). Defer — need explicit
  user ack to `umount` + move, or to run `pixi run ingest-lucie-proteomics`.
  (Surfaced 2026-05-21 during data-layout consolidation; tracked in
  `docs/plans/data_layout_consolidation_2026-05-21.md`.)
- **B2.** Factorial fragments (see Phase 3 above) — promoted to Phase 3,
  not a side blocker.

## Reference Documents

- Prior data-layout consolidation: `docs/plans/data_layout_consolidation_2026-05-21.md`
- Canonical data sources: `data/README.md`
- Live analysis charter: `docs/foundation/analysis_charter.md`
- Repository-level instructions: `CLAUDE.md`

## Success Criteria

The plan is "done" when:

- `docs/` reflects only live, authoritative material.
- The Song factorial coding lives in exactly one place.
- The four analysis modes have explicit canonical input contracts AND are
  defined as kedro pipelines.
- A new cohort can be onboarded by writing one ingest module + adding one
  `conf/<cohort>/parameters.yml`, with no changes to shared analysis code.

These are aspirational endpoints, not deadlines. Methodical beats fast.
