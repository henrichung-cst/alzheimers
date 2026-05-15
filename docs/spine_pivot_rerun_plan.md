# Levy-19 spine pivot — full rerun plan

**Trigger:** code pivoted from the 34 WMB-class spine to the 19-cluster Levy spine
(commit `4794ff1`, 2026-05-13). All downstream outputs computed before that point
are keyed on the wrong spine and must be regenerated.

**Companion docs:**
- Design / code-rewrite plan: `docs/snrna_46cluster_pivot_plan.md` (steps 1–11; mostly done)
- Foundation: `docs/foundation/analysis_charter.md`, `docs/foundation/live_pipeline_contract.md`

**Overarching discipline:** do not mix results from different spine eras in the same
deliverable directory. Each phase below ends with a verification check + an archive
boundary. Do not advance without both.

---

## Tracks recap

| Track | What it produces | Spine sensitivity |
|---|---|---|
| **1. Bulk MEA + bulk attribution** | `mea_stoichiometry.csv`, `unified_attribution.csv`, `kinase_hypothesis_table.csv` | Attribution step joins Song concordance + SEA-AD LFC + WMB specificity, all keyed on the spine |
| **2. Per-cluster decomposition + decomp MEA** | `outputs/reports/decomposition/levy19/` | Spine *is* the deliverable axis (one row per Levy-19 cluster) |
| **3. Incytr factorial (per-cluster)** | `per_cluster/{pr,ps,py}/<cluster>.parquet` | Spine determines sender × receiver cardinality (19² = 361 pairs) |

---

## Phase 0 — Audit current output state (before re-running anything)

Catalog what's on disk and stamp its spine era. Goal: zero ambiguity about which
files are stale.

```bash
stat -c '%y %n' outputs/reports/snrna_integration/*.csv \
                outputs/reports/wmb_expression/*.csv \
                outputs/reports/kinase_attribution/*.csv \
                outputs/reports/attribution_recovery/*.csv \
                outputs/reports/decomposition/levy19/*.csv 2>/dev/null
```

For each directory, decide: keep in place (will be overwritten), or move aside.
Default for any deliverable from before `4794ff1` (2026-05-13 22:43): **move aside**
into `outputs/archive/pre_levy19_<date>/` to avoid any chance of a downstream
script picking up a stale CSV via a soft path.

**Verification:** every file remaining under `outputs/reports/` has mtime
≥ 2026-05-13 or is explicitly noted as spine-agnostic (e.g. `data_ingest/` PCA plots).

---

## Phase 1 — Refresh spine-keyed reference matrices

These are inputs to bulk attribution and to per-cluster decomp. They must be
regenerated **first** because everything downstream consumes them.

| Output | Command | Done? |
|---|---|---|
| `wmb_kinase_expression.csv` (whole_brain) | `bash alz/runners/supporting/run_wmb_expression.sh` | ✅ 2026-05-14 08:40 |
| `wmb_proteome_expression.csv` (whole_brain) | (same) | ✅ 2026-05-14 08:40 |
| `pseudobulk_cpm.csv`, `song_expression_specificity.csv`, `song_concordance.csv` | `pixi run python alz/snrna_integration.py --run` | ✅ 2026-05-14 08:58 |

**Verification (done):**
- `wmb_*expression.scope.json` reports `scope: whole_brain` across all 13 regions.
- snrna outputs use 19 Levy-19 cluster labels (17 pass the 10-param OLS gate).
- `active_classes.csv` (2026-04-30, 34-class) is a stale audit sidecar from
  retired code — no consumer; safe to delete.

**Archive boundary:** none — these matrices are now in their canonical location
and downstream phases will read them in place.

---

## Phase 2 — Track 1: Bulk attribution rerun

`pixi run dual` was executed before Phase 1's Song refresh, so its
`kinase_attribute.py` step consumed stale Song concordance. The `enrich` and
`normalize` outputs are spine-agnostic (they sit upstream of the spine join) and
do **not** need re-running.

```bash
pixi run attribute && pixi run recover                              # males_only (primary)
KEDRO_ENV=full_cohort pixi run attribute && KEDRO_ENV=full_cohort pixi run recover  # sensitivity
```

**Why not re-run `pixi run dual` end-to-end:** ingest/normalize/enrich are
spine-agnostic and already current; re-running them wastes ~30 min and rewrites
files we trust.

**Verification:**
- `unified_attribution.csv` `cell_type` column contains exactly the 19 Levy-19
  cluster names (no `01 IT-ET Glut`, no `Other`, no `n/a`).
- Row count = `n_kinases × 9 contrasts × 19 clusters` (modulo SEA-AD `n/a` rows
  for non-MTG clusters — those carry SEA-AD blank but WMB + Song populated).
- `attribution_summary.json` cell-type counts sum to expectations.
- `kinase_hypothesis_table.csv` mtime is fresh; spot-check that a known kinase
  (CAMK2A, MAPK1) shows non-null Song LFC in cortical clusters.

**Archive boundary:** before running, move existing
`outputs/reports/kinase_attribution_{males_only,full_cohort}/` and
`attribution_recovery_{males_only,full_cohort}/` into
`outputs/archive/pre_levy19_<date>/` if the dual runner produced spine-mixed
versions. Verify the dual runner's archiving logic does not retain stale copies.

---

## Phase 3 — Track 2: Per-cluster decomposition + decomp MEA

This track is independently keyed on the Levy-19 spine. End-to-end runner:

```bash
bash alz/runners/main/run_pivot_smoke.sh
```

Stages it executes:
1. `alz/snrna_proportions.py --spine levy19` — per-(animal, cluster, gene) `f_c` weights
2. `alz/decomposition/build_celltype_decomposition.py --spine levy19 --track both`
3. `alz/decomposition/enrich_celltype.py --spine levy19 --track {st,py}`

**Pre-flight:** confirm `raw_phospho_normalized_pY.csv` exists from Stage 1
normalize (`pixi run normalize`), otherwise the smoke runner skips the `--track py`
half (per CLAUDE.md Gotcha). If pY is desired, run normalize first.

**Verification (`alz/decomposition/verify_decomposition.py --spine levy19 --all`):**
- Mass identity: `Σ_c [P_c × (N_c / N_total)] ≈ bulk` (per-cell-rate form)
- Spine coverage: all 19 clusters present, no silent drops
- Per-cluster vs bulk MEA agreement: Spearman ρ ≥ 0.7 per contrast under `f_c`-weighting
- Incytr-readiness: per-cluster contracts ready for Track 3 export

**Archive boundary:** if `outputs/reports/decomposition/levy19/` already contains
files from an earlier WIP iteration, archive first
(`mv outputs/reports/decomposition/levy19 outputs/archive/pre_levy19_<date>/decomposition_levy19/`).

---

## Phase 4 — Track 3: Incytr factorial (per-cluster, 19²=361 pairs)

Depends on Phase 3 completion (per-cluster expression contracts are the
factorial input fixture).

```bash
pixi run export-factorial-inputs   # writes data/incytr_factorial_inputs/
pixi run incytr-factorial          # R-side scoring, persists parquet
```

**Pre-flight:** `pixi run install-incytr` if the upstream R package version
moved. The wrapper hard-fails before loading data if
`Incytr::construct_factorial_paths` / `Incytr::score_factorial_paths` are not
exported (per CLAUDE.md Gotcha).

**Verification:**
- `data/incytr_factorial_inputs/MANIFEST.json` records spine source +
  git SHA + filter parameters; `celltype_taxonomy` field reads
  "Levy 19-cluster strict spine".
- `expression_metadata.csv` `labels` column = exactly the 19 spine names.
- `per_cluster/{pr,ps,py}/` contains 19 parquets per omic (one per cluster,
  list-dispatched by `Incytr::resolve_wide`).
- Receiver pair cache contains 361 sender × receiver entries
  (or fewer if `PAIR_FILTER` is intentionally applied).

**Archive boundary:** archive any pre-existing
`data/incytr_factorial_inputs/` and `outputs/incytr/per_cluster/` from earlier
spine versions before this phase, otherwise loader idempotency may silently
mix.

---

## Phase 5 — Viewer + bubble plots (optional, presentation layer)

Only after Phases 2–4 pass verification:

```bash
pixi run python alz/build_unified_viewer.py
pixi run python alz/plot_attribution_bubbles.py
```

Hard-refresh the browser (Ctrl+Shift+R) when reloading the viewer — per CLAUDE.md
Gotcha, the payload is inlined into `index.html` and soft reloads serve stale.

**Verification:**
- `PAYLOAD.meta.generated_at` (DevTools console) matches the latest build.
- Color palette has 19 distinct colors (was 34). If pre-existing assets reused
  a 34-color palette, regenerate.
- Cell-type axis on bubble plots lists 19 Levy-19 names in the documented order.

---

## Phase 6 — Docs + retire stale artifacts

Once all phases pass:

1. Update `CLAUDE.md`: replace remaining "34 WMB classes" / `WMB_CLASSES`
   references in user-facing prose with Levy-19 / `CLUSTER_SPINE` (the
   spine-pivot commit `4794ff1` updated code paths but some prose may lag).
2. Delete `outputs/reports/snrna_integration/active_classes.csv` (orphaned
   from pre-pivot code).
3. Confirm `outputs/archive/pre_levy19_<date>/` is gitignored or moved off-tree
   if not already.
4. Update the foundation docs noted in `snrna_46cluster_pivot_plan.md` §4.11
   if not yet refreshed (analysis_charter, concordance, live_pipeline_contract,
   statistical_constraints).

---

## Mixing-prevention checklist (apply at every phase boundary)

- [ ] No file in `outputs/reports/` is older than `4794ff1` (2026-05-13 22:43)
      unless explicitly spine-agnostic.
- [ ] Each phase output directory is single-spine — no leftover 34-class files
      alongside 19-cluster files.
- [ ] Every MANIFEST/sidecar JSON written in this rerun records
      `celltype_taxonomy: "Levy 19-cluster"` (or equivalent) + git SHA.
- [ ] When in doubt, archive then regenerate; do not edit in place.
