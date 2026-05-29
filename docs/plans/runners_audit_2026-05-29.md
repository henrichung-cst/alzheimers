# Runners Audit — 2026-05-29

**Status:** EXECUTED 2026-05-29 (full consolidation, approved). 14 `main/` runners → 7.
Outcome at the bottom (`## Execution outcome`).

**Scope:** All 28 tracked scripts under `alz/runners/` (`main/` 14, `supplementary/` 1,
`supporting/` 13). Audited for: closed-path / anti-shim violations, reachability (pixi task
or composite), duplication/drift, convention consistency, and the CLAUDE.md correctness gates.

---

## Headline verdict

**Clean on correctness, sprawling on structure.** No runner reintroduces a closed path; every
`deconvolution`/`factorial` token is either active stoichiometry or an explicit archived-path
note. `set -e` (mostly `set -euo pipefail`) is universal. The documented prerequisite gates
(normalize-before-outliers, pY-track skip tolerance, WMB 13-h5ad guard, WMB/Song existence
checks) are all honored.

The problem is **redundancy and an undocumented surface**: the mouse kinase chain has **four**
overlapping implementations, the decomposition chain **three**, and the dual-cohort analysis
**two** — and only **6 of 14** `main/` runners are documented in the README. A newcomer (or
future-me) cannot tell which entrypoint is canonical.

---

## Reachability map

`pixi` = wired to a pixi task · `run_all` = invoked inside `run_all.sh` (inlined, not called) ·
`live` = called by `run_live_pipeline.sh` · README = documented as an entrypoint.

| Runner | Wired via | README? | Verdict |
|---|---|---|---|
| `main/run_all.sh` | `pixi all` | ✗ | **KEEP** — canonical superset (resumable, sentinels). Document it. |
| `main/run_dual_analysis.sh` | `pixi dual` | ✓ | KEEP — canonical dual-cohort. |
| `main/run_pair_mode_pipeline.sh` | — | ✓ | KEEP — canonical pair-mode bundle (resumable). Fix stale header (see F6). |
| `main/run_pivot_smoke.sh` | — | ✓ | KEEP — documented per-cluster decomp+verify. |
| `main/run_live_pipeline.sh` | — | ✗ | **SUPERSEDED** by `run_all.sh` + `pixi live` (see F1). |
| `main/run_data_ingest.sh` | via `live` | ✗ | KEEP (sub-runner) or fold — see F1. |
| `main/run_kinase_attribution.sh` | via `live` | ✗ | KEEP (sub-runner) or fold — see F1. |
| `main/run_attribution_recovery.sh` | via `live` | ✗ | KEEP (sub-runner) or fold — see F1. |
| `main/rerun_mouse_kinase_chain.sh` | — | ✗ | **CONSOLIDATE** — 4th copy of kinase chain (F1). |
| `main/rerun_decomposition_chain.sh` | — | ✗ | **CONSOLIDATE/KEEP** — richest decomp variant (F2). |
| `main/run_phase2_spine_pivot.sh` | — | ✗ | **ARCHIVE** — one-time migration, ~dup of `run_dual` (F3). |
| `main/run_levy_t5_attribution_rebuild.sh` | — | ✗ | KEEP — narrow targeted rerun; document. |
| `main/run_pair_mode_viewer_build.sh` | — | ✗ | KEEP — narrow targeted rerun; document. |
| `main/run_mukesh_perdonor.sh` | — | ✗ | KEEP — narrow human rerun; overlaps `pixi human` (F5). |
| `supplementary/run_reviewer_diagnostics.sh` | — | ✓ | KEEP. |
| `supporting/*` (13) | mostly pixi / called by `run_*` | partial | KEEP — see F4/F5 for thin-wrapper notes. |

---

## Findings

### F1 — Mouse kinase chain: four implementations (primary consolidation target)

The `ingest → normalize → enrich → (mechanism) → attribute → recover` chain exists four ways:

| Entrypoint | Stages | Notes |
|---|---|---|
| `pixi live` (depends-on) | ingest, normalize, enrich, attribute, recover | **omits `mechanism`** |
| `run_live_pipeline.sh` → 3 sub-runners | ingest, normalize+enrich+attribute, recover | **omits `mechanism`**; adds atlas/WMB/Song prereq auto-resolution |
| `run_all.sh` `K-*` | map, wmb, snrna, norm, enrich, **mech**, attr, recover | superset; resumable sentinels; assumes data downloaded |
| `rerun_mouse_kinase_chain.sh` | norm, enrich, **mech**, attr, recover | + mtime-ordering guard |

**RESOLVED (dependency traced 2026-05-29):** `attribute.py` does **not** consume any
`mechanism.py` output — it reads `mea_stoichiometry{suffix}.csv`, `kinase_to_gene_mapping.csv`,
WMB, Song. The dependency is the **reverse**: `mechanism.py:166-171` reads
`unified_attribution.csv` (attribute's output) and merges annotations back into it, guarded by
`if os.path.exists(unified_path)`. **Correct order is `attribute → mechanism`.** This makes three
runners buggy, not just stylistically divergent:

| Runner | Order | `unified_attribution.csv` result |
|---|---|---|
| `run_dual_analysis.sh` | attribute → mechanism | ✅ correct (annotated) |
| `run_all.sh` (`K-mech` before `K-attr`) | mechanism → attribute | ❌ fresh build: `unified` absent → merge guard False → silently skipped → attribute overwrites with **no mechanism columns** |
| `rerun_mouse_kinase_chain.sh` (mech `[3/5]` < attr `[4/5]`) | mechanism → attribute | ❌ same silent-skip |
| `pixi live` | omits mechanism | ❌ no `mechanism_annotation.csv` / `mea_raw_phospho.csv` / mechanism columns |
| `run_live_pipeline.sh` | omits mechanism | ❌ same omission |

Partly silent because mechanism's *standalone* outputs (`mea_raw_phospho.csv`,
`mechanism_annotation.csv`) are produced in any order; only the **merge into `unified`** is lost
when mechanism runs first. Net: `run_all.sh` — the otherwise-canonical superset — currently emits
a `unified_attribution.csv` missing its mechanism columns on a clean build.

**Fix (deterministic):** run `mechanism` after `attribute` everywhere. Swap the two steps in
`run_all.sh` and `rerun_mouse_kinase_chain.sh`; add `mechanism` after `attribute` in `pixi live`
and `run_live_pipeline.sh`. `run_dual` is already correct.

**Recommendation:** make `run_all.sh` the single source of truth for the full build (after the
order fix); reduce `run_live_pipeline.sh` to a thin alias or archive it (its unique value — prereq
auto-resolution — should move into `run_all.sh`, which currently assumes downloads are present).

### F2 — Decomposition chain: three implementations

| Entrypoint | Steps | Notes |
|---|---|---|
| `rerun_decomposition_chain.sh` | pseudobulk → proportions → decomp → enrich(st,py) → per-animal-ols → verify | + post-run `enrich_audit.json`/`verification.json` skipped-cluster assertion |
| `run_pivot_smoke.sh` (README) | normalize → proportions → decomp → enrich(st,py) → verify | tolerant pY skip; no per-animal-ols |
| `run_all.sh` `D-*` | proportions → decomp → enrich(st,py) → per-animal-ols → verify | inside the full build |

`rerun_decomposition_chain.sh` is the most complete standalone (adds the pseudobulk step and the
JSON audit gate). **Recommendation:** keep `rerun_decomposition_chain.sh` as the canonical
standalone decomp rerun and **document it**; keep `run_pivot_smoke.sh` only if its
lighter "smoke" scope is intentionally distinct (it is README-documented). Confirm whether two
standalone decomp runners are wanted or one should fold into the other.

### F3 — `run_phase2_spine_pivot.sh` ≈ `run_dual_analysis.sh` (archive candidate)

Both produce the same `outputs/reports/*_{males_only,full_cohort}/` snapshots by running the
males-only chain then the `KEDRO_ENV=full_cohort` chain. `run_phase2` is self-labeled a one-time
"Phase 2 spine-pivot rerun"; the spine pivot to levy_t5 is long settled and `pixi dual` is the
documented path. **Recommendation: ARCHIVE `run_phase2_spine_pivot.sh`** → `archive/` (the dual
analysis is the live equivalent). Anti-shim: one canonical dual runner, not two.

### F4 — Convention drift (cosmetic, but worth one normalizing pass)

- **Three cd-to-root idioms**: `git rev-parse --show-toplevel` (run_all, pivot_smoke, levy_t5,
  pair_mode, hbca) vs `dirname BASH_SOURCE/../../..` (kinase, ingest, recovery, atlas, wmb*) vs
  `dirname $0/../../..` (mukesh, tcells_*, ingest_tcells). Pick one.
- **Python invocation inconsistent**: `${PYTHON:-pixi run --manifest-path … python}` override
  (~7 runners) vs bare `pixi run` (run_all et al.) vs — two outliers —
  - `run_snrna_integration.sh` calls **bare `python`** (no `pixi run`, no `$PYTHON`): only works
    if the env is already active; will use the wrong interpreter if invoked directly.
  - `rerun_decomposition_chain.sh` calls `.pixi/envs/default/bin/python` **directly** for its
    audit block, bypassing `pixi run`.
- No runner sets `CONDA_OVERRIDE_CUDA`/`DUCKDB_TEMP_DIR` — **correct**, both come from `.envrc`;
  noted only to confirm it's intentional, not a gap.

**Recommendation:** standardize on `cd "$(git rev-parse --show-toplevel)"` + `pixi run python`,
and fix the two interpreter outliers (correctness, not just style).

### F5 — Thin supporting wrappers / partial duplication (low priority)

- `run_wmb_download.sh` (`atlas.py --wmb-download`) duplicates step 2/2 of
  `run_atlas_reference.sh` (same command). Minor; keep if used as a standalone re-download.
- `run_snrna_integration.sh` duplicates `pixi snrna` but adds an h5ad existence guard (and the
  bare-`python` bug above). Either fix it or drop it in favor of the pixi task.
- `run_mukesh_perdonor.sh` overlaps `pixi human` (`human-perdonor` + `human-seaad`). Keep as a
  logged narrow rerun, or note it's a subset of `human`.
- `tcells_projectils_map.sh` / `tcells_scrna_extract.sh` loop over both donors — **not** pure
  pass-throughs; keep. (Note: `tcells_scrna_extract.sh` uses bare `Rscript`, sibling uses
  `pixi run Rscript` — same outlier class as F4.)

### F6 — `run_pair_mode_pipeline.sh` stale header

Header says "Phase-2 clean rerun for change requests 01-04 … see
docs/plans/change_requests_sequencing.md", but README markets it as the general full pair-mode
pipeline and that plan path is not among the kept plans. Update the header comment to match its
documented role (no code change).

---

## Clean bill (verified, no action)

- **No closed-path reintroduction** (factorial Incytr, direct deconvolution, WMB-34, Levy-19,
  two-compartment) in any runner.
- **`set -euo pipefail`** present in all 27 shell runners.
- **Correctness gates honored**: `run_dual` normalizes before `song.py --outliers`;
  `run_pivot_smoke` tolerates the pY-track skip; `run_live_pipeline` and `run_wmb_expression`
  gate on the 13 WMB h5ads; `run_extract_wmb_subset` guards the gene-list prerequisite.

---

## Proposed execution order (on approval)

1. **F1 ordering fix (RESOLVED — `attribute → mechanism`)** — swap `K-mech`/`K-attr` in
   `run_all.sh`; swap steps `[3/5]`/`[4/5]` in `rerun_mouse_kinase_chain.sh`; add `mechanism`
   after `attribute` in `pixi live` and `run_live_pipeline.sh`. This is a correctness fix, not a
   preference — `unified_attribution.csv` is currently missing mechanism columns on clean builds
   via `run_all`/`live`.
2. **F3 ARCHIVE** — `git mv run_phase2_spine_pivot.sh → archive/`.
3. **F1 consolidate** — reduce/alias `run_live_pipeline.sh`; migrate its prereq auto-resolution
   into `run_all.sh`; drop `rerun_mouse_kinase_chain.sh` if `run_all --rerun K-*` covers it.
4. **F4 normalize** — one cd idiom + `pixi run python`; fix the bare-`python`/direct-`.pixi`/bare-
   `Rscript` outliers.
5. **F2** — keep one or both decomp runners per your call; document the survivor.
6. **F6** — fix the `run_pair_mode_pipeline.sh` header comment.
7. **README + docs** — document the canonical entrypoints (`run_all`, the targeted reruns) so the
   surface is discoverable; refresh `docs/foundation/repo_retention_policy.md` if runner tiers move.

### Open decisions for you
- **F1 consolidation depth**: collapse to `run_all.sh` + `pixi live` only (archive
  `run_live_pipeline` + `rerun_mouse_kinase_chain`), or keep the granular sub-runners for
  targeted reruns?
- **F2**: one decomp runner or two (`rerun_decomposition_chain` vs `run_pivot_smoke`)?
- **Scope of this pass**: structural consolidation now, or only the safe wins (F3 archive, F4
  outlier fixes, F6 header, README doc) and defer the F1/F2 chain merges?

---

## Execution outcome (2026-05-29)

**Full consolidation applied. 14 `main/` runners → 7. All 20 on-disk runners pass `bash -n`; `pixi.toml` parses and `live` now lists `mechanism` between `attribute` and `recover`.**

**F1 (correctness + collapse):**
- `pixi.toml`: `live` now `ingest → normalize → enrich → attribute → mechanism → recover`.
- `run_all.sh`: `K-attr` now precedes `K-mech` (with an inline comment on why); migrated `run_live_pipeline.sh`'s SEA-AD/WMB-h5ad prereq auto-resolution into a block gated on the previously-dead `--skip-atlas`.
- Archived (de-tracked → `archive/runners_2026-05-29/`): `run_live_pipeline.sh`, `rerun_mouse_kinase_chain.sh`, and the granular trio `run_data_ingest.sh` / `run_kinase_attribution.sh` / `run_attribution_recovery.sh`. (`run_data_ingest.sh` was additionally stale — it called the removed `song.py --markers` step.)

**F2:** kept the fuller `rerun_decomposition_chain.sh`; archived `run_pivot_smoke.sh`. README's two `run_pivot_smoke` references repointed.

**F3:** archived `run_phase2_spine_pivot.sh` (completed one-time spine-pivot migration; `pixi run dual` is the live equivalent).

**F4 (interpreter correctness):** `run_snrna_integration.sh` bare `python` → `pixi run python` (+ canonical `cd`); `rerun_decomposition_chain.sh` `.pixi/.../python` → `pixi run python`; `tcells_scrna_extract.sh` bare `Rscript` → `pixi run Rscript`. The purely-cosmetic cd-idiom variation across the *other* runners was left as-is (no behavioral impact; mass churn not worth the diff).

**F6:** `run_pair_mode_pipeline.sh` header rewritten to describe its real (canonical pair-mode) role.

**Docs:** README (`pixi run all` documented, `mechanism` in the chain, decomp runner swap), `docs/INDEX.md` stage table (pixi-task runner column, mechanism promoted to numbered stage 5), `docs/foundation/repo_retention_policy.md` (canonical runner set + archival note), `CLAUDE.md` WMB-prerequisite gotcha.

**Surviving `main/` runners:** `run_all.sh`, `run_dual_analysis.sh`, `run_pair_mode_pipeline.sh`, `rerun_decomposition_chain.sh`, `run_levy_t5_attribution_rebuild.sh`, `run_pair_mode_viewer_build.sh`, `run_mukesh_perdonor.sh`.

Not committed — left staged-ready for review.
