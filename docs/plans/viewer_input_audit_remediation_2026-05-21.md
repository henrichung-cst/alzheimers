# Unified Viewer Input Audit — Remediation Plan (2026-05-21)

Follow-up to `viewer_input_audit_2026-05-21.md`. Phased plan to close every issue surfaced. Phases are ordered so that each step's outputs are usable by the next.

**Standing decisions** (apply across all phases):
- Research pivots replace, they do not coexist. Strip every backwards-compat residue.
- Hardfail every contract gate. No softfail. No skip-if-exists with stale upstream.
- Empty cells in the viewer where data is genuinely absent — no UX explanation needed.
- Delete all unused code. Do not preserve "in case" or behind flags.
- One end-to-end runner. Mouse + human in the same chain.
- `analysis_mode` and spine name get stamped in sidecars where missing — no new schema design.

**Deferred (not in scope for this plan):**
- `docs/plans/incytr_mea_seed_list_expansion_plan.md` — kept as a future workstream; not closed, not executed here.
- `docs/plans/evidence_tab_redesign_2026-05-21.md` — separate UI work.

---

## Phase 0 — Re-run the mouse pipeline to clear stale enrich outputs

Closes: CRITICAL #1 (stale enrich/mechanism artifacts on disk).

No code changes. Re-run in dependency order:

```
pixi run normalize
pixi run enrich
python alz/kinase_mechanism.py
pixi run attribute
pixi run recover
```

Verify mtime ordering after: `stoichiometry_matrix.csv < site_level_ols.csv < mea_stoichiometry.csv < unified_attribution.csv < kinase_hypothesis_table.csv`.

**Success:** all 14 stale files (ST + pY) are newer than `stoichiometry_matrix.csv`. Do *not* rebuild the viewer yet — later phases change what the viewer reads.

---

## Phase 1 — Scorched-earth removal of factorial Incytr + WMB-34/Levy-19 residue

Closes: CRITICAL #5, CRITICAL #6, WARN #13, WARN #14, WARN #15, NOTE #30, NOTE #32.

**Delete entirely:**
- `_write_incytr_pathways()` (factorial branch) in `alz/build_unified_viewer.py`. Remove the `INCYTR_SOURCE` env-var dispatch at lines 2860–2866 — call `_write_incytr_pair_pathways()` directly.
- `alz/build_unified_viewer.py` lines 1–25 — phantom backbone-edges docstring. Replace with a one-paragraph statement of what the viewer actually does (reads attribution + decomposition + Incytr pair-mode + human per-donor; writes inlined-payload HTML with per-entity shards under `edge_slices/{human_perdonor,decomp_ols,song_concordance,incytr_pathways}/`).
- `config.SEAAD_TO_WMB_CLASS_FILE` and any constants that point at deleted bridge files.
- `_annotate_trajectory_columns(..., source_label="factorial")` call site once `_write_incytr_pathways` is gone.

**Rename:**
- `outputs/reports/incytr_factorial/receiver_cache/` → `outputs/reports/incytr_pair_mode/receiver_cache/`. Update:
  - `alz/integration/pair_to_receiver_cache.py` output path.
  - `alz/runners/main/run_pair_mode_viewer_build.sh` (and its docstring comment claiming "no flags needed downstream").
  - `alz/viewer/paths.py` references.
  - `alz/build_unified_viewer.py` `INCYTR_FACTORIAL_*` constants → `INCYTR_PAIR_MODE_*`.
  - `alz/viewer/template/js/tabs/incytr_heatmap.js:311` fallback message.
  - `alz/decomposition/verify_decomposition.py:33`, `alz/integration/verify_pathway_round_trip.py:392` (`pair_metadata.parquet` path references).

**Fix docstring drift** (text-only, no behavior change):
- `alz/build_unified_viewer.py:152` — "Unified attribution (all 34 WMB classes)" → "Unified attribution (31 Levy-t5 clusters)".
- `alz/attribution_recovery.py:89` — "1/34 ≈ 0.029" → "1/31 ≈ 0.0323".
- `alz/kinase_attribute.py:25` — "n_kinases × 9 × 19" → "n_kinases × 9 × 31".
- `alz/config.py:471, 481` — "19 entries" → "31 entries" (in `load_cluster_to_*` docstrings).
- `CLAUDE.md:180` — replace "rolled up to 34 WMB classes via wmb_class_manifest.csv" with the actual `barcode_to_cluster.csv → Levy-t5` chain.

**Filesystem cleanup** (do at the end of the phase, after rename above):
- Delete `archive/incytr_integration/` references from any *live* path. Archive directory itself stays; it's already isolated.

**Success:** `grep -rE 'WMB[-_]34|levy[-_ ]?19|incytr_factorial|INCYTR_SOURCE|source_mode.*factorial' alz/ docs/` returns only archive paths or this plan document. No live code references the retired vocabulary or branch.

---

## Phase 2 — Hardfail every gate; remove skip-if-exists

Closes: CRITICAL #3 (softfail on verify), WARN #16 (skip-if-exists in pair-mode), WARN #25 (no scope validation in viewer).

**Edits:**
- `alz/runners/main/run_pair_mode_pipeline.sh:212` — change `run_step_softfail V` → `run_step V`. `verify_decomposition.py` becomes a hard gate.
- `alz/runners/main/run_pivot_smoke.sh` — same treatment if it uses `softfail` for verify.
- `alz/incytr/run_pair_mode.sh:62-65` — remove the `if [[ -s "$out_parquet" ]]` skip guard. Always re-run all 9 contrasts when the runner is invoked. (Solo-dev repo; runtime cost is acceptable in exchange for guaranteed freshness.)
- `alz/build_unified_viewer.py` — at viewer-build start, read `outputs/reports/wmb_expression/wmb_kinase_expression.scope.json` and `assert scope == config.WMB_REGION_SCOPE`. Abort with a clear message if not.
- `alz/build_unified_viewer.py` — read `outputs/reports/decomposition/levy_t5/spine.scope.json` and `enrich_audit.json` at viewer start; assert `spine == "levy_t5"`, `min_cells == 5`, `analysis_mode` matches expectation. Abort on mismatch.

**Success:** any contract violation in `verification.json` (`all_pass: false`) aborts the pipeline. Any scope/spine mismatch aborts the viewer build before any HTML is written.

---

## Phase 3 — Investigate and fix the 12-cluster sparsity bug

Closes: CRITICAL #4 (12 of 31 Levy-t5 clusters silently dropped from `mea_per_cluster.parquet`).

**Hypothesis to investigate first:** the spine was rebuilt with `min_cells = 5` and no rank gate precisely to avoid silent drops, but `enrich_celltype.py` has a *second* sparsity gate ("no sites with any data") that fires after decomposition. Either:
- (a) `build_celltype_decomposition.py` is producing all-zero or all-NaN columns for the 12 clusters even though they have ≥5 nuclei in `proportions.parquet`, or
- (b) `enrich_celltype.py`'s gate is too aggressive — should keep the cluster and emit NaN per-contrast, not drop it.

**Steps:**
1. Inspect `outputs/reports/decomposition/levy_t5/phospho_per_cluster.parquet` for the 12 named clusters from `verification.json`. Check whether they have non-zero entries at all.
2. If they have non-zero entries → bug is in `enrich_celltype.py`'s gate. Fix the gate to keep clusters with any signal, emit NaN per-contrast where OLS can't fit (matching the rank-deficient handling at `enrich_celltype.py:238-240`).
3. If they have all-zero/NaN entries → bug is upstream in `build_celltype_decomposition.py`. Check whether `proportions.parquet` weights for those clusters are degenerate; investigate `snrna_proportions.py`.
4. Re-run `decomposition → enrich_celltype → verify_decomposition`. `verification.json` must report `all_pass: true` with all 31 clusters present.

**Success:** `mea_per_cluster.parquet` has rows for all 31 Levy-t5 clusters (NaN where unestimable). `verification.json` `all_pass == true`. Incytr pair count = 31² = 961 per contrast.

---

## Phase 4 — Levy-t5 per-animal site-level OLS producer

Closes: CRITICAL #2 (orphan WMB-34 `per_animal/site_level_ols.parquet`).

**Build a new producer.** No regenerating the old WMB-34 file.

**Approach:**
- New script: `alz/decomposition/build_per_animal_site_ols.py --spine levy_t5`. Inputs: `levy_t5/phospho_per_cluster.parquet` (per-animal × per-cluster × per-site), `sample_mapping.csv`, `sample_exclusions.csv`. Output: `outputs/reports/decomposition/levy_t5/per_animal/site_level_ols.parquet` (singular new path; Levy-t5 vocabulary).
- Per-site, per-cluster factorial OLS with the standard 9-contrast design from `kinase_enrich.py`. NaN where rank-deficient (consistent with the rest of the Levy-t5 stack).
- Stamp `analysis_mode`, `spine`, and source-input mtimes into a `.audit.json` sidecar.
- Producer must be re-runnable as part of the end-to-end runner (Phase 8). Spine changes → script picks up the new spine automatically via `config.CLUSTER_SPINE`.

**Viewer changes:**
- `alz/build_unified_viewer.py:_write_decomp_ols_slices` reads the new path. Delete the old `outputs/reports/deconvolution/per_animal/site_level_ols.parquet` path reference. Column is now `cell_type` (Levy-t5), not `wmb_class`.
- `alz/decomposition/paths.py:41` `SITE_OLS_FILE` and `viewer/paths.py:DECOMP_OLS_PARQUET` agree on the new path.

**Filesystem:**
- After verification, delete `outputs/reports/deconvolution/per_animal/site_level_ols.parquet` from disk. The old `outputs/reports/deconvolution/` tree gets pruned of any file with no live producer (audit during Phase 9).

**Success:** viewer Attribution drawer renders Levy-t5 per-animal site-level evidence; no WMB-34 label appears anywhere in the column.

---

## Phase 5 — Promote human chain to first-class

Closes: CRITICAL #7 (seaad_agreement.csv path), CRITICAL #8 (no runner for seaad_human_agreement.py), CRITICAL #9 (off-runner canonical artifacts), WARN #19 (undocumented in CLAUDE.md).

**File moves:**
- `outputs/reports/kinase_attribution_human/seaad_agreement.csv` → `outputs/reports/kinase_attribution/human_seaad_agreement.csv`. Update `alz/seaad_human_agreement.py:44` output path and `alz/build_unified_viewer.py:1092–1108` reader. Mouse and human SEA-AD evidence live together under `kinase_attribution/`.
- Consider whether the rest of `kinase_attribution_human/` should also fold into `kinase_attribution/` with a `human_` prefix on each file. Decision deferred until move above is in — re-evaluate after one round.

**Runner:**
- `alz/runners/main/run_mukesh_perdonor.sh` — add `python alz/seaad_human_agreement.py` as a final step. Remove the "next: pixi run viewer" comment that implied the chain was done.
- Promote to a pixi task: `pixi run human` = `ingest_mukesh.py --reshape && ingest_mukesh_perdonor.py --track both && seaad_human_agreement.py`. Mirrors `pixi run live`.
- Once Phase 8's end-to-end runner exists, the human chain is one of its steps; `pixi run human` is still available for the human-only subgraph.

**CLAUDE.md:**
- Add a `Human-Cohort Pipeline` section under Architecture documenting: raw Mukesh CSVs → reshape → per-donor MEA → SEA-AD agreement. List the three scripts, the runner, the pixi task, the output paths.
- Add a `pixi run human` entry under "Running the Analysis".

**Stamps:**
- `donor_groups.json`, `seaad_agreement.csv`, and the perdonor CSV outputs should carry the same `analysis_mode`/`cohort` stamp as the mouse outputs (Phase 6).

**Off-runner artifact provenance:**
- Re-run the entire human chain from scratch via the new runner so the canonical artifacts have a matching log. Delete or archive the stale 2026-05-14 log file beforehand.

**Success:** the human chain has a runner, a pixi task, a CLAUDE.md section, and outputs whose mtime/log correspond to a single runner invocation.

---

## Phase 6 — Minimal provenance stamps

Closes: WARN #17 (no `analysis_mode` stamp in mouse kinase pipeline sidecars), part of WARN #10/#11 (downstream of mapping cache decision in Phase 7).

**No new schema.** Mirror the fields already in `enrich_audit.json`: `analysis_mode`, `spine`, `produced_at`, plus track-specific fields where relevant.

**Add stamps to:**
- `outputs/reports/kinase_attribution/normalization_summary{,_pY}.json` — add `analysis_mode` (note that normalize uses all 72; record the downstream intended mode read from Kedro params).
- `outputs/reports/kinase_attribution/attribution_summary.json` — add `analysis_mode`, `spine`, list of upstream artifact mtimes.
- `outputs/reports/attribution_recovery/recovery_summary.json` (create if absent) — same.
- Phase 4's `per_animal/site_level_ols.audit.json` — same.

**Viewer hard-assertions** (extends Phase 2):
- At viewer-build start, read every audit JSON and assert all `analysis_mode` values agree. Abort on mismatch.
- Assert all `spine` values are `levy_t5`.

**Success:** `grep -l analysis_mode outputs/reports/*/[a-z]*summary*.json` returns every summary JSON. Viewer aborts cleanly if any chain was generated under a different cohort.

---

## Phase 7 — Fix kinase→gene mapping coverage

Closes: WARN #10/#11 (mapping postdates outputs), WARN #27 (173 unmapped kinases), WARN #28 (broken producer).

**First, resolve cache vs frozen.** Read what `map_kinases_to_genes.py` is actually trying to do. Two paths:

- (a) If it queries MyGene and writes the CSV: treat the CSV as a **regenerable cache**. Fix the broken hardcoded input (`outputs/deconv/enrichment_summary_sig_kins.csv` no longer exists) to read from `kldata.csv`'s full kinase list (562 entries). Producer regenerates on every full pipeline run.
- (b) If it's been edited by hand and shouldn't be machine-regenerated: treat as **frozen**, delete the producer, version-control the CSV, and accept manual curation.

**Decision in this phase:** use (a). The producer should fill all 562 kinases. Manual entries (if any) get added to a `kinase_to_gene_overrides.csv` sidecar that takes precedence over MyGene results — that's where the manual curation lives.

**Steps:**
1. Fix `alz/map_kinases_to_genes.py:83` to read `data/datasets/song/kinase/kldata_pspy.csv` and iterate over `unique(GENE_NAME)`.
2. Add `kinase_to_gene_overrides.csv` (initially empty) under `data/datasets/song/analysis_cache/`. Merge logic: overrides win.
3. Run the producer. Verify all 562 kinases mapped. Manually review the ~173 newly-resolved entries — special attention to lowercase-prefix kinase aliases like `p38` (gene `MAPK14`), `p70S6K` (gene `RPS6KB1`), `p90RSK` (gene `RPS6KA1`). Move any wrong MyGene results into `overrides.csv`.
4. Replace the title-case heuristic in `alz/atlas_reference.get_all_kinase_genes()` with a strict lookup against `kinase_to_gene_mapping.csv`; if a kinase isn't in the mapping, fail loudly.
5. Re-run any downstream stage that consumes the mapping: WMB expression (`pixi run wmb-export`), Song concordance (`pixi run snrna-integration`), attribute, recover.

**Success:** `kinase_to_gene_mapping.csv` covers every kinase in `kldata.csv`. No silent heuristic fallback anywhere. WMB expression matrix's kinase row count matches `kldata.csv` unique count.

---

## Phase 8 — One end-to-end runner

Closes: NOTE #33 (`pixi run live` doesn't cover viewer chain), Theme D consolidation.

**One pixi task** that runs everything in dependency order:

```
pixi run all  # or pixi run viewer-full
```

Chain:

```
data_ingest
  → wmb_expression (export)
  → snrna_integration (pseudobulk, specificity, concordance)
  → kinase_normalize
  → kinase_enrich
  → kinase_mechanism
  → kinase_attribute
  → attribution_recovery
  → snrna_proportions (Levy-t5)
  → build_celltype_decomposition (Levy-t5)
  → enrich_celltype (Levy-t5)
  → build_per_animal_site_ols (Levy-t5, Phase 4)
  → verify_decomposition (hardfail, Phase 2)
  → incytr_inputs build (build_pair_seurat, build_input_gene_list, build_yuyu_kldata, export_decomposition_for_pair)
  → incytr pair-mode (run_pair_mode.sh, no skip-if-exists, Phase 2)
  → pair_to_receiver_cache
  → ingest_mukesh --reshape (human)
  → ingest_mukesh_perdonor (human)
  → seaad_human_agreement (human, Phase 5)
  → build_unified_viewer (with all assertions from Phases 2 + 6)
```

**Implementation:**
- Define as a single `pixi` task that shells to `alz/runners/main/run_all.sh`.
- `run_all.sh` is the script; every step is `run_step` (hardfail). Comments mark which existing runner each step replaces.
- Keep `pixi run live`, `pixi run human`, `pixi run viewer` as named subgraph shortcuts pointing at the same script with stage filters. Or kill them — decide once `run_all.sh` is working.
- Remove `alz/runners/main/run_live_pipeline.sh`, `run_pair_mode_pipeline.sh`, `run_pair_mode_viewer_build.sh`, `run_dual_analysis.sh` if `run_all.sh` supersedes them. Anything not superseded gets called from `run_all.sh`.

**Success:** a clean repo can produce a working viewer from `pixi install && pixi run all` (assuming raw data is mounted). No manual orchestration between stages.

---

## Phase 9 — Dead code deletion + doc reconciliation

Closes: Theme E (docs), Theme F (dead code), WARN #18 keeps but undocumented (note: per user, `_raw` human MEA outputs stay for future tab).

**Code deletions:**
- `config.KL_THRESH` (NOTE #29).
- `config.SEAAD_TO_WMB_CLASS_FILE` and any related dead constants (NOTE #30, done in Phase 1).
- `alz/decomposition/paths.py:41` `SITE_OLS_FILE` — fix or delete depending on Phase 4 resolution (WARN #23, done in Phase 4).
- Sweep `alz/` for any function/class with zero importers (use `grep` or `vulture`). Delete.
- Sweep `alz/viewer/template/js/` for any function/handler not wired into the active tabs.

**File deletions:**
- `outputs/reports/wmb_expression/wmb_kinase_expression_cortex_hpf.csv`
- `outputs/reports/wmb_expression/wmb_kinase_expression_whole_brain.csv`
- All `*.pre_levy_spine.bak` files under `outputs/reports/wmb_expression/`
- `outputs/reports/deconvolution/per_animal/site_level_ols.parquet` (after Phase 4 replacement is verified)
- Any `outputs/reports/decomposition/levy19/` subtree (CLAUDE.md says "may stay as historical record" but it has no live producer — confirm with user before deleting). **Default: delete.**
- Stale `outputs/reports/kinase_attribution_human/perdonor/run_mukesh_perdonor.log` (after Phase 5 regenerates a fresh log).

**Keep (per explicit user instruction):**
- `recurrence_raw*`, `recurrence_ctrl_raw*`, `mea_substrate_sets_raw*` in human per-donor output. Future tab consumer.

**Doc reconciliation:**
- `docs/plans/change_request_02_spine_rethreshold.md` — Phase 2 has run; mark complete and move to `docs/archive/plans/`.
- `docs/plans/viewer_audit_2026-05-20.md` — fold items 4–10 (perf/robustness) into a Phase 9b sub-list below. Then archive the audit doc.
- `docs/plans/incytr_mea_seed_list_expansion_plan.md` — leave open (deferred future workstream per user).
- `docs/plans/evidence_tab_redesign_2026-05-21.md` — leave open (separate UI work).
- `CLAUDE.md` — update the snrna_integration description (Phase 1), add the human pipeline section (Phase 5), and add `pixi run all` to "Running the Analysis" (Phase 8).

**Phase 9b — fold-in from viewer_audit_2026-05-20.md punch list:**
- Item 4: substrate_motifs / leading_substrates inlining — verify whether human-perdonor shards already covered this; if still inlined, shard.
- Item 5: replace `iterrows()` in `build_human_slice()` at line ~950 with a vectorized pass.
- Item 6: add LRU cap on `TranscriptTraceStore`.
- Item 7: make shard-clear robust to partial directory state.
- Item 8: filter unused columns from decomp-OLS shards before write.
- Item 9: review `_khPerdonorFor` indexing for correctness.
- Item 10: add LRU cap on `MeasurementTraceStore`.

**Success:** `grep -r "WMB-34\|levy_19\|incytr_factorial\|softfail" alz/` returns nothing live. No file in `outputs/reports/` has zero readers. `docs/plans/` contains only open plans.

---

## Phase 10 — Empty cells in viewer where data is sparse

Closes: WARN #26 (no indicator when Song concordance is NaN), CRITICAL #4 follow-on once Phase 3 lands.

Per user decision: empty cells. No badges, no footers, no "no data" UX.

**Edits:**
- `alz/viewer/template/js/widgets/evidence_row.js` and consumers — confirm that NaN/null in the source columns renders as blank, not as `"NaN"` or `"null"` or `"undefined"`.
- `alz/kinase_attribute.py:408-410` — remove the WMB-parent warning since it now applies uniformly across all sources (or keep as build-time log only, not user-facing).

**Success:** Levy-t5 cells with no Song concordance, no per-cluster MEA signal, or no per-animal OLS appear visually blank. No additional UX scaffolding.

---

## Execution order summary

| Phase | Depends on | Output |
|---|---|---|
| 0  | — | Fresh mouse kinase pipeline outputs |
| 1  | — | Vocabulary + factorial branch deleted |
| 2  | 1 | Hardfail gates wired |
| 3  | 0, 2 | 31-cluster `mea_per_cluster.parquet`, verify passes |
| 4  | 3 | Levy-t5 per-animal site-level OLS producer + output |
| 5  | — | Human chain first-class with runner + pixi task |
| 6  | 0, 3, 4, 5 | Provenance stamps in summaries; viewer asserts on mismatch |
| 7  | — | Complete kinase→gene mapping, no heuristic |
| 8  | 1–7 | Single `run_all.sh` runner + `pixi run all` |
| 9  | 1–8 | Dead code purged, docs reconciled, perf punch list closed |
| 10 | 3, 4 | Viewer renders blank cells for sparse data |

After Phase 10: rebuild the viewer with `pixi run all`; confirm `index.html` `PAYLOAD.meta.generated_at` reflects the new build and every assertion fires cleanly.

---

## What this plan does NOT include

- MEA seed-list expansion for Incytr (deferred — `incytr_mea_seed_list_expansion_plan.md` kept open).
- Evidence tab UI redesign (separate plan).
- `_raw` human MEA outputs cleanup (kept on disk for future tab).
- Resurrecting the `kinase_backbone_edges.parquet` chain (closed path; the docstring referencing it is being deleted).
