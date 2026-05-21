# Unified Viewer — End-to-End Input Provenance Audit (2026-05-21)

Audit of every input consumed by `alz/build_unified_viewer.py`, traced back to raw data under `data/datasets/` and `data/external/`. Goal: surface every coherency hazard — stale intermediates, schema drift, mismatched cohorts/spines/scopes, half-finished migrations, undocumented branches, broken provenance trails.

Seven Explore subagents ran in parallel against disjoint slices. This report consolidates their findings.

---

## Executive Summary — Top Issues

### CRITICAL (must fix or verify before trusting the current viewer)

1. **Mouse kinase core: enrich/mechanism outputs are older than their normalize inputs.** All 14 ST+pY downstream artifacts (`site_level_ols*.csv`, `mea_stoichiometry*.csv`, `mea_global_shift*.csv`, `winsorized_sites*.csv`, `mea_substrate_sets*.csv`, `mea_raw_phospho*.csv`) were written 09:19–09:21 on 2026-05-14, **before** `stoichiometry_matrix.csv` and `raw_phospho_normalized.csv` (09:26 same day). The downstream files therefore reflect a *previous* revision of the matrices. The viewer is currently serving these stale enrich outputs. → **Re-run `normalize → enrich → mechanism` in correct dependency order.**

2. **`per_animal/site_level_ols.parquet` is an orphan from the closed CTM deconvolution path.** Mtime 2026-05-10, predates the current `raw_phospho_normalized.csv` (2026-05-14). No live script produces this file — `run_pivot_smoke.sh` and `run_pair_mode_pipeline.sh` have no regen step. Worse, its `wmb_class` column carries WMB-34 labels (e.g. `"01 IT-ET Glut"`) while the viewer's other decomposition source (`mea_per_cluster.parquet`) uses Levy-t5 names (e.g. `"Astrocytes"`). `_write_decomp_ols_slices` writes both into the same column without reconciliation → **schema drift inside a single edge-slice column.**

3. **`verification.json` reports `all_pass: false` and the pipeline did not abort.** Two contracts fail: `per_cluster_vs_bulk_mea` (`min_rho=0.571 < 0.7`, `ApTt_2mo`) and `incytr_pair_count` (361 vs expected 930+). Verification is invoked as `run_step_softfail V` in `run_pair_mode_pipeline.sh:212`, so the build proceeded despite contract violations. **The viewer currently ships data that does not satisfy its own correctness gates.**

4. **`enrich_celltype.py` silently drops 12 of 31 Levy-t5 clusters** ("no sites with any data"). `mea_per_cluster.parquet` covers only 19 clusters. The viewer is built against a *claimed* 31-cluster spine but ~39% of clusters are absent from the per-cluster MEA tab with no user-facing indicator. Identical 12 clusters are missing from the Incytr pair-count check.

5. **Module docstring in `alz/build_unified_viewer.py:1–25` describes artifacts that do not exist.** Claims the viewer streams `kinase_backbone_edges.parquet` (7.14 GB / 2.23B rows), produces `kinase_backbone_edges_sig.parquet`, and writes `edge_slices/{kinase,backbone}/`. **None of these are produced or consumed by the live code.** The producer (`_build_backbone_edges` in `aggregate_cross_pair.py`) is archived under `archive/code/viewer_backbone_strip_20260507/` and `archive/incytr_integration/adapters/`. The on-disk shard layout is `edge_slices/{human_perdonor,decomp_ols,song_concordance,incytr_pathways}/` — entirely different. This docstring is actively misleading.

6. **Factorial Incytr branch still reachable via `INCYTR_SOURCE` env var.** `alz/build_unified_viewer.py:2860–2866` dispatches on `os.environ.get("INCYTR_SOURCE", "pair_mode")`; setting `factorial` invokes `_write_incytr_pathways()` against `outputs/reports/incytr_factorial/receiver_cache/`, which is currently populated (33 partitions, mtime 2026-05-20). Per CLAUDE.md, factorial Incytr was archived 2026-05-18 and the upstream API was deleted at commit `424119f`. This violates the project rule "research pivots replace, they do not coexist."

7. **`kinase_attribution/seaad_agreement.csv` does not exist.** Listed in audit scope; actual file lives at `outputs/reports/kinase_attribution_human/seaad_agreement.csv`. The viewer reads it from the `_human` directory at `build_unified_viewer.py:1092–1094`. The cross-track naming is a coherence trap.

8. **`seaad_human_agreement.py` has no runner and no orchestration.** Not invoked by any pixi task, `run_pair_mode_pipeline.sh`, or `run_mukesh_perdonor.sh`. `seaad_agreement.csv` is updated manually. If `ingest_mukesh_perdonor.py` is re-run, the viewer will embed stale SEA-AD agreement against fresh per-donor NES with no guard.

9. **Human per-donor run log is stale; canonical run was off-runner.** `perdonor/run_mukesh_perdonor.log` records a 2026-05-14 12:40 run, but all perdonor CSVs are dated 2026-05-16 01:50–01:52. The 2026-05-16 invocation did not go through `run_mukesh_perdonor.sh` (which would have overwritten the log) — provenance for the canonical artifacts is undocumented.

### WARN (correctness-adjacent or process hazards)

10. **`kinase_to_gene_mapping.csv` postdates the attribution outputs by ~5 hours** (mtime 2026-05-16 15:46 vs `unified_attribution*.csv` at 10:34). On-disk outputs may embed stale gene symbols; re-run `attribute` + `recovery` to confirm. Same file feeds `seaad_human_agreement.py` (which then ran at 15:46, after the mapping update — that branch is consistent).

11. **`song_concordance.csv` was written 2026-05-14 against a `kinase_to_gene_mapping.csv` that was then updated 2026-05-16.** The FDR is computed over a possibly-smaller kinase universe than the current mapping covers.

12. **`alz/wmb_expression.py` and `alz/snrna_integration.py` were modified 2026-05-20**, but their output CSVs are dated 2026-05-14 — 6-day lag. Viewer is serving artifacts from an earlier version of both producing scripts.

13. **Stale "WMB-34" labels in live code:**
    - `alz/build_unified_viewer.py:152` — audit-spec label `"Unified attribution (all 34 WMB classes)"` (actual: 31 Levy-t5).
    - `alz/attribution_recovery.py:89` — comment `"1/34 ≈ 0.029 under the WMB-class spine"` (actual threshold is `1/31 ≈ 0.0323`).
    - `alz/kinase_attribute.py:25` — docstring `"n_kinases × 9 × 19"` (Levy-19 residue).
    - `alz/config.py:471,481` — two `load_cluster_to_*` docstrings say "19 entries" (actual: 31).
    - `CLAUDE.md:180` — says snRNA integration is "rolled up to 34 WMB classes via wmb_class_manifest.csv"; the code actually uses `barcode_to_cluster.csv` → Levy-t5 directly.
    These are docstring/comment drift only — values are correct in code — but they collectively obscure which spine is canonical.

14. **`source_mode: "factorial"` stale label** in `_write_incytr_pathways()` return dict (`build_unified_viewer.py:2299`); inline comment says "becomes pair_mode" — the swap was never made.

15. **`incytr_heatmap.js:311` fallback message references `outputs/reports/incytr_factorial/receiver_cache/`** as the data location — wrong since the pair-mode pivot.

16. **Pair-mode wide parquets are 4.9 days older than the `data/derived/incytr_inputs/` they depend on** (2026-05-16 vs 2026-05-20). `run_pair_mode.sh:62-65` skips contrasts whose parquet already exists. If the May 20 input refresh changed anything materially, no re-run will happen automatically.

17. **No `analysis_mode` stamp** in `normalization_summary*.json`, `attribution_summary.json`, or any CSV header in the mouse kinase + attribution chain. By contrast, `enrich_audit.json` (decomposition) does stamp `"analysis_mode": "males_only"`. A `KEDRO_ENV=full_cohort` rerun would overwrite outputs without a programmatic way to detect the cohort.

18. **`recurrence_raw*`, `recurrence_ctrl_raw*`, `mea_substrate_sets_raw*` (human per-donor)** are produced (~113 MB) but not consumed by the viewer.

19. **Human per-donor branch is undocumented in CLAUDE.md** — `ingest_mukesh.py`, `ingest_mukesh_perdonor.py`, `seaad_human_agreement.py`, `run_mukesh_perdonor.sh` are not described under "Running the Analysis", "Architecture", or "Runners".

20. **Several open punch-list items in `docs/plans/viewer_audit_2026-05-20.md`** (items 4–10: substrate-motif inlining, iterrows in human-slice builder, TranscriptTraceStore LRU caps, shard-clear robustness, decomp-OLS column filter, etc.) — none of these were addressed in the last 20 commits.

21. **`docs/plans/change_request_02_spine_rethreshold.md`** header says "Phase 2 (full pipeline rerun) pending" but `enrich_audit.json` shows `spine: levy_t5, males_only` with artifacts dated 2026-05-15. Either Phase 2 ran and the plan is stale, or it ran partially. Document is out of sync with disk.

22. **`docs/plans/incytr_mea_seed_list_expansion_plan.md`** describes building `bench/build_mea_gene_list.R`; the builder does not exist anywhere. The current pair-mode wide parquets were generated *without* the planned MEA seed-list expansion. If this plan is considered active, Incytr outputs are stale relative to the intended gene universe.

23. **`_write_decomp_ols_slices` reads `outputs/reports/deconvolution/per_animal/site_level_ols.parquet`** — `alz/decomposition/paths.py:41` declares a *different* `SITE_OLS_FILE` path (`outputs/reports/deconvolution/site_level_ols.parquet`, no `per_animal/`) which does not exist. `variance_audit.py` will fail if run. Path inconsistency between viewer and decomposition module.

24. **`alz/wmb_expression.py:553-554` "mapping_fresh" guard is confusingly named** — it returns `True` when expression is newer than mapping (i.e. cache is fine to keep), but the variable name reads the other way. Current state: mapping is newer → cache should recompute on next run, which is the correct behavior, but only by coincidence with the variable semantics.

25. **`alz/build_unified_viewer.py` does not validate `wmb_kinase_expression.scope.json`.** If `WMB_REGION_SCOPE` were ever set to `cortex_hpf` at viewer-build time against a `whole_brain` cache, the viewer would silently serve mismatched data.

26. **`song_concordance.csv` has rows for only 17 of 31 Levy-t5 clusters** (14 clusters got NaN from rank-deficient OLS). `kinase_attribute.py:408-410` warns on missing WMB parent but **not** on missing Song concordance. The viewer surfaces no indicator that ~45% of clusters have no Song evidence.

27. **`kinase_to_gene_mapping.csv` has 389 entries but `kldata.csv` has 562 unique gene names.** The 173 unmapped kinases get a title-case heuristic in `atlas_reference.get_all_kinase_genes()`, which is unreliable (e.g. `p38 → P38`). Unmapped kinases may be silently excluded from WMB expression.

28. **`alz/map_kinases_to_genes.py:83`** hardcodes an input path (`outputs/deconv/enrichment_summary_sig_kins.csv`) that does not exist on disk. The mapping cache's 389 entries cannot currently be refreshed via this script.

### NOTE (cosmetic / documentation drift)

29. **`config.KL_THRESH = 15`** in `alz/config.py:77` is orphan dead code (no Python consumer; tracks read per-track from `PHOSPHO_TRACKS`). Hazard: silent divergence if `PHOSPHO_TRACKS["st"]["kl_thresh"]` is changed.

30. **`config.py:128–131` `SEAAD_TO_WMB_CLASS_FILE`** retained "pending kinase_attribute.py rewire" — no current consumer; half-finished migration marker.

31. **Orphaned scope-named files** in `outputs/reports/wmb_expression/`: `wmb_kinase_expression_cortex_hpf.csv` and `wmb_kinase_expression_whole_brain.csv` (mtime 2026-05-14 08:09–08:24, before the canonical 08:40 file). No live consumer. `pre_levy_spine.bak` files also exist; inert.

32. **`run_pair_mode_viewer_build.sh`** intentionally reshapes pair-mode output into `incytr_factorial/receiver_cache/` so downstream code doesn't need flags. Functions correctly today but means `INCYTR_FACTORIAL_OUTPUTS_DIR` is still written, perpetuating the factorial-name coupling.

33. **`pixi run live` and `pixi run dual` cover only `ingest → normalize → enrich → attribute → recover`.** They do not invoke `wmb-export`, the Levy-t5 decomposition stack, pair-mode Incytr, receiver-cache reshape, or the viewer build. A fresh viewer build requires multiple manual runner invocations — not documented as a single golden-path workflow.

34. **`docs/plans/evidence_tab_redesign_2026-05-21.md`** was created today and is unimplemented. The current `index.html` (07:43 build) predates it. Known intermediate state.

---

## All-Clear (confirmed coherent end-to-end)

- **SEA-AD variant routing** (`App→early`, `Tau→late`, `ApTt→full`) is consistent between `config.SEA_AD_PATHWAY_MAP`, `kinase_attribute.py:171–181`, and the three strata in `sea_ad_supertype_lfc.csv`. `seaad_human_agreement.py` correctly uses only `effect_sizes.h5ad` (full CPS) for the human cohort.
- **Cell-type spine = Levy-t5 (31 clusters), single source of truth** across `config.CLUSTER_SPINE`, `_assemble_unified` cross-join, `unified_attribution_full.csv` (31 unique `cell_type` values), `cluster_to_wmb_class.csv` (31 rows, full coverage), `cluster_to_seaad_supertype.csv`, decomposition pipeline, and pair-mode Incytr receiver-cache (31 receiver partitions). Reachable WMB-34 / Levy-19 mentions are docstring-only.
- **Row-count integrity of `unified_attribution_full.csv`** holds: 108,531 rows = (2,799 ST + 702 pY) × 31. The runtime assertion at `kinase_attribute.py:492–497` would have raised on a silent drop.
- **`WMB_REGION_SCOPE = whole_brain`** is stamped in `wmb_kinase_expression.scope.json` and matches the `config.py` default; no consumer hardcodes `cortex_hpf`.
- **No chained vocabulary crosswalks.** Levy-t5 → WMB class is one hop via `cluster_to_wmb_class.csv`; Levy-t5 → SEA-AD supertype is one hop via `cluster_to_seaad_supertype.csv`; Song concordance joins Levy-t5 directly by `cell_type` identity. Convention followed.
- **Mass identity contract** (`Σ_c [P_c × (N_c / N_total)] ≈ bulk`) passes in `verification.json` with `max_rel_err = 5.4e-8`. Forward-projection math is numerically correct for the 19 clusters that actually pass through.
- **Donor consistency (human per-donor)**: `donor_groups.json` matches `sample_mapping.csv` (10 AD + 7 CTRL); `kinase_donor_nes.csv` and `_pY.csv` headers are identical; ST↔pY mtime parity within the same continuous run.
- **All 9 Incytr pair-mode contrast parquets present** with the correct `ma_{2,4,6}mo_{AppP,Ttau,ApTt}_ma_<age>mo_WTyp_incytr_output.parquet` pattern. Sender/receiver vocabulary derived at runtime from the data, not hardcoded — no vocabulary mismatch in the active branch.
- **`kldata.csv` symlink resolves correctly** to `data/datasets/song/kinase/kldata_pspy.csv`.
- **`incytr_obj.rds` was rebuilt for the Levy-t5 spine** (2026-05-20), post-spine-migration.
- **Raw-data leaves all exist** on disk: Song proteomics + IMAC + pY xlsx (2026-04-03), Sample list xlsx, Mukesh raw CSVs, SEA-AD `effect_sizes*.h5ad` (2026-04-02), all 13 WMB-10Xv3 region h5ads under `data/external/allen_abc/`.
- **Mtime ordering coherent** from raw data → ingest → wmb_expression → kinase_attribution → attribution_recovery → decomposition → incytr_pair_mode → unified_viewer (with the explicit exceptions called out in CRITICAL #1 and #2).
- **`min_cells = 5`, no rank gate** is consistent across `build_cluster_spine.py`, `spine.scope.json`, `config.SONG_MIN_CELLS`, and `snrna_integration.py`.

---

## Recommended Action Priority

**Immediate (blocking trust in the current viewer):**

- Re-run `pixi run normalize && pixi run enrich` and `kinase_mechanism.py` to fix CRITICAL #1 stale enrich outputs.
- Decide on `per_animal/site_level_ols.parquet`: either retire from viewer entirely (CRITICAL #2), regenerate against Levy-t5 vocabulary, or treat as a historical-only audit slice with explicit user-facing labeling.
- Re-run `pixi run viewer` after the above.
- Either fix the 12-cluster sparsity gate (CRITICAL #4) or surface it in the UI as an explicit "no signal in these clusters" indicator.
- Convert `run_step_softfail V` → `run_step` for `verify_decomposition.py` (CRITICAL #3) so the build fails on contract violation.

**Process / consistency:**

- Add `analysis_mode` stamps to all output sidecars and CSV headers (WARN #17).
- Delete the archived backbone-edge docstring in `alz/build_unified_viewer.py:1–25` (CRITICAL #5).
- Remove the factorial Incytr branch and `INCYTR_SOURCE` env var (CRITICAL #6) or guard with `raise RuntimeError("factorial archived 2026-05-18")`.
- Wire `seaad_human_agreement.py` into `run_mukesh_perdonor.sh` and document the human chain in CLAUDE.md (CRITICAL #8, WARN #19).
- Fix the `seaad_agreement.csv` path naming (CRITICAL #7) — move it under `kinase_attribution/` or rename to `human_seaad_agreement.csv`.
- Replace all WMB-34 / Levy-19 residual docstring/comment references with Levy-t5 (WARN #13).
- Reconcile or close the stale plan documents (WARN #21, #22).
- Re-run `kinase_attribute.py` + `attribution_recovery.py` after the 2026-05-16 `kinase_to_gene_mapping.csv` update (WARN #10).

**Cleanup:**

- Remove `config.KL_THRESH`, `SEAAD_TO_WMB_CLASS_FILE`, orphan scope-named CSVs, `.bak` files (NOTE #29–31).
- Add scope-stamp validation to `build_unified_viewer.py` (WARN #25).
- Document the full viewer-build chain as a single golden-path workflow (NOTE #33).
