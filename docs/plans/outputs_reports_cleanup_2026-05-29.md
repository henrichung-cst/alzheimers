# `outputs/reports/` bloat audit — 2026-05-29

`outputs/` is **gitignored** — this is pure disk cleanup, not a code change. No
commit results from deletions. Total today: **31G across 30 report dirs.** Goal:
identify what is still relevant and reclaim the dead weight.

Method: per dir, counted live path references in `alz/ conf/ docs/ bench/ pixi.toml`
(excluding `archive/`), cross-checked mtime, and traced each producer/consumer.
The big levers are *inside* `incytr_pair_mode` (20G), not only whole dirs.

---

## Tier A — SAFE DELETE: closed paths, superseded, zero live refs (~1.4G)

| Dir | Size | mtime | Why dead |
|---|---|---|---|
| `unified_viewer_OLD` | 856M | 05-14 | Superseded by `unified_viewer`; viewer build writes/serves the non-OLD dir. |
| `incytr_factorial_5xfad_kldata` | 263M | 05-14 | Factorial Incytr **archived 2026-05-18 / closed path**. No live ref. |
| `deconvolution` | 232M | 05-10 | Direct (statistical) deconvolution is a **closed path**. No live producer/consumer (`pixi ingest-deconvolution-bulk` writes to `data/datasets/…`, not here). |
| `audits` | 19M | 05-14 | Phase2/3/4 spine-pivot logs + `kinase_mapping_rerun_2026-05-13`. Historical run logs, 0 refs. |
| `incytr_pair_levy_t5` | 2.8M | 05-22 | Earlier pair run (logs + probe subdirs), superseded by `incytr_pair_mode`. The one `bench/` ref points at `bench/incytr_pair_levy_t5/`, a different path. |
| `incytr_factorial_subset_perm_bench` | 620K | 05-12 | Factorial perm-count bench, closed path. |
| `spine_audit` | 20K | 05-15 | One-off cells-per-design-cell diagnostic CSVs, 0 refs. |

## Tier B — SAFE DELETE: smoke/probe intermediates inside the live `incytr_pair_mode` (~1.2G)

| Path | Size | Why dead |
|---|---|---|
| `incytr_pair_mode/wide_smoke2` | 1.2G | Smoke-test output of `run_pair_mode.sh` (`wide_smoke` target). Not read by the viewer. |
| `incytr_pair_mode/wide_smoke1` | 17M | Earlier smoke run. |
| `incytr_pair_mode/probe_*` | ~5M | `probe_gene_use_min`, `probe_short_list`, `probe_sce4_rule`, `probe_sender_batch` — sce4-investigation probe runs (bench history is captured in `bench/bench.md`). |

## Tier C — DECISION: stale sensitivity-mode copies (regenerable, ~252M)

`run_dual_analysis.sh` does `cp -r kinase_attribution → kinase_attribution_males_only`
and `… → kinase_attribution_full_cohort`. The canonical `kinase_attribution`
(05-21) is **newer** than both copies (05-14) — i.e. the copies are stale
snapshots from an older dual run, fully regenerable by re-running the dual script.

| Dir | Size | mtime |
|---|---|---|
| `kinase_attribution_full_cohort` | 129M | 05-14 |
| `kinase_attribution_males_only` | 122M | 05-14 |
| `attribution_recovery_full_cohort` | 988K | 05-14 |
| `attribution_recovery_males_only` | 984K | 05-14 |

Decision: delete now (stale + regenerable) vs keep as the sensitivity deliverable.

## Tier D — DECISION: `wide` vs `wide_nboot0` inside `incytr_pair_mode` (5–9G lever)

| Path | Size | Producer / consumer |
|---|---|---|
| `incytr_pair_mode/wide` | 5.0G | 9 bootstrap wide parquets (≈400–580M each). **Read by `build_unified_viewer.py` + `pair_to_receiver_cache.py`** — the current live viewer source. |
| `incytr_pair_mode/wide_nboot0` | 9.2G | nboot=0 (no permutation) run from `bench/perf/run_nboot0_w3.sh`; carries `COMPLETE.txt`, the retroactive `filter_significant_paths` logs, and regen/verify logs. |

These are two runs of the same 9 contrasts. The project has decided the **pair
pvalue is untrustworthy → rank on `|PDS|`** (CLAUDE.md), which makes the bootstrap
permutations in `wide` dead weight and points at `wide_nboot0` as the intended
production going forward. But the viewer **currently reads `wide/`**, so this is
not safe to flip unilaterally. One of three:
- keep `wide_nboot0`, delete `wide` (commit to nboot=0 production; repoint viewer) → reclaim 5.0G
- keep `wide`, delete `wide_nboot0` (treat nboot0 as a finished perf experiment) → reclaim 9.2G
- keep both (no reclaim)

## Tier E — KEEP (live)

`incytr_pair_mode/{wide-or-nboot0 per Tier D, receiver_cache}`, `unified_viewer`,
`decomposition/levy_t5` (2.2G, feeds Incytr inputs + MEA), `kinase_attribution`
(canonical), `kinase_attribution_human`, `snrna_integration`, `wmb_expression`,
`supplementary`, `change_requests`, `human_reference_expression`,
`atlas_reference`, `data_ingest`, `data_ingest_human`, `levy_t5_rebuild`,
`attribution_recovery`. All **tcells** dirs (`incytr_pair_mode_tcells` 2.1G,
`kinase_attribution_tcells`, `tcell_viewer`, `data_ingest_tcells`) — new cohort,
recent (05-27→29), active.

---

## Recoverable summary

| Action | Reclaim |
|---|---|
| Tier A + B (safe, no decision) | **~2.6G** |
| + Tier C (stale regenerable copies) | +0.25G |
| + Tier D (whichever wide variant) | +5.0G or +9.2G |

Outer bound if all approved: **~12G of 31G.**

## Status — DONE (2026-05-29)

Executed: **Tier A + B + C deleted; Tier D = keep `wide`, drop `wide_nboot0`**
(no viewer change — viewer still reads `wide/`). Reclaimed **31G → 19G (~12G)**.
Verified live dirs intact (`incytr_pair_mode/{wide,receiver_cache}`,
`unified_viewer`, `decomposition/levy_t5`, `kinase_attribution{,_human}`) and all
deletion targets absent. The 4 stale mode copies regenerate from
`run_dual_analysis.sh` if sensitivity outputs are next needed.
