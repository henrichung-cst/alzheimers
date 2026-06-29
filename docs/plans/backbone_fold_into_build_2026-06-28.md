# Backbone fold-into-build + B4.2 wiring

**Decision (2026-06-28):** Backbone counts are a build-step `GROUP BY`, not a standalone
pipeline artifact. One backbone definition, computed where the kinase join already lives
(B4's bridge). Only R-EM-T (2.78M rows) is materialized to a file — for B2's lazy read.
Everything payload-viable goes straight into the payload.

Supersedes the standalone B5 step (`backbone_reduction.py` as a pixi task / runner step J,
committed `ec91523`). Its `reduce()` aggregation SQL is reused as a library function — only
the standalone-step framing is removed.

## Why folding is correct (not just smaller)
- Backbone recurrence (`n_conditions_present`, `n_timepoints_present`, `backbone_rank`) is a
  cross-contrast `GROUP BY` over `wide/`, computed in ~9s — cheap to run inside the build.
- The two payload-viable groupings are tiny (R-EM 19,680; L-R-EM 27,927); the per-kinase
  `#Backbones` is a scalar. None needs a materialized intermediate.
- `#Backbones` is per-**kinase**, but B5's `backbone_table` has no kinase column — the count
  requires B4's kinase↔path join. So the recurrence aggregation and the kinase join must run
  together. B4's bridge is the natural home.
- B4 already computes an *approximate* `n_backbones` (`kinase_backbone_counts.csv`,
  "upper-bound estimate", no recurrence). Two definitions is the redundancy this removes.

## Semantic decision (RESOLVED 2026-06-28: show BOTH grains, any-node participation)

**The original "full-path grain" premise was falsified by the data and reversed.** The plan
claimed the kinase tab's shipped "15,028 chains" literal was a full-path count and that the
R-EM grain "would be hundreds." Both wrong (verified against the real artifacts):

| kinase | R-EM spine, any-node | full path, any-node | shipped literal |
|--------|---------------------:|--------------------:|----------------:|
| CAMK2D | 14,968               | 799,064             | 15,028          |
| CDK1   | 14,402               | 972,289             | 14,776          |
| CHK1   | 13,950               | 575,798             | 13,098          |

R-EM **any-node** reproduces the shipped numbers (≈0.4% off, ranking preserved); full-path is
53× larger and re-orders the table (CDK1 → #1). Only R-EM *EM-driver-only* is "hundreds"
(CAMK2D 1,461) — the figure the original premise confused with R-EM any-node. The full-path
number means a kinase "touches" ~20% of all 3.85M gated paths, dominated by downstream Target
fan-out the kinase does not control — a poor breadth signal.

**Decision (user, 2026-06-28): emit BOTH counts per kinase**, both any-node participation over
the gated `wide/` paths (canonical SigProb/PDS floor, pooled distinct across contrasts):

- `n_backbones` = distinct **(Sender, Receiver, Receptor, EM) spines** the kinase acts on —
  the breadth number; matches the shipped literal and the `#Backbones` name. Target fan-out
  collapsed.
- `n_paths` = distinct full **(Sender, Receiver, Ligand, Receptor, EM, Target) pathways** the
  kinase sits along — total end-to-end route involvement.

**Consequence for the fold:** both counts come from one `GROUP BY` over `wide/` joined to the
kinase-node attribution (unpivot positions → 4-key equi-join `(sender, receiver, role, gene)`,
then count distinct spines and distinct full paths per kinase). They do **not** need the
recurrence grouping. The recurrence table's only remaining consumer is **B2** (sankey); so the
reduction collapses to: materialize **R-EM-T only** for B2. The 3-grouping
`backbone_table.parquet` is fully retired.

## Phase 1 — Backend fold (gate before Phase 2) — DONE, verified

1. `kinase_incytr_bridge.py`: **replaced** `build_fan_and_backbone` with `build_recep_em_fan`
   (fan only) + `compute_participation_counts` — one DuckDB pass over the gated `wide/` parquets
   joined to the bridge's kinase-node attribution (unpivot positions → 4-key equi-join
   `(sender, receiver, role, gene)`), emitting per kinase **both** `n_backbones` (distinct
   R-EM spines) and `n_paths` (distinct full paths). DuckDB-streamed + memory-capped. Output:
   `kinase_participation.csv`. Kept `recep_em_fan.csv`. **Deleted** the `backbone_key` arg, the
   `full` branch, `--backbone-key`, the "provisional — pending B5" notes, and
   `kinase_backbone_counts.csv`.
2. `backbone_reduction.py`: collapsed to a single-grouping **R-EM-T** reducer, kept as a library
   (`reduce()`) **and called from the bridge's song branch** (the fold) — writes
   `outputs/reports/incytr_pair_mode/backbone/backbone_rem_t.parquet` (B2's lazy input).
   **Deleted** pixi task `incytr-backbone`, runner step J, and the 3-grouping
   `backbone_table.parquet`.
3. Verified: `n_backbones` reproduces the shipped literals (CAMK2D 14,968 vs 15,028, ranking
   preserved — the shipped column was the R-EM spine count all along); `n_paths` is the full
   route count alongside it; `backbone_rem_t.parquet` rows = 2,782,293, all sanity checks pass;
   memory capped throughout.
4. **Runtime / dead-code hardening (so the fold runs end-to-end).** Two pre-existing problems
   blocked an integrated bridge run; both removed:
   - `attach_node_fc` queried `{role}_st_log2FC`/`{role}_py_log2FC` columns that **do not exist**
     in the wide parquets (real names are `_sclog2FC`/`_pr_`/`_ps_`/`_py_log2FC`); every lookup
     threw and was swallowed by a bare `except`, so `node_log2FC`/`node_PDS` were **100% NULL**
     across all 9.57M rows while the step burned ~18 min in per-(role,sender,receiver,gene)
     full-parquet scans. No consumer reads those columns. Deleted the function, the columns, and
     the orphaned `_song_contrast_to_parquet`/`_fivexfad_parquet_name`/`_SONG_COND_MAP` helpers.
   - `annotate_celltype_match_{song,fivexfad}` looped `iterrows()` over the 9.57M-row hit table in
     pure Python. Vectorized to a melt → groupby-min-rank → left-join (`_apply_celltype_ranks`);
     `celltype_match` distribution preserved (90,331 matched, rank 1/2/3; rest None).
   With both fixed the full song bridge runs end-to-end in ~1–2 min and the fold call executes for
   real (bridge → `reduce()` → 2,782,293 backbones written). 5xFAD (cortex+hippocampus) regenerated
   on the same code path (removes their stale `node_*` columns, adds `kinase_participation.csv`).

## Phase 2 — B4.2 viewer wiring (consumes Phase 1) — wiring DONE, browser pass pending

4. `paths.py` + `song.py`: `KINASE_INCYTR_BRIDGE_DIR` added; `_build_kinases_slice` reads
   `song/kinase_participation.csv` and emits `n_backbones` + `n_paths` per kinase (NULL for
   kinases absent from the table — 92/389). Verified in the built payload: CAMK2D 14,968 /
   799,064, CDK1 14,402 / 972,289, CHK1 13,950 / 575,798 — byte-identical to the CSV.
5. `kinase_explorer.js`: two count columns — `#Backbones` (spine breadth) + `#Paths` (full
   routes), rendered via shared `_keCountCell` (thousands-separated, — when NULL), sorted via the
   fallback branch + `numCmp` (nulls always last). Added to the CSV export. Headers in
   `body.html` with `data-metric` tooltips (`nBackbones`/`nPaths` in `METRIC_DEFS`).
6. **Orphaned intro counts reconciled** — the kinase-tab intro's "CAMK2D 15,028 chains" literals
   (backed by no column) are replaced with the populated `n_backbones`/`n_paths` pairs, and the
   **wrong gate framing was corrected**: the count is any-node participation over the canonical
   `SigProb > 0.1 (either) AND |PDS| ≥ 0.2` floor, **not** an FDR < 0.25 chain pass. The intro,
   method, shows-lead, bullet, and the Receiver/Support toggle note now all state the counts are
   network-wide totals that do not respond to the FDR slider or filter bar (honesty-over-polish).
7. `pixi run viewer` ran clean (exit 0); payload + built HTML verified at the data/DOM level.
   **Browser click-through (human, authoritative) is the one remaining Phase 2 item.**
8. **5xFAD (MouseC2) wired too.** `fivexfad.py`: `_load_fivexfad_participation` reads
   `fivexfad_{cortex,hippocampus}/kinase_participation.csv` and stamps per-(kinase, tissue)
   `n_backbones`/`n_paths` onto each `supporting_5xfad` MEA row (NULL where the kinase is absent
   from that tissue's table — 2,104/6,224 rows). `kinase_fivexfad.js`: same two columns on the
   `#f5-table` (group rec fields, render cells, CSV export, empty-row colspan 12→14); sort via the
   `numCmp` fallback (nulls last); shared `nBackbones`/`nPaths` `METRIC_DEFS` tooltips. A bullet in
   the `fivexfadkinase` guide explains the columns are **per-tissue** network totals (counts are
   computed per cortex/hippocampus, not pooled). Verified in payload: 0 mismatches vs the two CSVs
   (e.g. AAK1/cortex 20,481 / 332,535).

## Deletions (anti-shim — no coexistence)
- pixi task `incytr-backbone`; runner step J (`run_pair_mode_pipeline.sh`).
- `backbone_key` param + `full` branch + `--backbone-key` flag + "provisional" notes in
  `kinase_incytr_bridge.py`.
- `kinase_backbone_counts.csv` as a standalone artifact; the 3-grouping
  `backbone_table.parquet` (replaced by R-EM-T-only `backbone_rem_t.parquet`).

## Out of scope
B2 sankey itself (separate tail item; this only leaves R-EM-T on disk for it); the fan
characterization (`recep_em_fan.csv` stays); any Incytr regen; t-cell cohort.
