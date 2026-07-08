# Incytr three-cohort re-run — KsG + PTM + backbone (orchestration plan)

**Status:** in progress (branch `feat/incytr-backbone-refactor`). **Date:** 2026-06-29. **Theme:** B.
**Companions:** [`ksg_kinase_imputed_nodes.md`](../../archive/archived_plans/orchestration/theme_b/ksg_kinase_imputed_nodes.md)
(KsG design — shipped, archived), [`backbone_incytr_track.md`](../foundation/backbone_incytr_track.md)
(authoritative backbone spec).

## Goal

Re-run pair-mode Incytr on all three cohorts (Song, 5xFAD, t-cell), memory-safe,
updating results to carry three feature layers:

1. **KsG** — kinase-inferred candidate genes added to `gene.use`.
2. **Acetylation + ubiquitination (Ack/KGG)** — extra PTM scoring tracks.
3. **Backbone** — per-grain (R-EM, L-R-EM, R-EM-T) sub-chain re-scores.

## Decisions taken (2026-06-29)

- **KsG = admission-only.** KsG only widens `gene.use`; a kinase-supported gene whose
  phospho is missing **stays missing**. The `|PDS| ≥ 0.2` floor self-limits over-emission,
  so **no per-cluster cap** and **no expressed-gate / calibration inputs**. This
  **overturns the prior locked decision 3 (imputation)** in `ksg_kinase_imputed_nodes.md`;
  that doc's decision 3, the cap (§KsG-set logic #4), Layer 2 (`preseed_phospho`), and the
  imputed-marker viewer work are removed, not toggled off (anti-shim).
- **T-cell KsG = donor1 only.** donor1 has an IMAC/ST track + `unified_attribution_tcells.csv`.
  donor2 is pY-only with no ST and no kinase→cell-type attribution → donor2 runs without KsG
  (still gets backbone + its pr/py paths). No fabricated attribution.
- **Scope this round = compute, not viewer surfacing.** Phase 0 + Phase 1 (+ compute-side
  Phase 3 builds/gates) ship the updated `wide`/`backbone` parquets, KsG labels, and 5xFAD
  PTM. Porting the backbone-grain UI to the 5xFAD + t-cell viewers (Phase 2) is a separate
  later pass; the parquets are the durable artifact.

## Current state — per cohort × feature

| Cohort | KsG inputs | PTM (Ack/KGG) | Backbone compute | Backbone in viewer |
|---|---|---|---|---|
| **Song** | ready (`mea_stoichiometry{,_pY}.csv`, `stoichiometry_matrix.csv`, `kinase_hypothesis_table.csv`) | **none** (no AcK/KGG data — gated out) | **done** (9×3 grains on disk) | **done** (`song.py`) |
| **5xFAD** | MEA + matrices per tissue/modality present; **attribution adapter needed** (`fivexfad_snrna_attribution.csv` → long `kinase,cell_type`) | **ready & wired** (`--ptm`: `ack/kgg_deconvoluted.csv`, both tissues; prior `wide_ptm/` exists) | not emitted (no per-cohort `BACKBONE_OUT_DIR`) | **not ported** (only `song.py` has grain payload) |
| **t-cell (donor1)** | `stoichiometry_matrix.csv` + `unified_attribution_tcells.csv` present; MEA is timecourse-shaped — confirm `kinase_substrate_gene` parses it | none | not emitted | not ported |
| **t-cell (donor2)** | n/a (no KsG) | none | not emitted | not ported |

Viewer topology: **Song + 5xFAD share the unified viewer** (`build_unified_viewer.py`,
tissue-gated; 5xFAD read from `incytr_pair_mode_5xfad/{tissue}/wide/`). **t-cell is
separate** (`build_tcell_viewer.py`). Bridge `kinase_incytr_bridge` already takes
`--cohort {song,fivexfad}` + `--tissue`.

## Phase 0 — wiring (no compute; review-gated before any run)

**0a. KsG attribution-long builder** — one script emitting per-cohort long-form
`kinase,cell_type` (replaces the scratchpad `ksg_attribution_long.csv`):
- Song: melt `kinase_hypothesis_table.top_celltype_{1,2,3}`.
- 5xFAD: adapter from `fivexfad_snrna_attribution.csv` (per tissue).
- t-cell donor1: from `unified_attribution_tcells.csv`.

**0b. Wire `KSG_*` into the three runners** (set only where KsG is active):
`KSG_MEA_FILE`(+`_PY_FILE`), `KSG_MOTIF_FILE`, `KSG_ATTRIBUTION_FILE`, `KSG_CONTRAST`.
Do **not** set `KSG_CAP`, `KSG_EXPR_FILE`, `KSG_MEASURED_FC_FILE` (admission-only).
Resolve the per-cohort `KSG_CONTRAST` label map (Song `ma_<age>_<geno>`→`<geno>_<age>`;
5xFAD `TG_<age>mo`→MEA label; t-cell `d<later>`→MEA label).

**0c. Per-cohort `BACKBONE_OUT_DIR`** in the 5xFAD + t-cell runners, alongside each
cohort's `wide/` (5xFAD `incytr_pair_mode_5xfad/{tissue}/backbone/`; t-cell
`incytr_pair_mode_tcells/{donor}/backbone/`). Today both default to the **shared Song
backbone path** — a misroute.

**0d. Anti-shim removal** (admission-only is now the only mode): delete the driver's
imputation seeding block + `KSG_EXPR_FILE`/`KSG_MEASURED_FC_FILE`/`KSG_CAP` reads from
`incytr_commandline.R`; update `ksg_kinase_imputed_nodes.md` decisions to admission-only
(no tombstones). The package method keeps its optional `measured_fc`/`cap` params (general
capability, exercised by its own tests) — the app stops passing them.

**0e. Memory-capped wrappers** for 5xFAD + t-cell, mirroring `regeneration/run_backbone_overnight.sh`
(`systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0`, `NPAIR_WORKERS=1`,
logged, resumable). 5xFAD + t-cell runners have **no cap today**.

## Phase 1 — compute (memory-safe, per cohort, gated between cohorts)

Each cohort = one capped overnight runner; **stop and report after each** before the next.

1. **Song** (KsG ON): guard with `verify-incytr-sce4` run KsG-**OFF** (byte-identical) →
   then production run KsG-ON, 9 contrasts × 961-pair grid → `wide/` + `backbone/` + ksg labels.
2. **5xFAD** (KsG ON + `--ptm`): 8 contrasts (cortex+hippocampus × {3,6,9,12}mo) →
   `wide_ptm/` + per-tissue `backbone/`. `derive_phospho_from_ptm.py` before filter.
3. **t-cell** (donor1 KsG ON, donor2 KsG OFF): 7 contrasts → `wide/` + per-donor `backbone/`.

Scale note: each pair peaks ~13–15 GB at `NPAIR_WORKERS=1`; the 24 G cap kills a runaway
job, not the box. KsG-ON widens `gene.use`, so more than Song's frozen-base 418 pairs may
enumerate — the `|PDS|≥0.2` filter is downstream and unchanged.

## Phase 2 — viewer surfacing (DEFERRED — separate later pass)

Out of scope this round. Future work: port `song.py`'s backbone-grain payload (heatmap
tensors, R-EM/L-R-EM inline + R-EM-T sharded, grain selector, spine index) to
`cohorts/fivexfad.py` and the t-cell viewer (repo rule: lift, don't rewrite), plus the KsG
badge (`_INCYTR_LABEL_VOCAB = ("DEG","prG","KsG")` + JS third color + driving-kinase tooltip).
Song's viewer already carries both.

## Phase 3 — bridge + builds + verification (compute-side this round)

Per cohort: `kinase-incytr-bridge --cohort <c>` → viewer build (Song unified + t-cell viewer
rebuild to refresh #Backbones/#Paths from the new `wide/`; the backbone-grain UI port is
Phase 2).

Gates: (1) Song `verify-incytr-sce4` + `-full` PASS with KsG OFF; (2)
`verify_backbone_counts.py` per cohort; (3) `verify_backbone_spine_index.py`; (4) builds
clean. PTM tracks surface **only** for 5xFAD (Song/t-cell gate Ack/KGG out — no empty columns).

## Launch protocol

Per the multi-hour-job rule: the capped overnight runners are run by the operator in
`tmux`, one cohort at a time, reviewed at each boundary — **not** launched as in-session
background processes.

## Open verification points (resolve at build, not assumed)

- Does `kinase_substrate_gene` parse t-cell's **timecourse-shaped** MEA (needs `Leading
  substrates` + `FDR` + `NES`)? The KsG doc claims `mea_timecourse.csv` carries leading
  substrates — confirm before donor1 KsG.
- Confirm the unified-viewer build reads 5xFAD backbone from the **5xFAD tree** once
  `BACKBONE_OUT_DIR` is set there (else Phase 2 payload won't find it).
- 5xFAD `KSG_CONTRAST` label: confirm the exact MEA contrast string in
  `cortex_st_mea_stoichiometry.csv` matches the runner's `TG_<age>mo` mapping.
