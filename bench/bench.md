# Pair-mode Incytr — sce4 reproduction (investigation summary)

> **Status: SOLVED (A37, 2026-05-25).** The derived 31-spine inputs reproduce
> sce4's `DEG_PRG_Top300` table through the corrected method; the only residual
> is the App transgene (§3, documented and gate-exempt).
>
> **Canonical production path:** the parity overrides derived here live in
> `alz/incytr_pair/incytr_commandline.R` and `alz/incytr_pair/build_input_gene_list.R`.
> Run via `bash alz/runners/main/run_pair_mode_pipeline.sh`; gate with
> `pixi run verify-incytr-sce4`. Propagation record:
> `docs/plans/sce4_fix_propagation_2026-05-23.md`. Performance / OOM work:
> `docs/plans/pairmode_perf_oom_2026-05-25.md`.
>
> This file is the consolidated investigation record. The blow-by-blow probe log
> (A5–A37, ~1.4k lines) is in git history prior to 2026-05-25. The curated probe
> scripts referenced below remain on disk under `bench/probes/` as the experimental
> record; their regenerable output dirs were discarded in the 2026-05-27 cleanup.
>
> **Layout (2026-05-27 reorg).** `bench/` keeps only this record, the ground-truth
> CSV, and four subdirs: `fidelity/` (the parity gate), `perf/` (perf/OOM work),
> `decomposition/` (yuyu decon ports), `probes/` (curated refuted-rule probes).
> ~1.5 GB of regenerable output dirs and superseded one-off scripts were deleted;
> live production inputs were unaffected (they live in `data/derived/incytr_inputs/`).

---

## 1. Goal & ground truth

**Goal:** drive the raw omics through the corrected Incytr method to reproduce
sce4's published per-pair output, without OOM on the shared 30 GB box.

**Ground truth (non-negotiable):** `bench/sce4_DEG_PRG_Top300_table_10302025.csv`
— sce4's `DEG_PRG_Top300` table. Benchmark pairs (contrast `ma_2mo_AppP` vs
`ma_2mo_WTyp`): **Microglia → Cholinergic-Neurons** and **Ndnf × Ndnf**, 600 paths each.

Intermediate files (`incytr_obj.rds`, `input_gene_list.csv`,
`pr_yuyu_deconvoluted.csv`, `kldata.csv`) are *derived artifacts*, not preconditions
— a missing intermediate is a regeneration task, not a blocker. Collaborator contact
is out of the solution space; everything proceeds from artifacts on disk.

---

## 2. The reproduction recipe (the answer)

### 2a. Gene-use construction (per cluster)

- **Receiver** `gu_R = DEG ∪ prG`, where `prG = {genes with |pr_log2FC| > 1}`.
  **`pr_log2FC` is the value Incytr's scorer computes**, i.e.
  `limma::normalizeBetweenArrays()` (quantile-normalize the floored `[cond1, cond2]`
  pr columns across all genes) **then** `|log2(c1/c2)| > 1` — **not** the raw column
  ratio (the A36 fix; see §4). This replaces the disqualified full-proteome receiver.
- **Sender** `gu_S = sender DEG ∪ sender prG` (DEG = `avg_log2FC>1.5 & p_val<1e-4` ∪ HEG,
  **no top_n(500) cap**).
- On sce4's own full per-pair enumeration this yields **1,283 paths** for
  Microglia→Cholin (3 L / 4 R / 6 EM / 1226 T) — receiver nodes are **100% prG**.

### 2b. The six parity overrides

Five are now upstream `Incytr` defaults (locked by
`tests/testthat/test-sce4_defaults.R`); two stay driver-side because they depend on
AD-project inputs. All six are required together — none implies any other.

| # | stage | param | value | why |
|---|---|---|---|---|
| 1 | DG construction (driver) | `top_n` cap | **none** | sce4 clusters carry >1,300 prG genes; any cap drops truth |
| 2 | pr preprocessing (driver) | `pmax(pr_*, 1)` floor | floor <1 → 1 | `Cal_foldchange`'s zero-correction only fires on exact 0; 1e-5 residuals → 13-log2 outliers |
| 3 | expression agg (upstream) | `Expr_bygroup(mean_method)` | `NULL` (trimean) | arithmetic mean is nonzero whenever any cell expresses; trimean = 0 for <25%-expressed genes |
| 4 | path SigProb (upstream) | `Cal_SigProb(correction)` | `0.01` | sets the log2FC floor for exact-zero SigProb paths |
| 5 | survival cutoffs (upstream) | `cutoff_SigProb`, `cutoff_PDS` | `0`, `0` | emit all paths; narrowing is the Top300 PDS ranking (§2d), not a cutoff |
| 6 | sender/receiver scFC (upstream) | `Cal_scFC(correction)` | `0.01` | default `1e-5` saturates sclog2FC; the 9.95-log2 gap was exactly `log2(0.01/1e-5)` |

> Downstream code that recomputes sclog2FC from the transcript substrate must use
> ε = 0.01 (`alz/integration/verify_pathway_round_trip.py`,
> `alz/integration/build_normalized_substrate.py`).

Override #5 is the *upstream* half of an emit-all-then-filter design: the driver
keeps `cutoff_SigProb=cutoff_PDS=0`, and the **downstream** production filter
`alz/incytr_pair/filter_significant_paths.py` applies the analysis cutoffs to the
driver output —
`(SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1) AND abs(PDS) >= 0.2`
(the collaborator's `cutoff_SigProb=0.1` / `cutoff_PDS=0.2`). Both cutoffs are
pure row subsets — they drop rows, never recompute `SigProb`/`PDS` — so applying
the filter to existing parquets is mathematically identical to re-running with
the cutoffs. It is **parity-preserving**: the verifier already applies
`SigProb>0.1`, and adding `abs(PDS)>=0.2` drops **0** recalled paths
(Microglia→Cholinergic 573/600, Ndnf×Ndnf 599/600 unchanged); sce4's Top300
survivors sit on or above these cutoffs (§selection-rule dead-ends), so the
filter reproduces sce4's own gating rather than altering it. Wired into
`run_pair_mode.sh` and `bench/run_nboot0_w3.sh` after the contrast loop, and
applied retroactively to `wide_nboot0` (54.7M → ~21M rows/contrast). This
**replaces** the viewer's former arbitrary `|PDS|>=0.30` storage cut.

### 2c. Decomposition feeding the Incytr inputs (A37, Option A)

`alz/incytr_pair/export_decomposition_for_pair.py` runs the **provenance deconvolution**
`P_c = (N_total/N_c) × bulk × (specific_c / Σ_46 specific)` with min/10000 imputation:

- **transcript share** from the frozen `aggexp.csv` — `AggregateExpression(slot="data")`,
  an SCT/model normalization baked into a Seurat object not on this box. It is the one
  unrecoverable upstream step: the h5ad `.X` cannot reproduce it (raw counts give 50.0,
  CPM 305.9, vs aggexp's 350.3 for Acvr1/Microglia/AppP).
- **size factors** regenerated byte-exactly from the Song h5ad.
- **bulk** from the frozen per-group medians (`pr/imac/py_median.csv`).

This **replaces** the levy_t5 `log2(CPM+1)×bulk` share for the Incytr inputs. (The
levy_t5 `f_c×bulk` forward projection still serves the MEA decomposition — different
consumer.) See `docs/plans/sce4_decomposition_reconciliation_2026-05-24.md`.

### 2d. Top300 selection

Per pair: **top-300 highest PDS ∪ top-300 lowest PDS**. On sce4's own 1,283-path list
→ **600/600, 0 extras**. There is no mystery filter and no pvalue step. The driver
emits all paths (cutoff 0,0); Top300 is the published-table selection applied downstream.
Rank/filter on `|PDS|`, never pvalue.

### 2e. Parity results (final)

| pair | recall | sclog2FC |
|---|---|---|
| Microglia → Cholinergic-Neurons | **573/600** (27 misses all App-ligand) | max \|Δ\| = 0 (R/EM/T) |
| Ndnf × Ndnf | **599/600** (PASS) | max \|Δ\| = 0 (R/T) |

Acvr1/Cholinergic ma_2mo = 50.70 / 28.21 (target 50.74 / 28.22; 0.08% float32 floor).
The frozen 46-cluster inputs (`bench/validate_frozen_parity.sh`) give the same.
`pixi run verify-incytr-sce4` → PASS on both pairs.

---

## 3. App transgene residual

**The only residual, documented and gate-exempt.** All 27 Microglia→Cholinergic
misses are App-ligand paths; all 7 Ndnf×Ndnf `sclog2FC` outliers are App-EM.

App is irrecoverable from any pr file on this box:
- `pr_log2FC = −0.345` → fails prG (`|log2|>1`).
- `fc2 = 1.157` ranks **4523/6687** in Microglia — far below sce4's top_n(500) cutoff
  (its App fc2 ≈ 41.9).
- App's scRNA `sclog2FC` also diverges (Ndnf EM 0.19 vs sce4 7.65).

sce4's Oct-2025 run pr ranked the transgene high; that pr is not on this box. This is
an isolated **input-provenance gap on the transgene**, NOT a method error — the frozen
46-cluster run also caps at 573. Per user decision, `verify_sce4_parity.py` PASSES a
sub-595 recall **iff every miss is an App-ligand path**, and tolerates Ligand/EM
`|Δ sclog2FC|` outliers only when the position gene is App (`--transgene`). **Do not
fabricate the App paths to force 600.**

---

## 4. Key empirical facts (load-bearing)

- **sce4's full pre-Top300 enumeration is 1,283 paths/pair, not ~110k.** Our former
  full-proteome receiver over-enumerated to 110,121 (86× too big). The divergence was
  always at the enumeration/input level, never a downstream selector.
- **prG uses the quantile-normalized `pr_log2FC`, not the raw ratio (A36 — the decisive
  fix).** Acvr1/Cholinergic raw `log2(50.74/28.22) = 0.846` (<1, excluded) but
  `normalizeBetweenArrays` over the full gene column → `2.04 ≈` sce4's reference 1.997
  (included). Computing prG on the raw ratio dropped exactly the genes the scorer flags
  pr-significant (Acvr1, Cr1l, Grm7, Sorl1) → Receptor overlap 0 → all 600 paths lost.
  This was the long-standing "Acvr1 4× gap" — not clustering, not bulk, not a missing file.
- **The frozen pr file is the faithful output of sce4's documented provenance.** A35
  replayed `protein-ms-by-cell-type.py` on `aggexp.csv` + `pr_median.csv` +
  `yuyu_clustersize.csv` and reproduced Acvr1 = 50.74/28.22 to 4 decimals.
- **Our scoring matches sce4 to corr 1.000** on every published column (SigProb, PDS,
  TPDS, PPDS, PhPDS) on shared paths. Scoring is NOT the bug.
- **Cholinergic-Neurons = 1 cell per condition** in ma_2mo. This makes the cell-label
  permutation pvalue degenerate (bimodal {0,1}) and makes Acvr1's ratio a single-cell vs
  single-cell quantity — robust one-sided genes (Sorl1, the EMs) match sce4; two-sided
  moderate genes (Acvr1, Grm7) are sensitive to which sparse cell survives.
- **Cost is driven by DB-active receptor×EM count, not raw gene-use size.** A 5.7k-gene
  presence-union enumerated 311k paths where a 6.4k-gene proteome gave 110k. Shrinking
  *signaling-competent* receiver genes is the lever, not gene count.

---

## 5. Closed theories & refuted directions — DO NOT RETRY

Every entry below was tested to a definitive negative. Reopening any is wasted compute.

### Original theories (pre-recipe)

| theory | verdict |
|---|---|
| T1–T6: pipeline params / vocab / contrast / DB layers / Seurat obj / vendored source differ | falsified — direct verification, all identical |
| T7: DG.Sender/Receiver content drifted from sce4's | **confirmed** → drove the gene-use work (§2a) |
| T8: `Cal_SigProb` has a DG-coupled normalizer | falsified — no normalizer over DG |
| T9: driver-side DG construction differs | moot — driver byte-identical to frozen |
| T10: per-group expression / membership drift | superseded by trimean fix #3 (sparse expression, not cell-membership) |

### Selection-rule dead-ends (the 110k → 600 narrowing)

The narrowing was never a downstream filter on our enumeration — our enumeration was
*wrong* (full-proteome receiver). With the correct ~1,283-path enumeration the Top300
PDS ranking reproduces 600 trivially (§2d). The following were all eliminated first:

| direction | why refuted |
|---|---|
| `top_n(N, |pr_fc|)` cap on receiver | capping excludes truth; no-cap needed |
| path-level magnitude ranking (PDS / multimodel / log2FC) | recall 0/600; truth sits mid-pack (~102,900/110,121) |
| pvalue selection (`cutoff_p_value`, pvalue ranking) | degenerate on 1-cell receiver — p bimodal {0,1}, 92.8% tie at 0; eliminated A28 |
| `cutoff_SigProb=0.1` / `cutoff_PDS=0.2` as the narrowing | near-no-op on our scores (which match sce4); sce4's survivors sit far above them (sp_max min 0.211, \|PDS\| min 0.765) |
| "scoring saturation is the bug" (A25) | **retracted** A26 — our scores match sce4 corr 1.000; sce4's are equally "saturated" |
| concordance-flag selection (`sc_*/pr_*/ps_*/py_*`) | no flag combo collapses to ~600 |

### Refuted receiver gene-use rules (the generative-filter search)

| rule | size | recall | verdict |
|---|---|---|---|
| **global_prG bolt-on (full proteome, "rule A")** | ~6.4k | 573–595/600 | **DISQUALIFIED** — not a filter (any superset hits parity) **and** cost: 110k paths/pair → ~17 days for the full grid (~180× the T7 minimum) |
| DEG only | ~3.4k | 0/600 | DEGs lack constitutive R/EM (Acvr1, Cr1l, Sorl1) |
| scRNA-presence proxy (≥X% detected) | ≥7.5k for full R/EM | — | refuted — smallest viable threshold larger than the proteome; truth genes intermixed with bystanders |
| DB-reachability prior | 16.4k | full | refuted — fan-out to >half the transcriptome at EM/Target |
| naive presence-union | 5.7k | 573/600 | works but **311k paths/pair** (3× the proteome) — smaller list, more compute |
| cluster `input_gene_list` as receiver | 3.2k | grid errors | refuted — 0/4 truth R/EM (constitutive, proteome-only, not DEGs) |
| R/EM position-significance | — | ≤24/600 | refuted — truth R/EM not top-ranked by any pr/ps/py/sc axis; significance gates out constitutive R/EM |

### Provenance mis-diagnoses (corrected by A36)

| claim | correction |
|---|---|
| A32: "sce4 used a different unavailable pr file (dot vs dash naming)" | wrong — `build_cond` rename artifact; column exists |
| A34/A35: "Acvr1 gap is irreducible sparse-detection noise / needs a different clustering" | wrong — the gap was computing prG on the raw ratio instead of the quantile-normalized `pr_log2FC` (A36, §4) |

---

## 6. Data-provenance notes

- **Frozen vs derived.** `data/incytr_frozen/v2_46clusters/incytr input/` is the
  parity-validated 46-cluster reference (the diagnostic oracle). The production driver
  reads the derived 31-spine inputs. A37 made the **derived inputs reproduce provenance**
  (§2c), so the gate need not point at frozen-only. A discrepancy between derived and
  frozen was always evidence of a primary-method defect, never a "which file" choice.
- **The 12 zero-prG clusters were a stale-pseudobulk bug.** Traced to
  `pseudobulk_cpm.csv` carrying only 19 of 31 spine clusters (generated pre-reorg) and a
  silent `fillna(0)` in `snrna_proportions._compute_animal_weights`. The fillna was
  replaced with an explicit `RuntimeError` (hardfail restored). Regenerating at
  `SONG_MIN_CELLS=0` recovers all 31 clusters; gate=5 null-routes Cholinergic-Neurons
  (only 1 stratum survives), so **gate=0 is the only viable choice** for this cohort.
- **Yuyu deconvolution = our forward projection** (`P_c = (N_total/N_c)×bulk×share`),
  differing only in min/10000 zero-imputation. Ported and validated; not a bespoke NMF.

---

## 7. Artifacts index

Paths are relative to `bench/`; runner scripts `cd` to the repo root and refer to
themselves as `bench/<subdir>/<file>`.

**Reference (root)**
- `sce4_DEG_PRG_Top300_table_10302025.csv` — ground truth (55 MB). Hardcoded
  dependency of `alz/incytr_pair/verify_sce4_parity.py`; do not move.

**`fidelity/` — the parity gate**
- `validate_frozen_parity.sh` / `.log` — parity on frozen 46-cluster inputs.
- `validate_derived_parity.sh` / `.log` — parity on derived 31-spine inputs.
- `compare_pair_outputs.R` — per-pair parquet diff tool.
- `run_one_pair_v2_frozen.R` — single-pair frozen-input driver.
- `parity_frozen_out/`, `parity_derived_out/` — small reference outputs.

**`decomposition/` — yuyu decon ports**
- `yuyu_decon_a.py` / `yuyu_decon_b.py` / `yuyu_decon_persample.py` — three impute variants.
- `export_pair_allsexes.py` — both-sex bench exporter (production is males-only).
- `regen_pseudobulk_gate.sh`, `regen_g0_bothsexes.sh` — gate sweeps.
- `verify_aggexp_foundation.py` — confirms h5ad→cluster join reproduces cell counts;
  shows `aggexp.csv` is SCT-normalized (unrecoverable from h5ad).

**`probes/` — gene-use & selection probes** (curated record; refuted rules per §5)
- `probe_a33_variants.R` — the two receiver-universe constructions.
- `probe_deg_prg_split.R` — confirmed positional DEG(sender)/prG(receiver) structure.
- `probe_score_calibration.R` — scores match sce4 corr 1.000.
- `probe_pvalue_real.R` — pvalue degeneracy proof (nboot=1000).
- `probe_geneset_enrichment.R`, `probe_dbgated_topn.R`, `probe_dbpct_survival.R`,
  `probe_scrna_presence.R`, `probe_db_reachability.R`, `probe_naive_presence_grid.R`,
  `probe_cluster_receiver.R`, `probe_rem_position_sig.R` — refuted rules (§5).
  (Their input/output dirs were discarded in the cleanup; kept as record only.)

**`perf/` — performance / OOM work** (see `docs/plans/pairmode_perf_oom_2026-05-25.md`)
- `measure_pairmode_slice.sh` / `.log`, `measure_pairmode_rand40.log` — pre-fix profiling.
- `profile_pair_one.R` / `.sh`, `probe_matrix_footprint.R` — single-pair profiling.
- `phase1_run.sh`, `parallel_pair_probe.R`, `parallel_sweep.sh` — OOM-guarded parallelism.
- `run_nboot0_w3.sh`, `launch_nboot0_w3.sh`, `notify_on_complete.sh` — nboot=0 wide run.

**Repair note.** `data/incytr_frozen/v2_46clusters/incytr input/kldata.csv` is a relative
symlink to `../../shared/kldata_5xad_fallback.csv`, restored from
`~/Projects/work/incytr/examples/5xad_data/kldata_pspy.csv`.
