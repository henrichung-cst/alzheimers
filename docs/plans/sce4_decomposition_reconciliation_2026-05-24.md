# sce4 end-to-end reproduction: decomposition reconciliation + residual close (2026-05-24)

## Goal

From **raw omics**, reproduce sce4's pair-mode output clearing the 595 recall gate on
the two benchmark pairs (Microglia→Cholinergic.Neurons, Ndnf×Ndnf, ma_2mo AppP/WTyp).

Two user decisions drive this plan:
1. **Make the derived (production levy_t5) decomposition reproduce the provenance pr
   values** — so the from-raw-omics pipeline matches sce4's inputs, not just sce4's method.
2. **Chase the ~5% residual** (App + boundary genes) to clear 595, rather than accepting it.

## What is already done (A36, committed to code)

- **prG bug fixed** (`incytr_commandline.R:161-172`): prG now mirrors the Incytr scorer —
  `limma::normalizeBetweenArrays()` on the cluster's `[c1,c2]` pr columns, then `|log2|>1`.
  The old raw-ratio prG dropped exactly the genes sce4 flags (Acvr1 raw 0.846 → excluded;
  normalized 2.04 → included).
- **`INPUTS_DIR_OVERRIDE`** added; **`bench/validate_frozen_parity.sh`** runs the 2 benchmark
  pairs against the frozen 46-cluster inputs.
- Result on **frozen** inputs: Microglia→Cholinergic **573/600**, Ndnf×Ndnf **569/600**,
  `*_sclog2FC` max|Δ|=0 on every joined path. (Derived inputs cap at ~314 because derived
  pr ≠ provenance pr — exactly what Phase 1 fixes.)

## The decomposition gap (audit findings)

Cluster bases reconcile cleanly: **31 (derived) ⊂ 46 (provenance)**; the benchmark clusters
are named and present in both. So this is **not** a re-clustering problem — it is making the
derived deconvolution *method* produce the provenance pr values. Five structural divergences:

| # | Dimension | Provenance (`protein-ms-by-cell-type.py`) | Production (`snrna_proportions.py` / `build_celltype_decomposition.py`) |
|---|-----------|-------------------------------------------|--------------------------------------------------------------------------|
| 1 | scRNA unit in the share ratio | **raw summed counts** (`aggexp.csv`) | log2(CPM+1) (`snrna_integration.py:138-144`) |
| 2 | zero handling | impute every zero → `global_min_nonzero/10000` | none; Σexpr=0 genes dropped (`snrna_proportions.py:145`), expr_c=0 → hard zero |
| 3 | denominator cluster scope | all **46** clusters (N_total, Σspecific) | filtered to **31** spine *before* denominators (`snrna_proportions.py:92-97`) |
| 4 | bulk source | per-group **median MS** (`pr_median.csv`) | per-animal **IRS-normalized** (`total_proteome_normalized.csv`) |
| 5 | aggregation order | aggregate bulk per-group → deconvolve once | deconvolve per-animal → median at export (`export_decomposition_for_pair.py:_pivot`) |

No leftover/flagged imputation code exists (A6/A7 variants were removed) — Phase 1b is a
clean replacement, not a shim removal.

The provenance deconvolution math (the target):
`P_c = (N_total/N_c) × bulk × (specific_c / Σ_clusters specific)`, imputed, group-level.

## FOUNDATION CHECK (2026-05-24) — Phase 1a premise DISPROVEN

`bench/verify_aggexp_foundation.py` + manual deconvolution probe established:

1. **Clustering is byte-reproducible.** h5ad barcode→46-cluster join reproduces
   `yuyu_clustersize.csv` cell counts EXACTLY (Cholinergic 1/1, Microglia 114/77
   for ma_2mo AppP/WTyp). The user's "byte-identical clustering" premise holds.
2. **aggexp is NOT raw counts and NOT reproducible from the h5ad.**
   `aggregate_expression.R` uses `AggregateExpression(slot="data")` =
   `Σ_cells expm1(LogNormalize(counts, scale=1e4))` per (cluster, group). The
   per-cell library size comes from an upstream `renamed_sobj.rds` (on the
   collaborator's machine, NOT on this box) whose gene-filter set differs from
   the h5ad's 30,567 genes. Evidence: for Microglia/ma_2mo_AppP/Acvr1, aggexp =
   350.3 but raw-count sum = 50.0 and h5ad-`.X` CPM sum (`expm1(.X)`) = 305.9;
   the aggexp/expm1 ratio varies per gene (Acvr1 1.15, App 1.08, Csf1r 0.62,
   Gfap 1.35) → a non-uniform per-cell normalization that cannot be undone from
   the h5ad counts layer alone.
3. **The provenance formula + FROZEN aggexp reproduces the target pr EXACTLY.**
   `P_c = (N_total/N_c) × bulk × (specific_c / Σ_46 specific)` with min/10000
   imputation, fed by `aggexp.csv` + `yuyu_clustersize.csv` + `pr_median.csv`,
   gives Acvr1/Cholinergic ma_2mo **AppP=50.737, WTyp=28.218** (targets 50.74,
   28.22). Cluster names carry a per-group numeric suffix (WTyp none, AppP "1",
   …) collapsed by `remove_number`.

**Consequence:** Phase 1a ("emit raw summed counts from the h5ad") is wrong —
raw counts don't reproduce aggexp, and the h5ad cannot reproduce aggexp's
normalization. Exact derived==provenance parity is achievable ONLY by feeding
the provenance deconvolution the frozen `aggexp.csv` (Option A). Regenerating
aggexp from the h5ad (Option B) requires the absent `renamed_sobj.rds` gene
filter and risks breaking the just-won 573/569 parity. See decision below.

### Option B feasibility — MEASURED, infeasible without the absent sobj (2026-05-24)

User chose Option B (regenerate from h5ad). Tested the only h5ad-derivable
normalization (standard Seurat LogNormalize = `Σ_cells counts/L × 1e4`):

- **L = total_counts** (= colSum of counts layer, verified): Acvr1/Cholinergic
  ma_2mo deconvolves to **AppP=43.48, WTyp=18.91** vs targets 50.74/28.22 —
  15%/33% undershoot, and the *driving ratio shifts 1.80 → 2.30* (would move
  pr_log2FC and regress parity).
- **L = total − mito/ribo/hb**: 305.9 → 306.3 vs aggexp 350.3 — no help
  (ribo flagged 0 in this h5ad; mito frac ~0.6%).
- **Not a scale factor**: aggexp/expm1(.X) ratio varies per gene (Acvr1 1.15,
  App 1.08, Csf1r 0.62, Itgam 0.84, Gfap 1.35). A global `scale.factor` would
  give a constant ratio. The per-gene/per-cell variation ⇒ a model-based
  normalization (SCTransform/Pearson residuals) baked into the *original* yuyu
  Seurat object, upstream of `renamed_sobj.rds`. `workflow.txt` confirms the
  normalization predates everything checked in here.

**Therefore Option B as a faithful reproduction is infeasible** — the only
h5ad-derivable normalization introduces a confirmed 15–33% gap and shifts the
ratio that drives the score. This collides with the project's no-gap principle
(same data + same method ⇒ same output): we do NOT have the same normalization
method on disk. The one genuinely from-raw, byte-identical piece is the
clustering / cell counts (already reproduced exactly). The expression
normalization is the single unrecoverable upstream step; the frozen `aggexp.csv`
encapsulates exactly that step.

## Phase 1 — Decomposition reconciliation

**Bulk-probe result (2026-05-24): #4 and #5 are NOT levers — ruled out.** Acvr1/Cholinergic
bulk ratios: provenance `pr_median` = 35.62/35.52 = **1.003**; production per-group median =
36.40/36.17 = **1.006**. Both flat and within 2.5% in magnitude. The earlier "provenance bulk
~1.6" claim was wrong. The **entire** 1.12× vs 1.80× gap is in the **share** (provenance
share×size = 1.79 vs ours 1.11). Phase 1 therefore targets ONLY the share math:

- **1a (dominant lever)** Feed **raw summed counts** per (cluster, group) into the share ratio,
  not log2(CPM+1). The log transform compresses the dynamic range and squashes the ratio toward
  1 (this is most of the lost signal). Production already sums raw counts in
  `snrna_integration.py:138` before the CPM/log step — emit the raw sums and branch the share
  computation off them.
- **1b** Add Yuyu zero-imputation: `global_min_nonzero/10000` for every zero and every
  synthesized missing cluster×group. Replaces the current drop/hard-zero logic outright.
- **1c** Compute `N_total` and `Σ_clusters specific` over **all 46 clusters**, then emit the
  31 spine clusters. Move the spine filter to *after* the denominators.
- ~~1d bulk source~~ — **dropped** (probe shows production bulk ≈ provenance bulk; keep current bulk).
- ~~1e aggregation order~~ — **dropped** (no-op for the benchmark; revisit only if a >1-animal
  group fails the gate).

**Phase 1 gate:** regenerate derived pr; assert Acvr1/Cholinergic ≈ 50.74/28.22 and a
column-wise correlation ≥ 0.99 vs frozen pr on the shared 31 clusters for ma_2mo. If 1a+1b+1c
alone hit the gate, bulk/aggregation stay untouched.

## Phase 2 — Residual (close to ≥595)

The residual splits by *how sce4 included each gene* (label evidence from the reference table):

- **prG-labeled, |pr_log2FC|>1** (Calm2 +2.15, Col4a2 −1.09, Olfm2 −6.72, Usp47 −1.003):
  these *should* be caught by the fixed normalized prG. They were missing only in the
  **derived** run (derived pr ≠ provenance). **Phase 1 is expected to recover them** — verify
  on the post-Phase-1 derived run; no new rule needed.
- **DEG-labeled, |pr_log2FC|≤1** (Kcnab3, Map1b, Septin11, Tmx4): not in *our* frozen
  `input_gene_list`, but sce4 labeled them DEG → sce4 used a **different DEG list**.
  **2a:** regenerate sce4's DEG via the provenance `run_input_gene_list.R` from raw and check
  whether it includes these. If yes, the residual was a stale input_gene_list, not a method gap.
- **App (transgene), prG-labeled but pr_log2FC=−0.345:** entered sce4 via raw-ratio
  `top_n(500, receiver_fc2/sender_fc2)` on **sce4's** pr file (`incytr_commandline.R:124-155`).
  On the regenerated frozen file App's raw ratio ranks too low. **2c:** test whether the
  Phase-1-corrected derived pr ranks App into the raw-ratio top_n(500); if the corrected pr
  reproduces sce4's pr exactly, sce4's literal rule should catch App. If App remains
  unrecoverable, document precisely *why* (the regenerated provenance pr diverges from sce4's
  Oct-2025 run file — a real input-provenance gap, isolated to the transgene).

**Decision embedded here:** sce4's *literal* gene.use is `DEG ∪ top_n(500, raw_ratio)`, but
on the available file the **normalized-prG** rule reconstructs the receiver set far better
(587/587 vs 0/4). Recommendation: **keep normalized-prG as the production rule** (it is the
better reconstruction and matches the scorer), and treat App via 2c rather than reverting to
raw top_n(500).

## RESULT (2026-05-25) — Phase 1 DONE, Phase 2 = isolated transgene gap

**Phase 1 (Option A) implemented.** `export_decomposition_for_pair.py` rewritten
to the provenance deconvolution: transcript share from frozen `aggexp.csv`,
size factors regenerated from the Song h5ad (byte-exact cell counts), bulk from
frozen per-group medians. Emits 31-spine × 12 male-group wide CSVs for pr/ps/py.
Acvr1/Cholinergic ma_2mo = **50.698 / 28.207** (target 50.74/28.22; 0.08% off,
float32 floor). The levy_t5 log2(CPM+1)×bulk share is replaced outright for the
Incytr inputs (anti-shim).

**Derived parity now reproduces provenance:**
- Microglia→Cholinergic **0 → 573/600** (== frozen)
- Ndnf×Ndnf **266 → 599/600 PASS** (> frozen's 569)
- `*_sclog2FC` max|Δ| = 0 on every joined path.

**Phase 2 — the residual is 100% the App transgene, and it is irrecoverable
from available inputs:**
- All 27 Microglia→Cholinergic misses are App-ligand paths (sce4 label = prG).
- App pr_log2FC = −0.345 → fails the |pr_log2FC|>1 prG rule.
- App fc2 (raw ratio) = 1.157 → **rank 4523/6687** in Microglia → fails sce4's
  `top_n(500)` rule too (cutoff fc2 = 41.9).
- App is in our gene.use for 28/31 clusters (DEG/HEG) but not Microglia — a
  per-cluster boundary exclusion.
- sce4 labeled App "prG" (= entered via its top_n side), which only works if
  App/Microglia ranked into top_n(500) in **sce4's Oct-2025 run pr**. Every pr
  artifact on this box (frozen aggexp→pr, regenerated provenance) gives App
  rank 4523. ⇒ sce4's transgene pr diverged from the frozen snapshot. This is
  an **isolated input-provenance gap on the transgene**, NOT a method error:
  we do not have sce4's original App pr. (The frozen 46-cluster run also caps
  at 573 for the same reason.)

**Gate decision (user, 2026-05-25): document + adjust.** `verify_sce4_parity.py`
now PASSES a sub-595 recall iff every miss is an App-ligand path, and tolerates
Ligand/EM `|Δ sclog2FC|` outliers only when the position gene is the `--transgene`
(default App). App paths are NOT fabricated. The dead `--ligand-max-residual`
count tolerance was removed (replaced by the App-aware check).

## Phase 3 — Re-validate + lock — DONE

- `pixi run verify-incytr-sce4` (derived production wide dir) → **PASS** on both
  benchmark pairs (573/600 App-only, 599/600). No re-pointing needed: the gate
  already runs on the production pipeline output, and `build_pair_inputs.sh`
  now feeds it the provenance-deconvolution inputs.
- `build_pair_inputs.sh` pre-flight updated: drops the levy_t5 decomposition
  parquets; checks frozen `aggexp.csv` + `yuyu_clustersize/samplekey.csv` +
  `pr/imac/py_median.csv` + the Song h5ad.
- Docs: `bench/bench.md §A37`, `CLAUDE.md §sce4 parity constants` updated.

## Original Phase 3 plan (superseded by the DONE block above)

- Run `validate_frozen_parity.sh` and a derived-parity run (derived now ≈ provenance);
  expect ≥595 on both benchmark pairs.
- Re-point `verify-incytr-sce4` at whichever inputs the user wants the canonical gate to use
  (derived-from-raw if Phase 1 succeeds; frozen otherwise).
- Update `bench/bench.md` (A37) and `CLAUDE.md` (decomposition method + gate inputs).

## Risks / open questions for approval

1. **Phase 1 reshapes the active levy_t5 decomposition.** Per anti-shim, the provenance method
   *replaces* the forward-projection share math — no flag to switch back. Confirm levy_t5's
   `P_c = f_c × bulk` framing is being superseded by the provenance `(N_total/N_c)×bulk×share`.
2. **Bulk normalization (1d)** is the least-certain lever; may need its own probe before the
   full rewrite.
3. **App** may be irrecoverable from the regenerated provenance (isolated transgene gap). If so,
   clearing 595 rests on Phase 1 recovering the prG-labeled genes + Phase 2a recovering the
   DEG-labeled genes (27 App-only misses on Cholinergic would still cap that pair at ~573 unless
   2c works). This is the one place the "we have all the same data" premise may not hold.
