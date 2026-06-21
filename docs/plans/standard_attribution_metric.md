# Standard attribution metric — one definition, every cohort

**Status:** standard CONFIRMED 2026-06-20 (decisions 1–3 below locked). No code touched yet;
rollout phases remain individually gated. Supersedes the per-cohort share localizers
cataloged in `attribution_specificity_audit.md`.

**Locked decisions:** (1) the standard as specified; (2) **linear** (de-logged) weights for
concentration / effective-N; (3) SEA-AD **exempt** — kept as a ratio-only human corroborating
reference, no detection gate; (4) keep familiar **2×/5×/10× bins** where the metric is a
fold (the concentration metric — see Q1).

## Goal

Answer, without being misleading: **"How specific is this kinase to a cell type?"** — where
"cell type" generalizes over cluster / label / state / supertype across cohorts.

The share (`mean_in_group / Σ mean`) is misleading for two reasons proven in the T-cell work:
it is computed *without gating on presence* (so a near-zero kinase scores high where
competition is lowest), and it is a *relative ranking on log means* (so log-compression makes
broad kinases look concentrated and the denominator's group composition drives the number).
We replace it, not patch it.

## The question is actually two questions

| | Question | Right instrument |
|---|---|---|
| **Q1** | Is this kinase *present* in cell type X, and how concentrated is it there? | detection + detected-set concentration |
| **Q2** | Overall, is this a one-cell-type kinase or a broadly-expressed one? | effective number of cell types |

Both are built from one foundation. The share survives **only** as the detected-set
concentration in Q1, and never as a standalone tier.

## Foundation — detection (per kinase × cell type)

- **`fraction_cells_expressing`** = (cells with count > 0) / (cells in the cell type).
  Count-based → **normalization-free → identical meaning across every pipeline**, immune to
  the share artifact.
- **`detected`** = `fraction_cells_expressing ≥ 0.10`. The single cross-cohort presence gate.
  Drops the `mean > 1` co-gate, which the audit (finding E) showed is scale-dependent and
  wrong on the Seurat log(CP10K+1) cohorts. Mean expression is kept as a displayed secondary,
  never in the gate.

## Q1 — specificity to a given cell type (per kinase × cell type)

Reported as a **pair, never a lone tier**:

- `detected` (✓/✗) — the gate. If ✗, the honest answer is "not present here," full stop.
- `concentration_c` = `wᶜ / Σ_{detected} w`, where `wᶜ` = **linear** per-cell mean (de-logged
  with the cohort's own base), summed **over detected cell types only**.

Two corrections vs. the old share, both load-bearing:
1. **Gated to detected types** → no phantom specificity from noise types (fixes the CDK1
   anomaly and the inverse-prediction finding).
2. **Linear weights** → no log-compression inflating apparent breadth/concentration.

**`concentration_tier`** — the familiar 2×/5×/10× bins, applied to `concentration_c`:

| tier | meaning |
|---|---|
| ≥10× | `concentration_c ≥ 10 / n_detected` |
| ≥5× | `concentration_c ≥ 5 / n_detected` |
| ≥2× | `concentration_c ≥ 2 / n_detected` |
| ≥1× | at or above an even share among present cell types |

The baseline is **`1 / n_detected`** — an even split among the cell types where the kinase is
*actually present* — **not** `1/N_total` padded with empty cell types. That single change is
what makes the bin honest: the old ladder's fold was inflated by the count of cell types the
kinase isn't in. The tier is meaningful only when `n_detected ≥ 2`; when the kinase is present
in one cell type, specificity is conveyed by `effective_n = 1` (the fold is trivially 1×).

Non-misleading rule: `concentration_c` / its tier are shown only when `detected`, and always
beside the effective-N below. A 100% concentration in a kinase present in one type then reads
as "present in 1 type," not "10× enriched."

## Q2 — overall cell-type specificity (per kinase)

All gated to detected cell types:

- `n_celltypes_detected` — crude breadth (count).
- **`effective_n_celltypes`** = `1 / Σ_{detected} concentration_c²` — magnitude-aware breadth.
  "Effectively present in N cell types." A kinase 10× in one + trace-but-detected in five reads
  ≈1.3, not 6.
- `top_celltype` + `top_concentration` — where it concentrates and how much.

Interpretation ladder (a count, not a fold — so **no** 2×/5×/10× bins here; the bins live on
`concentration_tier` above):

| effective_n | reading |
|---|---|
| ≈ 1 | cell-type-specific — knockout hits ~one cell type |
| ≈ n_celltypes_detected | broadly / evenly expressed |
| between | concentrated but present in several |

## Cross-cohort comparability

`effective_n` is in units of "cell types," so it is **resolution-dependent** (14 T-states ≠ 31
Levy clusters ≠ ~9 WMB classes). We standardize the *form*, not a single magic number. Where a
cohort has a meaningful coarse vs. fine distinction, report at **both** a fixed coarse grouping
(broad lineages) and native resolution — the data legitimately gives different answers (ZAP70 =
effectively ~1 at "T cells" merged, ~13 across T-varieties; both true). Never present one
`effective_n` as if it were cohort-comparable.

## The critical fix — the attribution claim gates on detection (finding A)

`top_celltype` and the confidence tier currently rank on the share. Under the standard:

- `top_celltype` is chosen **only among detected cell types**. A kinase is never attributed to
  a cell type where `detected = ✗`.
- "high" / "very_high" confidence **requires** detection at the attributed cell type.
- Directional concordance (LFC sign/magnitude) is retained as the tie-break **among detected
  types** — it ranks *which* present cell type, never resurrects an absent one.

## What is removed (anti-shim)

| Removed | Replaced by |
|---|---|
| `specificity_score` share as standalone localizer/tier (WMB, Song-within, NSCLC, T-cell) | detected-set `concentration_c`, always paired with `detected` + `effective_n` |
| `confidence.py` `song_location_high` (`song_specificity ≥ 2/31`) and `wmb_crosscheck` (≥2/9) share gates | detection at the attributed cluster |
| Viewer fold-over-uniform ladders `_wmbTier` / `_msTier` / `_tcellTier` / `_nsclcTier` — fold over `1/N_total` on the **ungated** share | same 2×/5×/10× bins, but on `concentration_c` (gated, linear) over `1/n_detected`; plus the per-cell-type detection column and `effective_n` |
| `song_tau` as the headline per-gene specificity | `effective_n_celltypes` (interpretable; Tau was rejected) |
| Two-ladder split (2× pipeline gate vs 10/5/2/1 viewer) — finding D | one definition computed once in Python, rendered in JS |

5xFAD's decomposition (per-cell-type OLS significance) is already presence-grounded and stays —
it is the conceptual model the others move toward. Its OLS significance is the 5xFAD analog of
`detected`.

## Known constraints (resolve during implementation, not now)

- **Detection needs per-cell counts.** Available & already computed: WMB, NSCLC. Needs a
  streamed recompute: Song-within (`snrna_integration.py`), T-cell-within
  (`tcells_scrna_extract.R`, the approved Phase 2). Each runs under a memory cap.
- **SEA-AD (finding C) is the hard case.** It currently emits only a `log2(ct/brain)` ratio
  from summarized per-supertype expression. Detection requires per-cell SEA-AD counts. If those
  are not on disk, SEA-AD either gets a streamed recompute from source, or is explicitly labeled
  "ratio-only — no detection available" and exempted from the detection gate (it is a corroborating
  human reference, not the primary attributor). Decision below.
- **De-log needs the right base per cohort** (log2 vs Seurat natural-log CP10K+1) for the linear
  concentration weights. Wrong base silently distorts `effective_n`.

## Rollout (gated phases, each its own approval)

1. **T-cell** — already-approved Phase 1 (reference detection + FP flag) + this breadth metric.
   The prototype; proves the standard end-to-end on one cohort.
2. **Song + WMB** — surface existing WMB detection as the primary column; add Song-within
   detection (streamed recompute); switch `top_celltype`/confidence gate to detection.
3. **Consistency pass** — remove the viewer fold ladders, unify the tier definition, resolve
   SEA-AD per the decision below.

## Decisions — RESOLVED 2026-06-20

1. **Standard** — confirmed. Detection foundation (`fraction ≥ 0.10`, normalization-free) +
   Q1 detected-set `concentration_c` (linear weights) with 2×/5×/10× `concentration_tier` +
   Q2 `effective_n_celltypes` + `top_celltype`/`top_concentration`, reported at coarse+native
   resolution. One metric, every cohort.
2. **Linear weights** — confirmed. De-log means to linear per-cell expression for the
   concentration / effective-N weights (correct base per cohort).
3. **SEA-AD** — confirmed exempt. Labeled "ratio-only — no detection," remains a human
   corroborating reference, not gated on detection.
4. **Bins** — retained on the fold metric (`concentration_tier`, baseline `1/n_detected`).

## Phase 1 — T-cell prototype — DONE 2026-06-20

The standard is implemented end-to-end on the T-cell cohort:

- `alz/cross_reference/specificity.py` — the single definition (detection gate, linear de-log,
  concentration + tier, effective number of cell types) reused by every cohort.
- `alz/reference/nsclc_expression.py` — share `specificity_score` removed; metric computed via
  the helper at native + coarse resolution; `nsclc_kinase_specificity.csv` written.
  `pixi run nsclc-metrics` recomputes from the existing expression CSV (no matrix re-stream).
- `alz/build_tcell_viewer.py` — state crosswalk + reference detection joined onto the
  attribution index (`nsclc_frac`, `nsclc_detected`, `mea_false_positive`); `top_celltype`
  gated to NSCLC-detected states.
- Viewer JS/CSS — per-state NSCLC **detection** column (✓/✗ + frac%) replaces the share tier;
  ⚠ MEA false-positive badge overrides a high within-cohort tier; drawer strip shows per-lineage
  detection + effective-N.

Verified: LCK/ZAP70 detected in all 14 states (0 FP, top cell retained); LRRK2 share-localizes
to Treg at 10× but is detected in 0 T-states → flagged FP, top cell blanked; FGFR2/EGFR likewise
blanked. 587 (kinase, state, contrast) FP candidates flagged. `pixi run tcell-viewer` exit 0,
payload 11.47 MB raw / 1.53 MB gz.

## Phase 2 — Song + WMB pipeline — DONE 2026-06-20

The bulk-MEA attribution pipeline now gates on detection instead of the share.

- `alz/reference/snrna_integration.py` — Song `step_pseudobulk` computes per-(gene, cluster)
  `fraction_cells_expressing` from the raw sparse matrix (count-based, before CPM) →
  `song_detection.csv`; `step_specificity` feeds it + per-cluster mean log2(CPM+1) to
  `specificity.compute()` and writes the standard metric (detection / concentration /
  concentration_tier / effective_n over 31 Levy clusters) to `song_expression_specificity.csv`.
  The share `specificity_score` and the rejected `tau` are gone. Recompute via
  `pixi run snrna` (reads the 606 MB h5ad; run under a memory cap). Verified: 12,192/30,567
  genes detected, median effective # cell types = 11.92.
- `alz/reference/wmb_expression.py` — `--metrics` (`pixi run wmb-metrics`) recomputes the
  standard metric from the existing `wmb_kinase_expression.csv` (detection already present, **no
  h5ad re-stream**); share removed; `wmb_kinase_specificity.csv` written (per-kinase breadth).
  Verified: 468/547 kinases detected, median effective # cell types = 4.79 over the 9 classes.
- `alz/cross_reference/evidence.py` — `prepare_{song,wmb}_specificity` now surface
  `*_detected` / `*_concentration` / `*_concentration_tier` (+ Song per-gene `effective_n` /
  `top_celltype`) instead of the share.
- `alz/bulk_mea/confidence.py` — `song_location_high` → `song_detected`, `wmb_crosscheck` →
  `wmb_detected` (frac ≥ 0.10 at the attributed cell type, **not** the legacy `binary_expressed`
  mean>1 co-gate). The `_tier_from_share` share-fold labels (`song_location_tier`,
  `wmb_crosscheck_tier`) are removed.
- `alz/bulk_mea/{attribute,recover}.py` — sort tiebreaks and the Table-2 row filter switched
  from share to detected-set `concentration` / `song_detected`; `wmb_fold_over_uniform` /
  `wmb_tier` → `wmb_concentration_tier`.

**Impact (males_only).** Eligible set (MEA-significant ∧ concordant) is unchanged at 12,678
(kinase, cluster, contrast) rows — detection only *redistributes* within it. Confidence shifts
moderate→high as concordant-and-detected rows that the share gate had suppressed promote:

| tier | before (share gate) | after (detection gate) |
|---|---|---|
| very_high | 114 | 1,617 |
| high | 231 | 1,980 |
| moderate | 9,560 | 6,914 |
| low | 2,773 | 2,167 |
| none | 95,853 | 95,853 |

Invariant verified: **0** high/very_high rows are undetected at their attributed cell type.
Broad kinases (CK2A1 effective_n≈30) are now "high" in many clusters but flagged broad by
`effective_n`; specific kinases (MYO3A/ALK1 effective_n=1.0) are "high" in one. Presence (the
tier) and specificity (effective_n) are no longer conflated — the larger high-count is the
honest consequence of dropping a gate that was both misleading and artificially restrictive.

## Phase 2C — Unified viewer (Song cohort) — DONE 2026-06-20

- `build_unified_viewer.py` — `attribution_index` / `kinase_celltype_evidence` load + emit the
  detection fields (`song_detected` / `song_concentration` / `song_concentration_tier` /
  `song_fraction_cells_expressing` / `song_effective_n` / `song_top_celltype` /
  `song_top_concentration` + `wmb_detected` / `wmb_concentration` / `wmb_concentration_tier`),
  share fields dropped. `meta.song_uniform` removed; `meta.wmb_uniform` kept (5xFAD).
- `viewer/cohorts/song.py` — both builders emit the new fields.
- JS (`kinase_explorer.js`, `kinase_audit.js`, `kinase_crosstable.js`, `styles.css`) — Song-tab
  share fold-pills (`_msTier`/`_msTierBadge`/`_KX_SONG_UNIFORM`) replaced with detection cells
  (✓/✗ + frac%), `concentration_tier` badge, and `effective_n`. The shared `_wmbTier` /
  `_wmbTierBadge` helpers are KEPT for the 5xFAD tab.

Verified: `pixi run viewer` exit 0, payload 57.74 MB raw / 6.06 MB gz; attribution_index carries
all 8 new detection fields, all 7 stale share fields absent (108,531 rows, 54,693 song-detected).

**Scope (Option A, deliberate):** only the **Song** cohort tab is migrated. The unified viewer
also hosts a **5xFAD** tab whose WMB cross-check is still a share (`wmb_specificity`, its own
unmigrated surface — 5xFAD's primary attribution is presence-grounded OLS decomposition) and the
human/Mukesh tab. Those keep the shared share-fold helpers. Migrating 5xFAD's WMB cross-check is a
follow-on. `alz/ingest/test_fivexfad.py` was therefore NOT changed (it asserts on the 5xFAD path,
which is unchanged).

## Phase 2D — Within-cohort T-cell detection — DONE 2026-06-21

- `tcells_scrna_extract.R` emits `pct_expressing.csv` (fraction of cells expressing per
  state×day). Capped re-run of both donors (`MemoryMax=20G`, `SwapMax=0`): donor1 25,678 cells ×
  27,486 genes, donor2 20,654 × 29,191; `readRDS` peaked ~6.5 GB, never near the cap.
- `tcell_within_cohort.py`: `_per_state_detection` pools `pct_expressing` over days **cell-weighted**
  (Σ pct·n_cells / Σ n_cells); `_compute_metric` feeds the per-state mean log2-expression + pooled
  detection into the shared `specificity.compute` (one cross-cohort definition). Emits
  `tcell_detected` / `tcell_fraction_expressing` / `tcell_concentration` / `tcell_concentration_tier`
  + per-gene `tcell_effective_n` / `tcell_top_celltype`. The share localizer (mean_in_state / Σ,
  binned over 1/N_states) is removed.
- donor1 result: 384,804 (gene×state) rows, 71,088 detected; unified grid 16,338 rows (guard ✓);
  concentration_tier dist `{2:129, 1:2637, 0:13572}` — broad detection, low concentration (no
  5×/10×), the honest signature of kinase mRNA being post-translationally decoupled from activity.
- `validate_cohort.py` contract → `tcell_detected` + `tcell_concentration`; `tcells: PASS=106 FAIL=0`.
- Viewer: `build_tcell_viewer.py` payload + `tcell_viewer` `kinase_explorer.js` / `kinase_audit.js` /
  `styles.css` migrated to detection cells (✓/✗ + % cells), `concentration_tier` badge, `effective_n`;
  `attribution_uniform` / `_tcellUniform` / `_tcellTierBadge` retired; MEA false-positive now keys on
  within-cohort **detection** vs reference-absent. Rebuilt: payload 11.99 MB raw / 1.49 MB gz, all 6
  detection fields present, zero stale share fields.

### Out-of-scope collateral surfaced (need a separate decision)
- `alz/reference/wmb_expression.py --proteome` still emits a share `specificity_score` — a
  distinct proteome-wide surface, not the kinase attribution path. Left as-is.
- `alz/supplementary/{aggregation_robustness,threshold_sensitivity}.py` test the *old*
  share-confidence model (read `specificity_score` / `wmb_specificity`); they will break/degrade
  and need porting to detection or retiring.
