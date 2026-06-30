# Audit — cell-type attribution & specificity statistics (all cohorts)

**Status:** audit only. No code touched. Decisions requested at the end before any edit.

## Why this audit

The T-cell NSCLC reference work proved that the *specificity share* used to localize a
kinase to a cell type is the **wrong instrument for presence**, and is in fact *inversely*
predictive of truth: on bulk-significant T-cell kinases, 83% of share-localized
(kinase, state) pairs (tier ≥2×) and 93% of the peak-localized pairs (tier =10×) are
**not detected** in an independent 897k-cell reference at the state the share assigned them
(`docs/plans/tcell/nsclc_within_cohort_detection_comparison.md`). The harder the share localizes,
the more likely it is a false positive.

Cell-type attribution is a load-bearing output of every cohort. If the share is misleading
for T-cells, the same statistic is used — with the same failure mode — across Song, Mukesh,
5xFAD, and both viewers. This audit catalogs every attribution statistic, identifies which
share the defect applies to, and surfaces four additional cross-cohort inconsistencies the
mapping turned up.

## The shared defect: a share is not presence

Four of the five references define specificity as a **share**:

```
specificity = mean_expr_in_group / Σ_groups(mean_expr)      # sums to 1 across groups
```

A share is a *relative ranking*, not presence/absence. A kinase with near-zero expression
everywhere still gets a *high* share in whichever group has the fewest competing counts —
the denominator is tiny, so noise dominates. Share rewards the absence of competition, which
is the opposite of detection. This is the mechanism behind the inverse-prediction finding.

**The attribution claim itself is share-gated.** It is not just a display column:

- **Song (unified):** `top_celltype_1` is sorted by `confidence_tier` → weighted |LFC|
  concordance → `wmb_fold` (`recover.py:323-326`). `confidence_tier="high"` *requires*
  `song_location_high` = `song_specificity ≥ 2/31` (`confidence.py:162`) — a share gate.
- **T-cell:** `top_celltype_1` sorted by `tcell_tier` DESC (a share tier) → concordance
  (`build_tcell_viewer.py:718-729`).

So the share doesn't merely color a badge — it decides which cell type the kinase is
attributed to, and which attributions earn "high"/"very_high" confidence.

## Catalog — every specificity/location statistic by cohort

| Cohort / ref | Statistic (CSV col) | Family | Denominator (N) | Detection computed? | Drives the claim? |
|---|---|---|---|---|---|
| **WMB / Song** | `wmb_specificity` (`specificity_score`) | **share** | ~9 WMB classes | `fraction_cells_expressing` + `binary_expressed` (mean>1 AND frac>0.10) **yes** | crosscheck gate (2/9) |
| **Song within-cohort snRNA** | `song_specificity` (`specificity_score`) | **share** | 31 Levy-t5 clusters | **no** | high-confidence gate (2/31) |
| " | `song_tau` (Yanai index) | per-gene index | N−1=30 | n/a | headline only |
| **SEA-AD / Mukesh** | `seaad_location_score` / `human_location_score` | **log2-ratio** | 139 supertypes (col-unweighted mean) | **no** | moderate-tier gate (≥1.0) |
| **NSCLC / T-cell (ext)** | `specificity_score` | **share** | 7–8 coarse groups | `fraction_cells_expressing` + `binary_expressed` **yes** | display tier |
| **T-cell within-cohort** | `tcell_specificity` | **share** | n_states (donor; 14) | **no** | top-cell sort + tier (1/14) |
| **5xFAD** | *(none)* — decomposition | per-cell-type OLS sig | n/a | n/a (presence-grounded) | OLS significance |

Detection data (`fraction_cells_expressing`, `binary_expressed`) — the right instrument —
**already exists** in the two largest references (WMB, NSCLC) but is not the comparison axis.
In the unified viewer it is demoted to a yellow "low expr" warning badge that can co-occur
with a ≥10× share tier; the raw fraction is in `attribution_index` and the CSV export but is
never shown as a verdict-table column. Song-within, SEA-AD, and T-cell-within compute **no
detection statistic at all**.

## Four cross-cohort inconsistencies (compounding the defect)

1. **SEA-AD is a different statistic family.** `seaad_location_score = log2(ct_mean /
   brain_mean)` — unbounded, can be negative, does **not** sum to 1. It is numerically
   incomparable to the four shares, yet sits in the same "location" role and is rolled into
   the same confidence ladder via a separately-calibrated `≥1.0` cut. A reader comparing the
   "Location" columns across tabs is comparing a ratio to a share.

2. **Denominator vocabularies differ 4×–15×.** WMB N≈9, NSCLC N≈7, T-cell N=14, Song N=31.
   A share of 0.5 is "5× uniform" in Song but "3.5× uniform" in NSCLC — the same number means
   different things, and the uniform baselines (1/9, 1/31, 1/7, 1/14) are all different.

3. **Two different tier ladders bin the same share.** The Python pipeline
   (`confidence.py:41-48`) bins `wmb_specificity`/`song_specificity` with a single **2× gate**
   (high / above_uniform / below_uniform) to drive `confidence_tier`. The viewer JS
   (`_wmbTier`, `_msTier`, `_tcellTier`, `_nsclcTier`) *independently* rebins the same shares
   into a **10×/5×/2×/1× ladder** for the displayed "tier" badge. So "WMB tier" in the table
   (10/5/2/1) is not the `wmb_crosscheck_tier` (2× gate) the confidence logic actually used.

4. **NSCLC mixes denominators inside one file, and expression scales differ across cohorts.**
   NSCLC `specificity_score` is over 7 coarse groups while its `fraction_cells_expressing`
   stays at 14-state resolution. Expression scales are not uniform either: WMB = log2(raw+1)
   pre-normed Allen, Song/NSCLC = log2(CPM+1), T-cell-within = Seurat log(CP10K+1) — yet the
   `binary_expressed` threshold `mean>1` (calibrated on log2(CPM+1)) is applied across all of
   them. On the Seurat scale that cut is not the same biological bar.

## The one cohort that already does it right

**5xFAD has no specificity share.** It attributes via decomposition — per-cell-type OLS on
the bulk phospho signal weighted by snRNA expression proportions
(`cohorts/fivexfad/celltype_mea.py`). A kinase is attributed to a cell type when its signal is
*significant in that cell type's decomposed phospho*, not when its transcript share peaks
there. This is presence-grounded by construction and is the model the share-based cohorts
should move toward.

## What "specificity" should be (recommendation)

Two distinct questions are currently conflated under one "specificity" share. Separate them:

1. **Presence — "is this kinase in this cell type at all?"** Use **detection**
   (`fraction_cells_expressing`, count>0) gated by `binary_expressed`. Normalization-free,
   identical meaning across pipelines, immune to the share artifact. This is what should gate
   the attribution claim and flag MEA false positives.

2. **Breadth — "will knocking it out affect more than one cell type?"** Use the
   **effective number of cell types** (`1/Σpᵢ²` over cell types where the kinase is genuinely
   present, gated first) + **% in the top cell type** + the top cell type's name. Magnitude-
   aware (a kinase that is "10× in one, trace in the rest" reads ≈1, not N), interpretable,
   and reported at two groupings where a coarse/fine distinction matters (e.g. T-varieties
   merged vs. separate). This replaces the rejected Tau idea.

The relative *share* is retained only as a labeled secondary descriptor, never as the gate.

## Severity & remediation (phased, gated — no edits yet)

| # | Finding | Severity | Fix |
|---|---|---|---|
| A | Share gates the attribution claim across Song + T-cell, and is inversely predictive | **critical** | Gate `top_celltype`/confidence on detection; never attribute to an undetected cell type |
| B | Detection exists (WMB, NSCLC) but is hidden behind a warning badge | high | Surface per-(kinase,celltype) detection as the primary verdict column; demote share tier |
| C | SEA-AD log2-ratio vs four shares, same role, separate calibration | high | Decide one location-statistic family; if detection-led, this resolves itself |
| D | Two tier ladders bin the same share (2× pipeline vs 10/5/2/1 viewer) | medium | One ladder, computed once in Python, rendered in JS |
| E | Mixed denominators (NSCLC 7 vs 14) + mixed expression scales vs one `mean>1` bar | medium | Single resolution per file; per-scale detection threshold or harmonize scale |
| F | Song-within / T-cell-within / SEA-AD compute no detection at all | medium | Add `fraction_cells_expressing` (Phase 2 of the T-cell plan; same R/py change pattern) |

Recommended order: **A+B first** on the T-cell cohort (Phase 1 of the existing plan is
already scoped and approved — it is the prototype for this), then generalize B+F to Song-within
and WMB display, then C/D/E as a consistency pass. Each lands as its own gated phase.

## Decisions requested

1. **Scope.** Adopt detection-as-primary-gate as a repo-wide direction (findings A–F), or keep
   it T-cell-only for now and treat the others as documented-but-deferred?
2. **Anti-shim on the share.** When detection becomes the gate, do we *remove* the share-based
   `confidence_tier` location gate (`song_location_high` etc.) and the viewer share ladders, or
   keep the share as a labeled secondary descriptor for one release? (CLAUDE.md anti-shim
   favors removal once the replacement is proven.)
3. **SEA-AD (finding C).** Bring SEA-AD onto the same detection/share footing, or leave the
   human cohort's log2-ratio as-is and only harmonize the mouse + T-cell cohorts?
4. **Breadth metric.** Confirm effective-number-of-cell-types + %-in-top (gated to presence)
   as the standard breadth readout, replacing Tau, everywhere — or T-cell-only.
