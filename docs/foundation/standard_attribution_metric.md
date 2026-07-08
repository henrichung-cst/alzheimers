# Standard attribution metric — one definition, every cohort

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
  Drops the `mean > 1` co-gate, which is scale-dependent and wrong on the Seurat log(CP10K+1)
  cohorts. Mean expression is kept as a displayed secondary, never in the gate.

## Q1 — specificity to a given cell type (per kinase × cell type)

Reported as a **pair, never a lone tier**:

- `detected` (✓/✗) — the gate. If ✗, the honest answer is "not present here," full stop.
- `concentration_c` = `wᶜ / Σ_{detected} w`, where `wᶜ` = **linear** per-cell mean (de-logged
  with the cohort's own base), summed **over detected cell types only**.

Two corrections vs. the old share, both load-bearing:
1. **Gated to detected types** → no phantom specificity from noise types (fixes the CDK1
   anomaly and the inverse-prediction finding).
2. **Linear weights** → no log-compression inflating apparent breadth/concentration.

**`concentration_of_total`** = `wᶜ / Σ_{all} w` — the cell type's share of the kinase's
**total** linear expression across *all* cell types (denominator over every cell type, not just
detected). This is the tier basis; `concentration_c` (detected-set, above) stays as the input to
`effective_n` and the confidence pill.

**`concentration_tier`** — the familiar 2×/5×/10× bins, applied to `concentration_of_total`
over a fixed **`1 / N_total`** baseline (an even split across *all* cell types in the cohort):

| tier | meaning |
|---|---|
| ≥10× | `concentration_of_total ≥ 10 / N_total` |
| ≥5× | `concentration_of_total ≥ 5 / N_total` |
| ≥2× | `concentration_of_total ≥ 2 / N_total` |
| ≥1× | at or above the all-cell-types even share |

The baseline is **`1 / N_total`** — fixed per cohort (31 Levy clusters, 14 T-states, …) — so a
10× pill means the *same bar* (≥10× the uniform share) for **every** kinase and is comparable at
a glance. This deliberately supersedes the earlier `1/n_detected` baseline, whose fold depended
on how many cell types each kinase happened to be detected in (a kinase present in one cell type
read as a trivial 1×, the *least* concentrated pill, despite being maximally specific). The
share-is-not-presence failure is still averted — the share is **linear** (no log inflation) and
the tier is assigned only to **detected** cells — so the fold is honest without being relative to
a per-kinase denominator.

Non-misleading rule: the tier is assigned only when `detected`, and shown beside both shares —
`concentration_of_total` (what the pill is built from) and the detected-set `concentration_c`
(what the confidence eff is built from) — so the two are never conflated.

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

## The attribution claim gates on detection

- `top_celltype` is chosen **only among detected cell types**. A kinase is never attributed to
  a cell type where `detected = ✗`.
- "high" / "very_high" confidence **requires** detection at the attributed cell type.
- Directional concordance (LFC sign/magnitude) is the tie-break **among detected
  types** — it ranks *which* present cell type, never resurrects an absent one.

## What is removed (anti-shim)

| Removed | Replaced by |
|---|---|
| `specificity_score` share as standalone localizer/tier (WMB, Song-within, NSCLC, T-cell) | detected-set `concentration_c`, always paired with `detected` + `effective_n` |
| `confidence.py` `song_location_high` (`song_specificity ≥ 2/31`) and `wmb_crosscheck` (≥2/9) share gates | detection at the attributed cluster |
| Viewer fold-over-uniform ladders `_wmbTier` / `_msTier` / `_tcellTier` / `_nsclcTier` — fold over `1/N_total` on the **ungated, log** share | same 2×/5×/10× bins over `1/N_total`, but on the **linear** `concentration_of_total` assigned only to **detected** cells; plus the per-cell-type detection column and `effective_n` |
| `song_tau` as the headline per-gene specificity | `effective_n_celltypes` (interpretable; Tau was rejected) |
| Two-ladder split (2× pipeline gate vs 10/5/2/1 viewer) | one definition computed once in Python, rendered in JS |

5xFAD's decomposition (per-cell-type OLS significance) is already presence-grounded and stays —
it is the conceptual model the others move toward. Its OLS significance is the 5xFAD analog of
`detected`.

## Constraints

- **Detection needs per-cell counts**, computed per cohort under a memory cap: WMB and NSCLC
  from their expression exports; Song-within (`snrna_integration.py`), T-cell-within
  (`tcells_scrna_extract.R`), and 5xFAD (`build_5xfad_snrna_attribution.R`) from their raw
  sparse matrices.
- **SEA-AD is exempt.** It emits only a `log2(ct/brain)` ratio from summarized per-supertype
  expression, with no per-cell counts available; it is labeled "ratio-only — no detection" and
  stays a corroborating human reference, not gated on detection.
- **De-log needs the right base per cohort** (log2 vs Seurat natural-log CP10K+1) for the linear
  concentration weights. Wrong base silently distorts `effective_n`.

## One metric, every cohort

The single definition lives in `alz/cross_reference/specificity.py` (detection gate, linear
de-log, concentration + tier, effective number of cell types) and is reused by every cohort's
within-cohort and reference expression surface — Song, WMB, NSCLC, T-cell, and 5xFAD.
