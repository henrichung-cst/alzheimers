# Cross-species specificity validation — calc, propagation, viewer

**Date:** 2026-06-05
**Scope (user-gated):** validate the EXISTING mouse/human cell-type specificity — (1) correctly
calculated, (2) faithfully propagated into the viewer, (3) filterable/visualizable. **No** new
cross-species "same peak-cluster" feature this round. Method: chain-trace + source spot-check
(memory-safe; no large-artifact loads).

**Status: AUDIT — read-only. No code changed. Awaiting approval before any fix.**

---

## TL;DR

The two specificities are computed by **different metric families** and the mouse tier is
**mis-calibrated**:

| | Mouse (WMB) | Human (SEA-AD MTG) |
|---|---|---|
| Metric | **share**: `mean_log2_expr(class) / Σ mean_log2_expr(retained classes)` → sums to 1 | **fold**: `log2(mean(supertype) / mean over all 139 supertypes)` → signed, centered 0 |
| Resolution | **11 WMB classes** (31 levy_t5 clusters collapse to 11) | 139 supertypes → rolled to levy_t5 |
| Scope | whole brain (13 regions) | MTG only |
| Tiered against | uniform `1/N` | `log2(2/5/10)` |
| `N` actually used | **1/29** (crosstable), **1/34** (explorer), **1/31** (pipeline) | n/a (self-consistent) |

**The headline bug:** the WMB share is normalized across **≤11 retained WMB classes** (true
uniform ≈ 1/9–1/11 ≈ 0.09–0.11), but every tier compares it to `1/29`–`1/34`. So a kinase at
*true uniform* is scored ~3× specific. Worked example, **Csnk1a1** (housekeeping, flat across
classes): `specificity_score ≈ 0.11` in every class, summing to 1.0 over the 9 classes it
appears in → at crosstable uniform `1/29`: `0.11/0.0345 = 3.2×` → **"≥2×" badge**. A
textbook non-specific kinase is labeled specific. This directly degrades the user's headline
use case ("kinases highly specific to individual tissues").

Propagation itself is **faithful** (round(4) passthrough; no vocab collision in the shipped
payload). Human specificity is **internally self-consistent** (log2 metric, log2 tiers). The
problems are (A) the mouse uniform baseline, (B) three inconsistent `N` constants across tabs,
(C) cross-species comparability of "×N specific in both", and a few minor viewer gaps.

---

## 1. How each is calculated (verified)

### Mouse — `alz/reference/wmb_expression.py:615-626`
```
specificity_score(gene, class) = mean_log2_expr(gene, class)
                                 ───────────────────────────────────────────────
                                 Σ_{c ∈ RETAINED_WMB_CLASSES} mean_log2_expr(gene, c)
```
- `RETAINED_WMB_CLASSES` = `config.load_cluster_to_wmb_class_map().values()` =
  **11 distinct WMB classes** (the 31 levy_t5 clusters map onto only 11 — verified from
  `data/derived/bridges/cluster_to_wmb_class.csv`: 31 rows → 11 distinct class labels).
- Denominator is the **sum over retained classes**, so per gene the score **sums to 1.0**
  (verified: Csnk1a1's 9 present classes sum to 1.000). It is a **share**, not a z-score/fold.
- Scope = whole brain (13 regions), `WMB_REGION_SCOPE=whole_brain` (config.py:408), stamped to
  `wmb_kinase_expression.scope.json`.
- Crosswalk WMB→levy_t5 is **direct, 1-hop** (attribute.py:175-176): clusters sharing a parent
  WMB class **inherit identical scores** (so the score cannot resolve below WMB-class level).

### Human — `alz/reference/human_expression.py:158-170`
```
specificity_score(gene, supertype) = log2( mean_expr(gene, supertype)
                                            / mean over ALL 139 MTG supertypes )
```
- Signed log2 fold-over-mean. Verified scale: range −8.23…+6.84, 60% negative, centered ~0.
- Rolled to levy_t5 by **weighted mean of log2 ratios** across mapped supertypes
  (human_celltype_attribution.py:103) — i.e. log-space (geometric-mean) averaging. Defensible.
- Scope = **MTG only** (single region) — inherent to the SEA-AD atlas. Asymmetric vs mouse
  whole-brain, but unavoidable.
- Crosswalk SEA-AD subclass→supertype→levy_t5 is **direct, 1-hop** (build_seaad_bridge.py).

---

## 2. Propagation (verified faithful)

`unified_attribution_full.csv` → `attribution_index` (build_unified_viewer.py:3076,
`round(4)`) → payload → crosstable `_KX_WMB_BY_KIN_CLUSTER` (keyed by levy_t5 cell_type).
Human: `seaad_kinase_specificity.csv` → `build_celltype_specificity_payload` (rolled to
levy_t5) → `human.celltype_specificity.seaad_mtg.ranked_by_kinase[name][].score`.

- **No rounding/scale corruption.** attribution carries 31 levy_t5 clusters.
- **Earlier worry disproven:** I suspected `decomposition_index.cell_type` shipped WMB-class
  labels (build line 3098 reads `decomp["wmb_class"]`). The **shipped payload** carries
  levy_t5 names (19 distinct), so the cluster dropdown does **not** mix vocabularies. Verified
  against `unified_viewer.payload.json`.

---

## 3. Viewer filtering & visualization (current state)

- **Toolbar filters** (added earlier this session): `M-spec` / `H-spec` dropdowns (Any / ≥1× /
  ≥2× / ≥5× / ≥10×), resolved by `_kxResolveSpec` — "Any cell type" → peak across clusters
  (with argmax cluster in tooltip); pinned cluster → value at that cluster. Filter + column
  share the resolver, so they agree.
- **Columns** M-spec / H-spec render `_kxWmbTierBadge` / `_kxLog2TierBadge`.
- **Detail "Cell-type Specificity" sub-tab** reuses the per-dataset renderers verbatim
  (`_renderKinaseCelltypeEvidence` mouse, `_khRenderAttribution` human) — Mouse | Human.

Works mechanically. Issues below.

---

## 4. Findings (ranked)

### F1 — WMB uniform baseline is mis-set (the share is normalized over ~11, tiered against ~29–34) — **HIGH, correctness**
The share sums to 1 over ≤11 retained WMB classes (true uniform ≈ 1/9–1/11 ≈ 0.09–0.11), but
tiers use `1/29` (crosstable), `1/34` (explorer), `1/31` (pipeline). Inflation ≈ 2.6–3.1×.
Csnk1a1 (flat, ~0.11) → "≥2×" badge. Over-calls specificity for housekeeping/broadly-expressed
kinases — the exact failure mode that matters for "highly specific to individual tissues".

### F2 — three inconsistent `N` for the SAME metric — **HIGH, consistency**
- pipeline `config.N_CELL_TYPES = 31` → `SPECIFICITY_LOW/HIGH = 1/31, 2/31`
- `kinase_explorer.js:37` `_WMB_UNIFORM = 1/34`
- `kinase_crosstable.js:188` `_KX_WMB_UNIFORM = 1/_KX_CLUSTERS.length` = **1/29** as shipped
The **same kinase shows a different M-spec tier** in the Kinase (mouse) tab vs the Crosstable
tab, and neither matches the pipeline's `high/low` confidence gate. The crosstable's `N` even
drifts with payload contents (it's `len(decomp clusters ∪ human-spec clusters)`).

### F3 — mouse vs human "×N" are different baselines — **MEDIUM, interpretation**
Mouse ×N = "N× its uniform **share**"; human ×N = "N× the cross-supertype **mean** (fold)".
Filtering "M-spec ≥5× AND H-spec ≥5×" intersects two unlike definitions. Acceptable for a
coarse co-filter, but should be **documented in the UI** so users don't read it as one metric.
(Relevant to the deferred "same-cluster" feature.)

### F4 — mouse specificity cannot resolve below WMB-class — **MEDIUM, interpretation**
31 levy_t5 clusters inherit 11 class-level shares (duplicated). A kinase confined to one WMB
class (e.g. `30 Astro-Epen` → Astrocytes/Ependymal/…) shows identical "specificity" for every
levy_t5 cluster in that lineage — so per-cluster mouse specificity is really per-WMB-class.
The viewer presents it at cluster granularity without signaling this.

### F5 — 2 levy_t5 clusters with WMB specificity are absent from the crosstable cluster pivot — **LOW**
`_KX_CLUSTERS` (29) = decomp ∪ human-spec clusters; attribution carries 31. Two clusters with
WMB scores but no decomp/human-spec row are not selectable for a pinned-cluster M-spec lookup.

### F6 — M-spec column header advertises the wrong denominator — **LOW (cosmetic of F1/F2)**
`_kxBuildHeader` prints "vs uniform 1/${_KX_CLUSTERS.length}" → "1/29", surfacing the mis-set
baseline to the user.

---

## 5. Proposed fixes (for approval — NOT yet applied)

Anti-shim: replace the wrong constant in one canonical place; do not add a toggle.

1. **F1+F2+F6 — one canonical WMB uniform = `1 / N_RETAINED_WMB_CLASSES` (= 11).**
   - Add `N_RETAINED_WMB_CLASSES = len(set(load_cluster_to_wmb_class_map().values()))` to
     `config.py`; redefine `SPECIFICITY_LOW = 1/N_RETAINED`, `SPECIFICITY_HIGH = 2/N_RETAINED`.
   - Emit it into the payload (e.g. `meta.wmb_uniform`) so JS reads ONE value.
   - `kinase_explorer.js` `_WMB_UNIFORM` and `kinase_crosstable.js` `_KX_WMB_UNIFORM` both read
     `PAYLOAD.meta.wmb_uniform`; delete the `1/34` and `1/_KX_CLUSTERS.length` literals.
   - Fix the header text to the same value.
   - **Re-validate:** Csnk1a1 → `0.11×11 = 1.2×` → "<2×" (correctly not specific); re-check the
     `high/low` confidence counts in attribution shift accordingly (expected: fewer "high").
2. **F3 — UI copy.** Tooltip/legend stating mouse = share-vs-uniform, human = log2 fold-over-
   mean; "×N" is a coarse co-filter across two metrics, not one scale.
3. **F4 — signal class-level resolution.** In the M-spec tooltip, name the WMB class and note
   "shared across levy_t5 clusters in this lineage."
4. **F5 — include all attribution clusters** in `_KX_CLUSTERS` (union in attribution cell_type
   too), so every cluster with a WMB score is pinnable.

**Open question for the user:** for F1, is the intended specificity reference (a) **1/11**
(uniform across the resolvable WMB classes — recommended, matches the metric's own
normalization), or (b) genuinely **1/31** (treat each levy_t5 cluster as an independent unit,
accepting that lineage-shared clusters are non-independent)? This changes every tier and the
`high/low` confidence gate, so it should be confirmed before editing.

---

## 6. Song-derived location specificity — NEW SIGNAL (approved direction: Song primary, WMB cross-check)

**Decision (user, 2026-06-05):** add a Song-derived per-cell-type *location* signal as the
headline mouse specificity, at full 31-cluster resolution, keeping WMB beside it as the
independent cross-check. This is **not** a back-compat shim — the two answer different
questions (WMB = "an outside atlas calls this region-specific"; Song = "within this study's
own cells, the gene concentrates here"), so both coexisting is the anti-shim exemption.

This supersedes the F1 "1/11 vs 1/31" patch as the *primary* fix: the calibration trap (an
even-split yardstick that inflates with bin count) is solved by adopting a **bin-count-robust
specificity index**, not by re-choosing the divisor. WMB keeps its existing tier (informational
cross-check); the Song column drives the headline "specific in both" use case.

### 6.1 What "Song location" is — and the module that ALREADY computes it

Song location = per-gene expression across cell types, the **expression** axis, orthogonal to
the AD-vs-WT **activity** (NES/LFC) axis already shipped.

**This is already computed.** `alz/reference/snrna_integration.py` (`step_specificity`) builds
it from the real h5ad (`170_gex_celltypes_00.h5ad`, 63,695 nuclei × 28 animals): pseudobulk per
(animal, spine cluster) with a 5-cell-per-(cluster,animal) gate, CPM+log2, pooled across
animals, → `outputs/reports/snrna_integration/song_expression_specificity.csv`
(`cell_type, gene_symbol, specificity_score, mean_expression`). This is **better provenance**
than the frozen `aggexp.csv` AggregateExpression (real counts vs a baked SCT normalization), so
the earlier aggexp-based prototype module was deleted as redundant.

Two facts established (2026-06-05):
- It targets the **full 31 spine** (`config.CLUSTER_SPINE`); `pseudobulk_cell_counts.csv`
  confirms all 31 clusters present (58–8,091 cells each, pooled). The shipped specificity CSV
  showing only 19 is **stale** (predates the current spine — the module's own `missing_spine`
  guard flags this); a re-run restores 31.
- It is **not wired into the viewer or any pipeline consumer** — the only reference to
  `SONG_EXPRESSION_FILE` outside its producer is `config.py`. So surfacing it is net-new.
- Its metric is the **share** (`mean_in_cluster / Σ means`) — the peak/even trap. Tau must be
  added here.

### 6.2 Prototype (verified 2026-06-05, memory-safe column-stream of `aggexp.csv`)

Three-way, three reference kinases — Song agrees with the independent WMB atlas on every clear
case and resolves **46 cell types vs WMB's 9**:

| Kinase | Song (46 types) | WMB (9 types) | Human SEA-AD (139) |
|---|---|---|---|
| Csnk1a1 *(housekeeping)* | flat, 46/46 express, top 15% | flat, top 12% | weak, +0.96 max |
| Camk2a *(neuronal)* | neuronal spread, top 16% | neuronal Glut, top 19% | neuronal L2/3, +1.19 |
| Csf1r *(microglial)* | **95% Microglia** | **90% Immune** | **+4.50 Micro-PVM** |

**Bin-count trap confirmed numerically:** housekeeping Csnk1a1 reads 7× over even-split in Song
(46 bins) but only ~1.1× in WMB (9 bins). A naive peak/even ratio mislabels it at high
resolution → the metric MUST be bin-count-robust (see 6.4).

### 6.3 Build plan (phased)

1. **Compute (tau)** — extend `snrna_integration.py step_specificity`: alongside the existing
   per-(gene,cluster) `specificity_score` (share) and `mean_expression`, add per-gene `tau`
   (Yanai index over the cluster mean vector), `top_cluster`, `top_share`, `n_expressing`.
   Share-per-cluster and tau-per-gene are different granularities (a per-cluster cell value vs
   an overall specificity tier), both needed — not a shim. Re-run `--pseudobulk --specificity`
   to refresh the stale 19→31 output.
2. **Propagate** — merge `tau` + per-cluster `share` into `unified_attribution_full.csv`
   (alongside `wmb_specificity`, not replacing it) → `attribution_index` → payload.
3. **Viewer** — Song location as the primary mouse M-spec column (crosstable + explorer),
   WMB demoted to a secondary "atlas cross-check" column. Tier on `tau`, not peak/even.
   Reuse existing badge classes; no new CSS.

### 6.4 The one open decision — the specificity metric (needs sign-off)

The yardstick must NOT be peak/even (bin-count-sensitive — proven in 6.2). Candidates:

- **τ (tau), Yanai et al. 2005** — the standard tissue-specificity index:
  `τ = Σ_i (1 − x_i/x_max) / (N − 1)`, 0 = uniform → 1 = single-cell-type. Normalized for N
  (bin-count-robust by construction), peer-reviewed, the field default for exactly this
  question. **Recommended.** Csf1r → ≈1, Csnk1a1 → low.
- **Top share (absolute)** — largest cell type's fraction (Csf1r 95%, Csnk1a1 15%). Trivially
  interpretable; keep as the human-readable companion column regardless.
- Gini — bin-robust but less standard in this domain than τ.

**Plan: τ as the tier-driving score + top-cluster/top-share as the readable columns.** Tiers
become τ-band thresholds (e.g. τ≥0.85 highly specific, ≥0.6 specific) — to be calibrated on the
Song distribution after compute, not guessed.

**Decided (user):** metric = τ; tiers = **0.85 / 0.60**; landing = **Song primary, WMB cross-check**.

### 6.5 Implementation status (2026-06-05)

- **Phase 1 (compute) — DONE.** `snrna_integration.py step_specificity` now emits per-gene `tau`
  + `top_cluster`/`top_share`/`n_expressing` alongside the per-(gene,cluster) `share`.
  Regenerated `song_expression_specificity.csv` (stale 19 → full 31 clusters). Verified:
  Csnk1a1 τ=0.07 (housekeeping, correctly not specific), Csf1r τ=0.96 (Microglia), NTRK1 τ=0.99
  (Cholinergic — WMB only 0.27, lumps cholinergic into a broad class → the case for Song-primary).
  Kinase τ median 0.35, range 0.05–0.99.
- **Phase 2 (propagate) — DONE.** `evidence.prepare_song_specificity` carries tau/top_cluster/
  top_share → `attribute.py` merge → `unified_attribution_full.csv` (cols `song_tau`,
  `song_specificity`, `song_top_cluster`, `song_top_share`). Ran attribute → mechanism → recover.
  `build_unified_viewer.py` emits these into `attribution_index` + `meta.wmb_uniform` (= 1/11,
  the F1/F2/F6 fix).
- **Phase 3 (viewer) — crosstable DONE.** M-spec column = Song τ tier (per kinase; ★ when the
  pinned cluster is the top cell type); WMB demoted to a "WMB" cross-check column (per-cluster,
  ×uniform); M-spec filter = τ bands (0.60/0.85). Explorer tab: WMB baseline fixed to
  `meta.wmb_uniform` (F2); Song-primary upgrade of the explorer's mouse Kinase tab is the
  remaining follow-on (its WMB tier column + filter at kinase_explorer.js:782/653/289).
