# T-cell viewer coherence fixes

Audit of the incoherences reported against the T-cell attribution viewer, the
mechanism behind each, and the proposed fix. **No code edited yet — this is for
approval.**

All findings reproduced against
`outputs/reports/kinase_attribution_tcells/donor1/unified_attribution_tcells.csv`
and `outputs/reports/nsclc_reference/nsclc_kinase_specificity.csv` on 2026-06-23.

---

## Finding 1 — Phantom cell-type home for undetected kinases (the "completely broken" symptom)

**Reported:** CAMK1G shows "concentrated in CD4", but 0 cell specificity, 0 cell
states, 0 cross-lineage, 0 within-cohort attribution. TTBK1 likewise carries a
CD4 home and a confidence tier while detected nowhere.

**Reproduced:**

| gene | max fraction expressing (any state) | tcell_detected | top_celltype | concentration_tier |
|---|---|---|---|---|
| CAMK1G | 0.043% | False in all 14 states | CD4 | 2 |
| TTBK1 | 0.376% | False in all 14 states | CD4 | 1 |

**137 of the 157 genes** that are undetected in every within-cohort state (at the
10% floor) still carry a non-zero cell-type concentration tier.

**Mechanism:** `specificity.compute()` derives the cell-type concentration tier
from linear-expression *shares*, and by explicit design
(`specificity.py:11` — *"Detection never filters the share basis"*) computes a
tier **even when the gene is detected in zero labels**. For a kinase detected
somewhere this is correct ("of where it lives, how concentrated"). For a brain
kinase with only trace counts, whichever state has the largest rounding-error
fraction wins, and its coarse type (CD4) gets a phantom tier. The state-enrichment
axis is already gated on detection (the `elig` mask, `tcell_within_cohort.py:257`);
the **cell-type axis was never gated**.

**Fix:** gate the cell-type axis on detection in the within-cohort producer
(`tcell_within_cohort.py`). The shallow-modality justification for gating the
*basis* (undetected = noise, not low-real-expression) is local to this cohort, so
the shared `specificity.compute` and the deep brain cohorts (Song / 5xFAD) are
**not** touched. Concretely: compute the coarse cell-type aggregation over
**detected labels only** — so `tcell_top_celltype`, `tcell_celltype_concentration_tier`,
and `tcell_celltype_effective_n` reflect only states where the transcript is
actually present, and a gene detected in zero states gets all three null. CAMK1G
and TTBK1 then render "—" (no home, no pill), consistent with their 0/0/0.

---

## Finding 2 — TTBK1 "confidence" while 0/7 cross-lineage

**Reported:** TTBK1 reads as confident in T_NK yet shows 0/7 lineages.

**Reproduced:** TTBK1 in the NSCLC reference is `group_detected == False` in **all
7 lineages** (max T_NK fraction 0.998%, just under the 1% floor); `n_detected_coarse = 0`.
So NSCLC does **not** corroborate it. Its only within-cohort "signal" is the
phantom tier from Finding 1.

**Fix:** same as Finding 1. Once the cell-type axis is detection-gated, TTBK1 has
no within-cohort home and no NSCLC corroboration → no confidence pill. The 0/7
cross-lineage and the (now-absent) pill become consistent.

---

## Finding 3 — Within-cohort detection still gated at 10%

**Reported:** "Cells expressing still uses a 10% cell filter, another completely
fucked metric."

**Reproduced:** within-cohort `tcell_detected = fraction_expressing >= 0.10`
(`specificity.DETECTION_FRAC_MIN` default, not overridden). Genes detected in ≥1
state: **232/369 at 10%, 253 at 5%, 298 at 1%**.

**Fix:** lower the within-cohort floor to **1%**, matching the NSCLC reference
(`NSCLC_DETECTION_FRAC_MIN = 0.01`) so the two detection columns use one
consistent rule. Implemented by passing `detection_frac_min=0.01` to the
within-cohort `specificity.compute()` call only — the shared default and the brain
cohorts stay at 10% (deep snRNA, no complaint, out of scope).
*Tradeoff (one line):* at shallow scRNA depth 1% admits sparser calls than 10%;
this is the consistency the report asks for. CAMK1G/TTBK1 remain undetected even
at 1%, so Finding 1's fix still holds.

---

## Finding 4 — Cross-lineage shows a count + one lineage, not the members

**Reported:** "6/7 Epithelial" shows a count and a single dominant lineage but not
*which* 6 lineages the kinase is found in.

**Mechanism:** the payload carries only `nsclc_lineages_detected` (count),
`nsclc_lineages_total` (count), `nsclc_top_lineage` (single dominant name) —
`build_tcell_viewer.py:_load_nsclc_coarse_breadth`. The member list is never
emitted, though `group_detected` per `spec_group` is present in the source file.

**Fix:** add a payload field `nsclc_lineage_list` = sorted names of the lineages
with `group_detected == True`, and render them in the cross-lineage cell tooltip
(and inline when the cell is expanded) so "6/7" is backed by the actual lineage
names.

---

## Finding 5 — "Δ vs d2" / "vs Bulk" show "—" for most state-specific kinases

**Reported:** the vast majority of high-state-specificity kinases show "—" for
both delta columns.

**Reproduced:** `tcell_lfc` (Δ vs d2) is a pseudobulk log-difference
`m[state,day] − m[state,d2]`. It is **undefined where the transcript is absent at
d2 or the contrast day**. Null rate at contrast d13: **51% where the state's
fraction is 0**, vs 15% where fraction > 0. A state-specific kinase is, by
definition, absent in most states — so most of its state rows have no baseline to
difference and correctly render "—".

This is **correct behavior, not a bug** (you can't fold-change a transcript that
isn't there), and these columns are explicitly info-only (concordance with bulk
kinase activity is at chance, OR≈1). The problem is presentation: "—" reads as
"missing data."

**Fix (presentation only, no data change):** clarify the empty-state — render "—"
with a tooltip "transcript absent at this timepoint — no fold defined" and keep
the existing info-only caption. No threshold or metric change.

---

## Finding 6 — "linear" terminology

**Reported:** "linear is a horrible term."

**Found:** the only user-facing "linear" in the T-cell surface is the
kinase-explorer comment/label describing state enrichment as "each state's linear
expression". The internal field is `linear_expression` (anti-logged mean). Rename
the visible label to plain language ("expression level"); leave the internal field
name (not user-facing).

---

## Regeneration & scope

Light regen — no heavy 897k re-stream and no full-matrix read:

1. `tcell_within_cohort.py` producer → re-run (reads per-state donor aggregates
   only) for Findings 1–3.
2. NSCLC already at 1% (`nsclc-metrics`); only re-read for Finding 4's member list.
3. `build_tcell_viewer.py` rebuild for Findings 4–6.

**Out of scope / explicitly not doing:** the shared `specificity.compute` contract,
the brain cohorts' 10% floor, and the mass docstring sweep (per prior instruction).
A handful of NSCLC tooltips still say "no minimum-fraction floor" (inaccurate since
the 1% change); flagged, not edited, unless approved.
