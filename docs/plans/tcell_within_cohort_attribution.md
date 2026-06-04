# T-cell within-cohort cell-type attribution (specificity + concordance)

> Status: implemented 2026-06-03. Module `alz/cross_reference/tcell_within_cohort.py`; payload + UI wired in `build_tcell_viewer.py` and the three T-cell viewer templates; runner task `tcell-within-cohort` (chained before `tcell-viewer`).

## Context

The mouse/AD unified viewer localizes a **bulk** kinase activity signal to a **cell type** using the cohort's own paired transcriptomics — the "Song" method (`alz/reference/snrna_integration.py` + `alz/bulk_mea/attribute.py`). It needs no disease reference: a kinase attributes to a cluster when its transcript is (a) preferentially expressed in that cluster (specificity) and (b) moving in the same direction the bulk activity is (concordance).

The T-cell viewer ports the entire attribution UI (the Attribution audit subtab, `getScopedAttribution`, the WMB-style tier badge `_wmbTier`/`_wmbTierBadge`) but it is **dormant**: `build_tcell_viewer.py` emits no `attribution_index`, the kinase list had the Specificity column/filter stripped, and the capability flags are hardcoded `False`. We will light it up with a within-cohort attribution computed from donor1's data.

**Two deliberate departures from the mouse design:**
1. **No per-cell significance test.** Donor1 is a single donor with one scRNA library per day — there are no biological replicates, so a per-cell `FindMarkers` test would be pseudoreplication (Squair 2021) and any per-`(state,day)` p-value would be fabricated. The mouse tier never consumed the Song p-value anyway (`_assign_confidence_and_basis_vectorized` gates on `|song_lfc|` + sign, not `song_fdr`). Concordance is therefore a **pseudobulk log2FC** (direction + magnitude), and credibility comes from **timecourse consistency** across d13/d17/d20, not a p-value. **No FDR column anywhere in the T-cell attribution.**
2. **Confidence tiers (high/moderate/low) are replaced by binned specificity (1×/2×/5×/10× of uniform)** — copying the unified viewer's WMB-tier design. Concordance (sign agreement vs bulk + timecourse consistency) is the orthogonal gate/display axis.

**Out of scope:** two-donor agreement (donor2 corroboration). Donor2 has no IMAC → no MEA → no attribution; existing capability gating already handles it.

## Method recap — what is copied from where

| Piece | Mouse source (copy in design) | T-cell adaptation |
|---|---|---|
| Specificity (share of total expression) | `snrna_integration.py::step_specificity` (:216) | per `(gene, state)` from `aggexp_data.csv ÷ cell_counts` |
| Tier binning 10/5/2/1× uniform | `kinase_explorer.js::_wmbTier/_wmbTierBadge` (:37-52); uniform = 1/34 | uniform = **1/N_states** (donor1 = 1/14); binned in **Python** because N varies per donor |
| Cross-join + assemble | `attribute.py::_assemble_unified` (:152) | same shape, tier/SEA-AD/WMB blocks removed |
| List column + sort | `body.html` `wmb_max_tier` th (:91); `_makeKeCompare` branch (:443) | `tcell_max_tier` |
| List min-tier filter | `body.html` select (:49-60) + `kinase_wiring.js` (:83-91) + gate `kinase_explorer.js` (:686) | `tcellMin` |
| Cell-types pill | `kinase_explorer.js::_renderCellTypesCell` (:558) | badge by `tcell_tier` instead of confidence string |
| Verdict subtab table | `kinase_audit.js::ATTR_VERDICT_COLS` (:567) + `_renderAttributionVerdict` (:607) | T-cell column set (below) |

## Data side — new computation

**New module `alz/cross_reference/tcell_within_cohort.py`** (sits beside `evidence.py` / `human_celltype_attribution.py`). Donor1 only. Inputs already on disk:
- `data/derived/tcells_incytr_inputs/donor1/scrna/aggexp_data.csv` (gene × `<state>__<day>`, **sum** of log-normalized expression)
- `.../scrna/cell_counts.csv` (`state, day, n_cells`)
- `outputs/reports/kinase_attribution_tcells/donor1/mea/kinase_timepoint_nes.csv` + `kinase_timepoint_fdr.csv` (bulk MEA, contrasts `D1_d13…D1_d20`)
- kinase→gene: T-cell MEA `kinase` names are gene symbols (AAK1, ACVR2A). Join on `UPPER(kinase)`; reuse `config.MAPPING_CACHE_FILE` if any names follow kinase-library vocab, else identity. **Implementer: verify the kinase vocabulary against the mapping cache before assuming identity.**

**Correctness trap:** `aggexp_data.csv` is a **sum** across cells, not a mean. Both computations must divide by the matching `n_cells` from `cell_counts.csv` to get per-cell mean log-expression `m[state,day]` — otherwise abundant states look artificially "specific".

Outputs under `outputs/reports/kinase_attribution_tcells/donor1/`:

1. **`tcell_specificity.csv`** — `(gene, state, tcell_specificity, tcell_mean_log2_expression)`. Static property pooled across **all** scRNA days (mirrors Song pooling all animals). For each gene: `mean_over_days(m[state,day])` per state, then `tcell_specificity = mean_in_state / Σ_states mean_in_state`. Formula mirrors `step_specificity` (`snrna_integration.py:241-246`). Uniform baseline = `1/N_states`.

2. **`tcell_concordance.csv`** — `(gene, state, day, tcell_lfc)` for day ∈ {d13, d17, d20} (MEA-days ∩ scRNA-days): `tcell_lfc = m[state,day] − m[state,d2]` (log-space difference = LFC analog). No p-value.

3. **`unified_attribution_tcells.csv`** (+ `_full`) — cross-join donor1 MEA `(kinase × day)` × states (the `attribute.py:152` pattern, **without** the tier/SEA-AD/WMB-merge logic). Per `(kinase, contrast=day, cell_type=state)`:
   - `NES`, `FDR` (bulk anchor)
   - `tcell_specificity` (repeated across day rows), `tcell_tier` ∈ {0,1,2,5,10} (binned in Python via `1/N_states`)
   - `tcell_lfc`, `tcell_concordance = sign(NES) · tcell_lfc`
   - `tcell_consistency` = count of {d13,d17,d20} where `sign(NES)·tcell_lfc > 0` (per-`(kinase,state)`, repeated)
   - `_full` keeps every row; `unified_attribution_tcells.csv` keeps **concordant** rows only (`tcell_concordance > 0`), mirroring the mouse `combined_confidence != "none"` drop.
   - Row-count guard: `len(_full) == n_kinases × n_states × n_contrast_days` (same assertion style as `attribute.py:262`).
   - Day-contrasts d15/d19 have no scRNA → **no attribution rows** (honest empty), only d13/d17/d20 attribute.

## Viewer payload wiring — `alz/build_tcell_viewer.py`

- **New `_build_tcell_attribution_index(donor)`**: reads `unified_attribution_tcells.csv`, returns the columnar dict the JS already consumes (`getScopedAttribution`, `kinase_explorer.js:91`): `{kinase_id, contrast_id, cell_type, tcell_specificity, tcell_tier, tcell_lfc, tcell_concordance, tcell_consistency, nes, fdr}`. Map `kinase`→`kinase_id` via the slice's `kid` map; `day`→`contrast_id` via the `short_contrasts` order from `_build_donor_kinases_slice` (:483).
- **Populate `top_celltype_1`** (currently `[""]`, :497): highest-`tcell_tier` concordant state per kinase.
- Add `attribution_index` to the payload dict (:1938) per donor context (donor1 only; donor2 omitted/empty).
- **Capability flag:** flip the donor1 attribution gate true. Reuse the already-wired `song_concordance` flag name **or** add a clearer `within_cohort_attribution` flag (:1867 per-context, :1910 global) — keep `human_reference`/`decomp_ols`/`subclass_breakdown` **False**. Update the `notes` to describe the within-cohort method.
- Reuse the existing `aggexp_data.csv` load path in `_write_tcell_transcript_trace` (:1497) so the specificity builder and trace writer share one read.

## Viewer UI — parity + Cell-types pill

**`alz/tcell_viewer/template/js/tabs/kinase_explorer.js`**
- Add `_tcellTierBadge(t)` (copy `_wmbTierBadge` :48) and `_kineMaxTcellTierScoped(kid, filter)` (copy `_kineMaxWmbTierScoped` :292) reading `e.tcell_tier` directly (tier is precomputed).
- New render branch for the `tcell_max_tier` column (`_tcellTierBadge(_kineMaxTcellTierScoped(...))`) and a `_makeKeCompare` sort branch (copy the `wmb_max_tier` branch :443).
- Filter gate: `tcellMin` compares `e.tcell_tier >= tcellMin` directly (simpler than the mouse `wmbMinScore` since tier is precomputed; copy the gate at :686).
- Adapt `_renderCellTypesCell` (:558) to badge by `tcell_tier` (10→vhi, 5→hi, 2→mid) instead of the confidence string.

**`alz/tcell_viewer/template/js/tabs/kinase_audit.js`** — replace `ATTR_VERDICT_COLS` (:567) with the T-cell set, drop the decomp super-group and `cross_rank`/SEA-AD/Song columns:
`[cell_type · tcell_tier (badge) · tcell_specificity · tcell_lfc · concordance vs bulk]`.
- In `_renderAttributionVerdict` (:607): remove the decomp/`cross_rank` computation; sort by `tcell_tier` then `tcell_concordance` desc; keep the bulk-MEA anchor banner (it uses `NES`). Replace the explainer text with the within-cohort method (specificity tier + sign-vs-bulk + timecourse consistency; state the no-p-value rationale).
- Evidence drawer (`_renderAttributionDrawer` :815) currently renders WMB dot-plot / SEA-AD heatmap / Song OLS — none exist for T-cells. Replace with the **per-state transcript trace** (the day-vs-baseline expression curve, data already emitted by `_write_tcell_transcript_trace`), or omit the drawer. Do **not** render empty WMB/SEA-AD/Song panels.

**`alz/tcell_viewer/template/body.html`** — re-add to the kinase table (:58-68) and toolbar (:26-55), mirroring the unified `body.html`:
- `<th data-col="tcell_max_tier">` Specificity column + `<th>` Cell-types pill column.
- `<select id="ke-filter-tcell">` min-tier filter (Any / ≥1× / ≥2× / ≥5× / ≥10×, labelled with the donor's `1/N_states` absolute values).
- Wire the select in `kinase_wiring.js` (copy unified :83-91, key `tcellMin`).

## Files

- **Create:** `alz/cross_reference/tcell_within_cohort.py`
- **Modify:** `alz/build_tcell_viewer.py`; `alz/tcell_viewer/template/js/tabs/kinase_explorer.js`; `.../kinase_audit.js`; `.../body.html`; `.../js/.../kinase_wiring.js` (or wherever the T-cell filter wiring lives)
- **Runner:** add a `pixi`/shell step to run `tcell_within_cohort.py` before `build_tcell_viewer.py` (find the existing tcell-viewer task via `pixi task list`).

## Verification

1. **Module:** run `tcell_within_cohort.py` on donor1; assert `_full` row count = `n_kinases × n_states × 3` (d13/d17/d20); assert every shipped (`unified_attribution_tcells.csv`) row has `tcell_concordance > 0`; print tier distribution.
2. **Biology spot-check:** confirm a known exhaustion-associated kinase attributes to **CD8Tex / CD8Tpex** with the expected sign and a sensible tier; sanity-check 2-3 states' top-specific genes against `aggexp`.
3. **Viewer:** rebuild the T-cell viewer (task from `pixi task list`), hard-refresh (Ctrl+Shift+R per CLAUDE.md), confirm in DevTools: `PAYLOAD.attribution_index` present for donor1, absent/empty for donor2; the Specificity column + filter + Cell-types pill render; the Attribution subtab shows the T-cell verdict columns (no SEA-AD/WMB/Decomp/Conf, no FDR).
4. **Donor2 negative:** confirm donor2 shows the "No IMAC kinase MEA" note and no attribution surfaces.
