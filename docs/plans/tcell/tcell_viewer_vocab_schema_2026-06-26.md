# T-cell viewer vocabulary + detection schema (2026-06-26)

The T-cell viewer's attribution surface accreted **13 fuzzy terms across 4 data axes**, built
at different times with un-reconciled detection philosophies. The result reads as shoddy:
the same word means different things, the same pixels (`✓ 6%`) encode opposite verdicts, and
the code contradicts its own tooltips. This doc is the consolidation: a fixed vocabulary, three
global invariants, the rename map, and the live bug it also closes.

## The four axes (one vocabulary each, disambiguated by SOURCE, never by inventing a noun)

| Source | Axis | Identity | Strength | Reads as |
|---|---|---|---|---|
| Own scRNA — presence | **Detection** | `detected` | `pct_expressing` | present at the floor? |
| Own scRNA — coarse type | **Cell type** (CD8/CD4/Treg) | `cell_type` | `confidence` | concentrated in one type? (the pill) |
| Own scRNA — fine state | **Cell state** (ProjecTILs) | `cell_state` | `enrichment` | where on the activation continuum? |
| NSCLC reference | **Specificity** | `nsclc_cell_type` | `nsclc_specificity` | restricted to how few reference types? |

Reserved words, each owned by exactly one source and never crossing it:
**detection** = presence · **confidence** = own-data type concentration ·
**enrichment** = own-data state continuum · **specificity** = reference restriction.

## Three invariants the rename MUST enforce (renaming alone fixes nothing)

### I1 — One detection rule, stated once; never two verdicts behind identical pixels
Today "detected" uses **three floors**: cohort `detected` ≥10%, NSCLC `detected` ≥1%,
NSCLC `specificity_count` ≥10%. So `✓ 6%` is detected on an NSCLC row but ✗ on a cohort row —
identical glyph, opposite meaning.
- The two cohort/reference detections legitimately differ (the 897k-cell reference is ~30× deeper,
  so a 1% floor there is real signal; the shallow cohort needs 10%). That is a **defensible**
  difference — but then they must **not share the word "detected"**. Name them by what they are
  and print the floor in the column header:
  - cohort → **`present`** (floor 10%, header "Present ≥10%")
  - reference → **`present`** (floor 1%, header "Present ≥1%")
  Same concept, explicit per-axis floor in the header, so `✓ 6%` is never ambiguous because the
  reader always sees which floor that column is under.
- NSCLC `specificity_count`'s ≥10% floor is a **separate** threshold (concentration, not presence)
  and stays 10% — but is relabeled so it never reads as "detection" (see I2).

### I2 — One strength encoding per kind; kill the duplicate tier helpers
Today strength is encoded **five incompatible ways**: confidence word-tiers
(`very_high…none`), cohort concentration `≥N×` (`_concTierCell`, 2/5/10×), state enrichment
`≥N×` (1.5/2/3×), NSCLC specificity raw count `N/7`, NSCLC concentration `≥N×`
(`_nsclcConcTierBadge`, 2/5/10× — a *duplicate* of `_concTierCell` in another file).
- **Concentration/enrichment** → a single fold (`×`) helper with **one** threshold set across
  cohort-type, cohort-state, and NSCLC. Delete `_nsclcConcTierBadge`; keep one shared badge fn.
- **Specificity** → render as a fold or tier consistent with the others, NOT a bare inverted count
  (see I3). If a count is shown, it is secondary evidence, not the headline strength.

### I3 — One direction convention; signal the one inversion
Higher = stronger everywhere **except** NSCLC specificity, where `1/7` is strong and `7/7` weak.
Nothing encodes the inversion, so `0/7` reads as failure when (for `nsclc_cell_type = T_NK`) it is
the strongest corroboration possible.
- Either invert specificity into a "higher = more specific" score, or color *low* counts as the
  strong end and state it in the header: **"NSCLC specificity (types — fewer = more specific)."**

### (corollary) One breadth name, one floor
`tcell_effective_n` / `tcell_celltype_effective_n` / `n_detected_native` / "NSCLC breadth" all =
"effective # buckets spanned," at two floors, under four names. Collapse to a single `breadth`
concept at the axis's stated floor; demote the rest to informational or drop.

## Rename / retire map

| New field · label | Was | Note |
|---|---|---|
| `present` · "Present ≥10%" | `tcell_detected` | floor in header (I1) |
| `pct_expressing` · "Cells expressing" | `tcell_fraction_expressing` | the % the floor gates on |
| `cell_type` · "Cell type" | `tcell_top_celltype` | drop "top"/"home" — it IS the cell type |
| `confidence` · "Confidence" | `confidence_tier` | the pill; no "specificity" in prose |
| `confidence_reason` | `confidence_basis` | resolves the "basis" overload |
| `cell_state` · "Cell state" | row col misnamed `cell_type` + `state` | one name for the fine axis |
| `enrichment` · "Enrichment" | `tcell_state_enrichment` | the column once mislabeled "Specificity" |
| `peak_enrichment` · "Peak enrichment" | `peak_state_enrichment` | kinase-level max rollup |
| `nsclc_present` · "Present ≥1%" | `nsclc_detected` | floor in header (I1) |
| `nsclc_cell_type` · "NSCLC cell type" | `nsclc_top_lineage` | **"lineage" retired** |
| `nsclc_specificity` · "NSCLC specificity (types — fewer = more specific)" | `nsclc_specificity_count` | direction cue (I3) |
| `breadth` | `tcell_effective_n` & friends | one name, one floor |

### Retired outright (no alias, no tombstone)
- **"Specificity" as any within-cohort label** → split to `confidence` (type) + `enrichment` (state).
- **"home" / `home_state` / `specificity_celltype`** as a user concept → the pill-anchor row becomes
  an internal render detail (`_pill_anchor`), never surfaced.
- **"lineage"** everywhere (labels, `nsclc_*_lineage*` fields, tooltips, Python emit) → "NSCLC cell type."
- **"subtype"** → it is a cell state. One word.
- **"NSCLC breadth" / `nsclc_lineages_detected`** (the 1% lenient count) → folded into the single
  `breadth` concept; raw membership kept only as an unlabeled list if needed, not a 2nd metric.
- **`_nsclcConcTierBadge`** → deleted; one shared concentration badge (I2).

## Live bug this also closes (independent of the rename — exists today)

The NSCLC detection floor contradicts itself three ways and ships a self-contradictory tooltip:
- `nsclc_expression.py:429–432` comment: *"NO minimum-fraction floor… 'detected' = frac > 0, NOT ≥10%."*
- `nsclc_expression.py:437–438` code: actually applies `>= 0.01` (1%). The described override was never implemented.
- `kinase_audit.js:538` tooltip: *"expressed in ≥1 cell — no minimum-fraction floor"*; **line 549 same file**: *"detection = ≥1% of cells expressing."*

So data=1%, comment=0%, UI=both. Decision needed: is NSCLC presence **>0** (any nonzero, comment's
intent) or **≥1%** (code's behavior)? Whichever — make code, comment, tooltip, and header agree, and
print the floor (I1). Plus stale floor comments: `kinase_explorer.js:37` still says `>= 0.01`.

## Files touched (one coordinated pass)
- `alz/cross_reference/tcell_within_cohort.py` — column emit + docstring.
- `alz/reference/nsclc_expression.py` — floor decision, comment, column names.
- `alz/tcell_viewer/slices_kinase.py` — payload keys.
- `alz/tcell_viewer/template/js/tabs/attribution_manifest_tcell.js`, `kinase_explorer.js`, `kinase_audit.js` — labels, keys, delete duplicate badge.
- `alz/viewer_shared/template/js/tabs/attribution_view.js` — `_detGateCell` header/floor (shared with Song; scope check before editing).
- `alz/tcell_viewer/template/body.html` — headers + tooltips.

## Decisions (resolved 2026-06-26) + status

1. **I1:** keep `detected`; force ONE floor everywhere = **10%**. ✅ DONE.
2. **Floor value:** 10% for the NSCLC reference too (the "deep reference, 1% is real" leniency is overruled for consistency). ✅ DONE.
3. **I3 specificity:** keep `N/7` (most easily understood); label the direction (fewer = more specific). ✅ count kept; direction label lands in the rename pass.
4. **`_detGateCell` consistent across ALL cohorts** — shared helper, not a fork. ✅ DONE (NSCLC routed through it; `_nsclcDetCell` deleted; null→n/a folded in).

### Implemented this pass (floor + detGateCell)
- Single floor via `specificity.DETECTION_FRAC_MIN` (10%); the divergent `NSCLC_DETECTION_FRAC_MIN=0.01` and the
  `tcell_within_cohort` local `0.10` copy now both reference it. NSCLC `_write_attribution_metrics` 1%-override block deleted.
- `nsclc-metrics` regenerated: **5,898 → 2,758** detected (kinase,cell_type) rows; 3,140 flipped ✓→✗. Payload: NSCLC `True 4,839 / False 10,827 / n/a 672`.
- Re-entrancy bug fixed in `_write_attribution_metrics` (strip to raw facts at entry — `--metrics` on an already-metriced CSV used to collide on `specificity_count_x/_y`).
- Shared `_detGateCell` extended with null→n/a; NSCLC column + `_nsclcDetCell` removed. The live self-contradicting NSCLC floor tooltips/comments (1% vs 0% vs 10%) reconciled to 10%.

### Implemented this pass (vocabulary field+label rename, 2026-06-26)

**Shared-contract reconciliation (key finding — corrects the rename map above).**
Three keys the map treated as tcell-local are actually the unified viewer's
cross-cohort contract, emitted by Song (`song.py`) + 5xFAD (`fivexfad.py`) +
`bulk_mea` and read by the shared `attribution_view.js` / `build_unified_viewer.py`:
- **`cell_type`** is the engine's universal row-identity key (`attribution_view.js`
  reads `r.cell_type` at 77/525/583/630…). For tcell the row entity *is* the
  ProjecTILs state, so renaming it `cell_state` forks the shared engine — it's a
  **label** problem, not a key problem. Fixed at label only: the state column reads
  **"Cell state"**, the coarse-type column reads **"Cell type"**. `tcell_top_celltype`
  keeps its name (its `→cell_type` rename collided with the shared key).
- **`specificity_celltype`** / **`confidence_basis`** are shared keys too. Kept.
  Tcell's `home_state` payload field → **`_pill_anchor`** (tcell-local), still mapped
  to `specificity_celltype` in `kinase_explorer.js` for the shared engine.

Renaming those three keys for real is a **separate cross-cohort migration** (every
viewer + the shared engine); out of scope for this tcell pass.

**Done (tcell/nsclc-local, non-breaking):**
- All **"lineage" → "cell type" / "NSCLC cell type"** (the user's chief objection):
  payload fields `nsclc_lineages_*`→`nsclc_cell_types_*`, `nsclc_top_lineage`→
  `nsclc_cell_type`, `nsclc_lineage_list`→`nsclc_cell_type_list`; JS fns
  `_renderNSCLCLineageStrip`→`_renderNSCLCCellTypeStrip`, `_tcellLineageGroup`→
  `_tcellCellTypeGroup`; CSS `.nsclc-lineage-*`→`.nsclc-celltype-*`; producer
  internals `LINEAGE_MARKERS`/`coarse_lineage`/`cluster_to_lineage` + the
  `coarse_cell_type` cluster-label column; all prose/tooltips/headers.
- **"specificity" reserved for NSCLC**: within-cohort column relabeled
  **Specificity → Enrichment**; §0 "Specificity verdict" → **Confidence verdict**
  (fn `_renderTcellSpecificityVerdict`→`_renderTcellConfidenceVerdict`, section id
  `specificity`→`confidence`); within-cohort prose "specificity" → confidence/enrichment.
- **"home" jargon retired**: `home_state`→`_pill_anchor`; "Cell-type home" dt → **Cell type**.
- **"subtype" → "cell state"** (e.g. "Subtype spread" → "Cell-state spread").
- **I2**: deleted `_nsclcConcTierBadge`; the strip now calls the shared `_concTierCell`
  via a thin `_concTierBadgeOrEmpty` wrapper (tier-0 suppressed).
- **I3**: NSCLC spec header now reads "NSCLC specificity (N/7, fewer = more specific)";
  tooltips state direction.

**Regen note (no viewer impact):** the NSCLC Stage-1 intermediate CSVs
(`NSCLC_CLUSTER_LABELS_FILE`, `NSCLC_CELL_LABELS_FILE`) now carry `coarse_cell_type`
instead of `lineage`/`coarse_lineage`. The on-disk copies are stale until a
`nsclc-label` (`--label-clusters`) regen; the viewer reads `nsclc_kinase_expression.csv`
(`cell_type`/`spec_group`), which is unaffected.
