# Attribution view consolidation — around the Song renderer

> **Status: IMPLEMENTED — all sprints S0–S4 complete.**

## Goal

The Attribution view in the Song, 5xFAD, and T-cell kinase explorers are three
divergent implementations. Consolidate them around the **Song** renderer: one
shared verdict-table + inline-accordion engine driven by a per-cohort manifest,
so the table/sort/dedup/row-visibility/bulk-anchor/accordion machinery and the
reusable detail sections are shared, while each cohort declares only its columns
and its genuinely cohort-specific sections.

**Unify the engine, the confidence definition, and the evidence axes; keep only
the cohort-specific data (which reference, which cell-type vocabulary).** The
fork rules in
[`viewer_frontend_contract.md`](../foundation/viewer_frontend_contract.md) list
`kinase_audit.js` as a consolidation candidate. We share the renderer AND the
`confidence_tier` definition across all three cohorts; what stays per-cohort is
only the corroborating reference and the cell-type spine — supplied as data, not
forked code. Sections that have no analog in a cohort (e.g. a mouse-brain WMB dot
plot for the T-cell donor cohort) are dropped at the manifest level, never
silently reimplemented.

## Scope

The **Attribution view only** — the verdict table + its detail expander — in all
three explorers. Out of scope: the rest of each kinase workbench (master list,
trajectory/NES plots, Measurement-Trace / MEA-Prep / MEA-Score subtabs). Those
are separate consolidation debt and are not touched here.

## Current state (3-way divergence)

| Aspect | Song (canonical) | 5xFAD | T-cell |
|---|---|---|---|
| Host file | `viewer/.../kinase_audit.js` + `kinase_explorer.js` helpers | `viewer/.../kinase_fivexfad.js` (own top-level tab) | `tcell_viewer/.../kinase_audit.js` + `kinase_explorer.js` (fork) |
| Tab placement | subtab in `KINASE_AUDIT_TABS` | subtab in bespoke `_F5_AUDIT_TABS` | subtab in `KINASE_AUDIT_TABS` |
| Detail pattern | **inline accordion** (`_renderAttributionDetail`) | side drawer (`_f5RenderAttributionDrawer`) | side drawer (`_renderAttributionDrawer`) |
| Verdict columns | 11 | 11 (already on Song cell renderers) | 7 (was 8 pre-`mea_false_positive` removal) |
| Leaf cell renderers | `_detGateCell` / `_concTierCell` / `_attrLfcColor` / `_attrConfidenceClass` | reuses Song's (same build) | **forked** `_tcellDetCell` / `_nsclcDetCell` + identical-but-copied `_attrLfcColor` |
| Row visibility | hide undetected; "Show all Levy-t5" toggle | hide low/none conf; "Show all" toggle | always show all (deliberate) |
| Confidence **definition** | cell-type exclusivity (`effective_n` over curated units) + WMB/human corroboration | bulk-sig + snRNA direction + location tier ≥2 (**a different, direction/location hybrid**) | **none** today — detection + NSCLC corroboration only |
| Location reference | WMB dot plot + SEA-AD heatmap + human score | WMB + SEA-AD | **NSCLC lineage strip** |
| Within-cohort direction | Song OLS (lazy `song_concordance` shard) | snRNA LFC + per-cell OLS shard | transcript trace, **no p-value** (single donor) |
| Mechanism cross-check | decomp OLS (lazy `decomp_ols` shard) | decomp MEA + celltype OLS shard | none |
| Extra scoping axes | contrast (disease × timepoint) | **+ age + tissue** (cortex/hippocampus) | day-vs-d2 |
| Dead code | — | — | `_renderSongOLSPanel` / `_renderDecompOlsTable` retained but never called |

### Accidental vs intentional divergence

**Accidental (consolidate away):**
- Drawer (5xFAD, T-cell) vs Song's inline accordion — pure presentation.
- 5xFAD reimplementing the verdict-table build / sort / comparator from scratch
  (`_f5SortAttrRows`, `_f5CmpAttr`) instead of the shared path.
- T-cell forking `_attrLfcColor` / `_attrSubGroupRow` / `_attrVerdictCmp`
  byte-for-byte and re-deriving `_tcellDetCell` from `_detGateCell`.
- T-cell dead code (`_renderSongOLSPanel`, `_renderDecompOlsTable` uncalled).

**Divergent confidence definitions (unify on Song's pill — see below):**
- The three cohorts compute `confidence_tier` three different ways today. This is
  the substantive inconsistency, not just a UI one. Consolidating "around Song"
  means all three adopt Song's **exclusivity** definition, with the corroborating
  reference swapped per cohort. Requires producer-side changes (§ Unified
  confidence).

**Intentional (preserve, express in the manifest):**
- T-cell: always-show-all rows; no p-values on the single-donor trace; NSCLC
  reference instead of WMB/SEA-AD; day vocabulary.
- 5xFAD: age + tissue axes; decomp-only fallback rows; gzip lazy sidecars.
- Song: curated specificity units / collapsed clusters; human references.
- All three: **direction concordance is a separate, info-only axis** (never the
  pill). Song already does this (`direction_tier`); T-cell's OR≈1 caveat is just
  this rule applied — concordance shown, never gating.

## Design — one engine, three manifests

### New shared module

```
alz/viewer_shared/template/js/tabs/attribution_view.js
```

Resolved by both builds via the existing `viewer_shared` include path. Exposes:

```js
AttributionView.render(hostId, ctx, manifest)
```

The engine owns (lifted verbatim from Song's `kinase_audit.js`):
- 3-row grouped header build (super-group → sub-group → leaf with sort arrows);
- sort (default key from manifest, click-to-sort, num/conf desc & str asc);
- defensive dedup by `contrast_id|cell_type` using a manifest comparator;
- row-visibility (manifest predicate + "Show all" toggle, or always-show);
- the bulk-MEA anchor block above the table;
- the **inline single-expand accordion** detail rows (auto-open top row).

The manifest declares (per cohort):

```js
{
  columns: [{ key, label, type, group, sub, subLabel, render(r) }],   // ordered
  superGroups: [{ label, span, group }],
  getRows(ctx),            // cohort data access (getScopedAttribution / 5xFAD shards)
  dedupCmp(a, b),          // canonical-row pick; T-cell = tier-only
  defaultSort,             // {key, dir}
  rowVisible,              // (r) => bool, or null for always-show
  bulkAnchor(ctx),         // {contrast, nes, fdr, signNote}
  sections: [{ id, title, render(host, ctx, row) }],   // accordion sections in order
}
```

### Shared leaf + section renderers → `viewer_shared`

Move the canonical leaf renderers into the shared module (single definition):
`_detGateCell`, `_concTierCell`, `_attrLfcColor`, `_attrConfidenceClass`,
`_attrSubGroupRow`, `_attrVerdictConfCell`, `_attrVerdictCmp`.

Move the **reusable** detail-section renderers (used by ≥2 cohorts) to shared:
`_renderSpecificityVerdict`, `_renderWMBDotPlot`, `_renderSEAADHeatmap`,
`_renderSongOLSPanel` (rename to a cohort-neutral `_renderWithinCohortOLSPanel`),
`_renderDecompOlsTable`. They already read from `ctx`/lazy shards, so the cohort
supplies the data via `ctx`, not via forked code.

Cohort-specific section renderers stay local and are referenced from that
cohort's manifest only:
- T-cell: transcript-trace section + `_renderNSCLCLineageStrip`.
- 5xFAD: per-cell-type decomp-MEA section (age-scoped) + `celltype_ols` table.

### Per-cohort manifests

```
alz/viewer/template/js/tabs/attribution_manifest_song.js       (Song)
alz/viewer/template/js/tabs/attribution_manifest_fivexfad.js   (5xFAD)
alz/tcell_viewer/template/js/tabs/attribution_manifest_tcell.js (T-cell)
```

Each is a thin declaration (columns + sections + data hooks). The cohort's host
tab calls `AttributionView.render(host, ctx, MANIFEST)` instead of its bespoke
`_renderAttributionVerdict` / drawer.

### Manifest contents (column spine)

Shared spine present in all three: **Cell type · Detected · Conc tier · a
direction column · a "vs bulk" column.** Cohort extras layer on:

- **Song:** + Conf pill, WMB Detected/Conc, SEA-AD LFC, Song LFC, Decomp NES/FDR, vs-Bulk.
- **5xFAD:** + Conf pill, WMB Detected/Conc, SEA-AD LFC, snRNA LFC, Decomp NES/FDR, Bulk-match.
- **T-cell:** + **Conf pill** (unified, see below), Δ-vs-d2 (info), Concordance (info),
  Timecourse (info), NSCLC Detected.

> **Field names stay per-cohort; the manifest maps them to a shared spine.** This
> plan does **not** rename `song_detected` / `fivexfad_detected` / `tcell_detected`
> (or the `_concentration_tier` / `_lfc` families) to a single uniform key.
> "Consolidate around Song" means a shared *renderer + confidence definition*, not a
> shared column schema in the payloads. The shared engine reads each cohort's
> existing field via the manifest's `key` (and `render(r)`), so the producer-side
> names remain forked by intent — abstracted over, not unified. Renaming the payload
> fields is explicitly out of scope.

### Unified confidence definition (Song's exclusivity pill, all three cohorts)

`confidence_tier` is computed the **same way** for every cohort — Song's
cell-type-exclusivity model
([`specificity_confidence.md`](../foundation/specificity_confidence.md) §4):

```
inputs (all from the shared standard metric — already cohort-uniform):
  detected      = fraction_cells_expressing ≥ 0.10          (within-cohort)
  eff           = effective number of cell types/states     (within-cohort)
  corroborated  = an independent reference agrees on the home cell class
tiers:
  very_high  detected, eff ≤ 1.5, corroborated
  high       (detected, eff ≤ 3, corroborated) OR (detected, eff ≤ 1.5, uncorroborated)
  moderate   detected, eff ≤ 3, uncorroborated
  low        detected, eff > 3            (broadly expressed)
  none       not detected
```

Only the **corroborating reference** is per-cohort (supplied as data, not code):

| Cohort | Within-cohort exclusivity | Corroborator |
|---|---|---|
| Song | Levy-t5 clusters → curated units | WMB class + human (SEA-AD/HBCA) |
| 5xFAD | `new_clusters` per tissue | WMB class + SEA-AD |
| T-cell | ProjecTILs states | NSCLC lineage detection |

> **`eff` is computed over curated specificity UNITS in every cohort (Song's
> convention) — not raw clusters.** The pill consumes a unit-level `effective_n`:
> over-split native clusters are collapsed onto curated specificity units
> (`config.load_specificity_unit_map`) before measuring exclusivity, so a pan-class
> kinase split across several sub-clusters is not mis-read as broad. Song reads
> `song_unit_effective_n`; 5xFAD reads `fivexfad_unit_effective_n` (computed in
> `snrna_specificity.py` via the SAME shared `unit_concentration_shares` /
> `unit_effective_n` helpers in `specificity_class.py` — 5xFAD uses the identical
> unit map, since its named cell types are the same Levy-t5 vocabulary; unmapped
> `cluster-NN` fall back to their own unit). The raw per-cluster `*_effective_n` is
> retained only as the displayed "subtype spread", never the tier input. T-cell's
> ProjecTILs states are already curated units, so `tcell_effective_n` needs no
> collapse. **An earlier draft of this note called per-cohort eff granularity
> "by design" — that was wrong (it would let over-split 5xFAD clusters depress the
> pill) and was corrected 2026-06-23.**

Direction concordance (`song_lfc` / `tcell_lfc` / `fivexfad_lfc`) is a **separate
info-only axis** in all three — shown, never gating the pill. Where the cohort's
reference reports **absence** at the home cell type, that is the "uncorroborated"
case and the pill cannot exceed `moderate`. (The prior T-cell `mea_false_positive`
flag — a separate verdict badge for this case — was removed entirely on
2026-06-22; the absence signal now lives only in the corroboration step, not as a
standalone flag/column.)

Producer changes this requires:
- **5xFAD** — replace `_assign_fivexfad_song_aligned_confidence` (bulk-sig +
  direction hybrid) with the exclusivity tier on `fivexfad_effective_n` +
  WMB/SEA-AD corroboration. The bulk-sig + direction gates move to the info-only
  direction axis (not lost — relocated).
- **T-cell** — add `confidence_tier`/`confidence_basis` to
  `_build_tcell_attribution_index` using `tcell_effective_n` + NSCLC corroboration
  (NSCLC absence → uncorroborated).
- **Song** — unchanged (it is the reference definition).

### Section drops (N/A at manifest level — never silently reimplemented)

| Section | Song | 5xFAD | T-cell |
|---|---|---|---|
| Specificity verdict (exclusivity pill) | ✓ (curated units) | ✓ (`new_clusters`, no curated-unit collapse) | ✓ (states; corroborator = NSCLC) |
| WMB dot plot | ✓ | ✓ | ✗ (not a mouse-brain cohort) |
| SEA-AD heatmap | ✓ | ✓ | ✗ |
| NSCLC lineage strip (corroborator) | ✗ | ✗ | ✓ |
| Within-cohort OLS direction | ✓ (Song OLS) | ✓ (per-cell OLS) | ✓ (transcript trace, no p-value) |
| Mechanism decomp OLS | ✓ | ✓ | ✗ (no decomp layer) |

## User-visible change

**5xFAD and T-cell detail panels move from a below-table drawer to Song's inline
single-expand accordion.** This is the concrete meaning of "consolidate around the
Song page." 5xFAD and T-cell users will click a verdict row and the detail expands
in place (top row auto-opens) instead of populating a separate drawer below. Same
content, Song's interaction model.

## Phased plan (gated — stop at each boundary)

**Phase 0 — unify the confidence definition in the producers (data layer).**
Add the exclusivity tier to T-cell (`_build_tcell_attribution_index`, NSCLC
corroborator) and replace 5xFAD's hybrid (`_assign_fivexfad_song_aligned_confidence`
→ exclusivity tier, WMB/SEA-AD corroborator; relocate bulk-sig + direction to the
info axis). Song unchanged. Pin the tier logic in one shared helper if practical.
*Checkpoint: regenerate both payloads; report per-cohort tier distributions
(this is a finding to read, not a number to tune) before any frontend work.*

**Phase 1 — shared engine + Song cutover (no behavior change for Song).**
Extract `AttributionView` + shared leaf/section renderers into
`viewer_shared/...attribution_view.js`; build `attribution_manifest_song.js`;
repoint Song's `kinase_audit.js` attribution subtab to
`AttributionView.render(..., SONG_MANIFEST)`. Song output must be pixel-identical.
*Checkpoint: diff the Song attribution DOM before/after; report.*

**Phase 2 — T-cell cutover.** Build `attribution_manifest_tcell.js` (now with the
unified Conf column + NSCLC-corroborator specificity-verdict section); repoint
T-cell's `kinase_audit.js` to the shared engine; delete the T-cell forks
(`_renderAttributionDrawer`, forked `_attrLfcColor`/`_attrSubGroupRow`, and the
dead `_renderSongOLSPanel`/`_renderDecompOlsTable`). T-cell adopts the accordion.
*Checkpoint: build T-cell viewer, smoke donor1/donor2, report.*

**Phase 3 — 5xFAD cutover.** Build `attribution_manifest_fivexfad.js` (with the
age/tissue pre-scoping hook and decomp-only fallback expressed as manifest
`getRows`); repoint `kinase_fivexfad.js`'s Attribution subtab to the shared
engine; delete `_f5RenderAttribution`/`_f5RenderAttributionDrawer`/`_f5SortAttrRows`/
`_f5CmpAttr`. 5xFAD's master list, trajectory, and other subtabs are untouched.
*Checkpoint: build unified viewer, smoke 5xFAD cortex/hippocampus × ages, report.*

**Phase 4 — docs.** Update `kinase_explorer_attribution.md` (single shared engine
+ per-cohort manifests; unified exclusivity pill with per-cohort corroborator;
drawer→accordion), `specificity_confidence.md` (the pill now spans all cohorts),
and `viewer_frontend_contract.md` (remove `kinase_audit.js` attribution from the
consolidation-candidate list; add `attribution_view.js` to shared modules).

## Sprint decomposition & agent assignment

The five phases map 1:1 to sprints. Dependencies force the order
**S0 → S1 → {S2, S3} → S4**. S2 and S3 are mutually independent (different cohort
files; both only *read* the shared engine produced in S1), but each is gated, so
they run one at a time with a report between rather than in parallel. Each sprint is
implemented by one subagent that stops at its checkpoint; the next sprint is
dispatched only after the prior gate is reported and cleared.

| Sprint | Phase | Files touched | Deliverable | Depends on | Gate (report before next) | Shipped notes |
|---|---|---|---|---|---|---|
| **S0** | 0 — producer confidence | `alz/build_tcell_viewer.py`, `alz/viewer/cohorts/fivexfad.py`, new shared tier helper under `alz/cross_reference/` (or `alz/bulk_mea/`) | unified `confidence_tier`/`confidence_basis` in both payloads; 5xFAD's bulk-sig + direction gate relocated to the info-only axis; Song unchanged | — | per-cohort tier distributions (a finding, not a number to tune) | Tier logic landed in `alz/bulk_mea/exclusivity_tier.py` (`exclusivity_tier(detected, eff, corroborated) → (tier, basis)`). 5xFAD: bulk-sig + snRNA direction + decomp-agrees now stored as info-only row fields (`bulk_mea_significant`, `direction_concordant`, `decomp_agrees_bulk`) — do not gate the pill. T-cell had no confidence pill before this sprint. Song output byte-identical. |
| **S1** | 1 — shared engine + Song | new `alz/viewer_shared/template/js/tabs/attribution_view.js`, new `alz/viewer/template/js/tabs/attribution_manifest_song.js`, repoint Song `kinase_audit.js` attribution subtab | `AttributionView.render` engine + shared leaf/section renderers; Song running on it, pixel-identical | S0 | Song attribution DOM diff before/after = none | Shipped as planned. Song's `_renderSpecificityVerdict` adapter currently lives inside `attribution_view.js` rather than `attribution_manifest_song.js` — not duplication, placement only. |
| **S2** | 2 — T-cell cutover | new `alz/tcell_viewer/template/js/tabs/attribution_manifest_tcell.js`, repoint tcell `kinase_audit.js`; delete T-cell forks (`_renderAttributionDrawer`, forked `_attrLfcColor`/`_attrSubGroupRow`) + dead `_renderSongOLSPanel`/`_renderDecompOlsTable` | T-cell on shared engine + inline accordion + unified Conf pill (NSCLC corroborator) | S1 | `build_tcell_viewer.py` build + donor1/donor2 smoke | Shipped as planned. |
| **S3** | 3 — 5xFAD cutover | new `alz/viewer/template/js/tabs/attribution_manifest_fivexfad.js`, repoint `kinase_fivexfad.js` Attribution subtab; delete `_f5RenderAttribution`/`_f5RenderAttributionDrawer`/`_f5SortAttrRows`/`_f5CmpAttr` | 5xFAD on shared engine + accordion (age/tissue scoping + decomp-only fallback via manifest `getRows`) | S1 | unified build + cortex/hippocampus × age smoke | `_f5CmpAttr` was **kept** — it is used by 5xFAD master-list functions outside the attribution subtab. The §0 verdict adapters for all three cohorts landed in S3.5 as one shared shell (`_renderSpecificityVerdictShell` + helpers in `attribution_view.js`) plus three thin per-cohort adapters in their respective manifests. |
| **S4** | 4 — docs | `kinase_explorer_attribution.md`, `specificity_confidence.md`, `viewer_frontend_contract.md` | docs reflect the single shared engine, unified pill, drawer→accordion; remove the consolidation-candidate flag | S2, S3 | final payload-contract verify | This doc. |

Each subagent is handed: this plan, its sprint row, the relevant Explore map from
the divergence audit, and the standing constraints (`pixi run` for all verification,
memory caps on any large-artifact step, **no commits**, anti-shim — the pivot
replaces, no dual code paths left behind). S0 is dispatched first.

## Anti-shim cleanup folded in
- Delete the drawer renderers once the accordion engine is live (no dual pattern).
- Delete T-cell dead code (`_renderSongOLSPanel`/`_renderDecompOlsTable` uncalled).
- Delete the per-cohort forked leaf renderers superseded by the shared ones.
- No "keep the old drawer behind a flag" — the pivot replaces.

## Verification (per the frontend contract)
```bash
node --check alz/viewer_shared/template/js/tabs/attribution_view.js
node --check alz/viewer_shared/template/js/tabs/attribution_view.js   # + each manifest
pixi run python alz/build_unified_viewer.py --html --validate --skip-verify
pixi run python alz/build_tcell_viewer.py --html --validate
pixi run python alz/viewer/verify_payload_contract.py \
  outputs/reports/unified_viewer/unified_viewer.payload.json \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json
```
Browser smoke: Song attribution accordion (4 sections), 5xFAD attribution
(cortex/hippocampus × ages, accordion), T-cell donor1 attribution (transcript
trace + NSCLC strip, accordion), donor2 no-kinase message.

## Decisions
1. **Drawer → accordion for 5xFAD and T-cell** — ✅ confirmed (accordion everywhere).
2. **Unified confidence pill across all three** — ✅ confirmed (all cohorts adopt
   Song's exclusivity tier; corroborator swapped per cohort; direction stays
   info-only). This is the Phase-0 producer work.
3. **Scope** — attribution view + the confidence *definition* feeding it. Still
   out of scope: the 5xFAD master list / non-attribution subtabs and the
   Song↔T-cell `kinase_audit.js` MEA-subtab duplication. **Confirm this boundary.**

## Honesty guardrails (Phase 0)
- T-cell exclusivity uses pooled within-cohort scRNA across days — a real
  detection quantity, not dependent on biological replicates. The single-donor
  caveat applies only to p-values/direction, which stay info-only. The pill is
  honest for T-cell.
- The Phase-0 tier-distribution shifts (especially 5xFAD, whose definition
  changes) are a **finding to report**, not a number to tune. If the new pill
  reads worse for some kinases, that is the corrected measurement, not a regression.
