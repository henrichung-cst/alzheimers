# Kinase Explorer Attribution Tab

This is the authoritative live contract for the Attribution views in the
Kinase Explorer family. Historical implementation plans may describe earlier
drawers, score names, or payload shapes; this document describes the current
architecture and should be updated when the implementation changes.

## Scope

The Attribution view is a single shared engine driven by per-cohort manifests.
All three kinase-explorer surfaces (Song AD/mouse, 5xFAD supporting cohort,
T-cell donor viewer) run `AttributionView.render(hostId, ctx, MANIFEST)` and
share the same verdict-table, sort, dedup, row-visibility, bulk-anchor, and
inline single-expand accordion machinery. The Human/Mukesh view is table-only
and is out of scope for the shared engine.

| Cohort surface | Frontend owner | Producer contract | Current UI shape |
|---|---|---|---|
| Song AD / mouse | `alz/viewer/template/js/tabs/kinase_audit.js` calling `AttributionView.render(..., SONG_MANIFEST)` | `PAYLOAD.attribution_index`, `PAYLOAD.specificity_units`, `PAYLOAD.decomposition_index`, lazy Song/decomp shards | sortable verdict table with inline expandable detail rows (accordion) |
| Human / Mukesh | `alz/viewer/template/js/tabs/kinase_human.js` | `PAYLOAD.human.celltype_specificity`, `PAYLOAD.human.perdonor_index` | sortable table; no drawer |
| 5xFAD supporting cohort | `alz/viewer/template/js/tabs/kinase_fivexfad.js` calling `AttributionView.render(..., F5_MANIFEST)` | `PAYLOAD.supporting_5xfad` rows plus attribution/decomp sidecars | sortable verdict table with inline accordion |
| T-cell donor viewer | `alz/tcell_viewer/template/js/tabs/kinase_audit.js` calling `AttributionView.render(..., TCELL_MANIFEST)` | T-cell-specific `PAYLOAD.attribution_index`; optional NSCLC reference joins | sortable verdict table with inline accordion |

The UI must use categorical calls plus raw evidence columns with units. It must
not invent or display synthetic numeric attribution scores. Numeric gates used
for implementation are shown only when they have an analytical interpretation
and are already part of the method, such as fraction of cells expressing, FDR,
NES, log2 fold change, or effective number of specificity units.

## Shared Engine

### Module

```text
alz/viewer_shared/template/js/tabs/attribution_view.js
```

Resolved by both builds via the `viewer_shared` include path. Exposes:

```js
AttributionView.render(hostId, ctx, manifest)
```

The engine owns:
- 3-row grouped header build (super-group → sub-group → leaf with sort arrows);
- sort (default key from manifest, click-to-sort, numeric/confidence desc and string asc);
- defensive dedup by `contrast_id|cell_type` using the manifest's `dedupCmp`;
- row-visibility (manifest predicate + "Show all" toggle, or always-show);
- the bulk-MEA anchor block above the table;
- the **inline single-expand accordion** detail rows (top row auto-opens).

Shared leaf renderers held in the engine module:
`_detGateCell`, `_concTierCell`, `_attrLfcColor`, `_attrConfidenceClass`,
`_attrSubGroupRow`, `_attrVerdictConfCell`, `_attrVerdictCmp`.

Shared section renderers (used by ≥ 2 cohorts):
`_renderWMBDotPlot`, `_renderSEAADHeatmap`, `_renderWithinCohortOLSPanel`,
`_renderDecompOlsTable`, `_refCrosswalkLine`.

Shared §0 verdict shell: `_renderSpecificityVerdictShell` plus helpers
`_specEffBand`, `_specPct`, `_specF2`. Song's §0 adapter
(`_renderSpecificityVerdict`) also lives in this file (placement, not
duplication — it was not split to the Song manifest).

### Per-Cohort Manifests

```text
alz/viewer/template/js/tabs/attribution_manifest_song.js
alz/viewer/template/js/tabs/attribution_manifest_fivexfad.js
alz/tcell_viewer/template/js/tabs/attribution_manifest_tcell.js
```

Each manifest declares:
```js
{
  columns: [{ key, label, type, group, sub, subLabel, render(r) }],
  superGroups: [{ label, span, group }],
  getRows(ctx),          // cohort data access
  dedupCmp(a, b),        // canonical-row pick
  defaultSort,           // {key, dir}
  rowVisible,            // (r) => bool, or null for always-show
  bulkAnchor(ctx),       // {contrast, nes, fdr, signNote}
  sections: [{ id, title, render(host, ctx, row) }],
}
```

Field names remain per-cohort (`song_*` / `fivexfad_*` / `tcell_*`); each
manifest maps them to the shared spine via the column `key` and `render(r)`
accessor. Payload schemas were not renamed.

### Build includes

The shared `index.html.j2` includes `attribution_view.js` (before
`kinase_audit.js`). Each build supplies its cohort manifest via
`_VIEWER_SPECIFIC_TAB_INCLUDES` (unified viewer: Song + 5xFAD manifests;
T-cell build: T-cell manifest).

## Shared Concepts

### Bulk MEA Anchor

Every attribution view is anchored to a bulk kinase MEA result. The anchor is
the selected kinase and contrast, tissue, age, donor, or day as appropriate for
the cohort. The UI shows bulk NES and FDR when available, and all directional
columns are interpreted relative to that bulk direction.

Positive NES means kinase substrates are concentrated among sites with higher
stoichiometry or phosphosite signal in the disease/treated/timepoint arm named
by the cohort. Negative NES means the opposite arm has higher signal.

### Location, Direction, and Mechanism

The tab separates three questions:

| Question | Evidence examples | Interpretation |
|---|---|---|
| Where is the kinase transcript present or enriched? | Song/WMB/T-cell/5xFAD detection, concentration tiers, human location log2 ratio | cell-type location prior |
| Does the transcript or reference disease effect move with the bulk kinase direction? | Song OLS LFC, T-cell day-vs-d2 LFC, SEA-AD LFC | direction support — info-only, not the confidence pill |
| Does cell-type projected phosphosite evidence agree with bulk MEA? | decomposition NES/FDR, substrate-site OLS | mechanism/activity cross-check |

Do not collapse these axes into a single numeric result in the viewer.

### Confidence Is Categorical and Spans All Three Cohorts

Confidence labels (`none / low / moderate / high / very_high`) are the same
**cell-type exclusivity** definition in every cohort — Song's exclusivity pill
extended to 5xFAD and T-cell. The shared helper is
`alz/bulk_mea/exclusivity_tier.py::exclusivity_tier(detected, eff, corroborated)`.

In top-level Kinase Explorer tables, location confidence is encoded in the
localization badge color/tooltips (`Cell type`, `Cell types`, or T-cell `Cell
states`) rather than shown as a standalone `Conf` column. The full confidence
basis remains in the attribution/detail views and CSV exports may retain the
categorical field as audit metadata.

Direction concordance is always a **separate, info-only axis** shown alongside
the pill but never gating it. This applies to all three cohorts: the old 5xFAD
hybrid (bulk-sig + snRNA direction gates) and the old T-cell absence-only display
have both been replaced by the unified pill. See
`docs/foundation/specificity_confidence.md` for the full tier definition and
per-cohort corroborator details.

## Song AD / Mouse Attribution

### Producer Contract

Producer:

```text
alz/viewer/cohorts/song.py::_build_attribution_index
alz/viewer/cohorts/song.py::_build_specificity_units
```

The builder emits:

- `PAYLOAD.attribution_index`: columnar rows from
  `unified_attribution_full.csv` when available, otherwise
  `unified_attribution.csv`. The row spine is kinase x contrast x Levy-t5 cell
  type.
- `PAYLOAD.specificity_units`: static Song-cluster to specificity-unit mapping,
  plus Song-cluster to WMB-class and SEA-AD-subclass crosswalks.
- `PAYLOAD.decomposition_index`: compact per-kinase/contrast/cell-type
  decomposition MEA values.
- `edge_slice_ref.decomp_ols` and `edge_slice_ref.song_concordance`: lazy
  evidence shards read by the detail expander.

Important `attribution_index` fields consumed by the frontend:

| Field group | Columns |
|---|---|
| Identity | `kinase_id`, `contrast_id`, `cell_type` |
| Categorical calls | `confidence_tier`, `confidence_basis`, `direction_tier`, `direction_basis`, `human_location_tier`, `decomp_agrees_bulk` |
| Specificity unit | `specificity_unit`, `specificity_unit_label`, `specificity_celltype`, `specificity_collapsed` |
| Song location | `song_detected`, `song_fraction_cells_expressing`, `song_concentration`, `song_concentration_of_total`, `song_concentration_tier`, `song_effective_n`, `song_unit_effective_n`, `song_top_celltype`, `song_top_concentration` |
| WMB location | `wmb_detected`, `wmb_concentration`, `wmb_concentration_tier`, `wmb_mean_log2_expression`, `wmb_fraction_cells_expressing`, `wmb_binary_expressed` |
| Human/reference direction and location | `sea_ad_lfc`, `seaad_location_score`, `hbca_location_score`, `human_location_score` |
| Decomposition and Song direction | `decomp_nes`, `decomp_fdr`, `song_lfc`, `song_pval`, `song_fdr`, `song_direction_support` |
| Bulk MEA copy | `nes`, `fdr`, `concordance_source` |

### Frontend Flow

Frontend:

```text
alz/viewer/template/js/tabs/kinase_explorer.js::getScopedAttribution
alz/viewer/template/js/tabs/kinase_audit.js  → AttributionView.render(host, ctx, SONG_MANIFEST)
alz/viewer/template/js/tabs/attribution_manifest_song.js
alz/viewer_shared/template/js/tabs/attribution_view.js
```

The Attribution subtab is selected from `KINASE_AUDIT_TABS`. It reads the audit
panel's contrast picker and is independent of the left-list KinaseFilter. For
the selected kinase and contrast, it:

1. Reads `PAYLOAD.attribution_index` through `getScopedAttribution`.
2. Deduplicates defensively by `contrast_id|cell_type`, keeping the canonical
   row with the best confidence/location/decomp evidence.
3. Adds decomposition NES/FDR from `PAYLOAD.decomposition_index`.
4. Computes a display-only bulk/decomp sign-agreement glyph from bulk NES and
   decomposition NES/FDR.
5. Sorts the verdict table by the active column, defaulting to confidence.
6. Shows rows where Song detects the kinase plus the dominant specificity cell
   type; "Show all Levy-t5 clusters" reveals the full grid.
7. Opens the top visible row by default. Row clicks use a single-expand
   accordion with inline detail rows.

### Current Song UI

The verdict table groups columns into attribution evidence and decomposition
cross-checks. It always shows the bulk MEA anchor above the table.

Each expanded detail row contains:

- `Section 0 - Specificity verdict`: explains why the categorical confidence
  pill has its value. Shows the clicked cell type's fraction of cells
  expressing, share of total expression, concentration tier, specificity unit,
  collapse children when applicable, detected breadth, unit-level effective
  number, native cluster effective number, and the confidence basis.
- `Section 1 - Expression - WMB reference`: WMB dot plot for mean log2
  expression and fraction of cells expressing. The clicked Song cluster is
  mapped to its WMB class through `specificity_units.cluster_to_wmb_class` before
  outlining. Human location score is displayed here as corroborating location
  evidence.
- `Section 2 - Disease direction - SEA-AD`: human AD-vs-control LFC heatmap,
  with the clicked Song cluster mapped to SEA-AD subclass through
  `specificity_units.cluster_to_seaad_subclass`.
- `Section 2 - Disease direction - Song OLS`: within-cohort Song pseudobulk OLS
  rows from the lazy concordance shard, with the selected contrast highlighted.
- `Section 3 - Mechanism`: per-cell substrate-site OLS rows from the lazy
  decomposition OLS shard.

The retired `specificity_score` field is not part of the live contract. Use
`*_detected`, `*_concentration`, `*_concentration_tier`, and effective-number
fields instead.

## Human / Mukesh Attribution

### Producer Contract

Producer:

```text
alz/cross_reference/human_celltype_attribution.py::build_celltype_specificity_payload
alz/viewer/cohorts/mukesh.py::build_human_slice
```

The human viewer reads `PAYLOAD.human.celltype_specificity`, which contains
ranked per-kinase cell-type lists for available human references:

- `seaad_mtg`
- `allen_hbca`

Reference cell types are rolled up to the Levy-t5 nomenclature before display.
SEA-AD rows may carry AD-vs-control LFC; HBCA rows do not have an SEA-AD LFC
analog. Per-donor MEA values come from `PAYLOAD.human.perdonor_index`.

### Frontend Flow

Frontend:

```text
alz/viewer/template/js/tabs/kinase_human.js::_khRenderAttribution
```

The human Attribution subtab:

1. Collects ranked rows for every available human reference.
2. Merges rows to one displayed row per Levy-t5 cell type.
3. Keeps the strongest location evidence and the largest-magnitude SEA-AD LFC
   available for that cell type.
4. Derives a categorical confidence tier in the browser:
   high when human location is at least log2(2) and SEA-AD LFC magnitude is at
   least 0.1, moderate when either condition holds, low when location is at or
   above reference-wide mean, none otherwise.
5. Broadcasts the selected donor's NES/FDR across rows for that kinase.

This view is table-only. It does not display Song, WMB, decomposition OLS, or
per-cell substrate-site details because those layers do not exist for the human
per-donor surface.

## 5xFAD Supporting-Cohort Attribution

### Producer Contract

Producer:

```text
alz/viewer/cohorts/fivexfad.py::build_supporting_5xfad_slice
alz/viewer/cohorts/fivexfad.py::_write_fivexfad_attribution_shards
```

The unified viewer embeds a supporting block:

```text
PAYLOAD.supporting_5xfad
```

Attribution data is split to control first-load size:

- `rows`: compact bulk MEA rows for the selected tissue/assay/age surface.
- `celltype_attribution_summary_shard`: one gzipped whole-list index used for
  table filters and badges.
- `celltype_attribution_shards`: per-kinase JSON sidecars with full attribution
  rows, including confidence-basis strings and snRNA sample/cell counts.
- `celltype_mea_plot_index_shard`: compact per-cell-type decomposition MEA
  values for no-fetch plot/table views.
- `celltype_mea_shards` and `celltype_ols_shards`: per-kinase detail sidecars
  for full decomposition rows and substrate-site OLS.

### Frontend Flow

Frontend:

```text
alz/viewer/template/js/tabs/kinase_fivexfad.js  → AttributionView.render(host, ctx, F5_MANIFEST)
alz/viewer/template/js/tabs/attribution_manifest_fivexfad.js
alz/viewer_shared/template/js/tabs/attribution_view.js
```

The 5xFAD Attribution tab loads attribution and decomposition sidecars lazily
for the selected kinase group. It displays:

- bulk MEA anchor for the selected age;
- native matched 5xFAD snRNA location evidence across `new_clusters`;
- WMB and SEA-AD cross-reference evidence;
- per-cell-type 5xFAD decomposition MEA NES/FDR;
- decomposition agreement with the bulk NES direction;
- an inline accordion detail with snRNA counts, sample counts, reference
  evidence, decomposition evidence, and per-cell substrate-site OLS.

When no native 5xFAD snRNA attribution row is packaged for a kinase, the tab
falls back to decomposition-only rows and says so explicitly.

5xFAD confidence uses the unified exclusivity tier (`fivexfad_effective_n` over
`new_clusters` per tissue; corroborators: WMB class and SEA-AD at the home cell
type). The bulk-MEA significance flag and snRNA direction concordance are
preserved as info-only row fields (`bulk_mea_significant`, `direction_concordant`,
`decomp_agrees_bulk`) and displayed in the detail section, but they do not gate
the pill. `_f5CmpAttr` is retained in `kinase_fivexfad.js` because it is used
by 5xFAD master-list functions outside the attribution subtab.

## T-Cell Donor Attribution

### Producer Contract

Producer:

```text
alz/build_tcell_viewer.py::_build_tcell_attribution_index
```

The T-cell viewer emits a T-cell-specific `PAYLOAD.attribution_index`. It is not
schema-compatible with the Song AD `attribution_index`, even though the frontend
uses the same top-level key name.

Fields consumed by the T-cell frontend include:

| Field group | Columns |
|---|---|
| Identity | `kinase_id`, `contrast_id`, `cell_type` |
| Categorical calls | `confidence_tier`, `confidence_basis` |
| Within-cohort location | `tcell_detected`, `tcell_fraction_expressing`, `tcell_concentration`, `tcell_concentration_tier`, `tcell_effective_n`, `tcell_top_celltype`, `tcell_top_concentration` |
| Direction/concordance | `tcell_lfc`, `tcell_concordance`, `tcell_concordant`, `tcell_consistency`, `nes`, `fdr` |
| Independent reference | `nsclc_frac`, `nsclc_detected` |

Confidence uses the unified exclusivity tier (`tcell_effective_n` over
ProjecTILs states; corroborator: NSCLC detection at the crosswalked home state).
When the kinase is absent from the NSCLC panel, the row is treated as
uncorroborated and the tier caps at `moderate`. T-cell had no confidence pill
before the consolidation; the exclusivity pill is new as of S0.

Each row joins within-cohort ProjecTILs/state attribution to the NSCLC
reference detection at the crosswalked state when available. NSCLC detection is
an independent corroborator of the within-cohort attribution; it is reported as
detection evidence, not as a derived verdict flag.

### Frontend Flow

Frontend:

```text
alz/tcell_viewer/template/js/tabs/kinase_explorer.js::getScopedAttribution
alz/tcell_viewer/template/js/tabs/kinase_audit.js  → AttributionView.render(host, ctx, TCELL_MANIFEST)
alz/tcell_viewer/template/js/tabs/attribution_manifest_tcell.js
alz/viewer_shared/template/js/tabs/attribution_view.js
```

The T-cell Attribution subtab:

1. Reads the audit picker day and selected kinase.
2. Shows one row per cell-type state for the selected day.
3. Does not hide rows; all states remain visible and sorting only reorders.
4. Shows the unified confidence pill (NSCLC corroborator), within-cohort
   detection, concentration tier, day-vs-d2 transcript LFC, sign concordance
   with bulk NES, timecourse consistency, and NSCLC reference detection.
5. Opens an inline accordion with the within-cohort transcript trace across days
   and an NSCLC reference detection strip by lineage.

The T-cell view does not display WMB dot plot, SEA-AD heatmap, or decomp OLS —
those sections are absent at the manifest level. Direction LFC and concordance
are info-only. The viewer copy explains that biological replicates are not
available for single-donor p-values.

## Effective-Number Granularity Is Per-Cohort by Design

The `eff` value fed to the exclusivity tier formula uses a different unit set in
each cohort:

| Cohort | `eff` reads | Unit vocabulary |
|---|---|---|
| Song | `song_unit_effective_n` | inverse-Simpson over 17 curated specificity units |
| 5xFAD | `fivexfad_effective_n` | inverse-Simpson over `new_clusters` per tissue |
| T-cell | `tcell_effective_n` | inverse-Simpson over ProjecTILs states |

The tier formula is identical but a `very_high` pill is not numerically
comparable across cohorts. The manifest names the unit set per cohort so this is
explicit.

## Cross-Links and Historical Inputs

Current companion specs:

- `docs/foundation/specificity_confidence.md` explains the exclusivity pill,
  the shared `exclusivity_tier` helper, and per-cohort corroborators.
- `docs/foundation/viewer_payload_contract.md` defines the shared payload and
  lazy sidecar conventions.
- `docs/foundation/viewer_frontend_contract.md` defines shared versus
  intentionally forked frontend modules.

Historical inputs that this document supersedes for current behavior:

- `docs/plans/attribution/attribution_drawer_redesign.md`
- `docs/integrations/5xfad-kinase-mea-viewer.md` Attribution sections
- `docs/plans/todo2_tcell_specificity_reference.md` viewer-integration section
- `docs/audits/cohort_abstraction_refactor/phase_5A_payload_inventory.md`
  attribution inventory notes

Those files remain useful for implementation history and audit provenance, but
this document is the source of truth for the live Attribution tab architecture.
