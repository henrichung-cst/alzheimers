# "Pathways" detail tab in the T-cell Kinase Viewer

## Intent

The Kinase Explorer and the Incytr Pathways viewer surface the **same**
phospho-omics ground truth, organized kinase-first vs. pathway-first. We just
migrated the Explorer's `#pathways`/`#backbones` columns to the observed
terminal-edge rule so the two agree. This plan adds the drill-down the columns
imply: a new **"Pathways"** tab in the kinase detail pane (right-hand
`ke-detail`), at the same level as **Attribution**, that for the focused kinase
gives the summary statistics behind its two column counts and (on demand) the
observed terminal edges it draws into Incytr pathways.

This is the kinase-first inverse of the pathway-first sidechain graph: the
sidechain answers "for this pathway, which kinases edge into it"; this tab
answers "for this kinase, which pathway edges does it draw, and where."

## Scope

**In**
- New detail-pane tab `{id:"pathways", label:"Pathways"}` in `KINASE_AUDIT_TABS`
  + a render branch in `renderActiveKinaseAuditTab`. The tab is included **only
  when** the participation payload block is present (donor1); dropped otherwise.
- **Always-visible summary** (inlined, works under `file://`): headline
  `#pathways` / `#backbones` (matching the Explorer columns exactly) + three
  pathway-row breakdowns — by role (Receptor/EM/Target), by contrast, by
  receiver cluster.
- **Edge table, summary-first**: collapsed behind an expander; the full observed
  terminal-edge table renders on expand. Lists **only pathway-participating
  edges** (per-edge pathway count ≥ 1 — the sidechain-drawable set). Columns:
  `target_gene`, `role`, `contrast`, `receiver`, `pathways` (per-edge count),
  `signed_nes`, `best_fdr`, `n_sites`, `edge_delta`, `n_significant_concordant`,
  and a motif-peer `detected/informative` badge.
- **Lazy-load** the edge table per kinase over HTTP via the existing
  `AuditDataStore` mechanism (one sidecar table, sliced by kinase name
  client-side). Under `file://` the expander shows the summary + a "serve over
  HTTP for the edge table" note (same degradation as the audit context today).
- Each edge row cross-links into the Incytr Pathways tab, narrowed to the
  pathways that edge participates in.
- New payload block `kinase_incytr_participation` (donor1-only): the small,
  inlined per-kinase summary (counts + three breakdowns).

**Out** (explicitly)
- **No new Cytoscape / graph.** Tables + stats only. The sidechain graph stays
  the only pathway visualization; this tab links to it.
- **No donor2.** Kinase MEA is donor1-only; donor2 renders the existing
  "switch to donor1" message and never shows this tab.
- **No 5xFAD / Song viewer changes.**
- **No change to the two Explorer columns' values or the terminal-edge rule.**
  This tab consumes the same computation.
- **No count-0 edges.** Observed changes on a pathway node that no scored
  pathway instantiates at that `(contrast, receiver)` are excluded (they are
  never drawn in any sidechain). They remain in `terminal_edges.csv` / the audit
  layer; this tab does not surface them.
- **No per-site phosphosite table and no full motif-peer roster** in this tab.
  The heavy `sites`/`motif_peer_roster` JSON stay out; the edge row's cross-link
  into the sidechain is where that depth already renders. The badge shows
  detected/informative **counts** only.

## Where it plugs in (verified)

- Detail pane host `ke-detail` → `renderKinaseDetail(kinase_id)`
  (`kinase_detail.js:28`) builds the tab bar from `KINASE_AUDIT_TABS` and calls
  `renderActiveKinaseAuditTab`.
- Tabs list + dispatch: `KINASE_AUDIT_TABS` (`kinase_audit.js:28`),
  `renderActiveKinaseAuditTab` (`kinase_audit.js:824`) — one `else if` branch per
  tab; "attribution" is the last (`kinase_audit.js:937`). Tab body host is
  `kinase-audit-body`; `_loadKinaseAuditContext` yields `ctx` (has `ctx.name`).
- Lazy-load precedent: `AuditDataStore.load(tableKey)` (`01_state.js:400`) fetches
  a sidecar `meta.relative_path` once and caches; under `file://`
  (`location.protocol === "file:"`) it returns the inlined `meta.preview`. The
  audit manifest is `_auditManifest()`.
- Cross-nav precedent (`incytr_pathways.js:8`): the heatmap click handler writes
  `pair + senderIn + receiverIn + disease + timepoint` into `IncytrFilter` and
  switches tabs — reuse this exact field set. Top-level tab id for Incytr is
  `"incytr"` (`02_ui_chrome.js:46`); switch via
  `Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"incytr"})`. Force the
  table pane with `_setIncytrPane("table")` (`incytr_heatmap.js:903`). `searchText`
  matches a gene against Ligand/Receptor/EM/Target of each pathway, **any-role**,
  exact (`incytr_pathways.js:1162`).

## Data source

`outputs/reports/kinase_kinase_edges/tcells_donor1/terminal_edges.csv`
(78 MB — the bulk is the `sites` + `motif_peer_roster` JSON columns, both
excluded from this tab). **53,280 rows; the `(kinase, target_gene, role,
contrast, owning_cluster)` tuple is already unique** (max 1 row/tuple — no
aggregation). 324 kinases carry edges. Distinct edges/kinase: median 94, p90
363, p99 1352, **max 1554**. Role skew: Target 47,874 / EM 5,307 / Receptor 99.
Roles are Target/EM/Receptor only (no Ligand); `owning_cluster == receiver`.

Produced by `alz/cross_reference/kinase_kinase_edges.py` (`build_terminal_map`,
`load_motif_edges`): each kinase's significant+concordant *changed* substrate
sites mapped onto Incytr pathway nodes; a t-cell row is emitted only when its
direct-change bridge count ≥ 1. Already read by `_read_terminal_edges(donor)`
(`slices_incytr.py:866`).

## New behavior

### Summary header (inlined, always visible)

- **Headline**: `#pathways` (N, and N/total %) and `#backbones` (M), identical to
  the Explorer columns for this kinase (same source block).
- **By role**: pathway rows reached via Receptor / EM / Target. A pathway row
  reached at more than one role is counted under each — role-membership, not a
  partition (labeled as such). `#backbones` is the distinct Receptor∪EM union,
  not the sum.
- **By contrast**: pathway rows per T-cell contrast — partitions `#pathways`
  (each row has one contrast), sum == `#pathways`.
- **By receiver cluster**: pathway rows per receiver — partitions `#pathways`.

All breakdown values are **distinct pathway-row counts** (global-index rows),
consistent with the headline, computed in the same scan that produces the
counts. Small (role ≤ 3, contrast ≤ 9, receiver ≤ 31) → inlined.

### Edge table (summary-first, lazy, HTTP-only)

Collapsed behind an expander. On expand (over HTTP): fetch the edge sidecar via
`AuditDataStore`, slice to this kinase, render with `AuditTable` (sortable;
default sort `pathways` desc, tiebreak `|signed_nes|`). Only pathway-participating
edges (per-edge count ≥ 1). No cutoff otherwise — mirrors the sidechain. Under
`file://`, the expander shows the note; the summary above always renders.

`contrast` is stored in row form via `_terminal_contrast_to_row` to match the
CONTRASTS vocabulary used elsewhere.

Empty state: a kinase with 0 pathway-participating edges renders "no observed
pathway edges for this kinase" (its headline counts are 0).

### Cross-navigation

Clicking an edge row switches to the Incytr Pathways tab and narrows it to the
pathways that edge participates in. Because one edge maps to **many** pathway
rows, this is a **filter handoff, not a single-row selection**. Reuse the
heatmap field set:

```
IncytrFilter.set({ pair:null, ipMode:"top", senderIn:[],
                   receiverIn:[receiver], searchText:target_gene,
                   /* disease, timepoint — only if the edge contrast maps */ });
Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"incytr"});
_setIncytrPane("table");
```

`receiverIn` + `searchText`(gene) always applies (gene match is any-role, so it
does not pin the edge's specific role — acceptable; the role is shown in this
tab). Add `disease`+`timepoint` from the edge contrast **only if** the MEA
contrast (`D1_d13_vs_d2`, row form) maps cleanly to the t-cell Incytr
`(disease, timepoint)` axis (`_ipAxisParts` / `ViewerPayload.contrastAxis`) —
**verify at implementation**; if it doesn't map, omit contrast (receiver+gene
still lands the right shortlist). Do not add a per-role search lever.

## Compute (memory-safe)

One scan of the ~1.87M-row global index (extends the existing composite-key scan
in `_incytr_pathway_participation`, `slices_kinase.py` — no new pass), two
aggregations:

1. **Distinct-row masks (headline + breakdowns).** As today: per kinase, per role
   r∈{Receptor,EM,Target}, a boolean hit mask over index rows via `isin` on
   composite keys `(receiverId*nContrast+contrastId)*nGene+nodeId_r`. `#pathways`
   = |union of role masks|; `#backbones` = |Receptor∪EM|. `by_role` = per-role
   mask sums; `by_contrast`/`by_receiver` = `np.bincount` of the union mask's
   rows' `contrastId`/`receiverId` (both u1 arrays already read). Map ids→names
   via manifest vocabs. Tens of MB, bounded.
2. **Per-edge pathway count + participation filter.** Build, per role, a
   composite-key→count map over the index (`np.unique(..., return_counts=True)`).
   For each terminal edge `(gene, role, contrast_row, receiver)`, look up its key
   → per-edge count. **Drop edges with count 0.** The surviving edges (with count
   and the evidence fields from `_read_terminal_edges`, minus `sites`/`roster`)
   are the edge sidecar rows.

## Delivery / payload split

- **Inlined** in PAYLOAD (small): `payload["kinase_incytr_participation"]` =
  `{ name -> {counts:{pathways,backbones,total}, by_role, by_contrast,
  by_receiver} }`, donor1-only. Drives the always-visible summary. The Explorer
  columns are **not** re-sourced from this block — the extended scan populates
  both the existing columnar arrays (`K.incytr_pathway_count` /
  `incytr_backbone_count`, consumed unchanged by `kinase_explorer.js:205`) and
  this name-keyed block from one computation. The block adds only what is
  genuinely new (the three breakdowns); no Explorer JS is edited.
- **Sidecar** (lazy, HTTP): one edge table registered in the audit manifest
  (`relative_path` to a written CSV/JSON of the surviving edges across all
  kinases, with per-edge `pathways` count), fetched once via `AuditDataStore` and
  sliced by kinase name in the render branch. `preview` empty → `file://` shows
  the note. No data files committed to git (build output only).

## Files to change

| File | Change |
|---|---|
| `alz/tcell_viewer/slices_kinase.py` | Extend the `_incytr_pathway_participation` scan to also return per-kinase `by_role`/`by_contrast`/`by_receiver` (distinct-row) buckets and per-edge pathway counts; drop count-0 edges. Anti-shim: one scan, richer return — no parallel pass. |
| `alz/tcell_viewer/slices_incytr.py` | From `_read_terminal_edges(donor)` build the compact participating-edge rows (gene, role, contrast_row, receiver, per-edge `pathways`, signed_nes, best_fdr, n_sites, edge_delta, n_significant_concordant, motif_peers_detected/informative — no `sites`/`roster`) for the sidecar. Reuses `_terminal_contrast_to_row`. |
| `alz/build_tcell_viewer.py` | Assemble inlined `payload["kinase_incytr_participation"]` (donor1-gated on `mea_kinase_donor`); write the edge sidecar and register it in the audit manifest with `relative_path` (+ empty `preview`). The extended scan populates both the existing `incytr_pathway_count`/`incytr_backbone_count` columnar arrays (Explorer JS unchanged) and the new block — one computation, no rewire. Update the participation log line. |
| `alz/tcell_viewer/template/js/tabs/kinase_audit.js` | Conditionally add `{id:"pathways", label:"Pathways"}` to `KINASE_AUDIT_TABS` when the block is present; add the `else if (tab === "pathways")` branch: render summary (headline + 3 breakdowns) always; edge table behind an expander that lazy-loads via `AuditDataStore` and slices by `ctx.name` (note under `file://`); wire edge-row cross-link; empty-state for 0 participating edges. |

## Verification

- `pixi run` the T-cell viewer build, memory-capped
  (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`).
- Headline `#pathways`/`#backbones` in the tab == the Explorer column values for
  the same kinase. `by_contrast`/`by_receiver` each sum to `#pathways`; `by_role`
  union == `#pathways`; `#backbones` ≤ `#pathways`.
- Every listed edge has `pathways` ≥ 1; count-0 edges are absent. Sum of per-edge
  `pathways` ≥ `#pathways` (shared rows) — sanity, not equality.
- Spot-check one edge's cross-link: the narrowed Incytr table contains a pathway
  with that gene at that role, that receiver (and contrast if wired).
- Payload delta: inlined `kinase_incytr_participation` keeps `index.html` growth
  small (summary only); the edge sidecar is a separate file, not inlined. Report
  both sizes.
- Serve over HTTP: expander fetches + renders. Under `file://`: summary renders,
  expander shows the note. `node --check` the JS; other four tabs unaffected;
  hard-refresh after build.

## Resolved (grilling)

- Edge tuple already unique; no aggregation. Edges/kinase up to 1554 (median 94).
- Summary-first layout; full edge table behind an expander.
- Lazy-load edges over HTTP (`AuditDataStore`), sliced by kinase; `file://` shows
  summary + note. Summary (counts + breakdowns) is inlined so it always renders.
- Per-edge "in N pathways" count shown; table sorted by it.
- Pathway-participating edges only (count ≥ 1) — the sidechain-drawable set;
  count-0 edges excluded (mechanism documented under Data source).
- Cross-nav reuses the heatmap `IncytrFilter` field set; receiver+gene always,
  contrast only if mappable.
- Tab included only when the participation block exists; per-kinase empty state.

## To verify at implementation

- MEA edge-contrast (`D1_d13_vs_d2`) → t-cell Incytr `(disease, timepoint)` axis
  mapping for the cross-nav contrast narrowing (omit contrast if it doesn't map).
- Sidecar format/size and the `AuditDataStore` manifest entry shape (match an
  existing audit table's registration).
